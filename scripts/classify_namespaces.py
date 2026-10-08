"""Record whether each Hugging Face namespace we cite is an organisation or a user.

Collects namespaces from model repos, MLX builds, sources and the providers'
official organisations, asks the Hub API, and writes data/hf_namespaces.csv.
validate.py and build.py read that cache, so CI needs no network access.

    python3 scripts/classify_namespaces.py
"""

import csv
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard import DATA, Data, split  # noqa: E402

FIELDS = ["namespace", "kind", "verified", "fullname", "checked"]


def _token():
    path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return path.read_text().strip() if path.exists() else os.environ.get("HF_TOKEN")


TOKEN = _token()


def api(path):
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    req = urllib.request.Request(f"https://huggingface.co/api/{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def classify(ns):
    for kind, path in (("org", f"organizations/{ns}/overview"), ("user", f"users/{ns}/overview")):
        try:
            info = api(path)
            return kind, str(bool(info.get("isVerified"))).lower(), info.get("fullname", "")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    return "unknown", "", ""


def namespaces(d):
    found = set()
    for m in d.models:
        found.add(m["hf_repo"].split("/")[0])
    for b in d.builds:
        found.add(b["repo"].split("/")[0])
    for p in d.providers.values():
        found.update(split(p.get("hf_orgs")))
    for s in d.sources:
        u = urlparse(s["url"])
        parts = [p for p in u.path.split("/") if p]
        if u.netloc == "huggingface.co" and parts:
            if parts[0] in ("datasets", "spaces") and len(parts) > 1:
                found.add(parts[1])
            elif parts[0] == "api" and len(parts) > 2:
                found.add(parts[2])
            elif parts[0] not in ("api", "collections", "papers", "docs"):
                found.add(parts[0])
    return sorted(found)


def main():
    d = Data()
    today = dt.date.today().isoformat()
    rows = []
    for ns in namespaces(d):
        kind, verified, fullname = classify(ns)
        rows.append(dict(namespace=ns, kind=kind, verified=verified, fullname=fullname, checked=today))
        print(f"{kind:8s} {'verified' if verified == 'true' else '':9s} {ns}  {fullname}")
    with open(DATA / "hf_namespaces.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
