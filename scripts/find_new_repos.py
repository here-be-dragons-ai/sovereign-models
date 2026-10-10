"""New model repos of the providers we track (#6): new sizes, versions, families.

Lists the models of every provider's Hub organisations (providers.csv,
hf_orgs) and compares them with data/known_repos.csv. New repos are printed
and appended there, so each one is reported once; the weekly check turns them
into a review issue. Quantized copies inside a provider org count too, since a
provider's own GGUF or MLX build is worth knowing about.

    python scripts/find_new_repos.py            # report and remember new repos
    python scripts/find_new_repos.py --init     # remember everything, report nothing
"""

import argparse
import csv
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SM_DATA", ROOT / "data"))  # SM_DATA: another data directory, e.g. a local overlay
KNOWN = DATA / "known_repos.csv"
FIELDS = ["repo", "provider_id", "created", "first_seen"]


def _token():
    path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return path.read_text().strip() if path.exists() else os.environ.get("HF_TOKEN")


TOKEN = _token()


def models(org):
    q = urllib.parse.urlencode({"author": org, "limit": 1000, "full": "false"})
    h = {"User-Agent": "sovereign-models/find_new_repos"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(f"https://huggingface.co/api/models?{q}", headers=h)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    args = ap.parse_args()
    known = {r["repo"] for r in csv.DictReader(open(KNOWN))} if KNOWN.exists() else set()
    today = dt.date.today().isoformat()
    new = []
    for p in csv.DictReader(open(DATA / "providers.csv")):
        for org in filter(None, (p["hf_orgs"] or "").split(";")):
            for m in models(org):
                if m["id"] not in known:
                    new.append(dict(repo=m["id"], provider_id=p["id"],
                                    created=(m.get("createdAt") or "")[:10], first_seen=today))
                    known.add(m["id"])
    write_header = not KNOWN.exists()
    with open(KNOWN, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        if write_header:
            w.writeheader()
        w.writerows(sorted(new, key=lambda r: r["repo"]))
    if not args.init:
        for r in sorted(new, key=lambda r: (r["provider_id"], r["repo"])):
            print(f"- [{r['repo']}](https://huggingface.co/{r['repo']}) ({r['provider_id']}, created {r['created']})")
    print(f"{len(new)} new repos", flush=True)


if __name__ == "__main__":
    main()
