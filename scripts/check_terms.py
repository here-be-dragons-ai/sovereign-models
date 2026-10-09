"""Hash every licence and use-policy document the scorecard relies on (#6).

Two kinds of documents: those cited as sources for license_class or
terms_can_change (licence files, use policies, legal pages), and the licence
files in every model's own repo (LICENSE*, USE_POLICY*, NOTICE*). Files are
hashed as published; HTML pages by their visible text, so layout changes do
not count. A changed hash means the terms need a review.

    python scripts/check_terms.py      # writes data/terms_hashes.csv
"""

import csv
import datetime as dt
import hashlib
import html
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "terms_hashes.csv"
FIELDS = ["url", "kind", "sha256", "bytes", "checked", "error"]
TERMS_FILE = re.compile(r"^(LICEN[CS]E|USE_POLICY|USAGE_POLICY|NOTICE|ACCEPTABLE_USE)", re.I)


def _token():
    path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return path.read_text().strip() if path.exists() else os.environ.get("HF_TOKEN")


TOKEN = _token()


def _get(url):
    h = {"User-Agent": "sovereign-models/check_terms"}
    if TOKEN and "huggingface.co" in url:
        h["Authorization"] = f"Bearer {TOKEN}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        return r.read(), r.headers.get("Content-Type", "")


def raw_url(url):
    """The downloadable form of a cited document; None for pages we do not watch."""
    if "github.com/" in url and "/blob/" in url:
        return url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")
    if "huggingface.co/" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    if re.match(r"https://huggingface\.co/[^/]+/[^/]+/?$", url):
        return None  # a model card: covered by the Hub facts, not a terms document
    return url


def visible_text(body):
    t = body.decode("utf-8", "replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>|<nav.*?</nav>|<footer.*?</footer>", " ", t, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip().encode()


def documents():
    urls = set()
    for s in csv.DictReader(open(ROOT / "data" / "sources.csv")):
        if s["field"] in ("license_class", "terms_can_change") and raw_url(s["url"]):
            urls.add(s["url"])
    for m in csv.DictReader(open(ROOT / "data" / "scorecard.csv")):
        try:
            info, _ = _get(f"https://huggingface.co/api/models/{m['hf_repo']}")
        except urllib.error.HTTPError:
            continue
        for f in json.loads(info).get("siblings", []):
            if "/" not in f["rfilename"] and TERMS_FILE.match(f["rfilename"]):
                urls.add(f"https://huggingface.co/{m['hf_repo']}/blob/main/{f['rfilename']}")
    return sorted(urls)


def main():
    today = dt.date.today().isoformat()
    rows = []
    for url in documents():
        row = dict(url=url, checked=today)
        try:
            body, ctype = _get(raw_url(url))
            if "html" in ctype:
                body, row["kind"] = visible_text(body), "html text"
            else:
                row["kind"] = "file"
            row.update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
        except (urllib.error.URLError, TimeoutError) as e:
            row["error"] = str(e)
        rows.append(row)
        print(f"{row.get('sha256', 'ERROR')[:12]:12s} {url}", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
