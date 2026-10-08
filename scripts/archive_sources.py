"""Add a Wayback Machine snapshot to every source in data/sources.csv that has none.

Asks the Wayback Machine to capture the page now (Save Page Now); if that fails,
falls back to the most recent existing snapshot. Requests are spaced out to
respect the rate limit. Rows are written back after every URL, so the script
can be stopped and resumed.

    python3 scripts/archive_sources.py [--delay 10]
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from scorecard import DATA

UA = {"User-Agent": "sovereign-models-scorecard/0.1 (+https://github.com/here-be-dragons-ai)"}


def save(url):
    req = urllib.request.Request("https://web.archive.org/save/" + url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        final = r.geturl()
    return final if "/web/" in final else None


def latest(url):
    q = "https://archive.org/wayback/available?url=" + urllib.parse.quote(url, safe="")
    with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=60) as r:
        snap = json.load(r).get("archived_snapshots", {}).get("closest", {})
    return snap.get("url", "").replace("http://", "https://", 1) or None


def write(rows, fields):
    with open(DATA / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=10)
    args = ap.parse_args()
    with open(DATA / "sources.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    done = {r["url"]: r["archive_url"] for r in rows if r["archive_url"]}
    # New rows citing an already archived URL reuse its snapshot.
    for r in rows:
        if not r["archive_url"] and r["url"] in done:
            r["archive_url"] = done[r["url"]]
    write(rows, fields)
    todo = sorted({r["url"] for r in rows if not r["archive_url"] and r["url"] not in done})
    print(f"{len(todo)} URLs to archive")
    for i, url in enumerate(todo, 1):
        snap = None
        try:
            snap = save(url)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  save failed ({e}); trying the latest snapshot")
        if not snap:
            try:
                snap = latest(url)
            except (urllib.error.URLError, TimeoutError):
                snap = None
        if snap:
            snap = snap.replace("http://", "https://", 1)
            done[url] = snap
            for r in rows:
                if r["url"] == url:
                    r["archive_url"] = snap
            write(rows, fields)
        print(f"[{i}/{len(todo)}] {'ok  ' if snap else 'FAIL'} {url}")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
