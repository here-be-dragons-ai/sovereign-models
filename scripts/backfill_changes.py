"""Rebuild data/changes.csv from the git history of dist/scorecard.csv (#6).

Every commit that changed dist/scorecard.csv is compared with the one before;
the reason is the commit's subject and short hash. Run once; later changes are
appended by build.py.

    python scripts/backfill_changes.py
"""

import csv
import io
import subprocess

from build import CHANGE_FIELDS, CHANGES, ROOT, diff_rows


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def main():
    commits = git("log", "--reverse", "--format=%h|%ad|%s", "--date=short", "--",
                  "dist/scorecard.csv").splitlines()
    prev, out = [], []
    for line in commits:
        sha, date, subject = line.split("|", 2)
        rows = list(csv.DictReader(io.StringIO(git("show", f"{sha}:dist/scorecard.csv"))))
        out += diff_rows(prev, rows, date, f"{subject} ({sha})")
        prev = rows
    with open(CHANGES, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, CHANGE_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} changes from {len(commits)} commits")


if __name__ == "__main__":
    main()
