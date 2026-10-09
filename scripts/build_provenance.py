"""Provenance of our own builds (#7, 3c): source and output files with SHA-256.

For every build in data/build_recipes.csv this records the exact revision of
the source repo and of the build repo and the SHA-256 of every file in both:
large files from the Hub's LFS pointer (which is their SHA-256), small files
by downloading and hashing them. The recipe itself (command, tool versions,
scripts, whether a reproduction matched) is kept by hand in build_recipes.csv.

    python scripts/build_provenance.py                        # writes data/build_provenance.csv
    python scripts/build_provenance.py --card <build repo>    # Markdown section for the model card
"""

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "data" / "build_recipes.csv"
OUT = ROOT / "data" / "build_provenance.csv"
FIELDS = ["build", "role", "repo", "revision", "path", "size", "sha256"]
SKIP = {".gitattributes"}


def _token():
    path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return path.read_text().strip() if path.exists() else os.environ.get("HF_TOKEN")


TOKEN = _token()


def _get(url):
    h = {"User-Agent": "sovereign-models/build_provenance"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        return r.read()


def files(repo, revision, role):
    """(path, size, sha256) of every file at a revision; the revision as a full commit id."""
    rev = json.loads(_get(f"https://huggingface.co/api/models/{repo}/revision/{revision}"))["sha"]
    tree = json.loads(_get(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"))
    out = []
    for e in tree:
        if e["type"] != "file" or e["path"] in SKIP:
            continue
        if role == "output" and e["path"] == "README.md":
            continue  # the card carries this list; hashing it would chase its own tail
        lfs = e.get("lfs")
        sha = lfs["oid"] if lfs else hashlib.sha256(
            _get(f"https://huggingface.co/{repo}/resolve/{rev}/{e['path']}")).hexdigest()
        out.append((e["path"], e.get("size") or (lfs or {}).get("size"), sha))
    return rev, sorted(out)


def collect():
    rows = []
    for r in csv.DictReader(open(RECIPES)):
        for role in ("source", "output"):
            rev, fl = files(r[f"{role}_repo"], r[f"{role}_revision"] or "main", role)
            for path, size, sha in fl:
                rows.append(dict(build=r["build"], role=role, repo=r[f"{role}_repo"],
                                 revision=rev, path=path, size=size, sha256=sha))
            print(f"{r['build']:45s} {role:6s} {rev[:12]}  {len(fl)} files", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def card(build):
    recipe = next(r for r in csv.DictReader(open(RECIPES)) if r["build"] == build)
    rows = [r for r in csv.DictReader(open(OUT)) if r["build"] == build]
    src = [r for r in rows if r["role"] == "source"]
    out = [r for r in rows if r["role"] == "output"]
    lines = [
        "## Provenance",
        "",
        f"- Source: [{recipe['source_repo']}](https://huggingface.co/{recipe['source_repo']}) "
        f"at revision `{src[0]['revision']}`",
        f"- Command: `{recipe['command']}`",
        f"- Tools: {recipe['tools']}",
        f"- Reproduced: {recipe['reproduced']}",
        "",
        "SHA-256 of the files of this build (the Hub stores the same value as the LFS object id):",
        "",
        "| file | size | SHA-256 |",
        "|---|---:|---|",
    ]
    lines += [f"| `{r['path']}` | {int(r['size']):,} | `{r['sha256']}` |" for r in out]
    lines += ["", f"Source files and their hashes: `data/build_provenance.csv` in the "
              "[sovereign-models-scorecard](https://huggingface.co/datasets/here-be-dragons-ai/"
              "sovereign-models-scorecard) dataset."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", help="print the model-card section for this build repo")
    args = ap.parse_args()
    if args.card:
        print(card(args.card))
    else:
        collect()


if __name__ == "__main__":
    main()
