"""Find GGUF quantizations of each model on the Hub and write data/gguf_builds.csv.

Uses the Hub's base-model link (`base_model:quantized:<repo>`), so only repos
that declare the original model as their base are found. A repo is kept only if
its name contains the model name (fine-tunes that point to the base are dropped)
and it actually contains a .gguf file (tags alone are not reliable). The publisher class
(official, curated, organisation, individual) is derived at build time from
data/providers.csv and data/hf_namespaces.csv, as for MLX builds.
"""

import csv
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from scorecard import DATA, read

OUT = DATA / "gguf_builds.csv"
FIELDS = ["id", "repo", "downloads", "likes", "created"]


def _token():
    path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return path.read_text().strip() if path.exists() else os.environ.get("HF_TOKEN")


TOKEN = _token()


def search(base):
    q = urllib.parse.urlencode([("filter", f"base_model:quantized:{base}"), ("filter", "gguf"),
                                ("sort", "downloads"), ("limit", "100"), ("full", "true")])
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    req = urllib.request.Request(f"https://huggingface.co/api/models?{q}", headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


# Name markers of modified weights (not plain quantizations) that still declare
# the original as their quantized base.
MODIFIED = re.compile(r"abliterat|uncensor|heretic|merge|lora|distill|finetun|[-_]rp\d*\b", re.I)


def same_model(original, candidate):
    """Quantizations keep the model name; fine-tunes that also point to the base
    (e.g. role-play merges) usually do not, and modified weights are named so."""
    stem = re.split(r"[-_]", original.split("/")[1].lower())[0]
    name = candidate.split("/")[1]
    return stem in name.lower() and not MODIFIED.search(name)


def main():
    rows = []
    for m in read("scorecard.csv"):
        found = [f for f in search(m["hf_repo"])
                 if same_model(m["hf_repo"], f["id"])
                 and any(x["rfilename"].endswith(".gguf") for x in f.get("siblings", []))]
        for f in found:
            rows.append(dict(id=m["id"], repo=f["id"], downloads=f.get("downloads", 0),
                             likes=f.get("likes", 0), created=(f.get("createdAt") or "")[:10]))
        print(f"{len(found):3d} GGUF  {m['id']}")
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
