"""Check which models the current llama.cpp release can convert to GGUF.

Downloads the converter sources (conversion/*.py) of the latest llama.cpp release,
collects the Hugging Face architectures they register, and compares them with each
model's `architectures` entry (data/hf_facts.csv). Writes data/llama_cpp_support.csv.
GGUF builds of unsupported architectures (data/gguf_builds.csv) are flagged at build
time: they need a patched llama.cpp.

Limits: "supported" means the converter registers the architecture; it does not
prove that every variant converts or runs.
"""

import csv
import datetime as dt
import json
import os
import re
import urllib.request

from scorecard import DATA, read

OUT = DATA / "llama_cpp_support.csv"
FIELDS = ["id", "llama_cpp", "architecture", "detail", "checked", "llama_cpp_rev"]
REPO = "ggml-org/llama.cpp"


def gh(url):
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read().decode("utf-8")


def converter_sources(tag):
    tree = json.loads(gh(f"https://api.github.com/repos/{REPO}/git/trees/{tag}?recursive=1"))
    paths = [t["path"] for t in tree["tree"]
             if t["path"] == "convert_hf_to_gguf.py" or re.fullmatch(r"conversion/.+\.py", t["path"])]
    return "\n".join(gh(f"https://raw.githubusercontent.com/{REPO}/{tag}/{p}") for p in paths)


def main():
    tag = json.loads(gh(f"https://api.github.com/repos/{REPO}/releases/latest"))["tag_name"]
    src = converter_sources(tag)
    registered = set()
    for m in re.finditer(r"register\(([^)]*)\)", src):
        registered.update(re.findall(r'"([A-Za-z0-9_]+)"', m.group(1)))
    facts = {r["hf_repo"]: r for r in read("hf_facts.csv")}
    today = dt.date.today().isoformat()
    rows = []
    for m in read("scorecard.csv"):
        f = facts.get(m["hf_repo"], {})
        arch = f.get("architecture", "")
        if not arch:
            status, detail = "not checked", "config.json not readable (gated?)"
        elif arch not in registered:
            status, detail = "unsupported", "architecture not registered in the converter"
        else:
            status, detail = "supported", ""
        rows.append(dict(id=m["id"], llama_cpp=status, architecture=arch, detail=detail, checked=today, llama_cpp_rev=tag))
        print(f"{status:12s} {m['id']:22s} {arch:38s} {detail}")
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"llama.cpp {tag}: {len(registered)} registered architectures")


if __name__ == "__main__":
    main()
