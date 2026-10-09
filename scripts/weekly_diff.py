"""What the weekly re-check changed, as Markdown for a review issue (#6).

Compares the working copy of the re-generated data files with the last
commit, row by row, ignoring the columns that only record when a check ran.
Exit code 1 when anything changed.

    python scripts/weekly_diff.py > changes.md
"""

import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WATCHED = {  # file: (key, columns to ignore)
    "hf_facts.csv": ("hf_repo", {"fetched"}),
    "template_checks.csv": ("repo", {"checked"}),
    "terms_hashes.csv": ("url", {"checked", "bytes"}),
    "mlx_support.csv": ("id", {"checked", "mlx_vlm_rev"}),
    "llama_cpp_support.csv": ("id", {"checked"}),
}


def rows(text, key):
    return {r[key]: r for r in csv.DictReader(io.StringIO(text))}


def main():
    out, changed = [], False
    for name, (key, ignore) in WATCHED.items():
        path = ROOT / "data" / name
        if not path.exists():
            continue
        old_text = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:data/{name}"],
                                  capture_output=True, text=True).stdout
        old, new = rows(old_text, key), rows(path.read_text(), key)
        lines = []
        for k in sorted(set(old) | set(new)):
            a, b = old.get(k), new.get(k)
            if a is None or b is None:
                lines.append(f"- `{k}`: {'new' if a is None else 'gone'}")
                continue
            diffs = [f"{c}: `{a.get(c, '')[:80]}` → `{b.get(c, '')[:80]}`"
                     for c in b if c not in ignore and a.get(c, "") != b.get(c, "")]
            if diffs:
                lines.append(f"- `{k}`: " + "; ".join(diffs))
        if lines:
            changed = True
            out += [f"### {name}", ""] + lines + [""]
    print("\n".join(out) if out else "No changes in the watched files.")
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
