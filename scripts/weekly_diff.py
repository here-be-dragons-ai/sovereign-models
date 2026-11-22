"""What the weekly re-check changed, as Markdown for the review issue (#6).

Compares the working copy of the re-generated data files with the last commit, row by
row, ignoring the columns that only record when a check ran. The first part is what
needs a look. The second (--record) is what is only noted: repos whose commit moved
without a change in the tracked facts, and results that were kept because a document
could not be fetched in this run.

    python scripts/weekly_diff.py > changes.md           # exit code 1 when anything differs
    python scripts/weekly_diff.py --record > record.md
    python scripts/weekly_diff.py mlx_support.csv        # only these files
"""

import csv
import datetime as dt
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WATCHED = {  # file: (title, key, columns to ignore), in the order of the report
    "terms_hashes.csv": ("Licence and policy documents", "url", {"checked", "bytes"}),
    "quote_checks.csv": ("Quotes of the rubric answers", "key", {"checked", "method"}),
    "template_checks.csv": ("Chat templates", "repo", {"checked"}),
    "hf_facts.csv": ("Hub facts", "hf_repo", {"fetched"}),
    "mlx_support.csv": ("mlx-vlm support", "id", {"checked", "mlx_vlm_rev"}),
    "llama_cpp_support.csv": ("llama.cpp support", "id", {"checked", "llama_cpp_rev"}),
}
# Checks that keep the last result when a document cannot be fetched (verify_quotes.py,
# check_terms.py): such a row still carries the date of its last real check.
KEPT = ("terms_hashes.csv", "quote_checks.csv")
STALE_DAYS = 14  # a kept result older than this makes the run count as changed
EXAMPLES = 3     # builds named per group of chat-template changes


def rows(text, key):
    return {r[key]: r for r in csv.DictReader(io.StringIO(text))}


def differences(a, b, ignore):
    return [(c, a.get(c, ""), b.get(c, "")) for c in b if c not in ignore and a.get(c, "") != b.get(c, "")]


def line(key, diffs):
    return f"- `{key}`: " + "; ".join(f"{c}: `{x[:80]}` → `{y[:80]}`" for c, x, y in diffs)


def plain(old, new, ignore):
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if a is None or b is None:
            out.append(f"- `{k}`: {'new' if a is None else 'gone'}")
        elif differences(a, b, ignore):
            out.append(line(k, differences(a, b, ignore)))
    return out


def hub_facts(old, new, ignore):
    """(lines, moved): a repo whose commit moved while nothing else we track changed is only noted."""
    out, moved = [], []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if a is None or b is None:
            out.append(f"- `{k}`: {'new' if a is None else 'gone'}")
            continue
        diffs = differences(a, b, ignore)
        if [c for c, _, _ in diffs] == ["revision"]:
            moved.append((k, a["revision"], b["revision"]))
        elif diffs:
            out.append(line(k, diffs))
    return out, moved


def templates(old, new, ignore):
    """Chat-template changes per model: one original that changes its template turns all its
    builds to `differs`, so builds are counted per model and change, with a few named."""
    out, groups, findings = [], {}, []
    originals = set()
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        mid = (b or a)["id"]
        if a is None or b is None:
            groups.setdefault((mid, "new" if a is None else "gone"), []).append(k)
        elif b["status"] == "reference":
            if a["sha256"] != b["sha256"]:
                originals.add(mid)
                out.append(f"- `{mid}`: the original's chat template changed (`{a['sha256']}` → `{b['sha256']}`)")
        elif a["status"] != b["status"]:
            groups.setdefault((mid, f"`{a['status']}` → `{b['status']}`"), []).append(k)
        elif differences(a, b, ignore):
            groups.setdefault((mid, "with a changed template, same status"), []).append(k)
        if a and b and a.get("findings", "") != b.get("findings", ""):
            findings.append(f"- `{k}` ({mid}): findings: `{a['findings'][:80]}` → `{b['findings'][:80]}`")
    for (mid, what), repos in sorted(groups.items()):
        named = ", ".join(f"`{r}`" for r in repos[:EXAMPLES])
        more = f" and {len(repos) - EXAMPLES} more" if len(repos) > EXAMPLES else ""
        why = " (the original's template changed)" if mid in originals and what not in ("new", "gone") else ""
        out.append(f"- `{mid}`: {len(repos)} build{'s' if len(repos) != 1 else ''} {what}{why}: {named}{more}")
    return out + findings


def kept(new):
    """Rows not re-checked in this run, as (url, last check, days behind the newest check)."""
    dates = [r["checked"] for r in new.values() if r.get("checked")]
    if not dates:
        return []
    latest = dt.date.fromisoformat(max(dates))
    return [(r["url"], r["checked"], (latest - dt.date.fromisoformat(r["checked"])).days)
            for r in new.values() if r.get("checked") and r["checked"] < max(dates)]


def report(old, new, only=()):
    """(what needs a look, what is only noted, anything changed) for {file: {key: row}} before and after."""
    look, record, changed = [], [], False
    for name, (title, _, ignore) in WATCHED.items():
        if name not in new or (only and name not in only):
            continue
        a, b = old.get(name, {}), new[name]
        if name == "hf_facts.csv":
            lines, moved = hub_facts(a, b, ignore)
            if moved:
                changed = True
                record += [f"### Repos at a new commit, tracked facts unchanged ({len(moved)})", "",
                           "<details><summary>list</summary>", ""]
                record += [f"- `{k}`: `{x[:12]}` → `{y[:12]}`" for k, x, y in moved] + ["", "</details>", ""]
        elif name == "template_checks.csv":
            lines = templates(a, b, ignore)
        else:
            lines = plain(a, b, ignore)
        if lines:
            changed = True
            look += [f"### {title} (`{name}`)", ""] + lines + [""]
        if name in KEPT and kept(b):
            changed = changed or any(days > STALE_DAYS for _, _, days in kept(b))
            record += [f"### {title}: not fetched in this run, last result kept ({len(kept(b))})", ""]
            record += [f"- `{url}`: last checked {day}" for url, day, _ in sorted(set(kept(b)))] + [""]
    return look, record, changed


def load():
    old, new = {}, {}
    for name, (_, key, _) in WATCHED.items():
        path = ROOT / "data" / name
        if not path.exists():
            continue
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:data/{name}"],
                              capture_output=True, text=True).stdout
        old[name], new[name] = rows(text, key), rows(path.read_text(), key)
    return old, new


def main():
    look, record, changed = report(*load(), only=[a for a in sys.argv[1:] if not a.startswith("--")])
    if "--record" in sys.argv:
        print("\n".join(record))
    else:
        print("\n".join(look) if look else "Nothing in the watched files needs a look.")
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
