"""Human spot check of the sources: a stratified random sample as a review sheet.

Every source was read and checked once (partly by AI agents). A person checks a
random sample again; the share of sources they confirm is the data-quality
figure in METHODOLOGY.md.

    python3 scripts/review_sample.py sheet [--n 40] [--seed 2026]   # Markdown checklist to stdout
    python3 scripts/review_sample.py apply review.md --reviewer NAME  # ticked boxes -> data/review_sample.csv
    python3 scripts/review_sample.py stats

The sample is stratified by issuer and scope, so rare kinds (official, academic,
press, provider-scope facts) are not left out; only checked sources are drawn.
In the sheet, tick a box if the source states the value; leave it unticked and
add a note after "--" if it does not.
"""

import argparse
import csv
import random
import re
from collections import defaultdict

from rubric import quote_key
from scorecard import DATA, Data

OUT = DATA / "review_sample.csv"
FIELDS = ["date", "reviewer", "scope", "subject", "field", "url", "value", "agrees", "comment"]


def values(d):
    """The stored value a source backs, per (scope, subject, field)."""
    out = {}
    for p in d.providers.values():
        for k, v in p.items():
            out[("provider", p["id"], k)] = v
    for f in d.families.values():
        for k, v in f.items():
            out[("family", f["id"], k)] = v
    for m in d.models:
        for k, v in m.items():
            out[("model", m["id"], k)] = v
    for t in d.teachers:
        out[("family", t["family_id"], "teacher:" + t["model"])] = f"{t['stage']}, {t['model_origin']}: {t['purpose']}"
    for e in d.evidence:
        out[(e["scope"], e["subject"], "q:" + e["question"])] = e["answer"]
    return out


def evidence_items(d):
    """Rubric answers as sample items: the reviewer checks that the page holds the
    quote (or, for "not stated", really says nothing) and that it supports the answer."""
    out = []
    for e in d.evidence:
        status = d.quote_status.get(quote_key(e["url"], e["quote"]), "not checked") if e["quote"] else "no quote"
        note = f'quote: "{e["quote"]}" [{status}]' if e["quote"] else "absence: the page should not state it"
        out.append(dict(scope=e["scope"], subject=e["subject"], field="q:" + e["question"], url=e["url"],
                        issuer="evidence", checked="yes", note=note))
    return out


def cmd_sheet(args):
    d = Data()
    val = values(d)
    strata = defaultdict(list)
    for s in d.sources + evidence_items(d):
        if s["checked"] == "yes":
            strata[(s["issuer"], s["scope"])].append(s)
    rnd = random.Random(args.seed)
    total = sum(map(len, strata.values()))
    sample = []
    for key, rows in sorted(strata.items()):
        k = max(1, round(args.n * len(rows) / total))
        sample += rnd.sample(rows, min(k, len(rows)))
    rnd.shuffle(sample)
    print(f"# Source spot check ({len(sample)} sources, seed {args.seed})\n")
    print("Tick a box if the linked page states the value. Otherwise leave it unticked and write "
          "what the page says after `--`. For rubric answers (`q:...`, issuer evidence), tick if the "
          "page holds the quote and the quote supports the answer; for an absence, if the page "
          "indeed does not state it.\n")
    for i, s in enumerate(sample, 1):
        v = val.get((s["scope"], s["subject"], s["field"]), "")
        print(f"- [ ] {i}. `{s['scope']}/{s['subject']}/{s['field']}` = **{v}** — {s['url']} "
              f"({s['issuer']}; note: {s['note'][:300 if s['issuer'] == 'evidence' else 120]}) --")


def cmd_apply(args):
    import datetime as dt

    d = Data()
    val = values(d)
    by_key = {(s["scope"], s["subject"], s["field"], s["url"]): s for s in d.sources + evidence_items(d)}
    line = re.compile(r"- \[( |x|X)\] \d+\. `([^/]+)/([^/]+)/([^`]+)` = \*\*(.*?)\*\* — (\S+) \(.*?\) --(.*)$")
    rows = []
    for raw in open(args.sheet, encoding="utf-8"):
        m = line.match(raw.strip())
        if not m:
            continue
        tick, scope, subject, field, value, url, comment = m.groups()
        if (scope, subject, field, url) not in by_key:
            print(f"skip (source no longer listed): {scope}/{subject}/{field} {url}")
            continue
        rows.append(dict(date=dt.date.today().isoformat(), reviewer=args.reviewer, scope=scope, subject=subject,
                         field=field, url=url, value=val.get((scope, subject, field), value),
                         agrees="yes" if tick.lower() == "x" else "no", comment=comment.strip()))
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} reviews added to {OUT}")
    cmd_stats(args)


def cmd_stats(args):
    if not OUT.exists():
        print("no reviews yet")
        return
    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    for label, part in (("sources", [r for r in rows if not r["field"].startswith("q:")]),
                        ("rubric answers", [r for r in rows if r["field"].startswith("q:")])):
        if part:
            ok = sum(r["agrees"] == "yes" for r in part)
            print(f"{ok} of {len(part)} sampled {label} confirmed by a second reviewer ({ok / len(part):.0%})")
    for r in rows:
        if r["agrees"] != "yes":
            print(f"  disagrees: {r['scope']}/{r['subject']}/{r['field']} = {r['value']} — {r['comment']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sheet")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("apply")
    p.add_argument("sheet")
    p.add_argument("--reviewer", required=True)
    sub.add_parser("stats")
    args = ap.parse_args()
    {"sheet": cmd_sheet, "apply": cmd_apply, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
