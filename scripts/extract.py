"""Answer the rubric questions for a family: a worksheet out, checked answers in.

Whoever reads the sources, a person or a language model, gets a worksheet with
the atomic questions of scripts/rubric.py and the URLs already cited for the
family, and returns one JSON object per answer:

    {"scope": "family", "subject": "<id>", "question": "<id>", "answer": "...",
     "url": "https://...", "quote": "verbatim text from the page"}

The reader never sets a value: values follow from the answers (rubric.derive).
`import` checks every answer (known question, allowed value, https URL, a quote
where the rubric needs one) and replaces the subject's earlier answers to the
same questions in data/evidence.csv. Run verify_quotes.py afterwards.

    python scripts/extract.py sheet <family-id> [...]   # worksheet (Markdown) to stdout
    python scripts/extract.py import answers.json [--by NAME]
    python scripts/extract.py status [<family-id> ...]   # derived vs curated values
"""

import argparse
import csv
import datetime as dt
import json
import sys

from rubric import FIELDS, QUESTIONS, check_answer, needs_quote
from scorecard import DATA, Data, read

OUT = DATA / "evidence.csv"
COLUMNS = ["scope", "subject", "question", "answer", "url", "quote", "retrieved", "by"]

INSTRUCTIONS = """\
Answer each question from the provider's own documents (model card in its Hub
organisation, website, technical report, licence), from official documents, or from
scholarly or press sources. Rules:

- Quote verbatim: copy the sentence or table cell exactly as it appears on the page
  (a short span is enough; skip text with "..."). Do not paraphrase, translate or fix
  typos. A quote that cannot be found on the page does not count.
- One answer per object; give several objects when several sources or sites apply.
- If the documents do not say, answer "not stated" (or "no" for yes/no questions) and
  give the URL of the document you read, with an empty quote.
- Answer only what the source says. Do not infer the value of a field; it is computed
  from the answers.
"""


def cmd_sheet(args):
    d = Data()
    print(INSTRUCTIONS)
    print("Questions (id: question [allowed answers]):\n")
    for q, (field, text, allowed, _) in QUESTIONS.items():
        print(f"- `{q}` ({field}): {text}" + (f" [{', '.join(sorted(allowed))}]" if allowed else ""))
    for fid in args.families:
        f = d.families.get(fid)
        if not f:
            sys.exit(f"unknown family {fid!r}")
        models = [m for m in d.models if m["family_id"] == fid]
        urls = sorted({s["url"] for s in d.sources if (s["scope"], s["subject"]) == ("family", fid)
                       or (s["scope"] == "model" and s["subject"] in {m["id"] for m in models})})
        print(f"\n## family `{fid}`: {f['name']} ({d.providers[f['provider_id']]['name']})\n")
        print("Models: " + ", ".join(f"{m['name']} (https://huggingface.co/{m['hf_repo']})" for m in models))
        print("\nDocuments already cited for this family (start here; others are welcome):\n")
        for u in urls:
            print(f"- {u}")


def cmd_import(args):
    answers = json.load(open(args.file, encoding="utf-8"))
    d = Data()
    subjects = {"family": set(d.families), "model": {m["id"] for m in d.models}}
    errors, rows = [], []
    for i, a in enumerate(answers, 1):
        a = {k: str(a.get(k, "")).strip() for k in ("scope", "subject", "question", "answer", "url", "quote")}
        tag = f"answer {i} ({a['subject']}/{a['question']})"
        if a["subject"] not in subjects.get(a["scope"], set()):
            errors.append(f"{tag}: unknown {a['scope']} {a['subject']!r}")
            continue
        err = check_answer(a["question"], a["answer"])
        if err:
            errors.append(f"{tag}: {err}")
        if not a["url"].startswith("https://"):
            errors.append(f"{tag}: url must be https")
        if needs_quote(a["question"], a["answer"]) and not a["quote"]:
            errors.append(f"{tag}: answer {a['answer']!r} needs a verbatim quote")
        rows.append({**a, "retrieved": dt.date.today().isoformat(), "by": args.by})
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    replaced = {(r["scope"], r["subject"], r["question"]) for r in rows}
    old = [r for r in read("evidence.csv") if (r["scope"], r["subject"], r["question"]) not in replaced]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(old + rows)
    print(f"{len(rows)} answers imported for {len({r['subject'] for r in rows})} subjects; "
          "now run scripts/verify_quotes.py")


def cmd_status(args):
    d = Data()
    for m in d.resolved():
        if args.families and m["family_id"] not in args.families:
            continue
        got = d.rubric(m)
        if all(v is None for v in got.values()):
            continue
        cells = []
        for f in FIELDS:
            cur = d.current(m, f)
            v = got[f]
            mark = "·" if v is None else "=" if v == cur else "≠"
            cells.append(f"{f}={cur}{'' if mark in '·=' else ' (rubric: ' + v + ')'} {mark}")
        print(f"{m['id']:26s} " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sheet")
    p.add_argument("families", nargs="+")
    p = sub.add_parser("import")
    p.add_argument("file")
    p.add_argument("--by", default="unknown", help="who answered: a person's handle or a model id")
    p = sub.add_parser("status")
    p.add_argument("families", nargs="*")
    args = ap.parse_args()
    {"sheet": cmd_sheet, "import": cmd_import, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
