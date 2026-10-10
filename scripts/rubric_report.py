"""List every model's value for the judgement fields, with the notes of its sources.

For the rating rubric in METHODOLOGY.md: values that need judgement (training
data, origin, teacher status, recipe, terms that can change, compute) side by
side across all models, so that equal evidence can be checked for equal ratings.

    python3 scripts/rubric_report.py [field ...] > rubric.md
"""

import sys
from collections import defaultdict

from scorecard import Data

FIELDS = ["data", "origin", "recipe", "teacher_status", "terms_can_change", "compute"]


def main():
    d = Data()
    models = d.resolved()
    fields = sys.argv[1:] or FIELDS
    by_scope = defaultdict(list)
    for s in d.sources:
        by_scope[(s["scope"], s["subject"], s["field"])].append(s)
    for field in fields:
        print(f"\n## {field}\n")
        groups = defaultdict(list)
        for m in models:
            groups[m.get(field, "")].append(m)
        for value in sorted(groups):
            print(f"### {field} = {value or '(empty)'} ({len(groups[value])})\n")
            for m in sorted(groups[value], key=lambda m: m["id"]):
                notes = []
                for scope, subject in (("model", m["id"]), ("family", m["family_id"]), ("provider", m["provider_id"])):
                    for s in by_scope[(scope, subject, field)]:
                        mark = "" if s["checked"] == "yes" else " (unchecked)"
                        notes.append(f"{s['note'][:160]}{mark}")
                tier = m.get("region", "")
                print(f"- **{m['id']}** [{tier}]: " + (" | ".join(notes) if notes else "no source"))
            print()


if __name__ == "__main__":
    main()
