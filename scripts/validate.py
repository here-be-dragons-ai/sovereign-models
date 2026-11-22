"""Check the data files and print the computed tier of every model.

Exits with status 1 if a check fails, so it can gate pull requests.
"""

import re
import sys

import rubric
from scorecard import Data, read


def main():
    d = Data()
    errors = d.validate()
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        sys.exit(1)
    models = d.resolved()
    for m in sorted(models, key=lambda m: (d.tier(m), m["id"])):
        print(f"{d.tier(m):12s} {m['id']:22s} {m['provider_country']}  {', '.join(d.flags(m))}")
    unarchived = sum(1 for s in d.sources if not s["archive_url"])
    revisions = {r["hf_repo"]: r["revision"] for r in read("hf_facts.csv")}
    unpinned = sum(1 for m in models if not re.fullmatch(r"[0-9a-f]{40}", revisions.get(m["hf_repo"], "")))
    decided = sum(v is not None for m in models for v in d.rubric(m).values())
    pairs = len(models) * len(rubric.FIELDS)
    print(f"\n{len(d.providers)} providers, {len(d.families)} families, {len(models)} models, "
          f"{len(d.sources)} sources ({unarchived} without archive snapshot); "
          f"{unpinned} models without a pinned Hub commit; "
          f"rubric decides {decided} of {pairs} values from verified quotes: OK")


if __name__ == "__main__":
    main()
