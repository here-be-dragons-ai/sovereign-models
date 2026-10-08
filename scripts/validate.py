"""Check the data files and print the computed tier of every model.

Exits with status 1 if a check fails, so it can gate pull requests.
"""

import sys

from scorecard import Data


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
    print(f"\n{len(d.providers)} providers, {len(d.families)} families, {len(models)} models, "
          f"{len(d.sources)} sources ({unarchived} without archive snapshot): OK")


if __name__ == "__main__":
    main()
