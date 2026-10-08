"""Check the data files and print the computed tier of every model.

Exits with status 1 if a check fails, so it can gate pull requests.
"""

import sys

from scorecard import flags, load, tier, validate


def main():
    models, sources, by_field = load()
    errors = validate(models, sources)
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        sys.exit(1)
    for m in sorted(models, key=lambda m: (tier(m, by_field), m["id"])):
        extra = ", ".join(flags(m))
        print(f"{tier(m, by_field):12s} {m['id']:22s} {m['provider_country']}  {extra}")
    print(f"\n{len(models)} models, {len(sources)} sources: OK")


if __name__ == "__main__":
    main()
