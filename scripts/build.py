"""Merge the data files into dist/scorecard.csv and dist/scorecard.json.

Adds the computed columns: tier, flags, sizes at 4 and 8 bit, mlx-vlm load
status and the most trusted MLX build. The JSON also carries the sources of
every model, for the detail view of the Space.
"""

import csv
import json
import sys

from scorecard import ROOT, TIERS, flags, load, read, tier, validate

DIST = ROOT / "dist"
# Effective bits per weight of MLX affine quantization with group size 64
# (weights plus a bf16 scale and bias per group).
BPW = {4: 4.5, 8: 8.5}
TRUST = ["official", "hbd", "mlx-community", "lmstudio-community", "community"]
COLUMNS = [
    "tier", "id", "name", "provider", "provider_country", "region", "control",
    "origin", "base_model", "license_class", "license", "data", "compute",
    "ai_act_summary", "cop_signatory", "modalities", "params_b", "context",
    "model_type", "size_4bit_gb", "size_8bit_gb", "mlx_vlm", "mlx_vlm_detail",
    "mlx_vlm_checked", "mlx_build", "mlx_build_bits", "mlx_build_publisher",
    "flags", "hbd_involvement", "notes", "hf_repo", "hf_created",
]


def main():
    models, sources, by_field = load()
    errors = validate(models, sources)
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    facts = {r["hf_repo"]: r for r in read("hf_facts.csv")}
    support = {r["id"]: r for r in read("mlx_support.csv")}
    builds = {}
    for b in read("mlx_builds.csv"):
        builds.setdefault(b["id"], []).append(b)

    rows, details = [], []
    for m in models:
        f = facts.get(m["hf_repo"], {})
        s = support.get(m["id"], {})
        params = float(f["params_b"]) if f.get("params_b") else None
        best = sorted(
            builds.get(m["id"], []),
            key=lambda b: (TRUST.index(b["publisher_kind"]), -float(b["bits"])),
        )
        row = dict(m)
        row.update(
            tier=tier(m, by_field),
            license=f.get("license_name") or f.get("license", ""),
            params_b=f.get("params_b", ""),
            context=f.get("context", ""),
            model_type=f.get("model_type", ""),
            hf_created=f.get("created", ""),
            size_4bit_gb=f"{params * BPW[4] / 8:.1f}" if params else "",
            size_8bit_gb=f"{params * BPW[8] / 8:.1f}" if params else "",
            mlx_vlm=s.get("mlx_vlm", "not checked"),
            mlx_vlm_detail=s.get("detail", ""),
            mlx_vlm_checked=f"{s.get('checked', '')} {s.get('mlx_vlm_rev', '')}".strip(),
            mlx_build=best[0]["repo"] if best else "",
            mlx_build_bits=best[0]["bits"] if best else "",
            mlx_build_publisher=best[0]["publisher_kind"] if best else "",
            flags="; ".join(flags(m)),
        )
        rows.append({c: row.get(c, "") for c in COLUMNS})
        details.append(
            {
                **{c: row.get(c, "") for c in COLUMNS},
                "sources": [
                    {k: x[k] for k in ("field", "url", "status", "retrieved", "note")}
                    for x in sources
                    if x["id"] == m["id"]
                ],
                "mlx_builds": builds.get(m["id"], []),
            }
        )

    order = list(TIERS)
    rows.sort(key=lambda r: (order.index(r["tier"]), r["id"]))
    details.sort(key=lambda r: (order.index(r["tier"]), r["id"]))
    DIST.mkdir(exist_ok=True)
    with open(DIST / "scorecard.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(DIST / "scorecard.json", "w", encoding="utf-8") as fh:
        json.dump({"tiers": TIERS, "models": details}, fh, indent=1, ensure_ascii=False)
    counts = {t: sum(r["tier"] == t for r in rows) for t in order}
    print(f"dist/: {len(rows)} models " + ", ".join(f"{t}={n}" for t, n in counts.items() if n))


if __name__ == "__main__":
    main()
