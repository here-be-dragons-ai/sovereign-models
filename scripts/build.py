"""Merge the data files into dist/scorecard.csv and dist/scorecard.json.

Resolves every model through its family and provider and adds the computed
columns: tier, flags, sizes at 4 and 8 bit, cache sizes at 8k/32k/128k context,
mlx-vlm and llama.cpp support, and the most trusted MLX and GGUF builds. The
JSON also carries the sources of every model (from all three scopes), its builds
with their publisher and the cache layout, so the Space can compute memory for
any context length.
"""

import json
import sys
import csv

from memory import cache_bytes, cache_layout
from scorecard import PUBLISHER_ORDER, ROOT, TIERS, Data, read

DIST = ROOT / "dist"
# Effective bits per weight of MLX affine quantization with group size 64
# (weights plus a bf16 scale and bias per group).
BPW = {4: 4.5, 8: 8.5}
GIB = 2**30
CONTEXTS = {"8k": 8192, "32k": 32768, "128k": 131072}
GGUF_IN_JSON = 8  # builds per model kept for the detail view
COLUMNS = [
    "tier", "id", "name", "family", "provider", "provider_country", "region",
    "control", "majority_owner", "origin", "base_model", "license_class",
    "license", "data", "compute", "ai_act_summary", "cop_signatory",
    "modalities", "params_b", "context", "model_type", "size_4bit_gb",
    "size_8bit_gb", "mlx_vlm", "mlx_vlm_detail", "mlx_vlm_checked",
    "mlx_build", "mlx_build_bits", "mlx_build_publisher", "llama_cpp",
    "llama_cpp_detail", "llama_cpp_checked", "gguf_build", "gguf_build_publisher",
    "gguf_builds", "cache_kind", "cache_kib_per_token", "cache_gib_8k",
    "cache_gib_32k", "cache_gib_128k", "flags",
    "hbd_involvement", "notes", "hf_repo", "hf_created", "last_reviewed",
]
SOURCE_KEYS = ("scope", "field", "url", "issuer", "checked", "retrieved", "archive_url", "note")


def main():
    d = Data()
    errors = d.validate()
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    facts = {r["hf_repo"]: r for r in read("hf_facts.csv")}
    support = {r["id"]: r for r in read("mlx_support.csv")}
    llama = {r["id"]: r for r in read("llama_cpp_support.csv")}

    rows, details = [], []
    for m in d.resolved():
        f = facts.get(m["hf_repo"], {})
        s = support.get(m["id"], {})
        params = float(f["params_b"]) if f.get("params_b") else None
        builds = []
        for b in d.builds:
            if b["id"] == m["id"]:
                ns = b["repo"].split("/")[0]
                builds.append({**b, "publisher": d.publisher(b),
                               "verified_org": d.namespaces.get(ns, {}).get("verified") == "true"})
        builds.sort(key=lambda b: (PUBLISHER_ORDER.index(b["publisher"]), -float(b["bits"])))
        ggufs = []
        for b in d.gguf_builds:
            if b["id"] == m["id"]:
                ns = b["repo"].split("/")[0]
                ggufs.append({**b, "publisher": d.publisher(b),
                              "verified_org": d.namespaces.get(ns, {}).get("verified") == "true"})
        ggufs.sort(key=lambda b: (PUBLISHER_ORDER.index(b["publisher"]), -int(b["downloads"] or 0)))
        lc = llama.get(m["id"], {})
        lc_status, lc_detail = lc.get("llama_cpp", "not checked"), lc.get("detail", "")
        if lc_status == "unsupported" and ggufs:
            lc_status = "fork only"
            lc_detail = f"{len(ggufs)} GGUF builds exist, but upstream llama.cpp does not support the architecture; they need a patched llama.cpp"
        layout = cache_layout(f.get("arch_json", "")) if f.get("arch_json") else None
        max_ctx = int(f["context"]) if f.get("context", "").isdigit() else None

        def cache_gib(ctx):
            if not layout:
                return ""
            if max_ctx and ctx > max_ctx * 1.05:  # 32k still counts for a 32,000-token model
                return "n/a"
            return f"{cache_bytes(layout, ctx) / GIB:.1f}"
        sources = d.all_sources(m)
        row = dict(m)
        row.update(
            tier=d.tier(m),
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
            mlx_build=builds[0]["repo"] if builds else "",
            mlx_build_bits=builds[0]["bits"] if builds else "",
            mlx_build_publisher=builds[0]["publisher"] if builds else "",
            llama_cpp=lc_status,
            llama_cpp_detail=lc_detail,
            llama_cpp_checked=f"{lc.get('checked', '')} {lc.get('llama_cpp_rev', '')}".strip(),
            gguf_build=ggufs[0]["repo"] if ggufs else "",
            gguf_build_publisher=ggufs[0]["publisher"] if ggufs else "",
            gguf_builds=len(ggufs),
            cache_kind=layout["kind"] if layout else "",
            cache_kib_per_token=f"{layout['growing'] / 1024:.1f}" if layout else "",
            cache_gib_8k=cache_gib(CONTEXTS["8k"]),
            cache_gib_32k=cache_gib(CONTEXTS["32k"]),
            cache_gib_128k=cache_gib(CONTEXTS["128k"]),
            flags="; ".join(d.flags(m)),
            last_reviewed=max((x["retrieved"] for x in sources), default=""),
        )
        rows.append({c: row.get(c, "") for c in COLUMNS})
        details.append({
            **{c: row.get(c, "") for c in COLUMNS},
            "provider_notes": m["provider_notes"],
            "family_notes": d.families.get(m["family_id"], {}).get("notes", ""),
            "repo_verified_org": d.namespaces.get(m["hf_repo"].split("/")[0], {}).get("verified") == "true",
            "sources": [{k: x[k] for k in SOURCE_KEYS} for x in sources],
            "mlx_builds": builds,
            "gguf_list": ggufs[:GGUF_IN_JSON],
            "cache_layout": layout,
            "max_context": max_ctx,
        })

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
