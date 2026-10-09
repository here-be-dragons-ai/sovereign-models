"""Publish the built scorecard to Hugging Face: dataset, Space and collections.

The dataset, the Space and the tier collections (A, B, C) are public (since 2026-10-09).
Running it again updates the files and syncs the collections (adds, removes
and re-orders items, rewrites notes).

    python scripts/build.py
    python scripts/publish_hf.py [--org here-be-dragons-ai]
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DATASET = "sovereign-models-scorecard"
SPACE = "sovereign-models"
COLLECTIONS = {
    "A": ("Sovereign models · Tier A",
          "European open-weight LLMs: own weights, open licence, open data or EU compute."),
    "B": ("Sovereign models · Tier B",
          "European open-weight LLMs with open licences; data, compute or control less transparent."),
    "C": ("Sovereign models · Tier C",
          "European open-weight LLMs on a non-European base model or under a custom licence."),
}

DATASET_CARD = """---
license: cc-by-4.0
pretty_name: Sovereign open-weight models scorecard
tags:
- sovereignty
- open-weight
- llm
- mlx
configs:
- config_name: default
  data_files: scorecard.csv
---

# Sovereign open-weight models scorecard

Open-weight language models rated on sovereignty from a European point of view: provider and
control, origin of the weights, licence, training data and compute, plus whether they load in
mlx-vlm on Apple Silicon. Tiers are computed by rules; every fact has a source.

- `scorecard.csv`: one row per model, including the computed tier and flags
- `scorecard.json`: the same plus all sources and MLX builds per model
- `data/`: the curated inputs (`providers.csv`, `families.csv`, `scorecard.csv`, `sources.csv`,
  `mlx_builds.csv`) and the generated `hf_facts.csv`, `hf_namespaces.csv` and `mlx_support.csv`.
  Every source records its issuer (provider, official, academic, press, community) and a Wayback
  snapshot; only checked provider, official, academic and press sources count for the tier
- `METHODOLOGY.md`: criteria, tier rules and limits

Source of truth: the GitHub repository `here-be-dragons-ai/sovereign-models`. Corrections with a
source are welcome there.
"""


def note(m):
    parts = [
        f"Tier {m['tier']}",
        m["provider_country"],
        m["provider"],
        m["origin"] + (f" from {m['base_model']}" if m["base_model"] else ""),
        f"licence {m['license_class']} ({m['license']})",
        f"data {m['data']}",
        f"compute {m['compute']}",
        f"{m['params_b']}B, ~{m['size_4bit_gb']} GB at 4 bit",
        f"mlx-vlm: {m['mlx_vlm']}",
    ]
    if m["mlx_build"]:
        parts.append(f"MLX: https://huggingface.co/{m['mlx_build']}")
    if m["flags"]:
        parts.append(f"Flags: {m['flags']}")
    text = " · ".join(parts)
    return text if len(text) <= 500 else text[:497] + "..."


def sync_collection(api, org, title, description, items):
    """items: list of (item_id, item_type, note) in the wanted order."""
    existing = [c for c in api.list_collections(owner=org, limit=100) if c.title == title]
    if existing:
        col = api.get_collection(existing[0].slug)
    else:
        col = api.create_collection(title=title, namespace=org, description=description, private=False)
    api.update_collection_metadata(col.slug, description=description, private=False)
    wanted = {(i, t) for i, t, _ in items}
    for it in col.items:
        if (it.item_id, it.item_type) not in wanted:
            api.delete_collection_item(col.slug, it.item_object_id)
    have = {(it.item_id, it.item_type): it for it in api.get_collection(col.slug).items}
    for item_id, item_type, text in items:
        if (item_id, item_type) not in have:
            api.add_collection_item(col.slug, item_id, item_type, note=text, exists_ok=True)
    col = api.get_collection(col.slug)
    by_key = {(it.item_id, it.item_type): it for it in col.items}
    for pos, (item_id, item_type, text) in enumerate(items):
        it = by_key[(item_id, item_type)]
        api.update_collection_item(col.slug, it.item_object_id, note=text, position=pos)
    return col.slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="here-be-dragons-ai")
    args = ap.parse_args()
    api = HfApi()
    data = json.loads((ROOT / "dist" / "scorecard.json").read_text(encoding="utf-8"))
    models = data["models"]

    dataset_id = f"{args.org}/{DATASET}"
    api.create_repo(dataset_id, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(dataset_id, repo_type="dataset", private=False)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "README.md").write_text(DATASET_CARD, encoding="utf-8")
        for name in ("scorecard.csv", "scorecard.json"):
            shutil.copy(ROOT / "dist" / name, tmp / name)
        shutil.copy(ROOT / "METHODOLOGY.md", tmp / "METHODOLOGY.md")
        shutil.copytree(ROOT / "data", tmp / "data")
        api.upload_folder(repo_id=dataset_id, repo_type="dataset", folder_path=tmp,
                          commit_message="Update scorecard")
    print(f"dataset  https://huggingface.co/datasets/{dataset_id}")

    space_id = f"{args.org}/{SPACE}"
    # Static Spaces are free for organisations; Gradio Spaces need a paid plan.
    api.create_repo(space_id, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.update_repo_settings(space_id, repo_type="space", private=False)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in (ROOT / "space").iterdir():
            if f.suffix in (".html", ".md"):
                shutil.copy(f, tmp / f.name)
        shutil.copy(ROOT / "dist" / "scorecard.json", tmp / "scorecard.json")
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=tmp,
                          commit_message="Update scorecard")
    print(f"space    https://huggingface.co/spaces/{space_id}")

    slugs = []
    for tier, (title, description) in COLLECTIONS.items():
        items = [(dataset_id, "dataset", "Scorecard, sources and methodology for every model in this collection.")]
        items += [(m["hf_repo"], "model", note(m)) for m in models if m["tier"] == tier]
        slug = sync_collection(api, args.org, title, description, items)
        slugs.append(slug)
        print(f"collection https://huggingface.co/collections/{slug}  ({len(items) - 1} models)")
    # Order on the organisation page: A, B, C. Set after all exist, since a
    # newly created collection is inserted at the top.
    for position, slug in enumerate(slugs):
        api.update_collection_metadata(slug, position=position)


if __name__ == "__main__":
    main()
