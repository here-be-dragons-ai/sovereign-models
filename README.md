# Sovereign open-weight models

A scorecard of open-weight language models rated on **sovereignty from a European point of
view** — who controls them, where the weights come from, what the licence allows, how
transparent data and compute are — and whether they run on Apple Silicon with MLX.

Status: **MVP, private.** See [#1](../../issues/1) for the overall plan and [#2](../../issues/2)
for the MVP scope.

## How it works

```
data/scorecard.csv     curated facts, one row per model          (edit by hand, review in PRs)
data/sources.csv       one source per fact, with status          (edit by hand)
data/mlx_builds.csv    known MLX builds per model                (edit by hand)
data/hf_facts.csv      licence, size, architecture from the Hub  (scripts/fetch_hf_facts.py)
data/mlx_support.csv   does mlx-vlm load it?                     (scripts/check_mlx_support.py)
        │
        ▼  scripts/build.py   (validates, computes tiers and sizes)
dist/scorecard.csv, dist/scorecard.json
        │
        ▼  scripts/publish_hf.py
HF dataset here-be-dragons-ai/sovereign-models-scorecard   (private)
HF Space   here-be-dragons-ai/sovereign-models             (private)
HF collections "Sovereign models · Tier A / Tier B"         (private)
```

Tiers are computed, never set by hand. The rules and their limits are in
[METHODOLOGY.md](METHODOLOGY.md); the columns and allowed values in
[data/schema.md](data/schema.md).

## Updating

```bash
python3 scripts/fetch_hf_facts.py                    # refresh Hub facts
PYTHONPATH=~/src/mlx-vlm-main \
  ~/src/mlx/.venv/bin/python scripts/check_mlx_support.py   # needs mlx and mlx-vlm
python3 scripts/validate.py                          # checks and prints the tiers
python3 scripts/build.py                             # writes dist/
python3 scripts/publish_hf.py                        # needs huggingface_hub and a token
```

`validate.py` and `build.py` need only the Python standard library.

## Adding or correcting a model

1. Add or edit the row in `data/scorecard.csv` (allowed values: `data/schema.md`).
2. Add a source for every field you set in `data/sources.csv`. Mark it `primary` (provider or
   official document), `press` or `secondary`. Only `primary` and `press` count for the tier.
3. Run `python3 scripts/fetch_hf_facts.py <org/repo>` and `python3 scripts/validate.py`.
4. Open a pull request.

## Licence

Code: MIT. Data and documentation: CC BY 4.0. See [LICENSE](LICENSE) and [LICENSE-DATA](LICENSE-DATA).

## Disclosure

here be dragons publishes MLX builds and contributes to model ports. The `hbd_involvement`
column marks where that touches a model; the same rules apply as for every other entry.
