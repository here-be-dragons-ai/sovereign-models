# Sovereign open-weight models

A scorecard of open-weight language models rated on **sovereignty from a European point of
view** — who controls them, where the weights come from, what the licence allows, how
transparent data and compute are — and whether they run on Apple Silicon with MLX.
For comparison, current open-weight models from providers outside Europe are listed with the
same facts but without a tier; the overview shows them on request (see
[METHODOLOGY](METHODOLOGY.md#comparison-models-from-outside-europe)).

Status: public, here and on Hugging Face (dataset, Space and tier collections). See
[#1](../../issues/1) for the overall plan.

## How it works

```
data/providers.csv     organisations: seat, control, owners, official namespaces   (by hand)
data/families.csv      model families: origin, licence, data, compute              (by hand)
data/scorecard.csv     models: repo, family, overrides                             (by hand)
data/teachers.csv      teacher models per family and training stage                (by hand)
data/sources.csv       one source per fact, with issuer, checked, archive          (by hand + scripts/archive_sources.py)
data/mlx_builds.csv    known MLX builds per model                                  (by hand)
data/hf_facts.csv      licence, size, architecture from the Hub                    (scripts/fetch_hf_facts.py)
data/hf_namespaces.csv Hub namespaces: organisation or user, verified             (scripts/classify_namespaces.py)
data/mlx_support.csv   does mlx-vlm load it?                                       (scripts/check_mlx_support.py)
data/llama_cpp_support.csv  does llama.cpp support the architecture?               (scripts/check_llama_cpp_support.py)
data/gguf_builds.csv   GGUF quantizations on the Hub                               (scripts/find_gguf_builds.py)
data/template_checks.csv  chat template of every build against the original      (scripts/check_chat_templates.py)
data/template_diffs/   one unified diff per build whose template differs
data/schema.md         columns and allowed values of every file
data/changes.csv       log of every published change, with reasons                 (scripts/build.py)
data/terms_hashes.csv  hashes of licence and use-policy documents                  (scripts/check_terms.py)
data/known_repos.csv   repos of tracked providers seen so far                      (scripts/find_new_repos.py)
data/build_recipes.csv, data/build_provenance.csv  provenance of our own builds    (scripts/build_provenance.py)
        │
        ▼  scripts/build.py   (validates, computes tiers and sizes)
dist/scorecard.csv, dist/scorecard.json
        │
        ▼  scripts/publish_hf.py
HF dataset here-be-dragons-ai/sovereign-models-scorecard   (public)
HF Space   here-be-dragons-ai/sovereign-models             (public, static)
HF collections "Sovereign models · Tier A / B / C"          (public, in that order)
```

Tiers are computed, never set by hand. The rules and their limits are in
[METHODOLOGY.md](METHODOLOGY.md); the columns and allowed values in
[data/schema.md](data/schema.md).

## Updating

```bash
python3 scripts/fetch_hf_facts.py                    # refresh Hub facts
python3 scripts/classify_namespaces.py               # organisation or user, per Hub namespace
python3 scripts/archive_sources.py                   # Wayback snapshots for new sources
PYTHONPATH=~/src/mlx-vlm-main \
  ~/src/mlx/.venv/bin/python scripts/check_mlx_support.py   # needs mlx and mlx-vlm
python3 scripts/check_llama_cpp_support.py           # latest llama.cpp release
python3 scripts/find_gguf_builds.py                  # GGUF builds via the Hub base-model link
python3 scripts/classify_namespaces.py               # again, for new GGUF publishers
python3 scripts/validate.py                          # checks and prints the tiers
python3 scripts/build.py                             # writes dist/
python3 scripts/publish_hf.py                        # needs huggingface_hub and a token
```

`validate.py` and `build.py` need only the Python standard library.

## Adding or correcting a model

1. Add the provider to `data/providers.csv` if it is new, with its official Hub organisations,
   GitHub organisations and domains.
2. Add the family to `data/families.csv` and the model to `data/scorecard.csv`
   (allowed values: `data/schema.md`).
3. Add a source for every fact in `data/sources.csv`, at the scope where it is stated, with its
   `issuer` (`provider`, `official`, `academic`, `press`, `community`) and `checked`. Only
   checked provider, official, academic and press sources count for the tier; `validate.py`
   checks the issuer against the URL.
4. Run `fetch_hf_facts.py <org/repo>`, `classify_namespaces.py`, `archive_sources.py` and
   `validate.py`.
5. Open a pull request.

## Licence

Code: MIT. Data and documentation: CC BY 4.0. See [LICENSE](LICENSE) and [LICENSE-DATA](LICENSE-DATA).

## Disclosure

here be dragons publishes MLX builds and contributes to model ports. The `hbd_involvement`
column marks where that touches a model; the same rules apply as for every other entry.
