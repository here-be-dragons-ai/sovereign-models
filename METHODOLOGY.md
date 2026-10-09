# Methodology

This scorecard rates open-weight language models on **sovereignty from a European point of
view**: who controls them, where they come from, what you may do with them, and how
transparent their making is. It also records whether they run on Apple Silicon with MLX.
It does **not** rate how good a model is. Capability will be shown next to the tier in a later
version, never mixed into it; otherwise every small open model would rank below every large
opaque one.

## Why these criteria

Once weights are downloaded and run offline, there is no runtime dependency on anyone:
cloud jurisdiction, data residency and the risk of a provider switching the service off no
longer apply. What remains is what is baked into the artefact: the **licence**, the
**provenance** (who trained it, from which base, on what compute) and the **transparency**
of data and training. The criteria follow from that.

There is no official EU definition of a "sovereign model". The ownership test borrows from the
"no control by non-EU entities" idea in the proposed Cloud and AI Development Act; the
openness tests follow the OSI Open Source AI Definition and the Model Openness Framework in
spirit, at a much coarser level.

## Criteria

| Criterion | Column | Values |
|---|---|---|
| Provider seat | `region` | `eu` (EU/EEA), `europe` (European, outside the EU: CH, NO, UK, ...), `non-europe` |
| Control | `control` | `independent`; `acquired-non-eu`; `merger-pending-non-eu` |
| Origin of the weights | `origin` | `scratch`; `continued-own` (the provider's earlier model); `continued-european` / `finetune-european` (a European base); `continued-foreign` / `finetune-foreign` (a non-European base) |
| Licence | `license_class` | `osi` (Apache-2.0, MIT, CC-BY, CC-BY-SA and similar); `osi-aup` (such a licence plus a binding use policy, including a third party's policy the card binds the model to, e.g. Helium and the Gemma Terms of Use); `custom`; `nc` (non-commercial); `unclear` |
| Training data | `data` | `open` (published); `documented` (mix described); `undisclosed`; `unknown` |
| Training compute | `compute` | `eu`, `europe`, `non-europe`, `unknown` |
| AI Act training summary | `ai_act_summary` | `yes`, `no`, `unknown` (informative) |
| GPAI Code of Practice | `cop_signatory` | `yes`, `no`, `unknown` (informative) |

The AI Act summary and the Code of Practice are shown but do **not** affect the tier: the
first is a legal duty with an open-source exemption, the second a voluntary commitment
mostly signed by large providers.

## Sources

Every criterion except `region` and `license_class` needs a source in `data/sources.csv`
unless its value is `unknown`. Facts are stored at the level where they are true (provider,
family or model, see `data/schema.md`), so one source can back a fact for every model of a family.

Each source records **who published it** (`issuer`) and **whether we checked it** (`checked`):

| Issuer | Counts for the tier |
|---|---|
| `provider`: the model maker itself (model card in its own Hub organisation, its website, report or licence) | yes |
| `official`: public authority or official document (Commission lists, stock-exchange filings) | yes |
| `academic`: scholarly publication by third parties | yes |
| `press`: editorial media | yes |
| `community`: individuals or third parties without editorial control | **no** |
| `hbd`: our own measurement | for the Mac columns only |

A source counts only if its issuer counts **and** `checked = yes`. Anything else is shown but
treated as `unknown` for the tier.

The issuer is not taken on trust: each provider lists its official Hub organisations, GitHub
organisations and web domains in `data/providers.csv`, and `scripts/validate.py` derives the
issuer from the URL where it can and fails on a mismatch. For the Hub, `scripts/classify_namespaces.py`
records whether a namespace is an organisation or a personal account and whether the Hub has
verified it. A Hub page outside the provider's own organisation is `community`.

Every source gets a Wayback Machine snapshot (`archive_url`), because model cards change. Licence,
parameter count, architecture and context come from the Hugging Face Hub API
(`scripts/fetch_hf_facts.py`).

## Tiers

Computed by `scripts/scorecard.py`, in this order:

| Tier | Rule |
|---|---|
| out of scope | provider outside Europe |
| excluded | non-commercial licence |
| pending | licence unclear |
| **A** | open licence (`osi` or `osi-aup`), own weights (`scratch` or `continued-own`), `independent`, and either open training data or European compute |
| **B** | open licence and European lineage (own weights or a European base model), but not A: data and compute undisclosed, or control outside Europe |
| **C** | everything else from a European provider: a non-European base model or a custom licence |

Flags are shown next to the tier and never change it: outside the EU, acquired / merger
pending, use policy on redistribution, custom licence, non-European base model, our own work
involved.

## Informative criteria

Shown next to the tier, sourced like everything else, and **not** part of the tier. Decided in
#5 (2026-10-09) for teachers and reproducibility: almost every family that names its teachers
uses non-European ones, including all of tier A, so a teacher rule would describe the field
rather than separate it. Both stay visible so readers can weigh them.

- **Teacher models.** A European model trained on data generated by a non-European model
  inherits part of its behaviour; steering travels mostly through post-training data. Recorded
  per stage (pre-, mid-, post-training, judge) in `data/teachers.csv`. Judges are shown but do
  not count for `teacher_origin`. A model that is only mentioned is not a teacher.
- **Reproducibility** (`recipe`): whether training code, data, recipes and intermediate
  checkpoints are published, in the spirit of the Model Openness Framework.
- **Terms that can change** (`terms_can_change`): licences or use policies that refer to a
  version "in force" or "as updated", so the conditions can change after release.
- **Languages**: the languages the provider names and the count of the 24 official EU
  languages. Hub metadata is used only when nothing better exists; it is often incomplete.
- **Training hardware**: the accelerator vendor used for pre-training. There is no European
  alternative today; the column documents the dependency.

## Running on a Mac

- `mlx_vlm`: whether the current mlx-vlm loads the original checkpoint, checked from
  `config.json` and safetensors headers without downloading weights
  (`scripts/check_mlx_support.py`). "loads" means the weights map onto the model; it does not
  test the output. The checked revision is recorded.
- `llama_cpp`: whether the latest llama.cpp release registers the architecture in its GGUF
  converter (`scripts/check_llama_cpp_support.py`). `fork only` means GGUF builds exist but
  upstream llama.cpp does not support the architecture, so they need a patched llama.cpp.
- `size_4bit_gb`, `size_8bit_gb`: parameters × 4.5 or 8.5 bits per weight (MLX affine
  quantization, group size 64).
- **Cache** (`scripts/memory.py`): bytes per token and totals at 8k, 32k and 128k, as MLX
  allocates them. The layout depends on the architecture: standard attention grows in every
  layer; sliding-window layers stop at their window; hybrid models (linear attention, Mamba)
  grow only in their attention layers; recurrent models (xLSTM) keep a constant state. MLA
  models are cached **decompressed** in MLX, so MLA saves no memory there (llama.cpp caches the
  compressed latent; its numbers would be lower). "n/a" marks contexts beyond the model's maximum.
  The Space adds the cache at the chosen context and cache precision to the 4-bit weights, plus
  1 GB, and compares the total with 75 % of the RAM.
- `mlx_build`, `gguf_build`: the most trusted existing build. The publisher is derived from the
  Hub namespace: `official` (the provider's own organisation), `hbd` (ours), `curated`
  (mlx-community, lmstudio-community, ggml-org, unsloth), `organisation` (any other
  organisation), `individual` (a personal account), in that order; ties by downloads.

## Limits

- The tiers describe artefacts and their makers, not model quality or safety.
- Absence of evidence is not evidence of absence: `unknown` often means "not stated on the
  model card", and providers may know more than they publish.
- Ownership changes. A model released before an acquisition stays usable under its licence;
  the flag tells you who controls future releases.
- Synthetic training data generated with other models is noted where the card states it, but
  is not yet a criterion.

## Corrections

Open an issue or a pull request against `data/scorecard.csv` and `data/sources.csv`, with a
source. Providers are invited to correct their entries; we ask them before publication.

## Our own work

here be dragons publishes MLX builds and contributes to model ports. Where that touches a
model, the `hbd_involvement` column says so, and the same rules apply as for everyone else.
