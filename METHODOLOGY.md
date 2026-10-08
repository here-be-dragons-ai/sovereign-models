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
| Licence | `license_class` | `osi` (Apache-2.0, MIT, CC-BY, CC-BY-SA and similar); `osi-aup` (such a licence plus a binding use policy); `custom`; `nc` (non-commercial); `unclear` |
| Training data | `data` | `open` (published); `documented` (mix described); `undisclosed`; `unknown` |
| Training compute | `compute` | `eu`, `europe`, `non-europe`, `unknown` |
| AI Act training summary | `ai_act_summary` | `yes`, `no`, `unknown` (informative) |
| GPAI Code of Practice | `cop_signatory` | `yes`, `no`, `unknown` (informative) |

The AI Act summary and the Code of Practice are shown but do **not** affect the tier: the
first is a legal duty with an open-source exemption, the second a voluntary commitment
mostly signed by large providers.

## Sources

Every criterion except `region` and `license_class` needs a source in `data/sources.csv`
unless its value is `unknown`. Each source has a status:

- `primary`: the provider or an official document (model card, licence file, technical
  report, regulator list, stock-exchange filing).
- `press`: reputable reporting.
- `secondary`: a research summary not yet checked against a primary source.

**Only `primary` and `press` count for the tier.** A value backed only by a `secondary`
source is treated as `unknown`. Licence, parameter count, architecture and context come from
the Hugging Face Hub API (`scripts/fetch_hf_facts.py`).

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

## Running on a Mac

- `mlx_vlm`: whether the current mlx-vlm loads the original checkpoint, checked from
  `config.json` and safetensors headers without downloading weights
  (`scripts/check_mlx_support.py`). "loads" means the weights map onto the model; it does not
  test the output. The checked revision is recorded.
- `size_4bit_gb`, `size_8bit_gb`: parameters × 4.5 or 8.5 bits per weight (MLX affine
  quantization, group size 64). The KV cache comes on top.
- `mlx_build`: the most trusted existing MLX build, in the order official, ours, mlx-community,
  lmstudio-community, other community.

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
