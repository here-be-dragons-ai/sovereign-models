"""Collect the machine-checkable facts for every model from the Hugging Face Hub.

Writes data/hf_facts.csv: licence, gating, parameter count, architecture,
context length and creation date from the Hub API and config.json, plus
model-card lines that mention training compute or training data. These
facts carry the Hub URL as their source, so they count as confirmed.

Usage:
    python scripts/fetch_hf_facts.py              # all repos in data/scorecard.csv
    python scripts/fetch_hf_facts.py org/repo ... # only these
"""

import csv
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SM_DATA", ROOT / "data"))  # SM_DATA: another data directory, e.g. a local overlay
OUT = DATA_DIR / "hf_facts.csv"
FIELDS = [
    "hf_repo", "fetched", "license", "license_name", "gated", "created", "revision",
    "params_b", "model_type", "context", "pipeline_tag", "architecture", "arch_json",
    "compute_evidence", "data_evidence",
]
# config.json keys the memory estimate needs (top level or text_config).
ARCH_KEYS = (
    "model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "hidden_size", "sliding_window", "kv_lora_rank", "qk_rope_head_dim",
    "qk_nope_head_dim", "v_head_dim", "num_global_key_value_heads", "global_head_dim",
    "attention_k_eq_v", "linear_num_value_heads", "linear_key_head_dim",
    "linear_value_head_dim", "mamba_num_heads", "mamba_head_dim", "ssm_state_size",
    "num_blocks", "num_heads", "embedding_dim", "qk_dim_factor", "v_dim_factor",
    "max_position_embeddings",
)


def arch_summary(cfg):
    """The architecture facts the memory estimate needs, as a compact dict."""
    text = cfg.get("text_config") or cfg
    out = {k: text[k] for k in ARCH_KEYS if text.get(k) is not None}
    if "model_type" in cfg:
        out["outer_model_type"] = cfg["model_type"]
    layer_types = text.get("layer_types")
    if layer_types:
        out["layer_types"] = {t: layer_types.count(t) for t in sorted(set(layer_types))}
    pattern = text.get("hybrid_override_pattern")
    if pattern:
        out["hybrid_pattern"] = {"attention": pattern.count("*"), "mamba": pattern.count("M"), "mlp": pattern.count("-")}
    return out
COMPUTE = re.compile(
    r"(EuroHPC|MareNostrum|LUMI|Leonardo|Jean Zay|GENCI|Alps|CSCS|JUWELS|JUPITER|"
    r"Helios|Karolina|Deucalion|Berzelius|H100|B200|MI250X|GH200|AI Factor)",
    re.I,
)
DATA = re.compile(
    r"(training data|pre-?training (data|corpus)|dataset[s]? (is|are) (publicly )?available|"
    r"open data|Common Corpus|FineWeb|DCLM|tokens? of)",
    re.I,
)


def _token():
    for p in ("~/.cache/huggingface/token", "~/.huggingface/token"):
        p = Path(os.path.expanduser(p))
        if p.exists():
            return p.read_text().strip()
    return os.environ.get("HF_TOKEN")


TOKEN = _token()


def get(url):
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def evidence(text, pattern, limit=3):
    hits = []
    for line in text.splitlines():
        line = re.sub(r"<[^>]+>|\s+", " ", line).strip()
        if 20 < len(line) < 400 and pattern.search(line) and line not in hits:
            hits.append(line)
        if len(hits) == limit:
            break
    return " | ".join(hits)


def facts(repo):
    row = {"hf_repo": repo, "fetched": dt.date.today().isoformat()}
    info = json.loads(get(f"https://huggingface.co/api/models/{repo}"))
    card = info.get("cardData") or {}
    row.update(
        license=card.get("license", ""),
        license_name=card.get("license_name", ""),
        gated=str(info.get("gated", False)).lower(),
        created=(info.get("createdAt") or "")[:10],
        # The repo's current commit: any change to card, config or weights moves it.
        revision=(info.get("sha") or "")[:12],
        pipeline_tag=info.get("pipeline_tag") or "",
    )
    total = (info.get("safetensors") or {}).get("total")
    row["params_b"] = f"{total / 1e9:.1f}" if total else ""
    try:
        cfg = json.loads(get(f"https://huggingface.co/{repo}/resolve/main/config.json"))
        text = cfg.get("text_config") or cfg
        row["model_type"] = cfg.get("model_type", "")
        ctx = text.get("max_position_embeddings") or cfg.get("max_position_embeddings")
        row["context"] = str(ctx or "")
        row["architecture"] = (cfg.get("architectures") or [""])[0]
        row["arch_json"] = json.dumps(arch_summary(cfg), sort_keys=True)
    except urllib.error.HTTPError as e:
        # Gated repos: the API still reports model_type and architectures
        # without accepting the gate; context and cache layout need config.json.
        api_cfg = info.get("config") or {}
        row["model_type"] = api_cfg.get("model_type") or f"(config: HTTP {e.code})"
        row["architecture"] = (api_cfg.get("architectures") or [""])[0]
    try:
        readme = get(f"https://huggingface.co/{repo}/raw/main/README.md")
        row["compute_evidence"] = evidence(readme, COMPUTE)
        row["data_evidence"] = evidence(readme, DATA)
    except urllib.error.HTTPError:
        pass
    return row


def main():
    repos = sys.argv[1:]
    if not repos:
        with open(DATA_DIR / "scorecard.csv", newline="") as f:
            repos = [r["hf_repo"] for r in csv.DictReader(f)]
    rows = {}
    if OUT.exists() and sys.argv[1:]:
        with open(OUT, newline="") as f:
            rows = {r["hf_repo"]: r for r in csv.DictReader(f)}
    for repo in repos:
        try:
            rows[repo] = facts(repo)
            print(f"ok   {repo}")
        except Exception as e:  # keep going; report at the end
            print(f"FAIL {repo}: {e}")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for repo in sorted(rows):
            w.writerow({k: rows[repo].get(k, "") for k in FIELDS})


if __name__ == "__main__":
    main()
