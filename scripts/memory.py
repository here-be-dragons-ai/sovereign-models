"""Memory estimate for running a model with MLX: weights plus cache at a context length.

The cache layout follows what mlx-lm / mlx-vlm actually allocate, which can differ
from other runtimes:
- standard attention: keys and values per layer and token, KV heads x head dim;
- sliding-window layers: at most `window` tokens;
- MLA (kv_lora_rank): MLX caches the decompressed keys and values per head,
  not the compressed latent, so MLA saves no memory in MLX (llama.cpp does);
- Gemma 4: global layers use their own KV heads and head dim;
- linear-attention, Mamba and xLSTM layers: a constant state, independent of
  the context length.

All sizes in bytes. `estimate(arch, ctx, kv_bits)` returns the parts.
"""

import json

BYTES_F16 = 2
STATE_BYTES = 4  # recurrent states are kept in float32
# MLX quantized KV cache: bits plus a 16-bit scale and bias per group of 64.
KV_BITS_EFFECTIVE = {16: 16.0, 8: 8.5, 4: 4.5}


def _kv(arch):
    heads = arch.get("num_key_value_heads") or arch.get("num_attention_heads")
    head_dim = arch.get("head_dim")
    if not head_dim and arch.get("hidden_size") and arch.get("num_attention_heads"):
        head_dim = arch["hidden_size"] // arch["num_attention_heads"]
    return heads, head_dim


def cache_layout(arch):
    """Per-layer cache sizes at 16 bit.

    Returns dict with:
      growing:  bytes per token for layers whose cache grows with the context
      windowed: list of (bytes per token, window) for sliding-window layers
      constant: bytes of state that does not depend on the context
      kind:     short description of the layout
    """
    if isinstance(arch, str):
        arch = json.loads(arch) if arch else {}
    mt = arch.get("model_type", "")
    layers = arch.get("num_hidden_layers") or 0
    types = arch.get("layer_types") or {}
    heads, head_dim = _kv(arch)
    out = {"growing": 0, "windowed": [], "constant": 0, "kind": "standard attention"}

    if mt == "xlstm":
        n, h, d = arch["num_blocks"], arch["num_heads"], arch["embedding_dim"]
        qk, v = d * arch.get("qk_dim_factor", 0.5) / h, d * arch.get("v_dim_factor", 1.0) / h
        out.update(constant=int(n * h * (qk * v + qk) * STATE_BYTES), kind="recurrent (xLSTM), constant state")
        return out

    if arch.get("kv_lora_rank"):
        per_head = (arch.get("qk_nope_head_dim", 0) + arch.get("qk_rope_head_dim", 0)) + arch.get("v_head_dim", head_dim)
        out.update(growing=layers * arch["num_attention_heads"] * per_head * BYTES_F16,
                   kind="MLA, cached decompressed in MLX")
        return out

    if "hybrid_pattern" in arch:
        p = arch["hybrid_pattern"]
        out["growing"] = p["attention"] * 2 * heads * head_dim * BYTES_F16
        out["constant"] = int(p["mamba"] * arch.get("mamba_num_heads", 0) * arch.get("mamba_head_dim", 0)
                              * arch.get("ssm_state_size", 0) * STATE_BYTES)
        out["kind"] = f"hybrid: {p['attention']} attention + {p['mamba']} Mamba layers"
        return out

    if "linear_attention" in types:
        full = types.get("full_attention", 0)
        out["growing"] = full * 2 * heads * head_dim * BYTES_F16
        out["constant"] = int(types["linear_attention"] * arch.get("linear_num_value_heads", 0)
                              * arch.get("linear_key_head_dim", 0) * arch.get("linear_value_head_dim", 0) * STATE_BYTES)
        out["kind"] = f"hybrid: {full} full + {types['linear_attention']} linear-attention layers"
        return out

    window = arch.get("sliding_window")
    if "sliding_attention" in types and window:
        full, sliding = types.get("full_attention", 0), types["sliding_attention"]
        g_heads = arch.get("num_global_key_value_heads") or heads
        g_dim = arch.get("global_head_dim") or head_dim
        out["growing"] = full * 2 * g_heads * g_dim * BYTES_F16
        out["windowed"] = [(sliding * 2 * heads * head_dim * BYTES_F16, window)]
        out["kind"] = f"{full} full + {sliding} sliding-window ({window}) layers"
        return out

    if window and window < (arch.get("max_position_embeddings") or 0):
        out["windowed"] = [(layers * 2 * heads * head_dim * BYTES_F16, window)]
        out["kind"] = f"sliding window ({window}) in all layers"
        return out

    out["growing"] = layers * 2 * heads * head_dim * BYTES_F16
    return out


def cache_bytes(layout, ctx, kv_bits=16):
    scale = KV_BITS_EFFECTIVE[kv_bits] / 16
    kv = layout["growing"] * ctx + sum(per_token * min(ctx, window) for per_token, window in layout["windowed"])
    return kv * scale + layout["constant"]


def estimate(arch, ctx, kv_bits=16):
    layout = cache_layout(arch)
    return {"cache": cache_bytes(layout, ctx, kv_bits), **layout}
