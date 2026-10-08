"""Cache estimates against known layouts (values measured or derived by hand)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from memory import cache_bytes, cache_layout  # noqa: E402

KIB = 1024


class TestMemory(unittest.TestCase):
    def test_standard_attention(self):
        # Apertus 1.5 8B: 32 layers x 8 KV heads x 128 x (K+V) x 2 bytes = 128 KiB/token
        a = {"model_type": "apertus1p5_text", "num_hidden_layers": 32, "num_key_value_heads": 8,
             "num_attention_heads": 32, "head_dim": 128}
        self.assertEqual(cache_layout(a)["growing"], 128 * KIB)

    def test_sliding_window_layers_are_capped(self):
        # Kolibri 1: only the 10 full-attention layers grow (20 KiB/token)
        a = {"model_type": "kolibri1", "num_hidden_layers": 50, "num_key_value_heads": 4, "head_dim": 128,
             "sliding_window": 513, "layer_types": {"full_attention": 10, "sliding_attention": 40}}
        layout = cache_layout(a)
        self.assertEqual(layout["growing"], 20 * KIB)
        window_part = 40 * 2 * 4 * 128 * 2 * 513
        self.assertEqual(cache_bytes(layout, 100_000) - cache_bytes(layout, 50_000), 20 * KIB * 50_000)
        self.assertEqual(cache_bytes(layout, 1000), 20 * KIB * 1000 + window_part)

    def test_mla_is_cached_decompressed_in_mlx(self):
        # Mistral Small 4: 36 layers x 32 heads x ((64+64) + 128) x 2 bytes = 576 KiB/token
        a = {"model_type": "mistral4", "num_hidden_layers": 36, "num_attention_heads": 32,
             "num_key_value_heads": 32, "head_dim": 128, "kv_lora_rank": 256,
             "qk_nope_head_dim": 64, "qk_rope_head_dim": 64, "v_head_dim": 128}
        self.assertEqual(cache_layout(a)["growing"], 576 * KIB)

    def test_recurrent_state_is_constant(self):
        a = {"model_type": "xlstm", "num_blocks": 32, "num_heads": 8, "embedding_dim": 4096,
             "qk_dim_factor": 0.5, "v_dim_factor": 1.0}
        layout = cache_layout(a)
        self.assertEqual(layout["growing"], 0)
        self.assertEqual(cache_bytes(layout, 8), cache_bytes(layout, 1_000_000))

    def test_quantized_cache_scale(self):
        a = {"model_type": "llama", "num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
        layout = cache_layout(a)
        self.assertAlmostEqual(cache_bytes(layout, 1000, kv_bits=8) / cache_bytes(layout, 1000), 8.5 / 16)


if __name__ == "__main__":
    unittest.main()
