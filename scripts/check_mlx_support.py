"""Check whether mlx-vlm loads each model, without downloading weights.

For every model in data/scorecard.csv this reads config.json and the
safetensors headers (names, shapes, dtypes) from the Hub, builds the mlx-vlm
model lazily, runs its sanitize step on zero arrays of the checkpoint
shapes, and compares the result with the model parameters. Results go to
data/mlx_support.csv together with the mlx-vlm revision that was checked.

Run it with the mlx-vlm checkout you want to check on PYTHONPATH:
    PYTHONPATH=~/src/mlx-vlm-main python scripts/check_mlx_support.py [id ...]

Limits: FP8 releases are checked without their scale tensors, because the
FP8 conversion happens inside load_model. A "loads" result means the weights
map onto the model; it does not test the output.
"""

import csv
import datetime as dt
import json
import os
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx_vlm
from mlx.utils import tree_flatten
from mlx_vlm.utils import get_model_and_args, sanitize_weights, update_module_configs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard import DATA, read  # noqa: E402

OUT = DATA / "mlx_support.csv"
FIELDS = ["id", "mlx_vlm", "model_type", "detail", "checked", "mlx_vlm_rev"]
DTYPES = {
    "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32, "I8": mx.int8,
    "U8": mx.uint8, "I32": mx.int32, "I64": mx.int64, "U32": mx.uint32,
    "BOOL": mx.bool_, "F8_E4M3": mx.uint8, "F8_E8M0": mx.uint8,
}
FP8_SCALES = ("weight_scale_inv", "weight_scale", "input_scale", "activation_scale")


def _token():
    path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return path.read_text().strip() if path.exists() else os.environ.get("HF_TOKEN")


TOKEN = _token()


def get(url, start=None, end=None):
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read()


def headers(repo):
    base = f"https://huggingface.co/{repo}/resolve/main/"
    try:
        index = json.loads(get(base + "model.safetensors.index.json"))
        files = sorted(set(index["weight_map"].values()))
    except urllib.error.HTTPError:
        files = ["model.safetensors"]
    shapes = {}
    for name in files:
        if "consolidated" in name:
            continue
        size = struct.unpack("<Q", get(base + name, 0, 7))[0]
        header = json.loads(get(base + name, 8, 7 + size))
        header.pop("__metadata__", None)
        shapes.update(header)
    return shapes


def check(repo):
    try:
        config = json.loads(get(f"https://huggingface.co/{repo}/resolve/main/config.json"))
    except urllib.error.HTTPError as e:
        return "not checked", "", f"config.json: HTTP {e.code} (gated?)"
    model_type = config.get("model_type", "")
    try:
        module, _ = get_model_and_args(dict(config))
    except Exception:
        return "unsupported", model_type, "model type not in mlx-vlm"
    cfg = dict(config)
    cfg.setdefault("text_config", cfg.pop("llm_config", {}))
    cfg.setdefault("vision_config", {})
    cfg.setdefault("audio_config", {})
    try:
        model_config = module.ModelConfig.from_dict(cfg)
        model_config = update_module_configs(
            model_config, module, cfg, ["text", "vision", "perceiver", "projector", "audio"]
        )
        model = module.Model(model_config)
    except Exception as e:
        return "fails", model_type, f"model init: {type(e).__name__}: {e}"[:200]
    fp8 = (config.get("quantization_config") or {}).get("quant_method") == "fp8"
    weights = {
        k: mx.zeros(v["shape"], DTYPES.get(v["dtype"], mx.float32))
        for k, v in headers(repo).items()
        if not (fp8 and k.endswith(FP8_SCALES))
    }
    try:
        weights = sanitize_weights(model, weights)
        for cls, attr in (("VisionModel", "vision_config"), ("LanguageModel", "text_config"), ("AudioModel", "audio_config")):
            if hasattr(module, cls) and hasattr(model_config, attr):
                weights = sanitize_weights(getattr(module, cls), weights, getattr(model_config, attr))
    except Exception as e:
        return "fails", model_type, f"sanitize: {type(e).__name__}: {e}"[:200]
    params = dict(tree_flatten(model.parameters()))
    missing = [k for k in params if k not in weights]
    extra = [k for k in weights if k not in params]
    wrong = [k for k in params if k in weights and tuple(weights[k].shape) != tuple(params[k].shape)]
    if not (missing or extra or wrong):
        return "loads", model_type, "FP8 release, scales not checked" if fp8 else ""
    parts = []
    if missing:
        parts.append(f"{len(missing)} missing, e.g. {missing[0]}")
    if extra:
        parts.append(f"{len(extra)} unexpected, e.g. {extra[0]}")
    if wrong:
        k = wrong[0]
        parts.append(f"{len(wrong)} shape mismatches, e.g. {k} {tuple(weights[k].shape)} vs {tuple(params[k].shape)}")
    return "fails", model_type, "; ".join(parts)[:200]


def revision():
    src = Path(mlx_vlm.__file__).resolve().parent.parent
    try:
        rev = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = ""
    from importlib.metadata import version

    return f"{version('mlx-vlm')}@{rev}" if rev else version("mlx-vlm")


def main():
    models = read("scorecard.csv")
    wanted = set(sys.argv[1:])
    rows = {r["id"]: r for r in read("mlx_support.csv")}
    rev, today = revision(), dt.date.today().isoformat()
    print(f"mlx-vlm {rev}")
    for m in models:
        if wanted and m["id"] not in wanted:
            continue
        status, model_type, detail = check(m["hf_repo"])
        rows[m["id"]] = dict(id=m["id"], mlx_vlm=status, model_type=model_type, detail=detail, checked=today, mlx_vlm_rev=rev)
        print(f"{status:12s} {m['id']:22s} {model_type:14s} {detail}")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for mid in sorted(rows):
            w.writerow(rows[mid])


if __name__ == "__main__":
    main()
