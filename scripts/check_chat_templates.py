"""Compare the chat template of every MLX and GGUF build with the original's (#7, 3a).

Poisoned chat templates are a documented attack class (arXiv 2602.04653):
Jinja logic in the template passes every weight scanner. For each model in
data/scorecard.csv this fetches the original's template and the template of
each build in data/mlx_builds.csv and data/gguf_builds.csv, compares them
after normalising whitespace, and scans every template for risky constructs.

    PY=~/src/mlx/.venv/bin/python   # needs jinja2 for the rendering comparison
    $PY scripts/check_chat_templates.py              # all models
    $PY scripts/check_chat_templates.py kolibri-1    # only these model ids

Writes data/template_checks.csv (one row per template) and, for every build
that differs, a unified diff in data/template_diffs/<repo>.diff.

WHERE THE TEMPLATE IS. Originals and MLX builds: chat_template.jinja,
chat_template.json, tokenizer_config.json or processor_config.json, in that
order; named templates (e.g. tool_use) are compared too. GGUF builds: the
tokenizer.chat_template* keys of the GGUF header, read with HTTP range
requests from the smallest .gguf file (the first shard of split files), so
no weights are downloaded.

STATUS.
  identical    same templates as the original after whitespace normalisation
  equivalent   the source differs (quoting, escapes, layout), but every template
               renders the same text as the original's for a fixed set of test
               conversations (with and without system prompt, multi-turn, tool
               definitions and calls, tool results; date fixed)
  differs      a template renders differently, or one cannot be rendered; the
               diff is linked
  suspicious   the build has a risky construct the original does not have
  not checked  no template found or the download failed (reason given)

Risky constructs (date or time, randomness, conditions on message content
that compare it with a literal, URLs, base64 or hex blobs, very long string
literals) found in an original are reported on its own row and do not make
its builds suspicious; many official templates insert the current date.
"""

import csv
import datetime as dt
import functools
import difflib
import hashlib
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SM_DATA", ROOT / "data"))  # SM_DATA: another data directory, e.g. a local overlay
OUT = DATA / "template_checks.csv"
DIFFS = DATA / "template_diffs"
FIELDS = ["id", "repo", "kind", "file", "status", "reason", "findings", "sha256", "checked"]
TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json", "tokenizer_config.json",
                  "processor_config.json")
CHUNK = 4 << 20
MAX_HEADER = 64 << 20

RISKS = {
    "date or time": re.compile(r"strftime_now|\bnow\s*\(|datetime|localtime|current_date|today"),
    "randomness": re.compile(r"\brandom\b|\bshuffle\b|\buuid"),
    "content condition": re.compile(
        r"\{%-?\s*(?:el)?if\b[^%]*(?:content|message)[^%]*"
        r"(?:==|!=|\bin\b|startswith|endswith|search|match)\s*['\"][^'\"]+['\"][^%]*%\}"
        r"|\{%-?\s*(?:el)?if\b[^%]*['\"][^'\"]+['\"]\s*(?:not\s+)?in\s+[^%]*content[^%]*%\}"),
    "URL": re.compile(r"https?://|www\.[a-z0-9-]+\.[a-z]"),
    "encoded blob": re.compile(r"[A-Za-z0-9+/]{100,}={0,2}|(?:\\x[0-9a-fA-F]{2}){16,}|\b[0-9a-fA-F]{64,}\b"),
    "long string literal": re.compile(r"'(?:[^'\\]|\\.){1500,}'|\"(?:[^\"\\]|\\.){1500,}\""),
}


def _token():
    path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return path.read_text().strip() if path.exists() else os.environ.get("HF_TOKEN")


TOKEN = _token()


def _get(url, headers=None):
    h = {"User-Agent": "sovereign-models/check_chat_templates"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        return r.read()


def repo_files(repo):
    info = json.loads(_get(f"https://huggingface.co/api/models/{repo}?blobs=true"))
    return {s["rfilename"]: s.get("size") for s in info.get("siblings", [])}


def hf_templates(repo, files):
    """Named templates of a transformers-style repo: {'default': ..., 'tool_use': ...}."""
    for name in TEMPLATE_FILES:
        if name not in files:
            continue
        raw = _get(f"https://huggingface.co/{repo}/resolve/main/{name}").decode("utf-8")
        if name.endswith(".jinja"):
            return {"default": raw}, name
        value = json.loads(raw).get("chat_template")
        if isinstance(value, str):
            return {"default": value}, name
        if isinstance(value, list):
            return {t["name"]: t["template"] for t in value}, name
    return {}, None


class RangeReader:
    """Sequential reads from a remote file, fetched in CHUNK-sized ranges."""

    def __init__(self, url):
        self.url, self.buf, self.start, self.pos = url, b"", 0, 0

    def read(self, n):
        while self.pos + n > self.start + len(self.buf):
            if self.start + len(self.buf) >= MAX_HEADER:
                raise ValueError("GGUF header larger than the read limit")
            lo = self.start + len(self.buf)
            data = _get(self.url, {"Range": f"bytes={lo}-{lo + CHUNK - 1}"})
            if not data:
                raise ValueError("unexpected end of file")
            # Keep only what is still needed.
            keep = self.pos - self.start
            self.buf, self.start = self.buf[keep:] + data, self.pos
        out = self.buf[self.pos - self.start : self.pos - self.start + n]
        self.pos += n
        return out


_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d"}


def _value(r, vtype, keep):
    if vtype == 8:
        n = struct.unpack("<Q", r.read(8))[0]
        s = r.read(n)
        return s.decode("utf-8", "replace") if keep else None
    if vtype == 9:
        itype, n = struct.unpack("<IQ", r.read(12))
        if itype in _SCALARS and itype != 8:
            r.read(n * struct.calcsize(_SCALARS[itype]))
        else:
            for _ in range(n):
                _value(r, itype, False)
        return None
    fmt = _SCALARS[vtype]
    return struct.unpack(fmt, r.read(struct.calcsize(fmt)))[0]


def gguf_templates(url):
    """tokenizer.chat_template* keys from a remote GGUF header."""
    r = RangeReader(url)
    if r.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version, _tensors, kv_count = struct.unpack("<IQQ", r.read(20))
    if version < 2:
        raise ValueError(f"GGUF version {version} not supported")
    out = {}
    for _ in range(kv_count):
        klen = struct.unpack("<Q", r.read(8))[0]
        key = r.read(klen).decode("utf-8", "replace")
        vtype = struct.unpack("<I", r.read(4))[0]
        keep = key.startswith("tokenizer.chat_template")
        value = _value(r, vtype, keep)
        if keep:
            name = key[len("tokenizer.chat_template"):].lstrip(".") or "default"
            out[name] = value
    return out


def pick_gguf(files):
    """The smallest model .gguf file (first shard of split files; no imatrix or mmproj)."""
    ggufs = {f: s for f, s in files.items() if f.endswith(".gguf")
             and not re.search(r"-0000[2-9]-of-|-000[1-9]\d-of-", f)
             and not re.search(r"mmproj|imatrix", f.lower())}
    return min(ggufs, key=lambda f: ggufs[f] or 0) if ggufs else None


def norm(t):
    """Whitespace-insensitive form; a \\n escape in a string literal equals a real newline."""
    return " ".join(t.replace("\\n", "\n").split())


def findings(templates):
    found = set()
    for t in templates.values():
        t = t.replace("\\n", "\n")
        for name, rx in RISKS.items():
            if rx.search(t):
                found.add(name)
    return found


def compare(orig, build):
    """identical / differs, and the unified diff of the differing templates."""
    if set(orig) != set(build) or any(norm(orig[k]) != norm(build[k]) for k in orig):
        diff = []
        for k in sorted(set(orig) | set(build)):
            a, b = orig.get(k, ""), build.get(k, "")
            if norm(a) != norm(b):
                diff += difflib.unified_diff(a.splitlines(), b.splitlines(),
                                             f"original/{k}", f"build/{k}", lineterm="")
        return "differs", "\n".join(diff) + "\n"
    return "identical", None


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
CASES = [
    ([{"role": "user", "content": "Hallo, wie geht es dir?"}], None),
    ([{"role": "system", "content": "Antworte kurz."},
      {"role": "user", "content": "Was ist 2+2?"},
      {"role": "assistant", "content": "4."},
      {"role": "user", "content": "Und 3+3?"}], None),
    ([{"role": "user", "content": "Wie ist das Wetter in Wien?"},
      {"role": "assistant", "content": "", "tool_calls": [{"id": "call0001x", "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "Wien"}}}]},
      {"role": "tool", "tool_call_id": "call0001x", "content": "{\"temp\": 14}"},
      {"role": "assistant", "content": "In Wien hat es 14 Grad."}], TOOLS),
]
SPECIAL = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"}


@functools.lru_cache(maxsize=None)
def _env():
    """A sandboxed Jinja environment like transformers', with a fixed date."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **k: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii)
    env.globals.update(raise_exception=raise_exception,
                       strftime_now=lambda fmt: dt.datetime(2026, 1, 15, 12, 0).strftime(fmt))
    return env


def renders(templates):
    """Rendered test conversations per template, or None if Jinja is missing."""
    try:
        env = _env()
    except ImportError:
        return None
    out = {}
    for name, src in templates.items():
        try:
            tpl = env.from_string(src)
        except Exception as e:  # noqa: BLE001 - a broken template is a finding
            out[name] = [f"<template error: {type(e).__name__}>"]
            continue
        res = []
        for msgs, tools in CASES:
            for gen in (True, False):
                try:
                    res.append(tpl.render(messages=msgs, tools=tools, add_generation_prompt=gen,
                                          **SPECIAL))
                except Exception as e:  # noqa: BLE001
                    res.append(f"<render error: {type(e).__name__}: {e}>")
        out[name] = res
    return out


def digest(templates):
    return hashlib.sha256(json.dumps(templates, sort_keys=True).encode()).hexdigest()[:16]


def check_model(mid, original, builds, today):
    rows = []
    try:
        orig, src = hf_templates(original, repo_files(original))
    except (urllib.error.URLError, ValueError, KeyError) as e:
        orig, src = {}, None
        rows.append(dict(id=mid, repo=original, kind="original", status="not checked",
                         reason=f"download failed: {e}", checked=today))
    if src:
        f = findings(orig)
        rows.append(dict(id=mid, repo=original, kind="original", file=src, status="reference",
                         findings=";".join(sorted(f)), sha256=digest(orig), checked=today))
    elif not rows:
        rows.append(dict(id=mid, repo=original, kind="original", status="not checked",
                         reason="no chat template in the repo", checked=today))
    orig_findings = findings(orig) if orig else set()

    for kind, repo in builds:
        row = dict(id=mid, repo=repo, kind=kind, checked=today)
        try:
            files = repo_files(repo)
            if kind == "gguf":
                fname = pick_gguf(files)
                if not fname:
                    raise ValueError("no .gguf file")
                tpl = gguf_templates(f"https://huggingface.co/{repo}/resolve/main/{fname}")
            else:
                tpl, fname = hf_templates(repo, files)
            row["file"] = fname
        except (urllib.error.URLError, ValueError, struct.error, KeyError) as e:
            rows.append(row | dict(status="not checked", reason=f"download failed: {e}"))
            continue
        if not tpl:
            rows.append(row | dict(status="not checked", reason="no chat template in the build"))
            continue
        f = findings(tpl)
        row.update(findings=";".join(sorted(f)), sha256=digest(tpl))
        if not orig:
            rows.append(row | dict(status="not checked", reason="original has no template to compare"))
            continue
        new = f - orig_findings
        status, diff = compare(orig, tpl)
        if status == "differs":
            ro, rb = renders(orig), renders(tpl)
            if ro is not None and ro == rb:
                status = "equivalent"
        if diff:
            DIFFS.mkdir(parents=True, exist_ok=True)
            path = DIFFS / (repo.replace("/", "__") + ".diff")
            path.write_text(diff)
            row["reason"] = f"diff: {path.relative_to(DATA.parent)}"
        if new:
            status = "suspicious"
            row["reason"] = "; ".join(filter(None, [f"new in the build: {', '.join(sorted(new))}",
                                                    row.get("reason")]))
        rows.append(row | dict(status=status))
    return rows


def main():
    today = dt.date.today().isoformat()
    models = list(csv.DictReader(open(DATA / "scorecard.csv")))
    builds = {}
    for kind, name in (("mlx", "mlx_builds.csv"), ("gguf", "gguf_builds.csv")):
        if not (DATA / name).exists():
            continue
        for b in csv.DictReader(open(DATA / name)):
            builds.setdefault(b["id"], []).append((kind, b["repo"]))
    wanted = set(sys.argv[1:])
    old = []
    if wanted and OUT.exists():
        old = [r for r in csv.DictReader(open(OUT)) if r["id"] not in wanted]
    rows = []
    for m in models:
        if wanted and m["id"] not in wanted and m["family_id"] not in wanted:
            continue
        got = check_model(m["id"], m["hf_repo"], builds.get(m["id"], []), today)
        for r in got:
            print(f"{r['status']:12s} {r['kind']:8s} {r['repo']}  {r.get('reason', '')}", flush=True)
        rows += got
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        w.writeheader()
        for r in old + rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
