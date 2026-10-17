"""Teuken v0.4 commercial (the exact GGUF MÖVE used), two prompt formats, MÖVE's GermanQuAD sample."""
import json, re, sys, string, collections, urllib.request
import pandas as pd

SERVER = "http://127.0.0.1:8099"
df = pd.read_parquet("/tmp/sm-teuken/moeve/dataset.parquet")
tmpl = open("/tmp/sm-teuken/moeve/user_prompt.txt").read()
# Ollama's legacy template of fl0id/teuken-7b-instruct-commercial-v0.4 with the build's SYSTEM blob
# (literal backslash-n as stored), used because MÖVE sends no system message.
OLLAMA_SYSTEM = r"System: A chat between a human and an artificial intelligence assistant.\nUser: Hi!\nAssistant."
def chatml(p):
    return f"<|im_start|>system\n{OLLAMA_SYSTEM}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
DE = ("Ein Gespräch zwischen einem Menschen und einem Assistenten mit künstlicher Intelligenz."
      " Der Assistent gibt hilfreiche und höfliche Antworten auf die Fragen des Menschen.")
def official(p):  # model card / gptx_tokenizer.py, chat_template "DE"
    return f"System: {DE}\nUser: {p}\nAssistant: "

def post(path, body):
    r = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation) | {"„", "“", "–"})
    s = re.sub(r"\b(der|die|das|ein|eine|einen|dem|den|des)\b", " ", s)
    return " ".join(s.split())
def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g); n = sum(common.values())
    if not p or not g or n == 0: return 0.0
    pr, rc = n / len(p), n / len(g); return 2 * pr * rc / (pr + rc)

if sys.argv[1:] == ["tokens"]:
    for s in ["<|im_start|>system", "<|im_end|>", "System: Hallo\nUser:"]:
        toks = post("/tokenize", {"content": s, "with_pieces": True})["tokens"]
        print(repr(s), "->", len(toks), "tokens:", [t.get("piece") for t in toks][:12])
    sys.exit()

res = {"chatml (MÖVE/Ollama)": chatml, "official Teuken format": official}
out = {}
for name, fmt in res.items():
    rows = []
    for _, r in df.iterrows():
        gold = r["answer"] if isinstance(r["answer"], str) else list(r["answer"])[0]
        prompt = fmt(tmpl.format(context=r["context"], question=r["question"]))
        o = post("/completion", {"prompt": prompt, "n_predict": 96, "temperature": 0, "cache_prompt": False,
                                 "stop": ["\nUser:", "<|im_end|>", "</s>"]})
        ans = o["content"].strip()
        rows.append(dict(q=r["question"], gold=gold, ans=ans, contains=norm(gold) in norm(ans), f1=f1(ans, gold)))
    out[name] = rows
    c = sum(x["contains"] for x in rows); m = sum(x["f1"] for x in rows) / len(rows)
    print(f"{name:26s} answer contained {c}/{len(rows)}   mean token F1 {m:.2f}")
json.dump(out, open("/tmp/sm-teuken/results.json", "w"), ensure_ascii=False, indent=1)
for i in range(len(df)):
    print(f"\nQ{i+1}: {out[list(res)[0]][i]['q'][:90]}\n   gold:     {out[list(res)[0]][i]['gold'][:90]}")
    for name in res: print(f"   {name[:7]:8s}: {out[name][i]['ans'][:150]!r}")
