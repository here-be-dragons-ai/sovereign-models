import json, sys, importlib.util
import sentencepiece as spm
spec = importlib.util.spec_from_file_location("r", "/tmp/sm-teuken-scripts/run.py")
sys.argv = ["run.py", "lib"]; r = importlib.util.module_from_spec(spec)
src = open("/tmp/sm-teuken-scripts/run.py").read().split('if sys.argv[1:] == ["tokens"]:')[0]
exec(src, r.__dict__)
sp = spm.SentencePieceProcessor(model_file="/tmp/sm-teuken/orig/tokenizer.model")
conds = [("chatml + GGUF tokenizer (MÖVE)", r.chatml, False), ("official + GGUF tokenizer", r.official, False),
         ("chatml + original tokenizer", r.chatml, True), ("official + original tokenizer", r.official, True)]
out = {}
for name, fmt, orig in conds:
    rows = []
    for _, x in r.df.iterrows():
        gold = x["answer"] if isinstance(x["answer"], str) else list(x["answer"])[0]
        p = fmt(r.tmpl.format(context=x["context"], question=x["question"]))
        o = r.post("/completion", {"prompt": sp.encode(p) if orig else p, "n_predict": 96, "temperature": 0,
                                   "cache_prompt": False, "stop": ["\nUser:", "<|im_end|>", "</s>"]})
        a = o["content"].strip(); rows.append(dict(gold=gold, ans=a, contains=r.norm(gold) in r.norm(a), f1=r.f1(a, gold)))
    out[name] = rows
    print(f"{name:34s} contained {sum(z['contains'] for z in rows)}/10   mean F1 {sum(z['f1'] for z in rows)/10:.2f}")
json.dump(out, open("/tmp/sm-teuken/results2.json", "w"), ensure_ascii=False, indent=1)
for i in range(10):
    print(f"\nQ{i+1} gold: {out[conds[0][0]][i]['gold'][:70]}")
    for name, *_ in conds: print(f"   {name[:30]:30s}: {out[name][i]['ans'][:120]!r}")
