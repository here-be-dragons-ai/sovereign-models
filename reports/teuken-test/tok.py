import json, urllib.request
import pandas as pd, sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="/tmp/sm-teuken/orig/tokenizer.model")
def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:8099" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())
df = pd.read_parquet("/tmp/sm-teuken/moeve/dataset.parquet")
texts = ["völkerrechtlichen Vertrages", "Nordrhein-Westfalen und Niedersachsen", "amerikanischen Geschichte", "2,5 km", "Waldelefanten"] + [df.iloc[0]["context"][:400]]
print("vocab: original", sp.get_piece_size())
same = 0
for t in texts:
    o = sp.encode(t, out_type=str)
    g = [x["piece"] for x in post("/tokenize", {"content": t, "with_pieces": True})["tokens"]]
    ok = [p.replace("▁", " ") for p in o] == g
    same += ok
    if len(t) < 60 or not ok:
        print(f"\n{t[:60]!r}  identical={ok}\n  original ({len(o)}): {o[:16]}\n  gguf     ({len(g)}): {g[:16]}")
print(f"\nidentical tokenization: {same}/{len(texts)}")
