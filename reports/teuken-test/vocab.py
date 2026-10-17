import json, urllib.request, random
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="/tmp/sm-teuken/orig/tokenizer.model")
def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:8099" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())
props = json.loads(urllib.request.urlopen("http://127.0.0.1:8099/props").read())
print("bos/eos ids in sp:", sp.bos_id(), sp.eos_id(), "pad", sp.pad_id(), "unk", sp.unk_id())
random.seed(0); ids = random.sample(range(300, sp.get_piece_size()), 400) + [sp.encode("amerikanischen")[-1], sp.encode("Niedersachsen")[-1]]
bad = 0
for i in ids:
    want = sp.id_to_piece(i).replace("▁", " ")
    got = post("/detokenize", {"tokens": [i]})["content"]
    if got != want and not want.startswith("<"): bad += 1; (bad <= 5 and print("  differs", i, repr(want), repr(got)))
print(f"id->piece identical for {len(ids)-bad}/{len(ids)} sampled ids")
