"""OpenAI-compatible proxy in front of llama-server for two Teuken conditions.

model "teuken-moeve":   what MÖVE ran. Ollama's legacy template of fl0id/teuken-7b-instruct-commercial-v0.4
                        (ChatML markers, the build's SYSTEM blob when no system message is sent), sent as text,
                        so the GGUF's tokenizer splits it; BOS added by llama-server as for any text prompt.
model "teuken-correct": Teuken's documented format (gptx_tokenizer.py, chat_template "DE"), tokenized with the
                        original SentencePiece model and sent as token ids (no BOS, like the original tokenizer).
"""

import json
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sentencepiece as spm

LLAMA = "http://127.0.0.1:8099"
SP = spm.SentencePieceProcessor(model_file="/tmp/sm-teuken/orig/tokenizer.model")
OLLAMA_SYSTEM = r"System: A chat between a human and an artificial intelligence assistant.\nUser: Hi!\nAssistant."
DE = ("Ein Gespräch zwischen einem Menschen und einem Assistenten mit künstlicher Intelligenz."
      " Der Assistent gibt hilfreiche und höfliche Antworten auf die Fragen des Menschen.")


def text(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def render(model, messages):
    system = next((text(m["content"]) for m in messages if m["role"] == "system"), None)
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    if model == "teuken-moeve":
        # Ollama legacy template: {{ if .System }}…{{ end }}{{ if .Prompt }}…{{ end }}<|im_start|>assistant\n
        # .Prompt is the last user message; .System the request's system message or the Modelfile SYSTEM.
        sys_text = system if system is not None else OLLAMA_SYSTEM
        prompt = next((text(m["content"]) for m in reversed(turns) if m["role"] == "user"), "")
        out = f"<|im_start|>system\n{sys_text}<|im_end|>\n" if sys_text else ""
        out += f"<|im_start|>user\n{prompt}<|im_end|>\n" if prompt else ""
        return out + "<|im_start|>assistant\n", None
    if model == "teuken-correct":
        out = f"System: {system or DE}\n"
        for m in turns:
            role = m["role"].capitalize()
            out += f"{role}: {text(m['content'])}" + ("</s>" if role == "Assistant" else "") + "\n"
        out += "Assistant: "
        return out, SP.encode(out)
    raise ValueError(f"unknown model {model!r}")


def complete(model, messages, n_predict, temperature):
    prompt, ids = render(model, messages)
    body = {"prompt": ids if ids is not None else prompt, "n_predict": n_predict,
            "temperature": temperature, "cache_prompt": False}
    if model == "teuken-correct":
        body["stop"] = ["\nUser:"]
    req = urllib.request.Request(LLAMA + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return r["content"], r.get("tokens_evaluated", 0), r.get("tokens_predicted", 0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("proxy: " + fmt % args + "\n")

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in ("teuken-moeve", "teuken-correct")]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            n = req.get("max_completion_tokens") or req.get("max_tokens") or 512
            t = req.get("temperature")
            content, pt, ct = complete(req["model"], req["messages"], int(n), 0.0 if t is None else float(t))
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": {"message": str(e)}})
        self._send(200, {
            "id": f"chatcmpl-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
            "model": req["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        })


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8100), Handler).serve_forever()
