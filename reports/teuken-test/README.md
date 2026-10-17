# Teuken-Test zum MÖVE-Abgleich (2026-10-17)

- Modell: GGUF-Blob von `fl0id/teuken-7b-instruct-commercial-v0.4` aus der Ollama-Registry,
  SHA-256 `5605681719e54f3d8d789a4290254222b6447a65a713e4ed2a576033c07f104b`.
- Ausführung: `llama-server -c 4096 --port 8099` (llama.cpp, Build 11378), Temperatur 0.
- Daten: `datasets/german-quad-sample` aus Bundesdruckerei-GmbH/moeve-lmbench (10 Zeilen aus GermanQuAD),
  Prompt `user_prompt.txt` von dort.
- Original-Tokenizer: `tokenizer.model` aus openGPT-X/Teuken-7B-instruct-commercial-v0.4, gelesen mit `sentencepiece`.

`tok.py` vergleicht die Tokenisierung, `vocab.py` die Zuordnung von ID zu Token, `run2.py` misst die vier
Kombinationen aus Template und Tokenizer; `results2.json` enthält alle Antworten.

## MÖVE LMBench

`proxy.py` stellt dieselbe GGUF-Datei als OpenAI-kompatible API mit zwei Modellnamen bereit:
`teuken-moeve` (Ollamas Template und Standard-System-Prompt des Builds, Text an den GGUF-Tokenizer)
und `teuken-correct` (offizielles Format, Token-IDs des Original-Tokenizers). `bench.py` ruft
`EvaluationRunner` von MÖVE LMBench (Commit `2cb4012`) mit allen Metriken der Aufgabe außer den
Judge-Metriken auf (kein OpenAI-Schlüssel); `bench-all.sh` läuft beide Modelle auf beiden
öffentlichen Stichproben. Ergebnisse: `lmbench/<modell>/<datensatz>/metrics.json` und `output.csv`.

Die Skripte erwarten die Dateien unter `/tmp/sm-teuken` (Modell, `tokenizer.model`, Stichprobe)
und LMBench unter `/tmp/sm-lmbench/src`; Pfade bei Bedarf anpassen.
