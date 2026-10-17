# MÖVE und Sovereign Models: Abgleich

Stand: 17.10.2026. MÖVE-Daten: öffentliche API von moeve.bundesdruckerei.de (`/api/results/models`,
`/api/results/tasks/transparency`, `generated_at` 2026-10-08). Sovereign Models: Datensatz
[here-be-dragons-ai/sovereign-models-scorecard](https://huggingface.co/datasets/here-be-dragons-ai/sovereign-models-scorecard)
(CC BY 4.0), Methodik in `METHODOLOGY.md`.

## Kurzfassung

- Beide Projekte lassen sich über einen gemeinsamen Schlüssel verbinden: MÖVE führt pro Modell
  eine `model_card_url` auf dem Hugging Face Hub, Sovereign Models das `hf_repo` des Originals.
- Die Kriterien ergänzen sich. MÖVE misst Leistung auf deutschen Verwaltungsaufgaben,
  Halluzination, Energie, Werte und Anbieter-Transparenz. Sovereign Models erfasst Sitz und
  Kontrolle des Anbieters, Trainingsort, Offenheit der Daten, AI-Act-Zusammenfassung,
  Teacher-Modelle, Reproduzierbarkeit und prüft MLX- und GGUF-Builds gegen das Original.
- Befunde zu den Testbedingungen betreffen gerade die europäischen Modelle (Abschnitt 1): Der
  getestete Teuken-Build hat einen defekten Tokenizer und ein falsches Prompt-Format; die
  europäischen Open-Weight-Modelle laufen meist als 4-Bit-Builds Dritter, die US-Modelle über
  Anbieter-APIs.
- 23 europäische Open-Weight-Modelle mit Deutsch fehlen in MÖVE, davon 11 aus unserem Tier A (Abschnitt 2).
- Die Faktenangaben stimmen weitgehend überein; eine Lizenzangabe weicht ab (Abschnitt 3).

## 1. Welche Builds MÖVE getestet hat

MÖVE testet die US-Anbietermodelle (GPT, Claude) über deren APIs in voller Präzision. Die
europäischen Open-Weight-Modelle laufen fast alle als 4-Bit-Quantisierungen Dritter:

| Modell in MÖVE | getesteter Build | Herausgeber | Template gegen Original |
|---|---|---|---|
| Teuken 7B Commercial v0.4 | `fl0id/teuken-7b-instruct-commercial-v0.4` (Ollama) | Einzelperson | **falsch, Tokenizer defekt**, siehe 1.1 |
| EuroLLM 1.7B | `cas/eurollm-1.7b-instruct-q8` (Ollama) | Einzelperson | ChatML, passt zum Format |
| EuroLLM 9B | `bartowski/EuroLLM-9B-Instruct-GGUF` Q4_K_M | Einzelperson | nicht prüfbar (Original gated) |
| EuroLLM 22B 2512 | `mradermacher/EuroLLM-22B-Instruct-2512-GGUF` Q4_K_M | Einzelperson | identisch |
| Apertus 8B / 70B | `unsloth/Apertus-*-Instruct-2509-GGUF` Q4_K_M | Organisation | abweichend (Unsloth-Fixes für Tool-Calls; für Einzel-Turns ohne Tools wohl ohne Wirkung) |
| SauerkrautLM Nemo 12B | `cyberwald/sauerkrautlm-nemo-12b-instruct:q4_0` (Ollama) | Einzelperson | Mistral-[INST]-Format |
| SauerkrautLM Mixtral 8x7B | `TheBloke/SauerkrautLM-Mixtral-8x7B-AWQ` | Einzelperson | kein Chat-Template im Build |
| SmolLM3 3B | `unsloth/SmolLM3-3B-128K-GGUF` | Organisation | abweichend (Unsloth-Fixes) |

### 1.1 Teuken: defekter Tokenizer und falsches Prompt-Format

Geprüft mit genau der Datei, die MÖVE verwendet hat (GGUF-Blob aus der Ollama-Registry,
SHA-256 `5605681719e5…`, gleich dem Registry-Digest), ausgeführt mit llama.cpp.

**Tokenizer.** Der Build zerlegt Text anders als Teukens Original-Tokenizer (`tokenizer.model`
von openGPT-X). Von sechs Testtexten wird keiner gleich tokenisiert, z. B.:

| Text | Original | GGUF-Build |
|---|---|---|
| amerikanischen Geschichte | `▁amerikanischen` `▁Geschichte` (2) | ` am` `er` `ika` `nischen` ` G` `es` `chich` `te` (8) |
| Nordrhein-Westfalen und Niedersachsen | 5 Tokens | 11 Tokens |
| völkerrechtlichen Vertrages | 5 Tokens | 9 Tokens |
| 2,5 km (geschütztes Leerzeichen) | `▁km` | zwei rohe Bytes |

Die Zuordnung von ID zu Token ist dagegen gleich (402 von 402 Stichproben); der Fehler liegt in
der Zerlegung. Das Modell erhält so Token-Folgen, die es im Training nicht gesehen hat. Sichtbar
wird das an Schreibfehlern, auch beim Abschreiben aus dem Kontext („völkerrechltichen“,
„Nordrhien-Westfalen und Nidersachsen“, „amerikaanischen“, „letzzten“). Mit den Token-IDs des
Original-Tokenizers verschwinden sie.

**Prompt-Format.** Teuken erwartet laut Model Card und `gptx_tokenizer.py`
`System: …\nUser: …\nAssistant: ` ohne Sondertokens. Der Build setzt ein ChatML-Template
(`<|im_start|>…<|im_end|>`); diese Marker sind in Teukens Vokabular keine Sondertokens und werden
in 7 bis 10 Textstücke zerlegt. Da MÖVE keine System-Nachricht sendet, setzt Ollama den
Standard-System-Prompt des Builds ein, der eine englische Beispielrunde enthält
(`System: A chat between a human and an artificial intelligence assistant.\nUser: Hi!\nAssistant.`).
MÖVE nutzt die Chat-API von Ollama (`lmbench/models/ollama.py`), das Template greift also.

**Messung auf MÖVEs öffentlicher Stichprobe** (`datasets/german-quad-sample`, 10 Fragen,
MÖVE-Prompt, Temperatur 0, je eine Kombination aus Template und Tokenizer):

| Bedingung | Antwort enthalten | Token-F1 |
|---|---|---|
| ChatML + GGUF-Tokenizer (wie in MÖVE) | 4/10 | 0,27 |
| offizielles Format + GGUF-Tokenizer | 4/10 | 0,35 |
| ChatML + Original-Tokenizer | 4/10 | 0,25 |
| offizielles Format + Original-Tokenizer | 5/10 | 0,27 |

Auf zehn Fragen unterscheiden sich die Bedingungen nicht verlässlich; eine Verzerrung der
MÖVE-Werte für Teuken lässt sich damit weder belegen noch ausschließen. Belegt sind der
Tokenizer-Defekt und seine Folgen für die Textqualität. Ob er Kennzahlen wie BERTScore,
LLM-Judge-Urteile oder die Sprachprüfung bei Zusammenfassungen spürbar senkt, müsste ein Lauf
auf den vollständigen Datensätzen zeigen.

**Messung mit MÖVE LMBench selbst** (Commit `2cb4012`, beide öffentlichen Stichproben, alle
Metriken der Aufgabe außer den Judge-Metriken, die einen OpenAI-Schlüssel brauchen; Temperatur 0;
dieselbe GGUF-Datei hinter einem OpenAI-kompatiblen Proxy, einmal wie in MÖVE, einmal korrekt
tokenisiert im offiziellen Format):

| Datensatz | Metrik | wie in MÖVE | korrekt | korrekt besser |
|---|---|---|---|---|
| EUR-Lex-Sum (Zusammenfassung) | BERTScore F1 | 0,704 | 0,720 | 7 von 10 Dokumenten |
| | SemScore | 0,796 | 0,832 | 6 von 10 |
| | ROUGE-1 / ROUGE-Lsum | 0,203 / 0,172 | 0,263 / 0,226 | (Korpuswert) |
| GermanQuAD (Fragen) | BERTScore F1 | 0,682 | 0,688 | |
| | ROUGE-1 | 0,258 | 0,253 | |

Beim Zusammenfassen liegt die korrekte Konfiguration durchgehend vorn, beim Beantworten von
Fragen nicht. Mit zehn Dokumenten ist auch das nur ein Hinweis (Vorzeichentest 7/10: p ≈ 0,34),
kein Nachweis; MÖVEs vollständige und private Datensätze könnten es klären.

Außerdem ist die Datei laut Ollama-Manifest `F16` (14,9 GB); MÖVE führt sie mit `quantization: 4`.

### 1.2 Quantisierung und Herkunft als Störgröße

Wenn Open-Weight-Modelle in Q4 von Drittanbietern gegen Anbieter-APIs in voller Präzision
antreten, vermischt das Rangliste und Testbedingung. Für die europäischen Modelle gibt es
meist offizielle oder kuratierte Builds (z. B. GGUF in der Organisation des Herstellers), oder
sie lassen sich in BF16/FP8 betreiben. Sovereign Models führt pro Modell die Herkunft der
Builds (official, curated, organisation, individual) und das Ergebnis des Template-Abgleichs.

## 2. Europäische Modelle mit Deutsch, die in MÖVE fehlen

Aus der Scorecard (Tier nach unserer Methodik; „ältere Version“: MÖVE testet eine frühere
Generation desselben Anbieters):

| Tier | Modell | Hub-Repo | |
|---|---|---|---|
| A | ALIA 40B | BSC-LT/ALIA-40b-instruct-2606 | |
| A | Apertus 1.5 70B | swiss-ai/Apertus-v1.5-70B | ältere Version in MÖVE |
| A | Apertus 1.5 8B | swiss-ai/Apertus-v1.5-8B | ältere Version in MÖVE |
| A | Lucie 7B | OpenLLM-France/Lucie-7B-Instruct-v1.1 | |
| A | Luciole 23B | OpenLLM-France/Luciole-23B-Instruct-1.1 | |
| A | Luciole 23B Thinking | OpenLLM-France/Luciole-23B-Thinking-1.1 | |
| A | Luciole 8B | OpenLLM-France/Luciole-8B-Instruct-1.1 | |
| A | Pleias RAG 1B | PleIAs/Pleias-RAG-1B | |
| A | Salamandra 7B | BSC-LT/salamandra-7b-instruct | |
| A | TildeOpen 30B | TildeAI/TildeOpen-30b-64k | |
| A | Velvet 14B | AlmawaveLabs/Velvet-14B | |
| B | Bielik 11B v3 | speakleash/Bielik-11B-v3.0-Instruct | |
| B | Domyn Small | domyn/Domyn-Small-v1.0 | |
| B | EuroLLM 9B 2512 | utter-project/EuroLLM-9B-Instruct-2512 | ältere Version in MÖVE |
| B | Helium 1 2B | kyutai/helium-1-2b | |
| B | Magistral Small 2509 | mistralai/Magistral-Small-2509 | ältere Mistral-Modelle in MÖVE |
| B | Ministral 3 14B | mistralai/Ministral-3-14B-Instruct-2512 | ältere Mistral-Modelle in MÖVE |
| B | Mistral Small 3.2 | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | MÖVE: Small 3.1 |
| B | Mistral Small 4 | mistralai/Mistral-Small-4-119B-2603 | MÖVE: Small 3.1 |
| B | sui 1 24B | ellamind/sui-1-24b | |
| C | Llama-Krikri 8B v1.5 | ilsp/Llama-Krikri-8B-Instruct-v1.5 | |
| C | OpenEuroLLM 9B | openeurollm/oellm-9b-256k-sft | |
| excluded | Teuken 7B v0.6 (nicht kommerziell) | openGPT-X/Teuken-7B-instruct-v0.6 | MÖVE: v0.4 commercial |

## 3. Abgleich der Faktenangaben

- **Code of Practice:** Die Werte von `code_of_practice` in MÖVE stimmen für alle 51 Modelle mit
  der Unterzeichnerliste der Kommission (Stand 07.10.2026) überein.
- **Lizenz:** DeepSeek R1 und die R1-Distill-Modelle stehen in MÖVE mit „Apache 2.0“; laut
  Hugging Face (`deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`) ist es MIT.
- **Gemeinsame Modelle:** Kolibri 1 und EuroLLM 22B 2512 sind in beiden Datensätzen exakt gleich.

## 4. Mögliche Zusammenarbeit

1. **Verlinkung über den Hub-Schlüssel:** Sovereign Models zeigt bei gemeinsamen Modellen einen
   Link auf das MÖVE-Ergebnis; MÖVE kann auf die Souveränitäts-Angaben (Sitz, Kontrolle,
   Trainingsort, Daten, Tier) verweisen oder den CC-BY-Datensatz einbinden.
2. **Testbedingungen:** offizielle oder geprüfte Builds für die europäischen Modelle, gleiche
   Präzision für alle Open-Weight-Modelle, Prompt-Format gegen das Original prüfen
   (wir stellen den Template-Abgleich gern bereit).
3. **Modellauswahl:** Kandidaten aus Abschnitt 2, vor allem die aktuellen Generationen
   (Apertus 1.5, EuroLLM 2512, Mistral Small 3.2/4) und die Tier-A-Modelle.
4. **Offene Frage an MÖVE:** Unter welcher Lizenz stehen die Ergebnisdaten der API?
