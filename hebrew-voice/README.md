# Hebrew low-latency voice pipeline prototype

`audio (he) -> STT -> LLM (streamed) -> chunker -> TTS`, with per-stage latency and **time-to-first-audio (TTFA)**
measured from the *end of user speech*. Everything below ran on this box (8 vCPU, 15 GB RAM, **no GPU**; Gemini API key added 2026-10-26).

```
. .venv/bin/activate
python pipeline.py test_audio/*_avri.wav --stt sherpa --llm gemma4e2b --tts blue --first-words 4 --reps 2
python eval_stt_and_tts_roundtrip.py          # WER + TTS intelligibility
```

| flag | options |
|---|---|
| `--stt` | `sherpa` = ivrit.ai Hebrew streaming Zipformer (local, CPU, int8) · `fw` = ivrit-ai Whisper-large-v3-turbo CT2 via faster-whisper (`HV_FW_COMPUTE=float32|int8`) |
| `--llm` | `gemma4e2b`, `gemma4b`, `gemma1b` (local llama.cpp GGUF) · `openai` (OPENAI_API_KEY, OPENAI_MODEL) · `groq` (GROQ_API_KEY) · `gemini:<model>[:minimal|budget0]` (GEMINI_API_KEY, native streaming API, persistent connection, thinking minimised; e.g. `gemini:gemini-3.1-flash-lite:minimal`) · `gemini` (legacy, OpenAI-compat endpoint) · `gguf:/path` |
| `--tts` | `gemini` = Gemini TTS (GEMINI_API_KEY, `GEMINI_TTS_MODEL` default gemini-3.8-flash-lite-tts) · `blue` = BlueTTS 2.5 ONNX (local, MIT) · `edge` = Edge he-IL voices (online, unofficial, keyless, testing only) · `elevenlabs` (ELEVENLABS_API_KEY, model eleven_v3/eleven_v4_turbo – Flash v2.5 has no Hebrew) |
| `--voice` | BlueTTS voice: `noa`, `adam`, `daniel`, `lily`, `female`, `libri_*`, `witch_gacrux`, `witch_pulcherrima` or a .json path (for `--tts gemini`: a Gemini voice name, e.g. `Gacrux`) |
| `--voice-fx` | `none` · `witch` · `witch-lite` · `witch-rb` (see voice_fx.py) |
| `--persona` | `none` · `witch` (spooky old-witch system prompt, still 1-2 short sentences) |
| `--speed` | BlueTTS speaking rate (default 1.0) |
| `--first-words N` | start TTS on the first N words even before punctuation |

Files: `pipeline.py` (CLI/harness), `hv_backends.py` (backends + chunker), `eval_stt_and_tts_roundtrip.py`, `eval_results.txt`,
`results_*.json` (raw per-turn numbers), `out/*.wav` (reply audio), `test_audio/` (6 clips generated with edge-tts he-IL Avri/Hila).

## Measured results (this box, CPU only)
Test input = 6 short synthetic clips (3-5 s) -> optimistic for STT. TTFA excludes VAD end-of-speech hangover (add ~300-500 ms live).

| STT | LLM | TTS | n | TTFA median (min-max) |
|---|---|---|---|---|
| sherpa Zipformer (streaming) | Gemma-4 E2B Q4 | BlueTTS (local) | 12 | **940 ms (716-1476)** |
| sherpa Zipformer | Gemma-3 4B Q4 | BlueTTS | 6 | 1379 ms (1058-1423) |
| sherpa Zipformer | Gemma-4 E2B | edge-tts (cloud) | 6 | 1424 ms (567-2636; TTS first-byte 160-2044 ms, very jittery) |
| faster-whisper ivrit turbo fp32 | Gemma-4 E2B | BlueTTS | 2 | ~5.9 s (STT alone 4.3-5.7 s) |
| faster-whisper ivrit turbo int8 | Gemma-3 1B | BlueTTS | 3 | 2769 ms but **int8 hallucinates English tokens (WER 0.43)** |

Stage numbers (best local combo): STT final after end of speech 1-39 ms (streaming; partials decoded while speaking);
LLM first token 55-400 ms, first speakable chunk 410-1020 ms; BlueTTS first chunk 250-460 ms (RTF ~0.13 after warm-up).
STT accuracy on the clips: sherpa mean WER 0.045 (no punctuation); fw float32 ~0.03; fw int8 0.43.
TTS round-trip (reply audio re-transcribed by Whisper) mostly intelligible; occasional garble on short words.

## Gemini Flash-Lite vs local Gemma-4 E2B (2026-10-26, same session, same 6 clips x 2 reps)
STT = sherpa Zipformer, TTS = BlueTTS, `--first-words 4`. Raw: `results_oct26_sherpa_*_blue.json`.

```
python pipeline.py test_audio/q*_{avri,hila}.wav --reps 2 --stt sherpa --tts blue --first-words 4 \
   --llm gemini:gemini-3.1-flash-lite:minimal --json results_oct26_sherpa_gemini-gemini-3.1-flash-lite-minimal_blue.json
```

| LLM | TTFA median (min-max) | LLM first token med (min-max) | first speakable chunk med (min-max) | TTS first chunk med (min-max) | STT final |
|---|---|---|---|---|---|
| Gemma-4 E2B Q4 (local CPU) | **872 ms (572-1215)** | 144 ms (40-271)* | 590 ms (339-853) | 276 ms (232-334) | 1-44 ms |
| gemini-3.1-flash-lite, thinkingLevel=minimal | 1170 ms (934-1502) | 814 ms (614-1085) | 848 ms (614-1141) | 310 ms (275-350) | 1-30 ms |
| gemini-3.5-flash-lite, thinkingLevel=minimal | 1168 ms (904-1457) | 767 ms (549-1003) | 819 ms (608-1071) | 302 ms (247-359) | 1-38 ms |

\* 2nd reps hit llama.cpp prompt cache; 1st-rep Gemma TTFT 174-271 ms (rep-1 TTFA median 890 ms).
Gemini streams in large pieces, so the first SSE chunk already contains a speakable phrase (first chunk ~= first token).

Network: warm HTTPS round trip to generativelanguage.googleapis.com (GET model metadata / countTokens on a kept-alive
connection) = median ~116-122 ms (min 103, max 333, n=15); cold TLS handshake ~20 ms, cold TTFB 131 ms. So of Gemini's
~550-1085 ms first token, roughly 100-120 ms is network path from this box and the rest is server-side queue/prefill.
Box location / egress is unknown (likely a datacenter proxy), so RTT from Israel may differ.

Model access check (models.list with this key): flash-lite IDs = gemini-3.5-flash-lite, gemini-3.1-flash-lite(-preview),
gemini-flash-lite-latest, gemini-2.5-flash-lite (listed but returns 404 "no longer available to new users").
Flash: gemini-2.5-flash, 3-flash-preview, 3.5/3.6/3.7/3.8-flash. Thinking control: 3.x Flash-Lite accept `thinkingLevel=minimal`
(3.5-flash-lite rejects `thinkingBudget=0` with 400; 3.1 accepts both). gemini-3.8-flash rejects `minimal`, and with
`thinkingBudget=0` still produced 140 thought tokens once; probe TTFT 1.7-3.1 s, so it was not run in the pipeline.

Hebrew quality (same inputs): Gemini replies are fluent, idiomatic, colloquial and on-topic (e.g. "מזג אוויר כזה מושלם לפיקניק
בפארק או לטיול קליל בטבע..."), but longer (2 full sentences, LLM total ~1.8 s of text) and occasionally slightly awkward
("מהערב לפני", mixed הכן/תתחיל register). Gemma-4 E2B is shorter and mostly grammatical but has more semantic oddities
("קח את השבוע הזה לטיול", "תתחיל לאט ותקצב את הקצב שלך", "תעוף מהמיטה מהר") and less helpful content.

## Scary-witch voice (2026-10-29)
```
python pipeline.py test_audio/q*_{avri,hila}.wav --reps 2 --stt sherpa --first-words 4 \
   --llm gemma4e2b --tts blue --voice witch_gacrux --voice-fx witch --persona witch
```
**Voice.** BlueTTS 2.5 has 9 stock voices (4 in `third_party_BlueTTS/voices`, 5 in `onnx_models/voices`), with controls for `speed`,
`total_step` and `cfg_scale` only (no pitch/style knobs). Measured median F0 on a test line: female voices 173-225 Hz (`female` lowest at 173),
male 115-128 Hz; all similarly "clean" (spectral flatness 0.012-0.015). None sounds old or raspy, so I cloned a witch timbre:
1. Gemini `gemini-3.8-flash-lite-tts` (voice Gacrux / Pulcherrima) reading the Hebrew line with inline tags `[raspy] [whispering] [sinister] ... [cackles]`
   (reference wavs in `voices_witch/ref_*.wav`).
2. Extract a BlueTTS style with the zero-shot encoders from `notmax123/blue-onnx-v2` (`onnx_v2_enc/`, `blue_onnx.style.VoiceStyleExtractor`)
   -> `voices_witch/witch_gacrux.json`, `witch_pulcherrima.json`. These load straight into BlueTTS 2.5 (same style shapes). Not an officially
   supported v2->2.5 combination, but it works: the clone keeps the low pitch (F0 119 Hz) and breathiness (flatness 0.040, ~3x stock).
   Synthesis speed is unchanged (~0.4 s per sentence).
   Intelligibility cost (Zipformer re-transcription of 12 persona replies, whole-reply synth, 'חה' excluded): mean WER noa 0.19, female 0.16,
   witch_gacrux 0.28, witch_pulcherrima 0.31. The FX did not make it worse (witch_gacrux + witch 0.23, + witch-lite 0.22).

**Gemini TTS directly** (`--tts gemini`): the 3.8 TTS models reject `systemInstruction` ("Developer instruction is not enabled") and read a plain
English style prompt aloud; inline bracket tags work. First audio in probes: flash-lite-tts 432-599 ms, flash-tts 645-851 ms, 3.1-flash-tts-preview 717 ms.
Delivery is very slow and theatrical (a 4.6 s BlueTTS line came out as 11-14 s with pauses and cackles).

**FX (`voice_fx.py`, pedalboard 0.9.26 + scipy, per TTS chunk).** Pitch -2 st by resampling (the voice also gets ~12% slower), vibrato (Chorus), tremolo,
parallel grit (distortion + 8-bit crush, high-passed), whisper layer (band-passed noise following the speech envelope), echo, reverb and limiter, plus a 0.3 s tail
per chunk. Offline cost per chunk (8 vCPU): `witch` 10 ms / 1 s of audio, 17 ms / 2 s, 40 ms / 5.2 s; `witch-lite` 8/14/33 ms; `witch-rb` (Rubber Band tempo-preserving pitch shift)
35/57/126 ms. Rubber Band was ~33 ms of that, which is why resampling is the default.

**12-turn runs (6 clips x 2 reps, same session).** Raw: `results_witch_*.json`. Samples: `samples_witch/` (see its README).

| config | TTFA median (min-max) | LLM first chunk med | TTS first chunk med (raw + fx) | fx on first chunk med (min-max) | reply audio med |
|---|---|---|---|---|---|
| 1 plain: Gemma + noa | 836 ms (635-1033) | 561 ms | 270 ms | - | 2.9 s |
| 2 FX only: Gemma + noa + witch | 799 ms (600-1066) | 544 ms | 266 ms (255 + 12) | 11.8 ms (9.4-19.2) | 3.9 s |
| 3 full: Gemma + witch_gacrux + witch + persona | 929 ms (591-1259) | 624 ms | 281 ms (267 + 12) | 11.8 ms (9.9-18.0) | 6.1 s |
| 4 full: gemini-3.5-flash-lite + witch_gacrux + witch + persona | 1174 ms (1058-1382) | 851 ms | 306 ms (290 + 14) | 13.7 ms (9.4-26.0) | 9.7 s |
| 5 Gemma + Gemini TTS (Gacrux, tags) + persona | 1122 ms (834-1517) | 618 ms | 501 ms (first audio) | - | 16.1 s |

TTFA differences between 1, 2 and 3 are mostly LLM-side variance (the persona replies have a slightly later first chunk); the FX itself adds ~10-26 ms.
Persona quality: Gemma's witch replies are short and spooky but often dodge the question ("החושך הוא הבית שלי, ילד קטן"). Gemini 3.5 Flash-Lite stays in character
and still answers, but runs longer than 2 sentences, has some odd words ("טיפות זרת ממרתח עלי דודנה"), mixes singular and plural address, and gave one truncated reply.

## Caveats / what was not tested
- Only the Gemini text API was tested with a real key. OpenAI Realtime, Gemini Live, Deepgram, ElevenLabs, Azure, Google TTS/STT, Groq/Cerebras LLMs were **not** run. Backends for OpenAI-compat LLMs and ElevenLabs are coded but untested.
- RenikudPlus `datastore.json` returns 404 on HF today; `BlueTTSLocal` disables that optional rescorer (G2P still works) – Hebrew pronunciation may be slightly worse than upstream default.
- LLM ttft of 55-60 ms on 2nd reps is llama.cpp prompt-cache reuse; use the 1st-rep numbers (150-400 ms) as realistic.
- Whisper-class models on CPU are too slow for this use (4-6 s); they need a GPU.
- Box has no real mic/speaker and no network-path realism (TCP connect ~1-2 ms = proxy); Gemini HTTPS RTT measured as above.
- Not implemented yet: VAD/endpointing, barge-in, audio streaming out of the TTS (BlueTTS synthesises per chunk), WebRTC transport.
