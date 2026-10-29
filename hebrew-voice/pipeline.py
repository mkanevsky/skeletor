#!/usr/bin/env python
"""Hebrew voice pipeline latency harness:  audio file -> STT -> LLM (streamed) -> sentence chunker -> TTS.

Reports per-stage latency and 'time to first audio' (TTFA) measured from END OF USER SPEECH.
TTFA = STT finalisation + LLM time to first speakable chunk + TTS time to first audio.
(Not included: end-of-speech detection delay / VAD hangover - add ~200-500 ms in a live system.)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
import hv_backends as B

def build_stt(a):
    return B.SherpaStreamingSTT() if a == "sherpa" else B.FasterWhisperSTT(compute_type=os.environ.get("HV_FW_COMPUTE", "float32"))

def build_llm(a):
    if a.startswith("gguf:"):
        return B.LocalLlamaLLM(a[5:])
    if a == "gemma1b": return B.LocalLlamaLLM(B.HERE / "llm/gemma-3-1b-it-Q4_K_M.gguf")
    if a == "gemma4b": return B.LocalLlamaLLM(B.HERE / "llm/gemma-3-4b-it-Q4_K_M.gguf")
    if a == "gemma4e2b": return B.LocalLlamaLLM(B.HERE / "llm/gemma-4-E2B-it-Q4_K_M.gguf")
    if a == "openai":  return B.OpenAICompatLLM(os.environ.get("OPENAI_MODEL", "gpt-5-mini"), api_key=os.environ["OPENAI_API_KEY"])
    if a == "groq":    return B.OpenAICompatLLM(os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), "https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"])
    if a.startswith("gemini:"):  # gemini:<model>[:minimal|budget0]  (native API, persistent connection)
        parts = a.split(":"); return B.GeminiNativeLLM(parts[1], parts[2] if len(parts) > 2 else "minimal")
    if a == "gemini":  return B.OpenAICompatLLM(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"), "https://generativelanguage.googleapis.com/v1beta/openai/", os.environ["GEMINI_API_KEY"])
    raise SystemExit(f"unknown llm {a}")

def build_tts(a, voice=None, speed=1.0, fx="none"):
    if a == "blue": t = B.BlueTTSLocal(voice or "noa", speed=speed)
    elif a == "gemini": t = B.GeminiTTS(os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts"), voice or "Gacrux")
    else: t = {"edge": B.EdgeTTS, "elevenlabs": B.ElevenLabsTTS}[a]()
    if fx != "none":
        from voice_fx import VoiceFX, FXTTS
        t = FXTTS(t, VoiceFX(fx))
    return t

def run_turn(stt, llm, tts, audio, realtime=True, out=None, first_words=0):
    r = {}
    text, r["stt_ms"] = stt.transcribe(audio, realtime=realtime)
    r["transcript"] = text
    t0 = time.perf_counter()
    first_tok = first_chunk = None; chunks = []; full = ""
    def toks():
        nonlocal first_tok, full
        for t in llm.stream(text):
            if first_tok is None: first_tok = (time.perf_counter() - t0) * 1000
            full += t; yield t
    pcm, sr_out, tts_first, tts_total = [], 24000, None, 0.0
    for ch in B.chunker(toks(), first_words=first_words):
        if first_chunk is None: first_chunk = (time.perf_counter() - t0) * 1000
        a, sr_out, ms = tts.synth(ch)
        if tts_first is None:
            tts_first = ms
            if hasattr(tts, "last_fx_ms"): r["fx_first_ms"] = tts.last_fx_ms; r["tts_raw_first_ms"] = tts.last_raw_ms
        tts_total += ms; chunks.append(ch); pcm.append(a)
    r.update(llm_ttft_ms=first_tok, llm_first_chunk_ms=first_chunk, tts_first_audio_ms=tts_first,
             ttfa_ms=r["stt_ms"] + first_chunk + tts_first,
             llm_total_ms=(time.perf_counter() - t0) * 1000, tts_total_ms=tts_total,
             reply=full.strip(), first_chunk_text=chunks[0] if chunks else "", n_chunks=len(chunks))
    if out and pcm:
        sf.write(out, np.concatenate(pcm), sr_out); r["out_wav"] = out; r["reply_audio_s"] = sum(len(p) for p in pcm) / sr_out
    return r

def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="+"); p.add_argument("--stt", default="fw", choices=["fw", "sherpa"])
    p.add_argument("--llm", default="gemma1b"); p.add_argument("--tts", default="blue", choices=["blue", "edge", "elevenlabs", "gemini"])
    p.add_argument("--voice", default=None, help="BlueTTS voice name/json (e.g. noa, female, witch_gacrux) or Gemini TTS voice name")
    p.add_argument("--speed", type=float, default=1.0, help="BlueTTS speaking rate")
    p.add_argument("--voice-fx", default="none", choices=["none", "witch", "witch-lite", "witch-rb"])
    p.add_argument("--persona", default="none", choices=["none", "witch"])
    p.add_argument("--tag", default="", help="extra tag for output wav names")
    p.add_argument("--first-words", type=int, default=0, help="emit first TTS chunk after N words even without punctuation")
    p.add_argument("--reps", type=int, default=1); p.add_argument("--no-realtime", action="store_true")
    p.add_argument("--outdir", default="out"); p.add_argument("--json", default=None)
    a = p.parse_args()
    Path(a.outdir).mkdir(exist_ok=True)
    if a.persona != "none": B.SYSTEM_PROMPT = B.PERSONAS[a.persona]
    stt, llm, tts = build_stt(a.stt), build_llm(a.llm), build_tts(a.tts, a.voice, a.speed, a.voice_fx)
    print(f"STT={stt.name}  LLM={llm.name}  TTS={tts.name}", flush=True)
    for x in (stt, llm, tts): x.warmup()
    rows = []
    for f in a.audio:
        audio = B.load_wav16k(f)
        for i in range(a.reps):
            r = run_turn(stt, llm, tts, audio, realtime=not a.no_realtime, first_words=a.first_words, out=f"{a.outdir}/{Path(f).stem}_{a.stt}_{a.llm.replace(":","-")}_{a.tts}{a.tag}_{i}.wav")
            r.update(file=f, rep=i, audio_in_s=len(audio) / 16000, stt=stt.name, llm=llm.name, tts=tts.name, persona=a.persona, voice_fx=a.voice_fx); rows.append(r)
            print(f"\n[{Path(f).name} #{i}] in={r['audio_in_s']:.1f}s\n  STT   {r['stt_ms']:7.0f} ms  -> {r['transcript']}"
                  f"\n  LLM   ttft {r['llm_ttft_ms']:.0f} ms, first chunk {r['llm_first_chunk_ms']:.0f} ms, total {r['llm_total_ms']:.0f} ms -> {r['reply']}"
                  f"\n  TTS   first audio {r['tts_first_audio_ms']:.0f} ms (first chunk: {r['first_chunk_text']!r}), all chunks {r['tts_total_ms']:.0f} ms"
                  + (f" (fx {r['fx_first_ms']:.1f} ms of first)" if 'fx_first_ms' in r else "") + f"\n  ==> TTFA from end of speech: {r['ttfa_ms']:.0f} ms", flush=True)
    if len(rows) > 1:
        v = np.array([r["ttfa_ms"] for r in rows]); print(f"\nTTFA over {len(v)} turns: median {np.median(v):.0f} ms, min {v.min():.0f}, max {v.max():.0f}")
    if a.json: Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
