"""Pluggable STT / LLM / TTS backends for the Hebrew voice latency prototype.

Every backend is optional; the CLI picks what is available. Nothing here needs a
key unless you select a cloud backend (see README for env vars).
"""
from __future__ import annotations
import asyncio, io, json, os, re, time, wave
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SYSTEM_PROMPT = ("אתה עוזר קולי ידידותי. ענה תמיד בעברית, בקצרה (משפט אחד עד שניים), "
                 "בשפה מדוברת וטבעית, בלי רשימות ובלי סימנים מיוחדים.")

PERSONAS = {
    "witch": ("את מכשפה זקנה, מרושעת ומפחידה שגרה בבקתה חשוכה ביער. ענו תמיד בעברית, במשפט אחד או שניים קצרים בלבד, "
              "בטון מאיים, מסתורי ולוחש, ופני למשתמש בכינויים כמו 'יקירי' או 'ילד קטן'. "
              "שלבי מדי פעם רמזים לקדרה, שיקויים, עכבישים או ירח מלא, ואפשר לסיים לפעמים בצחוק 'חה חה חה'. "
              "עדיין עני על השאלה עצמה. בלי רשימות, בלי אימוג'ים ובלי סימנים מיוחדים."),
}

# ----------------------------------------------------------------- audio utils
def load_wav16k(path: str) -> np.ndarray:
    import subprocess
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

def mp3_bytes_to_f32(b: bytes, sr=24000) -> np.ndarray:
    import subprocess
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
                         input=b, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

# ------------------------------------------------------------------------ STT
class FasterWhisperSTT:
    """ivrit-ai Hebrew Whisper (CTranslate2) - batch after end-of-speech."""
    name = "faster-whisper"
    def __init__(self, model="ivrit-ai/whisper-large-v3-turbo-ct2", compute_type="int8", threads=8):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device="cpu", compute_type=compute_type, cpu_threads=threads)
        self.name = f"faster-whisper:{model.split('/')[-1]}:{compute_type}"
    def warmup(self):
        self.transcribe(np.zeros(16000, dtype=np.float32))
    def transcribe(self, audio16k: np.ndarray, realtime=False):
        t0 = time.perf_counter()
        segs, _ = self.m.transcribe(audio16k, language="he", beam_size=1, vad_filter=False,
                                    condition_on_previous_text=False, without_timestamps=True)
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, (time.perf_counter() - t0) * 1000

class SherpaStreamingSTT:
    """ivrit.ai Hebrew streaming Zipformer (sherpa-onnx, int8, CPU). Audio is fed
    in real-time-paced chunks; the reported latency is the time AFTER the last
    audio chunk is delivered until the final text is available."""
    name = "sherpa-zipformer2-66m-he"
    def __init__(self, d=HERE / "zipformer_he" / "sherpa-onnx", threads=4):
        import sherpa_onnx
        self.r = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"), joiner=str(d / "joiner.int8.onnx"),
            model_type="zipformer2", num_threads=threads, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search")
    def warmup(self):
        self.transcribe(np.zeros(16000, dtype=np.float32), realtime=False)
    def transcribe(self, audio16k, realtime=True, chunk_ms=100):
        s = self.r.create_stream()
        n = int(0.016 * 16000 * chunk_ms / 16)  # chunk samples
        n = int(16000 * chunk_ms / 1000)
        start = time.perf_counter()
        for i in range(0, len(audio16k), n):
            s.accept_waveform(16000, audio16k[i:i + n])
            if realtime:  # pace like a live mic
                due = start + (i + n) / 16000
                while self.r.is_ready(s):
                    self.r.decode_stream(s)
                time.sleep(max(0, due - time.perf_counter()))
            else:
                while self.r.is_ready(s):
                    self.r.decode_stream(s)
        t_end = time.perf_counter()           # last audio delivered == end of speech
        s.accept_waveform(16000, np.zeros(int(16000 * 0.66), dtype=np.float32))  # tail padding
        s.input_finished()
        while self.r.is_ready(s):
            self.r.decode_stream(s)
        text = self.r.get_result(s)
        text = text if isinstance(text, str) else getattr(text, "text", str(text))
        return text.strip(), (time.perf_counter() - t_end) * 1000

# ------------------------------------------------------------------------ LLM
FIRST_CHUNK_RE = re.compile(r"[.!?;:,،\n…]")
def chunker(token_iter, first_min=12, later_min=25, first_words=0):
    """Yield speakable chunks ASAP. First chunk: at clause punctuation after `first_min` chars, or (if
    first_words>0) as soon as that many complete words have arrived. Later chunks: sentence ends."""
    buf, first = "", True
    sent = re.compile(r"[.!?\n…]")
    for tok in token_iter:
        buf += tok
        cut = None
        if first:
            for mm in FIRST_CHUNK_RE.finditer(buf):
                if mm.end() >= first_min: cut = mm.end(); break
            if cut is None and first_words:
                words = list(re.finditer(r"\S+\s", buf))      # complete words (followed by whitespace)
                if len(words) > first_words: cut = words[first_words - 1].end()
        else:
            for mm in sent.finditer(buf):
                if mm.end() >= later_min: cut = mm.end(); break
        if cut:
            yield buf[:cut].strip(); buf = buf[cut:]; first = False
    if buf.strip():
        yield buf.strip()

class LocalLlamaLLM:
    """llama.cpp (GGUF) on CPU, e.g. Gemma-3 1B/4B."""
    def __init__(self, path, threads=8, ctx=1024):
        from llama_cpp import Llama
        self.l = Llama(model_path=str(path), n_ctx=ctx, n_threads=threads, verbose=False)
        self.name = "llama.cpp:" + Path(path).name
    def warmup(self):
        for _ in self.stream("שלום"): break
    def stream(self, user_text):
        msgs = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_text}]  # gemma has no system role
        for ch in self.l.create_chat_completion(messages=msgs, stream=True, max_tokens=80, temperature=0.6):
            d = ch["choices"][0]["delta"].get("content")
            if d: yield d

class OpenAICompatLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, Groq, Cerebras, Gemini-compat, local vLLM...)."""
    def __init__(self, model, base_url=None, api_key=None):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model; self.name = f"openai-compat:{model}"
    def warmup(self):
        for _ in self.stream("שלום"): break
    def stream(self, user_text):
        for ch in self.c.chat.completions.create(model=self.model, stream=True, max_tokens=120,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]):
            if ch.choices and ch.choices[0].delta.content:
                yield ch.choices[0].delta.content

# ------------------------------------------------------------------------ TTS
class BlueTTSLocal:
    """BlueTTS 2.5 (MIT, ONNX, CPU) + RenikudPlus G2P. Synthesises a whole chunk."""
    name = "blue-tts-2.5-onnx"
    def __init__(self, voice="noa", speed=1.0):
        import renikud_onnx
        self.speed = speed
        mp = renikud_onnx.download_model()
        orig = renikud_onnx.G2P.__init__           # datastore.json is 404 on HF right now -> disable rescorer
        def patched(self_, *a, **k):
            k.setdefault("datastore", None); return orig(self_, *a, **k)
        renikud_onnx.G2P.__init__ = patched
        from blue_onnx import BlueTTS
        self.t = BlueTTS(onnx_dir=str(HERE / "onnx_models"),
                         style_json=self.voice_path(voice), renikud_path=mp)
        self.name = f"blue-tts-2.5-onnx:{Path(self.voice_path(voice)).stem}"
    @staticmethod
    def voice_path(voice):
        """voice name (searched in third_party_BlueTTS/voices, onnx_models/voices, voices_witch) or a .json path"""
        if voice.endswith(".json"): return voice
        for d in ("third_party_BlueTTS/voices", "onnx_models/voices", "voices_witch"):
            if (HERE / d / f"{voice}.json").exists(): return str(HERE / d / f"{voice}.json")
        raise FileNotFoundError(voice)
    def warmup(self):
        self.t.synthesize("שלום", lang="he")
    def synth(self, text):
        t0 = time.perf_counter()
        a, sr = self.t.synthesize(text, lang="he", speed=self.speed)
        return np.asarray(a, dtype=np.float32), sr, (time.perf_counter() - t0) * 1000

class EdgeTTS:
    """Microsoft Edge online voices (he-IL-Avri/Hila Neural). Unofficial, keyless; NOT for production
    (use Azure Speech with a key for the same voices). Measures time to first audio byte."""
    name = "edge-tts:he-IL-AvriNeural"
    def __init__(self, voice="he-IL-AvriNeural"):
        self.voice = voice; self.name = f"edge-tts:{voice}"
    def warmup(self): self.synth("שלום")
    def synth(self, text):
        import edge_tts
        async def go():
            t0 = time.perf_counter(); first = None; buf = b""
            async for c in edge_tts.Communicate(text, self.voice).stream():
                if c["type"] == "audio":
                    if first is None: first = (time.perf_counter() - t0) * 1000
                    buf += c["data"]
            return buf, first
        buf, first = asyncio.run(go())
        a = mp3_bytes_to_f32(buf, 24000)
        return a, 24000, first

class ElevenLabsTTS:
    """Needs ELEVENLABS_API_KEY. Default model eleven_flash_v2_5 does NOT list Hebrew; use eleven_v3 / eleven_v4_turbo."""
    def __init__(self, voice_id=None, model=None):
        self.key = os.environ["ELEVENLABS_API_KEY"]; self.voice = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
        self.model = model or os.environ.get("ELEVENLABS_MODEL", "eleven_v3"); self.name = f"elevenlabs:{self.model}"
    def warmup(self): pass
    def synth(self, text):
        import httpx
        t0 = time.perf_counter(); first = None; buf = b""
        with httpx.stream("POST", f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}/stream?output_format=pcm_24000",
                          headers={"xi-api-key": self.key}, json={"text": text, "model_id": self.model, "language_code": "he"}, timeout=30) as r:
            r.raise_for_status()
            for c in r.iter_bytes():
                if first is None: first = (time.perf_counter() - t0) * 1000
                buf += c
        return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768, 24000, first

class GeminiNativeLLM:
    """Gemini API (native REST, SSE streaming) with a persistent HTTP connection and thinking minimised.
    Needs GEMINI_API_KEY. thinking: 'minimal' -> thinkingLevel=minimal (Gemini 3.x Flash-Lite);
    'budget0' -> thinkingBudget=0."""
    def __init__(self, model="gemini-3.1-flash-lite", thinking="minimal", max_tokens=150):
        import httpx
        self.c = httpx.Client(timeout=30); self.model = model; self.key = os.environ["GEMINI_API_KEY"]
        self.tc = {"thinkingLevel": "minimal"} if thinking == "minimal" else {"thinkingBudget": 0}
        self.max_tokens = max_tokens; self.name = f"gemini:{model}:{thinking}"
        self.last_thought_tokens = None
    def warmup(self):
        for _ in self.stream("שלום"): pass
    def stream(self, user_text):
        body = {"contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "generationConfig": {"maxOutputTokens": self.max_tokens, "temperature": 0.7, "thinkingConfig": self.tc}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse"
        with self.c.stream("POST", url, headers={"x-goog-api-key": self.key}, json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.read()[:300]!r}")
            for line in r.iter_lines():
                if not line.startswith("data:"): continue
                d = json.loads(line[5:])
                self.last_thought_tokens = d.get("usageMetadata", {}).get("thoughtsTokenCount", self.last_thought_tokens)
                for p in d.get("candidates", [{}])[0].get("content", {}).get("parts", []):
                    if p.get("text") and not p.get("thought"):
                        yield p["text"]


class GeminiTTS:
    """Gemini TTS (gemini-3.8-flash[-lite]-tts), SSE streaming PCM 24 kHz. Style via inline audio tags
    (these models reject systemInstruction and read a plain-text style prompt aloud). ms = time to first audio
    chunk; the whole chunk is still collected (harness plays per chunk)."""
    def __init__(self, model="gemini-3.8-flash-lite-tts", voice="Gacrux", tags="[raspy] [whispering] [sinister] "):
        import httpx
        self.c = httpx.Client(timeout=60); self.key = os.environ["GEMINI_API_KEY"]
        self.model, self.voice, self.tags = model, voice, tags; self.name = f"gemini-tts:{model}:{voice}"
    def warmup(self): self.synth("שלום")
    def synth(self, text):
        import base64
        body = {"contents": [{"role": "user", "parts": [{"text": self.tags + text}]}],
                "generationConfig": {"responseModalities": ["AUDIO"],
                                     "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}}}}
        t0 = time.perf_counter(); first = None; pcm = b""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse"
        with self.c.stream("POST", url, headers={"x-goog-api-key": self.key}, json=body) as r:
            if r.status_code != 200: raise RuntimeError(f"Gemini TTS HTTP {r.status_code}: {r.read()[:300]!r}")
            for line in r.iter_lines():
                if not line.startswith("data:"): continue
                for p in json.loads(line[5:]).get("candidates", [{}])[0].get("content", {}).get("parts", []):
                    if "inlineData" in p:
                        if first is None: first = (time.perf_counter() - t0) * 1000
                        pcm += base64.b64decode(p["inlineData"]["data"])
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768, 24000, first
