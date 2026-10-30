#!/usr/bin/env python
"""Live Hebrew voice conversation: mic -> high-pass -> energy gate + Silero VAD -> endpointer ->
streaming Zipformer STT -> LLM (streamed) -> chunker -> TTS -> speakers (chunks streamed as produced).

Half-duplex echo protection (default): the mic is gated from the moment a user turn ends until the speakers
have finished playing the bot's reply + --tail-ms. Frames captured while gated are dropped (by *capture* time,
so frames still queued from the playback period are dropped too), and VAD / endpointer / STT stream state is
reset when the gate re-opens, so the bot never hears itself.

Real hardware:   python live.py --llm gemma4e2b --voice witch_gacrux --voice-fx witch --persona witch
Simulation:      python live.py --sim-input in.wav --sim-output out.wav [--sim-echo-gain 0.5] ...
  (feeds the wav as if from the mic in real time, writes the speaker output to a wav and mixes the bot's own
   output back into the simulated mic while it "plays", exercising the same gating code).
"""
from __future__ import annotations
import argparse, collections, json, math, os, queue, re, sys, threading, time
from pathlib import Path
import numpy as np

import hv_backends as B

SR = 16000
FRAME = 512                       # 32 ms; Silero VAD window at 16 kHz
FRAME_S = FRAME / SR
FILLERS = {"אה", "אהה", "אההה", "אמ", "אממ", "אמממ", "הממ", "המממ", "מממ", "ממ", "אום", "אוו", "הא", "אהם", "אא", "ההה"}
HEB = re.compile(r"[\u05d0-\u05eaA-Za-z0-9]")
now = time.monotonic


def dbfs(x):
    return 20.0 * math.log10(math.sqrt(float(np.mean(x * x))) + 1e-10)


def resample(x, sr_in, sr_out):
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


# --------------------------------------------------------------------------- front-end
class HighPass:
    """Streaming Butterworth high-pass (removes hum / rumble / desk thumps below ~90 Hz)."""
    def __init__(self, hz, order=2):
        self.sos = None
        if hz and hz > 0:
            from scipy.signal import butter
            self.sos = butter(order, hz, "highpass", fs=SR, output="sos")
            self.reset()
    def reset(self):
        if self.sos is not None:
            self.zi = np.zeros((self.sos.shape[0], 2))
    def __call__(self, x):
        if self.sos is None:
            return x
        from scipy.signal import sosfilt
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


class SileroVAD:
    """Silero VAD v4 ONNX (vad/silero_vad.onnx, the file sherpa-onnx distributes), run with onnxruntime.
    Returns a per-32 ms-frame speech probability."""
    URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    def __init__(self, path):
        import onnxruntime as ort
        path = Path(path)
        if not path.exists():
            import urllib.request
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading Silero VAD -> {path}", flush=True)
            urllib.request.urlretrieve(self.URL, path)
        o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        self.reset()
    def reset(self):
        self.h = np.zeros((2, 1, 64), np.float32); self.c = np.zeros((2, 1, 64), np.float32)
    def __call__(self, frame):
        p, self.h, self.c = self.s.run(None, {"x": frame[None].astype(np.float32), "h": self.h, "c": self.c})
        return float(p.reshape(-1)[0])


class Endpointer:
    """idle --(>= min_speech of speech frames, gaps <= onset_gap)--> speech --(>= end_silence non-speech)--> end.
    Keeps a pre-roll so the STT also gets the onset that happened before the turn was confirmed."""
    def __init__(self, a):
        f = lambda ms: max(1, int(round(ms / 1000 / FRAME_S)))
        self.min_speech, self.end_sil, self.onset_gap = f(a.min_speech_ms), f(a.end_silence_ms), f(a.onset_gap_ms)
        self.max_frames = f(a.max_utt_s * 1000)
        self.preroll = collections.deque(maxlen=f(a.preroll_ms) + self.min_speech)
        self.reset()
    def reset(self):
        self.state = "idle"; self.run = 0; self.gap = 0; self.sil = 0; self.n = 0; self.voiced = 0
        self.preroll.clear(); self.t_last_speech = None
    def push(self, frame, speech, t):
        """returns None | 'start' | 'end'"""
        if self.state == "idle":
            self.preroll.append(frame)
            if speech:
                self.run += 1; self.gap = 0
            else:
                self.gap += 1
                if self.gap > self.onset_gap:
                    self.run = 0
            if self.run >= self.min_speech:
                self.state = "speech"; self.sil = 0; self.n = len(self.preroll); self.voiced = self.run
                self.t_last_speech = t + FRAME_S
                return "start"
            return None
        self.n += 1
        if speech:
            self.sil = 0; self.voiced += 1; self.t_last_speech = t + FRAME_S
        else:
            self.sil += 1
        if self.sil >= self.end_sil or self.n >= self.max_frames:
            self.state = "idle"; self.run = 0; self.gap = 0
            return "end"
        return None


class Gate:
    """Half-duplex mic gate. Closed while a turn is being answered (hold) and while the speaker plays;
    re-opens --tail-ms after playback ends. Frames are judged by their CAPTURE time."""
    def __init__(self, tail_s, enabled=True):
        self.tail_s, self.enabled = tail_s, enabled
        self.lock = threading.Lock(); self.token = 0; self.busy = False; self.open_at = 0.0
    def hold(self):
        with self.lock:
            self.token += 1; self.busy = True; return self.token
    def release(self, t_play_end=None, token=None, tail=True):
        with self.lock:
            if token is not None and token != self.token:
                return
            self.busy = False
            if t_play_end is not None:
                self.open_at = max(self.open_at, t_play_end + (self.tail_s if tail else 0.0))
    def accepts(self, t_cap, playing):
        if not self.enabled:
            return True
        with self.lock:
            return not self.busy and not playing and t_cap >= self.open_at and now() >= self.open_at


# --------------------------------------------------------------------------- audio I/O
class SoundDeviceMic:
    """16 kHz mono capture via PortAudio. Falls back to the device's native rate + resampling if 16 kHz is refused."""
    def __init__(self, device=None):
        import sounddevice as sd
        self.q = queue.Queue(); self.buf = np.zeros(0, np.float32)
        def cb(indata, frames, t, status):
            t_cap = now() - frames / self.sr - self.latency
            self.q.put((t_cap, indata[:, 0].copy()))
        try:
            self.sr = SR
            self.st = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME, device=device,
                                     latency="low", callback=cb)
        except Exception:
            self.sr = int(sd.query_devices(device, "input")["default_samplerate"])
            print(f"mic: 16 kHz not supported, capturing at {self.sr} Hz and resampling", flush=True)
            self.st = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", device=device,
                                     latency="low", callback=cb)
        self.latency = float(self.st.latency or 0.0)
        self.t_buf = None
    def start(self): self.st.start()
    def close(self): self.st.stop(); self.st.close()
    def frames(self):
        """yields (capture_time, 512-sample float32 frame at 16 kHz); never ends."""
        while True:
            t, x = self.q.get()
            if self.sr != SR:
                x = resample(x, self.sr, SR)
            if self.t_buf is None or len(self.buf) == 0:
                self.t_buf = t
            self.buf = np.concatenate([self.buf, x])
            while len(self.buf) >= FRAME:
                yield self.t_buf, self.buf[:FRAME]
                self.buf = self.buf[FRAME:]; self.t_buf += FRAME_S


class SoundDeviceSpeaker:
    """Callback output stream; write() appends TTS chunks, which start playing immediately."""
    def __init__(self, sr, device=None):
        import sounddevice as sd
        self.sr = sr; self.lock = threading.Lock(); self.chunks = collections.deque(); self.pos = 0
        self.play_end = 0.0
        def cb(out, frames, t, status):
            out.fill(0); i = 0
            with self.lock:
                while i < frames and self.chunks:
                    c = self.chunks[0]; n = min(frames - i, len(c) - self.pos)
                    out[i:i + n, 0] = c[self.pos:self.pos + n]; i += n; self.pos += n
                    if self.pos >= len(c):
                        self.chunks.popleft(); self.pos = 0
                if i:  # audio leaves the speaker ~latency after this callback
                    self.play_end = now() + self.latency + i / self.sr
        self.st = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", device=device, latency="low", callback=cb)
        self.latency = float(self.st.latency or 0.0)
        self.st.start()
    def write(self, audio, sr):
        a = resample(np.asarray(audio, np.float32), sr, self.sr)
        with self.lock:
            self.chunks.append(a)
            self.play_end = max(self.play_end, now() + self.latency + len(a) / self.sr)  # provisional until played
    def is_playing(self):
        with self.lock:
            return bool(self.chunks) or now() < self.play_end
    def end_time(self):
        with self.lock:
            return self.play_end
    def stop(self):
        with self.lock:
            self.chunks.clear(); self.pos = 0; self.play_end = now() + self.latency
    def close(self): self.st.stop(); self.st.close()


class EchoBus:
    """Simulated acoustic path speaker -> mic: delayed, attenuated copy of everything the sim speaker plays."""
    def __init__(self, gain, delay_s):
        self.gain, self.delay = gain, delay_s; self.segs = []; self.lock = threading.Lock()
    def add(self, t_start, audio16k):
        with self.lock:
            self.segs.append([t_start + self.delay, (audio16k * self.gain).astype(np.float32)])
    def cut(self, t):
        with self.lock:
            for s in self.segs:
                keep = int(max(0, (t + self.delay - s[0])) * SR)
                s[1] = s[1][:keep]
    def get(self, t0, n):
        out = np.zeros(n, np.float32)
        with self.lock:
            for ts, a in self.segs:
                i0 = int(round((t0 - ts) * SR))
                if i0 >= len(a) or i0 + n <= 0:
                    continue
                s0, s1 = max(0, i0), min(len(a), i0 + n)
                out[s0 - i0:s1 - i0] += a[s0:s1]
        return out


class SimSpeaker:
    """Plays 'virtually' in real time: tracks a play-out clock, keeps a timeline for the output wav, feeds the echo bus."""
    def __init__(self, sr, echo=None):
        self.sr, self.echo = sr, echo; self.lock = threading.Lock(); self.play_end = 0.0; self.segs = []
    def write(self, audio, sr):
        a = resample(np.asarray(audio, np.float32), sr, self.sr)
        with self.lock:
            t = max(now(), self.play_end); self.play_end = t + len(a) / self.sr; self.segs.append([t, a])
        if self.echo is not None:
            self.echo.add(t, resample(a, self.sr, SR))
    def is_playing(self):
        with self.lock:
            return now() < self.play_end
    def end_time(self):
        with self.lock:
            return self.play_end
    def stop(self):
        t = now()
        with self.lock:
            for s in self.segs:
                s[1] = s[1][:int(max(0, t - s[0]) * self.sr)]
            self.play_end = min(self.play_end, t)
        if self.echo is not None:
            self.echo.cut(t)
    def render(self, t0, dur_s):
        out = np.zeros(int(dur_s * self.sr) + 1, np.float32)
        for ts, a in self.segs:
            i = int(round((ts - t0) * self.sr))
            if i >= len(out): continue
            a = a[:len(out) - i]; out[i:i + len(a)] += a
        return out
    def close(self): pass


class SimMic:
    """Feeds a wav as if it came from a mic in real time (+ echo of the sim speaker)."""
    def __init__(self, path, echo=None):
        self.x = B.load_wav16k(path); self.echo = echo; self.t0 = None
        self.heard = np.zeros_like(self.x)
    def start(self): self.t0 = now()
    def close(self): pass
    def frames(self):
        for k, i in enumerate(range(0, len(self.x) - FRAME + 1, FRAME)):
            t_cap = self.t0 + i / SR
            due = t_cap + FRAME_S
            d = due - now()
            if d > 0:
                time.sleep(d)
            f = self.x[i:i + FRAME].copy()
            if self.echo is not None:
                f += self.echo.get(t_cap, FRAME)
            self.heard[i:i + FRAME] = f
            yield t_cap, f


# --------------------------------------------------------------------------- backends
def build_llm(name):
    import pipeline as P
    paths = {"gemma1b": "gemma-3-1b-it-Q4_K_M.gguf", "gemma4b": "gemma-3-4b-it-Q4_K_M.gguf",
             "gemma4e2b": "gemma-4-E2B-it-Q4_K_M.gguf"}
    if name in paths:   # bigger context than pipeline.py's default so the conversation history fits
        return B.LocalLlamaLLM(B.HERE / "llm" / paths[name], ctx=2048)
    if name.startswith("gguf:"):
        return B.LocalLlamaLLM(name[5:], ctx=2048)
    return P.build_llm(name)


class LiveSTT:
    """Incremental use of the sherpa-onnx streaming Zipformer from hv_backends.SherpaStreamingSTT."""
    def __init__(self, threads=2):
        self.m = B.SherpaStreamingSTT(threads=threads); self.r = self.m.r; self.name = self.m.name
    def warmup(self): self.m.warmup()
    def new(self): return self.r.create_stream()
    def feed(self, s, x):
        s.accept_waveform(SR, x)
        while self.r.is_ready(s):
            self.r.decode_stream(s)
    def partial(self, s):
        t = self.r.get_result(s); return (t if isinstance(t, str) else getattr(t, "text", str(t))).strip()
    def final(self, s):
        s.accept_waveform(SR, np.zeros(int(SR * 0.66), np.float32)); s.input_finished()
        while self.r.is_ready(s):
            self.r.decode_stream(s)
        return self.partial(s)


FINALS = str.maketrans("םןץףך", "מנצפכ")
def is_junk(text, min_chars):
    words = [re.sub(r"[^\u05d0-\u05eaA-Za-z0-9]", "", w).translate(FINALS) for w in text.split()]
    words = [w for w in words if w]
    if sum(len(HEB.findall(w)) for w in words) < min_chars:
        return "too short"
    if all(w in FILLERS for w in words):
        return "filler"
    return None


# --------------------------------------------------------------------------- app
class Live:
    def __init__(self, a):
        self.a = a
        if a.persona != "none":
            B.SYSTEM_PROMPT = B.PERSONAS[a.persona]
        import pipeline as P
        self.stt = LiveSTT(threads=a.stt_threads)
        self.llm = build_llm(a.llm)
        self.tts = P.build_tts(a.tts, a.voice, a.speed, a.voice_fx)
        print(f"STT={self.stt.name}  LLM={self.llm.name}  TTS={self.tts.name}  persona={a.persona}", flush=True)
        print("warming up...", flush=True)
        self.stt.warmup(); self.llm.warmup()
        if hasattr(self.tts, "warmup"): self.tts.warmup()
        _, self.tts_sr, _ = self.tts.synth("שלום")
        self.vad = SileroVAD(a.vad_model); self.hpf = HighPass(a.highpass_hz)
        self.ep = Endpointer(a); self.gate = Gate(a.tail_ms / 1000, enabled=not a.no_echo_gate)
        # append-only window, trimmed only when it overflows: keeps the prompt prefix stable so
        # llama.cpp can reuse its KV cache (a sliding deque re-prefills the whole history every turn)
        self.history = []
        self.jobs = queue.Queue(); self.abort = threading.Event()
        self.events = []; self.t0 = None; self.lock = threading.Lock()
        self.stats = collections.Counter()
        if a.sim_input:
            self.echo = EchoBus(a.sim_echo_gain, a.sim_echo_delay_ms / 1000) if a.sim_echo_gain > 0 else None
            self.mic = SimMic(a.sim_input, self.echo); self.spk = SimSpeaker(self.tts_sr, self.echo)
            self.shadow_vad = SileroVAD(a.vad_model)    # stats only: what would the VAD have said about dropped frames
        else:
            self.mic = SoundDeviceMic(a.input_device); self.spk = SoundDeviceSpeaker(self.tts_sr, a.output_device)
            self.shadow_vad = None
        self.worker = threading.Thread(target=self._responder, daemon=True); self.worker.start()

    # ---- logging
    def log(self, kind, t=None, **kw):
        t = now() if t is None else t
        e = dict(t=round(t - self.t0, 3), ev=kind, **kw)
        with self.lock:
            self.events.append(e)
        return e

    # ---- responder thread: STT final -> filter -> LLM -> TTS -> speaker
    def _responder(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self._respond(job)
            except Exception as ex:  # never leave the gate stuck closed
                print(f"  ! error: {ex!r}", flush=True); self.gate.release(now(), job["token"], tail=False)
                self.log("error", err=repr(ex))

    def _respond(self, job):
        a = self.a; self.abort.clear()
        t_s = now(); text = self.stt.final(job["stream"]); stt_ms = (now() - t_s) * 1000
        why = is_junk(text, a.min_chars)
        if why:
            print(f"  (ignored: {text!r} - {why})", flush=True)
            self.log("ignored", text=text, why=why, stt_ms=round(stt_ms, 1))
            self.gate.release(now(), job["token"], tail=False)
            return
        print(f"heard: {text}   (stt {stt_ms:.0f} ms)", flush=True)
        self.log("heard", text=text, stt_ms=round(stt_ms, 1), utt_s=job["utt_s"])
        t0 = job["t_detect"]; first_audio = None; reply = ""; first_tok = None
        def toks():
            nonlocal reply, first_tok
            try:
                it = self.llm.stream(text, history=list(self.history))
            except TypeError:
                it = self.llm.stream(text)
            for tk in it:
                if self.abort.is_set(): return
                if first_tok is None: first_tok = now()
                reply += tk; yield tk
        sys.stdout.write("bot: "); sys.stdout.flush()
        for ch in B.chunker(toks(), first_words=a.first_words):
            if self.abort.is_set(): break
            audio, sr, _ = self.tts.synth(ch)
            if self.abort.is_set(): break
            self.spk.write(audio, sr)
            if first_audio is None:
                first_audio = now(); self.log("first_audio", t=first_audio, chunk=ch)
            sys.stdout.write(ch + " "); sys.stdout.flush()
        print(flush=True)
        while self.spk.is_playing() and not self.abort.is_set():
            time.sleep(0.01)
        t_end = self.spk.end_time()
        if first_audio is not None:
            lat = (first_audio - t0) * 1000
            print(f"  latency: first audio {lat:.0f} ms after end-of-turn detect "
                  f"(+{a.end_silence_ms} ms end-silence wait = {lat + a.end_silence_ms:.0f} ms after you stopped); "
                  f"stt {stt_ms:.0f} / llm 1st tok {((first_tok or now()) - t_s) * 1000:.0f} ms", flush=True)
        self.log("bot", text=reply.strip(), aborted=self.abort.is_set(),
                 latency_ms=round((first_audio - t0) * 1000, 1) if first_audio else None,
                 latency_from_speech_end_ms=round((first_audio - job["t_speech_end"]) * 1000, 1) if first_audio else None,
                 stt_ms=round(stt_ms, 1), llm_ttft_ms=round((first_tok - t_s) * 1000, 1) if first_tok else None)
        self.log("playback_end", t=t_end)
        if reply.strip() and not self.abort.is_set():
            if self.a.history > 0:
                self.history.append((text, reply.strip()))
                if len(self.history) > self.a.history:
                    self.history = self.history[-max(1, self.a.history // 2):]
        self.gate.release(t_end, job["token"])

    # ---- main audio loop
    def speech_frame(self, x, p, db, in_speech, barge=False):
        a = self.a
        thr_db, thr_p = self.thr_db, a.vad_threshold
        if in_speech:   # hysteresis: easier to stay in speech than to enter it
            thr_db -= a.hangover_db; thr_p -= a.vad_hangover
        if barge:
            thr_db = max(thr_db + a.barge_margin_db, a.barge_db); thr_p = max(thr_p, 0.85)
        return db >= thr_db and p >= thr_p

    def run(self):
        a = self.a
        self.mic.start(); self.t0 = self.mic.t0 if a.sim_input else now()
        calib = []; n_cal = int(a.calibrate_s / FRAME_S); self.thr_db = a.energy_min_db
        stream = None; need_reset = True; gated = False; barge_active = False; barge_run = 0; token = None
        last_status = None
        print("calibrating ambient noise..." if n_cal else "listening...", flush=True)
        try:
            for t_cap, raw in self.mic.frames():
                x = self.hpf(raw); db = dbfs(x)
                self.stats["frames"] += 1
                # ---------- ambient noise calibration
                if n_cal and len(calib) < n_cal:
                    calib.append(db); self.vad(x)
                    if len(calib) == n_cal:
                        floor = float(np.percentile(calib, 50))
                        if floor > a.calib_max_db:
                            print(f"  ! calibration floor {floor:.1f} dBFS looks like speech/very loud noise; capping at {a.calib_max_db}", flush=True)
                            floor = a.calib_max_db
                        self.thr_db = max(a.energy_min_db, floor + a.energy_margin_db); self.noise_floor = floor
                        self.log("calibrated", floor_db=round(floor, 1), thr_db=round(self.thr_db, 1))
                        print(f"noise floor {floor:.1f} dBFS -> energy gate {self.thr_db:.1f} dBFS", flush=True)
                        need_reset = True
                    continue
                playing = self.spk.is_playing()
                # ---------- half-duplex gate
                if not barge_active and not self.gate.accepts(t_cap, playing):
                    if not gated:
                        gated = True; self.log("gate_closed", t=t_cap)
                    self.stats["dropped_frames"] += 1
                    if self.shadow_vad is not None:
                        sp = self.shadow_vad(x)
                        if db >= self.thr_db: self.stats["dropped_loud"] += 1
                        if db >= self.thr_db and sp >= a.vad_threshold: self.stats["dropped_speechlike"] += 1
                        if playing: self.stats["dropped_during_playback"] += 1
                    need_reset = True
                    if a.barge_in and playing:     # experimental: much stricter detector during playback
                        p = self.vad(x); self.ep.preroll.append(x)
                        barge_run = barge_run + 1 if self.speech_frame(x, p, db, False, barge=True) else 0
                        if barge_run * FRAME_S * 1000 >= a.barge_min_ms:
                            print("\n  >> barge-in", flush=True); self.log("barge_in", t=t_cap)
                            self.spk.stop(); self.abort.set(); barge_active = True; barge_run = 0
                            stream = self.stt.new()
                            for f in self.ep.preroll: self.stt.feed(stream, f)
                            self.ep.state = "speech"; self.ep.n = len(self.ep.preroll); self.ep.sil = 0
                            self.ep.t_last_speech = t_cap + FRAME_S; gated = False; need_reset = False
                    continue
                if gated:
                    gated = False; self.log("gate_open", t=t_cap)
                if need_reset:   # fresh state after the gate re-opens: nothing heard during playback survives
                    self.vad.reset(); self.ep.reset(); stream = self.stt.new(); need_reset = False; barge_run = 0
                    if last_status != "listening":
                        print("listening...", flush=True); last_status = "listening"
                p = self.vad(x)
                sp = self.speech_frame(x, p, db, self.ep.state == "speech")
                ev = self.ep.push(x, sp, t_cap)
                if ev == "start":
                    self.log("turn_start", t=t_cap - (self.ep.n - 1) * FRAME_S, db=round(db, 1), p=round(p, 2))
                    for f in self.ep.preroll: self.stt.feed(stream, f)
                    print("  ...speech", flush=True); last_status = "speech"
                elif self.ep.state == "speech" or ev == "end":
                    self.stt.feed(stream, x)
                    if a.show_partials and self.ep.n % 10 == 0:
                        pt = self.stt.partial(stream)
                        if pt: print(f"  ... {pt}", flush=True)
                if ev == "end":
                    t_det = t_cap + FRAME_S
                    utt_s = round(self.ep.n * FRAME_S, 2)
                    self.log("turn_end", t=t_det, utt_s=utt_s, speech_end=round(self.ep.t_last_speech - self.t0, 3))
                    token = self.gate.hold(); barge_active = False
                    self.jobs.put(dict(stream=stream, t_detect=t_det, t_speech_end=self.ep.t_last_speech,
                                       utt_s=utt_s, token=token))
                    stream = self.stt.new(); need_reset = True; last_status = None
        except KeyboardInterrupt:
            print("\nbye", flush=True)
        finally:
            self.finish()

    def finish(self):
        a = self.a
        if a.sim_input:   # let the last reply finish
            while not self.jobs.empty() or self.gate.busy or self.spk.is_playing():
                time.sleep(0.05)
        self.jobs.put(None); self.worker.join(timeout=5)
        self.mic.close(); self.spk.close()
        if a.sim_input:
            import soundfile as sf
            dur = max(len(self.mic.x) / SR, self.spk.end_time() - self.t0) + 0.5
            out = self.spk.render(self.t0, dur)
            sf.write(a.sim_output, out, self.tts_sr)
            mic = np.zeros(int(dur * SR) + 1, np.float32); mic[:len(self.mic.heard)] = self.mic.heard
            bot16 = resample(out, self.tts_sr, SR)[:len(mic)]; bot16 = np.pad(bot16, (0, len(mic) - len(bot16)))
            sf.write(Path(a.sim_output).with_name(Path(a.sim_output).stem + "_mic_vs_bot.wav"),
                     np.stack([mic, bot16], 1), SR)   # L = what the mic heard (incl. echo), R = bot output
            print(f"wrote {a.sim_output} (+ _mic_vs_bot.wav: L=mic incl. echo, R=bot)", flush=True)
        st = dict(self.stats); print("stats:", st, flush=True)
        if a.log_json:
            Path(a.log_json).write_text(json.dumps(dict(args=vars(a), stats=st, events=self.events,
                                                         thr_db=self.thr_db), ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("backends (same as pipeline.py)")
    g.add_argument("--llm", default="gemma4e2b"); g.add_argument("--tts", default="blue", choices=["blue", "edge", "elevenlabs", "gemini"])
    g.add_argument("--voice", default=None); g.add_argument("--speed", type=float, default=1.0)
    g.add_argument("--voice-fx", default="none", choices=["none", "witch", "witch-lite", "witch-rb"])
    g.add_argument("--persona", default="none", choices=["none"] + sorted(B.PERSONAS))
    g.add_argument("--first-words", type=int, default=4, help="first TTS chunk after N words (default 4)")
    g.add_argument("--history", type=int, default=3, help="previous turns sent to the LLM (0 = single-turn)")
    g.add_argument("--stt-threads", type=int, default=2)
    g = p.add_argument_group("audio devices")
    g.add_argument("--list-devices", action="store_true"); g.add_argument("--input-device", default=None)
    g.add_argument("--output-device", default=None)
    g = p.add_argument_group("noise filtering / turn taking")
    g.add_argument("--vad-model", default=str(B.HERE / "vad" / "silero_vad.onnx"))
    g.add_argument("--vad-threshold", type=float, default=0.5, help="Silero speech probability to count a frame as speech")
    g.add_argument("--vad-hangover", type=float, default=0.15, help="prob threshold drop while already in speech")
    g.add_argument("--energy-min-db", type=float, default=-50.0, help="absolute minimum frame RMS (dBFS) for speech")
    g.add_argument("--energy-margin-db", type=float, default=10.0, help="speech must be this far above the calibrated noise floor")
    g.add_argument("--hangover-db", type=float, default=4.0, help="energy threshold drop while already in speech")
    g.add_argument("--calibrate-s", type=float, default=1.0, help="ambient noise calibration at start (0 = off)")
    g.add_argument("--calib-max-db", type=float, default=-35.0)
    g.add_argument("--highpass-hz", type=float, default=90.0, help="0 = off")
    g.add_argument("--min-speech-ms", type=float, default=250, help="speech needed to START a turn")
    g.add_argument("--onset-gap-ms", type=float, default=100, help="max gap inside the onset run")
    g.add_argument("--end-silence-ms", type=float, default=500, help="silence that ENDS a turn")
    g.add_argument("--preroll-ms", type=float, default=300); g.add_argument("--max-utt-s", type=float, default=15)
    g.add_argument("--min-chars", type=int, default=2, help="discard transcripts with fewer letters (and pure fillers)")
    g.add_argument("--show-partials", action="store_true")
    g = p.add_argument_group("echo protection")
    g.add_argument("--tail-ms", type=float, default=300, help="mic stays closed this long after playback ends")
    g.add_argument("--barge-in", action="store_true", help="EXPERIMENTAL: allow interrupting with loud speech (use headphones/AEC)")
    g.add_argument("--barge-db", type=float, default=-25.0); g.add_argument("--barge-margin-db", type=float, default=12.0)
    g.add_argument("--barge-min-ms", type=float, default=400)
    g.add_argument("--no-echo-gate", action="store_true", help="TESTING ONLY: disable half-duplex gate")
    g = p.add_argument_group("simulation (no hardware)")
    g.add_argument("--sim-input", default=None, help="wav fed as the mic in real time")
    g.add_argument("--sim-output", default="live_sim_out.wav")
    g.add_argument("--sim-echo-gain", type=float, default=0.5, help="bot output mixed back into the sim mic (0 = none)")
    g.add_argument("--sim-echo-delay-ms", type=float, default=40)
    g.add_argument("--log-json", default=None)
    a = p.parse_args()
    for k in ("input_device", "output_device"):
        v = getattr(a, k)
        if v is not None and v.isdigit(): setattr(a, k, int(v))
    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices()); return
    Live(a).run()


if __name__ == "__main__":
    main()
