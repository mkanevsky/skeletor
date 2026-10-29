"""Real-time-friendly 'witch' voice effects for TTS chunks (numpy + pedalboard).

Applied per TTS chunk right after synthesis, so it adds (processing time) to time-to-first-audio and
nothing else. Chain (preset 'witch'):
  1. pitch down. pitch_mode 'resample' (default, ~1 ms): resample so the chunk plays lower AND ~12% slower
     (formants drop too; the slower delivery suits the character). 'rubberband': tempo-preserving
     Rubber Band pitch shift via pedalboard (~25-35 ms per second of audio).
  2. vibrato wobble (Chorus, mix=1 => pure delay modulation) + tremolo (amplitude LFO) -> shaky crone voice
  3. grit: soft-clip distortion + light bitcrush, high-passed and mixed in parallel -> raspy
  4. whisper layer: band-passed noise shaped by the speech envelope -> breathy/whispery
  5. reverb (dark room) + slapback echo                    -> cave / hut ambience
Each chunk gets a short silent tail so the reverb rings out instead of being cut.
"""
from __future__ import annotations
import time
import numpy as np
from pedalboard import (Pedalboard, PitchShift, Chorus, Distortion, Bitcrush, Reverb, Delay,
                        HighpassFilter, LowpassFilter, Gain, Compressor, Limiter)

PRESETS = {
    "witch-rb": dict(semitones=-2.0, pitch_mode="rubberband", vibrato_rate=5.5, vibrato_depth=0.18, trem_rate=6.5, trem_depth=0.22,
                     grit_drive=18.0, grit_bits=8, grit_mix=0.30, whisper_mix=0.22,
                     reverb_room=0.55, reverb_wet=0.22, echo_s=0.11, echo_mix=0.12, tail_s=0.30),
    "witch": dict(semitones=-2.0, pitch_mode="resample", vibrato_rate=5.5, vibrato_depth=0.18, trem_rate=6.5, trem_depth=0.22,
                  grit_drive=18.0, grit_bits=8, grit_mix=0.30, whisper_mix=0.22,
                  reverb_room=0.55, reverb_wet=0.22, echo_s=0.11, echo_mix=0.12, tail_s=0.30),
    # lighter version: less wobble/grit, better intelligibility
    "witch-lite": dict(semitones=-1.0, pitch_mode="resample", vibrato_rate=5.0, vibrato_depth=0.10, trem_rate=6.0, trem_depth=0.12,
                       grit_drive=12.0, grit_bits=10, grit_mix=0.18, whisper_mix=0.14,
                       reverb_room=0.4, reverb_wet=0.15, echo_s=0.10, echo_mix=0.06, tail_s=0.25),
}

class VoiceFX:
    def __init__(self, preset="witch", seed=0):
        self.name = preset; p = self.p = PRESETS[preset]
        self.pitch = None
        if p["semitones"] and p.get("pitch_mode") == "rubberband":
            self.pitch = Pedalboard([PitchShift(semitones=p["semitones"])])
        elif p["semitones"]:
            from fractions import Fraction
            r = Fraction(2 ** (-p["semitones"] / 12)).limit_denominator(40)   # e.g. -2 st -> 55/49
            self.up, self.down = r.numerator, r.denominator
        self.wobble = Pedalboard([Chorus(rate_hz=p["vibrato_rate"], depth=p["vibrato_depth"], centre_delay_ms=6.0,
                                         feedback=0.0, mix=1.0)])
        self.grit = Pedalboard([HighpassFilter(700), Distortion(drive_db=p["grit_drive"]),
                                Bitcrush(bit_depth=p["grit_bits"]), LowpassFilter(7000)])
        self.space = Pedalboard([Delay(delay_seconds=p["echo_s"], feedback=0.15, mix=p["echo_mix"]),
                                 Reverb(room_size=p["reverb_room"], damping=0.6, wet_level=p["reverb_wet"],
                                        dry_level=1.0 - p["reverb_wet"] * 0.5, width=0.8),
                                 LowpassFilter(9000), Compressor(threshold_db=-18, ratio=3), Limiter(-1.0)])
        self.rng = np.random.default_rng(seed)

    def _whisper(self, x, sr):
        # envelope follower (~10 ms) -> modulate band-passed white noise (2.5-7 kHz)
        k = max(1, int(sr * 0.01)); c = np.cumsum(np.abs(x), dtype=np.float64)
        env = ((c - np.concatenate([np.zeros(k), c[:-k]])) / k).astype(np.float32)
        n = self.rng.standard_normal(len(x)).astype(np.float32)
        n = Pedalboard([HighpassFilter(2500), LowpassFilter(7000)])(n, sr)
        return n * env * 3.0

    def process(self, audio: np.ndarray, sr: int):
        """Returns (processed_audio, ms)."""
        t0 = time.perf_counter(); p = self.p
        x = np.concatenate([audio.astype(np.float32), np.zeros(int(sr * p["tail_s"]), np.float32)])
        if self.pitch is not None: x = self.pitch(x, sr)
        elif p["semitones"]:
            from scipy.signal import resample_poly
            x = resample_poly(x, self.up, self.down).astype(np.float32)
        x = self.wobble(x, sr)
        t = np.arange(len(x)) / sr
        x = x * (1.0 - p["trem_depth"] * (0.5 + 0.5 * np.sin(2 * np.pi * p["trem_rate"] * t))).astype(np.float32)
        x = x + p["grit_mix"] * self.grit(x, sr) + p["whisper_mix"] * self._whisper(x, sr)
        x = self.space(x, sr)
        return x.astype(np.float32), (time.perf_counter() - t0) * 1000

class FXTTS:
    """Wraps any TTS backend: synth(text) -> (audio, sr, ms) with ms including the FX time; last_fx_ms kept."""
    def __init__(self, tts, fx: VoiceFX):
        self.tts, self.fx = tts, fx; self.name = f"{tts.name}+fx:{fx.name}"; self.last_fx_ms = None; self.last_raw_ms = None
    def warmup(self):
        self.tts.warmup(); a, sr, _ = self.tts.synth("שלום לכם"); self.fx.process(a, sr)
    def synth(self, text):
        a, sr, ms = self.tts.synth(text)
        y, fx_ms = self.fx.process(a, sr)
        self.last_raw_ms, self.last_fx_ms = ms, fx_ms
        return y, sr, ms + fx_ms
