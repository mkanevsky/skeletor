#!/usr/bin/env python
"""Simulation test for live.py (no mic/speaker needed).

Builds a 'mic' wav = steady background (pink noise + 50 Hz hum) + transient noises (clicks, knocks, coughs,
claps, typing, door slam) + a burst of loud fan/AC noise + the 6 Hebrew test clips separated by long gaps, then runs
live.py on it in real time with the bot's own output mixed back into the sim mic (echo gain 0.5 = loud speakers).

Checks
 (a) noise / steady background never starts a turn          -> 0 turn_start outside speech windows
 (b) every Hebrew clip starts exactly one turn, is transcribed (WER) and gets a spoken reply
 (c) the bot's echo during playback never starts a turn      -> 0 self-triggered turns
     + control run with --no-echo-gate showing the same echo DOES trigger turns without the gate.
usage: python test_live.py [--llm gemma4e2b] [--skip-control] [--extra "live.py flags"]
"""
import argparse, json, re, shlex, subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf
import hv_backends as B
import live as L

SR = 16000
REF = {"q1_greeting": "שלום מה שלומך היום", "q2_weather": "מה אתה ממליץ לעשות בסוף השבוע אם מזג האוויר יהיה יפה",
       "q3_instruction": "תגיד לי בבקשה שלושה טיפים קצרים כדי להתחיל לרוץ בבוקר"}
norm = lambda s: re.sub(r"[^\u05d0-\u05ea\s]", "", s).split()
def wer(r, h):
    r, h = norm(r), norm(h); d = np.zeros((len(r) + 1, len(h) + 1), int); d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1): d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))

rng = np.random.default_rng(7)
def rms_to(x, db): return x * (10 ** (db / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12))
def pink(n):
    w = rng.standard_normal(n); f = np.fft.rfft(w); k = np.arange(len(f)); k[0] = 1
    return np.fft.irfft(f / np.sqrt(k), n)
def bp(x, lo, hi):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, [lo, hi], "bandpass", fs=SR, output="sos"), x)
def lp(x, hz):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, hz, "lowpass", fs=SR, output="sos"), x)
def env(n, att, dec):
    t = np.arange(n) / SR; return np.minimum(1, t / max(att, 1e-4)) * np.exp(-t / dec)

def click():  return np.concatenate([[0.9, -0.6, 0.3], np.zeros(200)]) * env(203, 1e-4, 0.002)
def knock():  n = int(0.08 * SR); return 0.6 * lp(rng.standard_normal(n), 900) * env(n, 0.002, 0.02)
def clap():   n = int(0.06 * SR); return 0.7 * bp(rng.standard_normal(n), 800, 6000) * env(n, 0.001, 0.012)
def slam():   n = int(0.4 * SR);  return 0.8 * lp(rng.standard_normal(n), 400) * env(n, 0.003, 0.08)
def cough():
    out = []
    for dur, amp in ((0.14, 0.35), (0.06, 0), (0.22, 0.25)):
        n = int(dur * SR)
        if amp == 0: out.append(np.zeros(n)); continue
        x = bp(rng.standard_normal(n), 300, 3500) + 0.6 * bp(rng.standard_normal(n), 500, 900)   # broadband + low formant
        out.append(amp * x / np.abs(x).max() * env(n, 0.008, dur / 2.5))
    return np.concatenate(out)
def voiced_cough():
    """'uh-hm' style cough: glottal pulse train (~140 Hz) + breath noise, two bursts ~120/200 ms."""
    out = []
    for dur, amp in ((0.12, 0.35), (0.05, 0), (0.2, 0.3)):
        n = int(dur * SR)
        if amp == 0: out.append(np.zeros(n)); continue
        tt = np.arange(n) / SR; f0 = 140 * (1 - 0.3 * tt / dur)
        pulses = np.sign(np.sin(2 * np.pi * np.cumsum(f0) / SR)) * 0.5
        x = bp(pulses, 200, 2500) + 0.8 * bp(rng.standard_normal(n), 300, 4000)
        out.append(amp * x / np.abs(x).max() * env(n, 0.006, dur / 2))
    return np.concatenate(out)
_short = {}
def short_speech():
    """180 ms slice of real speech (shorter than --min-speech-ms 250)."""
    x = B.load_wav16k("test_audio/q2_weather_avri.wav"); i = int(np.argmax(np.abs(x))); a = x[max(0, i - 1440): i + 1440]
    return rms_to(a.astype(np.float64) * np.hanning(len(a)), -24)
def filler():
    """'אממממ...' (BlueTTS noa, 0.77 s, cached). Allowed to START a turn, must be discarded by the STT filter."""
    p = Path("out_live/filler_amm.wav")
    if not p.exists():
        t = B.BlueTTSLocal("noa"); a, sr, _ = t.synth("אממממ..."); sf.write(p, L.resample(a, sr, SR), SR)
    return rms_to(B.load_wav16k(str(p)).astype(np.float64), -24)
def typing(dur=2.5):
    n = int(dur * SR); x = np.zeros(n)
    for t in np.sort(rng.uniform(0, dur - 0.02, int(dur * 8))):
        c = 0.25 * bp(rng.standard_normal(160), 1500, 7000) * env(160, 1e-4, 0.003); i = int(t * SR); x[i:i + 160] += c
    return x
def fan(dur=3.0):
    n = int(dur * SR); x = rms_to(lp(rng.standard_normal(n), 2500) + 0.5 * pink(n), -32)
    f = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.3 * SR)); return x * f

def build(path, gap_s=17.0, clips=None):
    clips = clips or sorted(Path("test_audio").glob("q*_*.wav"))
    plan = [("bg", 2.0)]
    noise1 = [("click", click), ("click", click), ("knock", knock), ("cough", cough), ("click", click), ("knock", knock),
              ("clap", clap), ("cough", cough), ("typing", typing), ("knock", knock), ("fan", fan), ("slam", slam),
              ("click", click), ("cough", cough), ("clap", clap), ("voiced_cough", voiced_cough), ("short_speech", short_speech),
              ("voiced_cough", voiced_cough), ("filler", filler)]
    noise2 = [("cough", cough), ("knock", knock), ("voiced_cough", voiced_cough), ("click", click), ("slam", slam), ("typing", typing),
              ("short_speech", short_speech), ("cough", cough)]
    segs, gt, t = [], dict(speech=[], noise=[]), 0.0
    def add(x, kind, name=None):
        nonlocal t
        segs.append((t, x)); (gt[kind].append(dict(name=name, t0=round(t, 3), t1=round(t + len(x) / SR, 3))) if kind else None)
        t += len(x) / SR
    add(np.zeros(int(2.0 * SR)), None)
    for nm, fn in noise1:
        add(fn(), "noise", nm); add(np.zeros(int(rng.uniform(0.6, 1.2) * SR)), None)
    add(np.zeros(int(1.5 * SR)), None)
    for c in clips:
        x = rms_to(B.load_wav16k(str(c)).astype(np.float64), -22); add(x, "speech", c.stem); add(np.zeros(int(gap_s * SR)), None)
    for nm, fn in noise2:
        add(fn(), "noise", nm); add(np.zeros(int(rng.uniform(0.6, 1.2) * SR)), None)
    add(np.zeros(int(2.0 * SR)), None)
    n = int(t * SR) + 1; y = np.zeros(n)
    for t0, x in segs:
        i = int(round(t0 * SR)); y[i:i + len(x)] += x[:n - i]
    tt = np.arange(n) / SR
    bg = rms_to(pink(n), -55) + rms_to(np.sin(2 * np.pi * 50 * tt) + 0.3 * np.sin(2 * np.pi * 150 * tt), -50)
    y = np.clip(y + bg, -1, 1).astype(np.float32)
    sf.write(path, y, SR, subtype="PCM_16"); gt["dur_s"] = round(n / SR, 2)
    return gt

def frame_level(path, gt, a):
    """Offline: what the front-end (HPF + energy gate + Silero) thinks of each noise event."""
    x = B.load_wav16k(path); vad = L.SileroVAD(B.HERE / "vad/silero_vad.onnx"); hpf = L.HighPass(90)
    dbs, ps = [], []
    for i in range(0, len(x) - 512 + 1, 512):
        f = hpf(x[i:i + 512]); dbs.append(L.dbfs(f)); ps.append(vad(f))
    dbs, ps = np.array(dbs), np.array(ps); fs = 512 / SR
    floor = np.percentile(dbs[:int(1 / fs)], 50); thr = max(-50, floor + 10)
    rows = []
    for ev in gt["noise"]:
        i0, i1 = int(ev["t0"] / fs), int(ev["t1"] / fs) + 4
        e_ok, v_ok = dbs[i0:i1] >= thr, ps[i0:i1] >= 0.5; both = e_ok & v_ok
        run = mx = 0
        for b in both: run = run + 1 if b else 0; mx = max(mx, run)
        rows.append(dict(name=ev["name"], t=ev["t0"], peak_db=round(float(dbs[i0:i1].max()), 1), max_p=round(float(ps[i0:i1].max()), 2),
                         loud_frames=int(e_ok.sum()), vad_frames=int(v_ok.sum()), max_run_ms=round(mx * fs * 1000)))
    return floor, thr, rows

def classify(log, gt, tail_s):
    ev = log["events"]; speech = gt["speech"]
    play = []; fa = None
    for e in ev:
        if e["ev"] == "first_audio": fa = e["t"]
        if e["ev"] == "playback_end" and fa is not None: play.append((fa, e["t"] + tail_s)); fa = None
    res = dict(turn_starts=0, starts_speech=0, starts_noise=[], starts_self=[], starts_filler=[], per_clip={s["name"]: [] for s in speech})
    for e in ev:
        if e["ev"] != "turn_start": continue
        res["turn_starts"] += 1; t = e["t"]
        hit = [s for s in speech if s["t0"] - 0.4 <= t <= s["t1"] + 0.2]
        if hit: res["starts_speech"] += 1; res["per_clip"][hit[0]["name"]].append(t); continue
        if any(p0 - 0.1 <= t <= p1 for p0, p1 in play): res["starts_self"].append(t); continue
        if any(n["name"] == "filler" and n["t0"] - 0.4 <= t <= n["t1"] + 0.2 for n in gt["noise"]): res["starts_filler"].append(t); continue
        res["starts_noise"].append(t)
    return res

def run_live(args, inp, out, logp, extra):
    cmd = [sys.executable, "live.py", "--sim-input", inp, "--sim-output", out, "--log-json", logp, "--llm", args.llm,
           "--voice", "witch_gacrux", "--voice-fx", "witch", "--persona", "witch"] + extra
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(Path(logp).read_text())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--llm", default="gemma4e2b"); ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--extra", default=""); ap.add_argument("--outdir", default="out_live")
    args = ap.parse_args(); od = Path(args.outdir); od.mkdir(exist_ok=True)
    inp = str(od / "sim_mic_in.wav"); gt = build(inp)
    fill = [n for n in gt["noise"] if n["name"] == "filler"]
    (od / "sim_mic_in_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1))
    floor, thr, rows = frame_level(inp, gt, None)
    print(f"\nsim input {gt['dur_s']} s: {len(gt['speech'])} speech clips, {len(gt['noise'])} noise events; floor {floor:.1f} dBFS, gate {thr:.1f} dBFS")
    print("noise event front-end view (energy>=gate & Silero>=0.5; a turn needs a 250 ms run):")
    for r in rows: print(f"   {r['name']:7s} t={r['t']:6.1f}s peak {r['peak_db']:6.1f} dBFS  max p={r['max_p']:.2f}  loud={r['loud_frames']:3d} vad={r['vad_frames']:3d} frames  longest run {r['max_run_ms']:4d} ms")
    report = dict(input=gt, frontend=rows)
    log = run_live(args, inp, str(od / "sim_out_gated.wav"), str(od / "log_gated.json"), shlex.split(args.extra))
    c = classify(log, gt, log["args"]["tail_ms"] / 1000)
    heard = [e for e in log["events"] if e["ev"] == "heard"]; bots = [e for e in log["events"] if e["ev"] == "bot"]
    ign = [e for e in log["events"] if e["ev"] == "ignored"]
    print("\n================ RESULTS (gated, default flags) ================")
    print(f"(a) noise-triggered turn starts: {len(c['starts_noise'])}  {c['starts_noise']}")
    print(f"    filler 'אממממ' turn starts: {len(c['starts_filler'])}, discarded by STT filter: {[(e['text'], e['why']) for e in ign]}")
    print(f"(c) self-triggered turn starts during/after playback: {len(c['starts_self'])}  {c['starts_self']}")
    st = log["stats"]
    print(f"    frames dropped by gate: {st.get('dropped_frames',0)} ({st.get('dropped_frames',0)*0.032:.1f} s), during playback {st.get('dropped_during_playback',0)}, "
          f"of which loud {st.get('dropped_loud',0)} / loud+speech-like {st.get('dropped_speechlike',0)} (would have counted as speech)")
    print(f"(b) speech clips: {len(gt['speech'])}, with exactly one turn: {sum(len(v)==1 for v in c['per_clip'].values())}, heard: {len(heard)}, replies: {sum(1 for b in bots if b['text'])}")
    wers, lats, lats2 = [], [], []
    for s in gt["speech"]:
        h = [e for e in heard if s["t0"] <= e["t"] <= s["t1"] + 3.0]
        b = [e for e in bots if e["t"] >= s["t1"] and (not h or e["t"] >= h[0]["t"])][:1]
        ref = REF["_".join(s["name"].split("_")[:2])]
        w = wer(ref, h[0]["text"]) if h else 1.0; wers.append(w)
        if b and b[0]["latency_ms"]: lats.append(b[0]["latency_ms"]); lats2.append(b[0]["latency_from_speech_end_ms"])
        print(f"   {s['name']:22s} heard: {h[0]['text'] if h else '-'}  (WER {w:.2f})\n   {'':22s} bot:   {b[0]['text'] if b else '-'}  "
              f"[first audio {b[0]['latency_ms'] if b else '-'} ms after detect, {b[0]['latency_from_speech_end_ms'] if b else '-'} ms after speech end]")
    if lats: print(f"    latency first audio after end-of-turn detect: median {np.median(lats):.0f} ms ({min(lats):.0f}-{max(lats):.0f}); "
                   f"after actual end of speech: median {np.median(lats2):.0f} ms ({min(lats2):.0f}-{max(lats2):.0f}); mean WER {np.mean(wers):.3f}")
    report["gated"] = dict(classify=c, stats=st, wer=wers, latency_ms=lats, latency_from_speech_end_ms=lats2, heard=heard, bots=bots, ignored=ign)
    filler_replied = [e for e in heard if any(n["t0"] <= e["t"] <= n["t1"] + 3 for n in fill)]
    print(f"    filler turns that got a reply: {len(filler_replied)}")
    ok = not filler_replied and not c["starts_noise"] and not c["starts_self"] and all(len(v) == 1 for v in c["per_clip"].values()) and len(heard) == len(gt["speech"])
    if not args.skip_control:
        inp2 = str(od / "sim_mic_in_short.wav"); gt2 = build(inp2, gap_s=12.0, clips=sorted(Path("test_audio").glob("q*_hila.wav"))[:2])
        log2 = run_live(args, inp2, str(od / "sim_out_nogate.wav"), str(od / "log_nogate.json"), ["--no-echo-gate"] + shlex.split(args.extra))
        c2 = classify(log2, gt2, 0.0)
        print(f"\nCONTROL (--no-echo-gate, same echo): turn starts {c2['turn_starts']}, speech {c2['starts_speech']}, "
              f"self/echo-triggered {len(c2['starts_self'])}, other {len(c2['starts_noise'])}; "
              f"heard events: {[e['text'] for e in log2['events'] if e['ev']=='heard']}")
        report["control_nogate"] = dict(classify=c2, events=log2["events"])
    (od / "test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("\nPASS" if ok else "\nFAIL", flush=True); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
