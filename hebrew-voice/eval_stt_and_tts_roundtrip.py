"""(1) WER of both local STT engines on the 6 edge-tts test clips (synthetic voices - optimistic!),
(2) intelligibility check of the local TTS: transcribe the pipeline's own reply audio with ivrit-ai Whisper."""
import glob, re, time, json, numpy as np, hv_backends as B
REF = {"q1_greeting": "שלום מה שלומך היום", "q2_weather": "מה אתה ממליץ לעשות בסוף השבוע אם מזג האוויר יהיה יפה",
       "q3_instruction": "תגיד לי בבקשה שלושה טיפים קצרים כדי להתחיל לרוץ בבוקר"}
norm = lambda s: re.sub(r"[^\u05d0-\u05ea\s]", "", s).split()
def wer(r, h):
    r, h = norm(r), norm(h); d = np.zeros((len(r)+1, len(h)+1), int); d[:,0]=range(len(r)+1); d[0,:]=range(len(h)+1)
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1): d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(r[i-1]!=h[j-1]))
    return d[-1,-1]/max(1,len(r))
fw, sh = B.FasterWhisperSTT(), B.SherpaStreamingSTT(); fw.warmup(); sh.warmup()
res = {"fw": [], "sherpa": []}
for f in sorted(glob.glob("test_audio/*.wav")):
    a = B.load_wav16k(f); ref = REF[[k for k in REF if k in f][0]]
    for n, m in (("fw", fw), ("sherpa", sh)):
        t, ms = m.transcribe(a, realtime=False) if n=="fw" else m.transcribe(a, realtime=True)
        res[n].append((f, wer(ref, t), ms, t))
for n in res:
    print(n, "mean WER %.3f" % np.mean([x[1] for x in res[n]]), "| post-speech latency ms: median %.0f max %.0f" % (np.median([x[2] for x in res[n]]), max(x[2] for x in res[n])))
    for x in res[n]: print("   ", x[0].split('/')[-1], "WER %.2f %5.0fms  %s" % (x[1], x[2], x[3]))
print("\nTTS round-trip (reply audio -> ivrit-ai whisper):")
for f in sorted(glob.glob("out/*sherpa_gemma4e2b_blue_0.wav"))[:6]:
    a = B.load_wav16k(f); t, _ = fw.transcribe(a); print("  ", f.split('/')[-1], "->", t)
