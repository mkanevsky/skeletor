"""Probe Gemini TTS models for Hebrew witch voice; measures time to first audio (streaming SSE). Key read from env, never printed."""
import os,sys,time,json,base64,httpx,numpy as np,soundfile as sf
K=os.environ["GEMINI_API_KEY"]; c=httpx.Client(timeout=60)
TXT="ברוכים הבאים לבקתה שלי, יקירי. התקרב... אל תפחד, הקדרה כבר רותחת."
STYLE="Say in Hebrew, as a scary old witch: raspy, whispery, sinister, with a creepy cackle: "
SYS="You are a voice actor. Read the user's Hebrew text aloud exactly, word for word, and nothing else, in the voice of a scary old witch: raspy, whispery, sinister, slow, with a short creepy cackle at the end."
def tts(model,voice,text=TXT,style=STYLE,out=None,mode="sys"):
    if mode=="sys": body={"systemInstruction":{"parts":[{"text":SYS}]},"contents":[{"role":"user","parts":[{"text":text}]}],
          "generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice}}}}}
    elif mode=="tags": body={"contents":[{"role":"user","parts":[{"text":"[raspy] [whispering] [sinister] "+text+" [cackles]"}]}],
          "generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice}}}}}
    else: body={"contents":[{"role":"user","parts":[{"text":style+text}]}],
          "generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice}}}}}
    t0=time.perf_counter(); first=None; pcm=b""; mime=None
    with c.stream("POST",f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse",headers={"x-goog-api-key":K},json=body) as r:
        if r.status_code!=200: return f"HTTP {r.status_code}: {r.read()[:250]!r}"
        n=0
        for line in r.iter_lines():
            if not line.startswith("data:"): continue
            d=json.loads(line[5:])
            for p in d.get("candidates",[{}])[0].get("content",{}).get("parts",[]):
                if "inlineData" in p:
                    if first is None: first=(time.perf_counter()-t0)*1000
                    mime=p["inlineData"].get("mimeType"); pcm+=base64.b64decode(p["inlineData"]["data"]); n+=1
    tot=(time.perf_counter()-t0)*1000
    a=np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768
    if out: sf.write(out,a,24000)
    return f"first_audio={first:.0f}ms total={tot:.0f}ms chunks={n} audio={len(a)/24000:.2f}s mime={mime}"
if __name__=="__main__":
    for spec in sys.argv[1:]:
        m,v,mode=spec.split("/")
        print(m,v,mode,tts(m,v,out=f"witch_work/gem_{m}_{v}_{mode}.wav",mode=mode),flush=True)
