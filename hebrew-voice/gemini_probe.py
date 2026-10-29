import os, time, json, httpx, sys
K=os.environ["GEMINI_API_KEY"]; c=httpx.Client(timeout=30, http2=False)
def run(model, tc, text="שלום, מה שלומך היום?"):
    body={"contents":[{"role":"user","parts":[{"text":text}]}],
          "systemInstruction":{"parts":[{"text":"ענה בעברית בקצרה, משפט או שניים."}]},
          "generationConfig":{"maxOutputTokens":150,"thinkingConfig":tc}}
    t0=time.perf_counter(); first=None; out=""; usage=None
    with c.stream("POST",f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse",
                  headers={"x-goog-api-key":K},json=body) as r:
        if r.status_code!=200: return f"HTTP {r.status_code}: {r.read()[:200]!r}"
        for line in r.iter_lines():
            if not line.startswith("data:"): continue
            d=json.loads(line[5:]); usage=d.get("usageMetadata",usage)
            for p in d.get("candidates",[{}])[0].get("content",{}).get("parts",[]):
                if p.get("text") and not p.get("thought"):
                    if first is None: first=(time.perf_counter()-t0)*1000
                    out+=p["text"]
    return f"ttft={first:.0f}ms total={(time.perf_counter()-t0)*1000:.0f}ms thoughts={usage.get('thoughtsTokenCount')} | {out.strip()}"
for m in sys.argv[1:]:
    for tc in ({"thinkingBudget":0},{"thinkingLevel":"minimal"}):
        for i in range(2): print(m, tc, run(m,tc))
