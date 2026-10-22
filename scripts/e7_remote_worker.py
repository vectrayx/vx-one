#!/usr/bin/env python3
"""Worker remoto E7 (fleet): lee un slice de requests.jsonl.
Formato del slice: línea 1 = {"k":-1,"questions":...} (meta), resto = {"k":<test-idx>,"state":...}.
Append incremental en results.jsonl (flush). Honora retry-after de 429.
Uso: python3 e7_remote_worker.py <slice.jsonl> <results.jsonl>   (KEY en env OPENCODE_API_KEY)
PACE en env (default 1.0s).
"""
import json, os, sys, time, urllib.error, urllib.request

KEY = os.environ["OPENCODE_API_KEY"]
REQ, RES = sys.argv[1], sys.argv[2]
URL = "https://opencode.ai/zen/v1/systemone"
PACE = float(os.environ.get("PACE", "1.0"))

done = set()
if os.path.exists(RES):
    for line in open(RES):
        if line.strip():
            done.add(json.loads(line)["k"])

rows, questions = [], None
for line in open(REQ):
    if not line.strip():
        continue
    r = json.loads(line)
    if "questions" in r and questions is None:
        questions = r["questions"]
    else:
        rows.append(r)

def ask(state):
    body = json.dumps({"model": "jev-1.13-free", "state": state,
                       "questions": questions}, separators=(",", ":")).encode()
    req = urllib.request.Request(URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "vxone-e7/0.1"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read()), (time.perf_counter() - t0) * 1000

todo = [r for r in rows if r["k"] not in done and r["k"] >= 0]
print(f"worker: {len(todo)}/{len(rows)} por hacer (pace {PACE}s)", flush=True)
t_start = time.time()
stalled = False
with open(RES, "a") as fh:
    for j, r in enumerate(todo):
        backoff = 2.0
        resp, lat = None, None
        for attempt in range(30):
            try:
                resp, lat = ask(r["state"])
                break
            except urllib.error.HTTPError as e:
                ra_h = e.headers.get("retry-after")
                ra = None
                if ra_h:
                    try: ra = float(ra_h)
                    except ValueError: ra = None
                print(f"HTTP {e.code} ra={ra} fila {j+1}/{len(todo)}", flush=True)
                if e.code == 429 and ra and ra > 1800:
                    print(f"cooldown largo {ra/60:.0f} min — duermo", flush=True)
                    time.sleep(min(ra + 5, 24*3600))
                    backoff = 2.0
                    continue
                time.sleep(min(backoff, 120)); backoff *= 2
            except Exception as e:
                print(f"err {type(e).__name__}: {e}; backoff {backoff:.0f}s", flush=True)
                time.sleep(backoff); backoff = min(backoff*2, 120)
        if resp is None:
            print("ATASCO FINAL: salgo (resume ok)", flush=True)
            stalled = True
            break
        ans = resp["answers"]
        fh.write(json.dumps({
            "k": r["k"], "gate": ans.get("gate", {}).get("noul"),
            "family": ans.get("family", {}).get("probabilities"),
            "sev": ans.get("severity", {}).get("probabilities"),
            "lat_ms": round(lat, 1), "tok": resp["usage"]["input_tokens"],
            "model": resp.get("model")}) + "\n")
        fh.flush()
        if (j+1) % 50 == 0:
            print(f"{j+1}/{len(todo)} {(time.time()-t_start)/60:.1f} min", flush=True)
        time.sleep(PACE)
print("WORKER DONE" if not stalled else "WORKER STALLED", flush=True)
