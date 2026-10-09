#!/usr/bin/env python3
"""Measure a llama-server unit at depth: does --parallel > 1 help? (docs/06 §5, pitfall #34)

Run: python3 bench-parallel-depth.py   # ~10 min against :8088 (BENCH_URL), occupies its slots
"""
import json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

# :8088 (Ornith-35B-A3B, the Open Notebook worker) was the unit measured. Env, not a flag:
# the script has no argparse and the arms below are the experiment.
URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8088")
# Was a Claude Code session scratchpad path, which no longer exists; cwd by default now.
OUT = os.environ.get("BENCH_OUT", "depth_bench.json")

def post(path, payload, timeout=1800):
    req = urllib.request.Request(URL+path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])

WORDS = ("survey instrument calibration drift telemetry aggregate quarterly variance "
         "sediment coastal gradient turbine maintenance schedule anomaly threshold "
         "inventory reconciliation ledger provisional estimate downstream consumer "
         "latency percentile regression baseline corpus annotation guideline").split()

def make_prompt(target, seed):
    rnd = random.Random(seed)
    parts, n = [], 0
    while n < target * 1.4:
        s = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(8, 18)))
        parts.append(s.capitalize() + ".")
        n += 12
    text = " ".join(parts)
    # binary-trim to target tokens
    lo, hi = 0, len(text)
    while lo < hi - 200:
        mid = (lo + hi) // 2
        if ntok(text[:mid]) < target: lo = mid
        else: hi = mid
    text = text[:lo]
    return text, ntok(text)

def run(prompt, tag, npred=128):
    t0 = time.time()
    r = post("/completion", {"prompt": prompt, "n_predict": npred, "temperature": 0,
                             "cache_prompt": False, "stream": False})
    wall = time.time() - t0
    t = r["timings"]
    return {"tag": tag, "wall_s": round(wall, 2),
            "prompt_n": t["prompt_n"], "pp_tps": round(t["prompt_per_second"], 2),
            "pred_n": t["predicted_n"], "tg_tps": round(t["predicted_per_second"], 2)}

def arm(name, specs):
    print(f"\n### {name}", flush=True)
    prompts = []
    for i, (depth, seed) in enumerate(specs):
        p, n = make_prompt(depth, seed)
        prompts.append((p, f"req{i}(d{n})"))
        print(f"  built {n} tokens", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        rows = list(ex.map(lambda a: run(*a), prompts))
    total = round(time.time() - t0, 2)
    for r in rows: print("   ", r, flush=True)
    print(f"    ARM WALL: {total}s", flush=True)
    return {"arm": name, "wall_s": total, "rows": rows}

out = []
out.append(arm("A. one request at d32k, alone", [(32768, 1)]))
out.append(arm("B. two concurrent at d32k each", [(32768, 2), (32768, 3)]))
out.append(arm("C. two concurrent at d65k each (fills the 131072 pool)", [(65000, 4), (65000, 5)]))
out.append(arm("D. one request at d130k, alone (whole window to one slot)", [(130000, 6)]))
json.dump(out, open(OUT, "w"), indent=2)
print("\nDONE", flush=True)
