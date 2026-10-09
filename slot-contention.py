#!/usr/bin/env python3
# Was scripts/hermes-slot-contention.py in halo-prep (renamed 2026-10-09).
"""Measure head-of-line blocking on a llama-server slot: how long a short request waits
behind a long one. The A/B for `--parallel 1` vs `--parallel 2 --kv-unified` on :8092.

  python3 slot-contention.py http://127.0.0.1:8092/v1 ornith-1.5-9b --label np1
  python3 slot-contention.py http://127.0.0.1:8092/v1 ornith-1.5-9b --label np2 --json results.json

WHY THIS FILE EXISTS. Hermes on :8092 started timing out (stream stale 900 s, no chunks)
on 2026-09-20 while a cron job held the only slot, and the 2026-08-29 `:8088` experiment --
which rejected -np 2 -- measured prefill-bound batch work, not Hermes's mixed workload
(short Signal turns arriving while a 15-minute cron turn decodes). Whether a second slot
helps depends on *where* the short request lands, so this measures both cases separately:

  A. short request arrives during the long one's PREFILL  (GPU compute-bound)
  B. short request arrives during the long one's DECODE   (GPU memory-bound, batch has room)

and reports the short request's time-to-first-token (what a Signal user feels) and the
long request's total (what the cron job pays for sharing). Every request carries the same
~9K-token "system prompt" so the prefix-cache behaviour is realistic: with one slot the
short request re-uses the cache the long one just built; with two, it may land on a cold
slot and re-prefill the prefix. That miss is part of the cost being measured, not noise.

Idle GPU only -- a live podcast on :8088 or a Hermes turn contaminates every number here.
"""
import argparse
import json
import pathlib
import sys
import threading
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent


def corpus(chars):
    # Deterministic filler from the repo's own docs, so both arms prefill identical tokens.
    text = ""
    for p in sorted((REPO / "docs").glob("*.md")):
        text += p.read_text(errors="replace")
        if len(text) >= chars:
            break
    return text[:chars]


SYSTEM = ("You are a careful assistant on a Linux workstation. Reference material follows; "
          "use it only if asked.\n\n" + corpus(36000))          # ~9K tokens: Hermes-sized prefix
LONG_USER = ("Summarise the following runbook section by section, one paragraph each, "
             "in order, without skipping any:\n\n" + corpus(48000)[36000:])  # +~3K tokens
SHORT_USER = "In one sentence: what is 17 multiplied by 23? Then say why."


def stream(base, model, user, max_tokens, out):
    body = json.dumps({"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    out["start"] = t0
    first = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                d = json.loads(line[6:])
                delta = d["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    n += 1
                    if first is None:
                        first = time.time()
                if d.get("timings"):
                    out["timings"] = d["timings"]
    except Exception as e:  # noqa: BLE001 -- record it; a failed arm is a result, not a crash
        out["error"] = str(e)[:200]
    out["ttft"] = (first - t0) if first else None
    out["total"] = time.time() - t0
    out["tokens"] = n


def scenario(base, model, delay, label):
    long_r, short_r = {}, {}
    tl = threading.Thread(target=stream, args=(base, model, LONG_USER, 400, long_r))
    tl.start()
    time.sleep(delay)
    stream(base, model, SHORT_USER, 48, short_r)
    tl.join()
    print(f"  {label:<38} short TTFT {short_r['ttft'] or float('nan'):7.1f} s  "
          f"short total {short_r['total']:6.1f} s  long total {long_r['total']:6.1f} s"
          + (f"  ERR short={short_r['error']}" if short_r.get("error") else "")
          + (f"  ERR long={long_r['error']}" if long_r.get("error") else ""))
    return {"label": label, "short": short_r, "long": long_r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", help="append a result record to this file")
    a = ap.parse_args()

    # Warm the prefix once so arm A measures contention, not a cold cache on the long request.
    w = {}
    stream(a.base, a.model, SHORT_USER, 8, w)
    print(f"[{a.label}] warm-up: TTFT {w['ttft']:.1f} s (prefix prefill), server={a.base}")
    # Solo baselines: what the short request costs with nobody else on the server.
    s = {}
    stream(a.base, a.model, SHORT_USER, 48, s)
    print(f"  {'short alone (warm prefix)':<38} short TTFT {s['ttft']:7.1f} s  short total {s['total']:6.1f} s")
    l = {}
    stream(a.base, a.model, LONG_USER, 400, l)
    long_prefill = l["ttft"]
    print(f"  {'long alone':<38} long  TTFT {l['ttft']:7.1f} s  long  total {l['total']:6.1f} s")

    results = [scenario(a.base, a.model, 1.0, "A: short arrives 1 s into long prefill"),
               scenario(a.base, a.model, long_prefill + 3.0,
                        "B: short arrives 3 s into long decode")]
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"label": a.label, "base": a.base, "model": a.model,
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "short_alone": s, "long_alone": l, "scenarios": results}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
