#!/usr/bin/env python3
# Was scripts/hermes-sampler-check.py in halo-prep (renamed 2026-10-09).
"""Re-run the tool harness at the sampler the unit actually serves, not at temp 0.

  python3 sampler-check.py http://127.0.0.1:8092/v1 ornith-1.5-9b \
      --sampler '{"temperature":1.0,"top_k":20,"top_p":0.95,"min_p":0.0,"presence_penalty":1.5}' \
      --kwargs '{"terse":true}' --repeats 10

WHY THIS EXISTS, AND WHY IT IS A SEPARATE FILE. `tool-harness.py` forces
`temperature: 0` on purpose -- that is what makes its numbers comparable across
runs and across models, and it must not change. But Hermes sends no sampler
parameters of its own, so what a real Signal turn gets is whatever the .service
file sets, and for both models on this box that is `temperature 1.0`. The two
are not the same measurement, and on 2026-08-27 they disagreed:

    ornith-1.5-9b, terse, same three prompts, same box, same hour
      temp 0                      30/30
      unit's own sampler          24/30   <- `max_results` dropped 4/10

That is pitfall #25's failure -- a well-formed call with a quietly dropped
bound -- appearing ONLY at the sampler the box actually runs. A greedy 15/15
therefore does not license "tool calling is fine"; it licenses "tool calling is
fine when we ask it a way we never ask it". Run both.

The repeats default to 10 rather than the harness's 3-5 because at temperature 1
the question stops being pass/fail and becomes a rate; 5 repeats cannot
distinguish 6/10 from 9/10.

Everything else -- the prompts, the argument-level checks, the toolset -- is
imported from `tool-harness.py` so the two cannot drift apart. That file
remains the definition of the experiment.
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

_HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tool-harness.py")  # hermes-tool-harness.py until 2026-10-09
_spec = importlib.util.spec_from_file_location("hermes_tool_harness", _HARNESS)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def run_arm(base, model, sampler, kwargs, repeats):
    # Warm the slot -- pitfall #27, the first request carries cold-cache noise.
    try:
        h.post(base, {"model": model, "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 8, "temperature": 0})
    except Exception as e:
        print(f"  warm-up failed: {e}", file=sys.stderr)

    rows = []
    for label, prompt, check in h.PROMPTS:
        toks, secs, oks, fails = [], [], [], []
        for _ in range(repeats):
            body = {"model": model, "max_tokens": 2400,
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": h.TOOLS}
            if kwargs:
                body["chat_template_kwargs"] = kwargs
            body.update(sampler)
            t0 = time.time()
            try:
                d = h.post(base, body)
            except Exception as e:
                oks.append(False); fails.append(f"ERROR {e}"); continue
            secs.append(time.time() - t0)
            msg = d["choices"][0]["message"]
            calls = h.extract(msg)
            toks.append(d.get("usage", {}).get("completion_tokens", 0))
            ok, note = check(calls[0]) if calls else (
                False, "NO TOOL CALL: " + (msg.get("content") or "")[:70])
            oks.append(ok)
            if not ok:
                fails.append(note)
        rows.append({"prompt": label, "pass": sum(oks), "of": repeats,
                     "tokens": statistics.median(toks) if toks else 0,
                     "seconds": round(statistics.median(secs), 2) if secs else 0,
                     "fails": fails})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model")
    ap.add_argument("--sampler", required=True,
                    help="JSON of the sampler the unit serves, e.g. "
                         "'{\"temperature\":1.0,\"top_p\":0.95}'. Copy it from the "
                         ".service file rather than retyping it from memory.")
    ap.add_argument("--kwargs", default=None, help="JSON for chat_template_kwargs")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--json", help="append a result record to this file")
    a = ap.parse_args()

    sampler = json.loads(a.sampler)
    kwargs = json.loads(a.kwargs) if a.kwargs else None
    print(f"\n=== {a.model} @ {json.dumps(sampler)}"
          f"{' / ' + json.dumps(kwargs) if kwargs else ''}, {a.repeats} repeats ===")
    rows = run_arm(a.base, a.model, sampler, kwargs, a.repeats)

    tt = sum(r["tokens"] for r in rows); ts = sum(r["seconds"] for r in rows)
    p = sum(r["pass"] for r in rows); o = sum(r["of"] for r in rows)
    for r in rows:
        print(f"    {r['prompt']:20} {r['pass']}/{r['of']}  {r['tokens']:4.0f} tok "
              f" {r['seconds']:5.2f}s")
        for fl in r["fails"][:5]:
            print(f"        FAIL: {fl[:100]}")
    print(f"    {'TOTAL':20} {p}/{o}  {tt:4.0f} tok  {ts:5.2f}s")
    if p < o:
        print("    --> a failure rate here is a REAL failure rate; the greedy harness "
              "cannot see it")

    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"model": a.model, "sampler": sampler, "kwargs": kwargs,
                                "repeats": a.repeats, "pass": p, "of": o, "tokens": tt,
                                "seconds": round(ts, 2), "detail": rows}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
