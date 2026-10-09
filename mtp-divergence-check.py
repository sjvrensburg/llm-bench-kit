#!/usr/bin/env python3
"""Check whether --spec-type draft-mtp changes output at temperature 0.

  python3 mtp-divergence-check.py http://127.0.0.1:8091/v1 cybertiel-coder-35b-a3b

WHY THIS EXISTS. Speculative decoding is supposed to be lossless: at temp 0 the
sampled token must be byte-identical to running without it. llama-ornith.service's
"MTP IS DELIBERATELY OFF" block found it is NOT, on this box, on this llama.cpp
build, for Ornith-1.5-9B and (more subtly) for Gemma-4-26B-A4B -- see pitfall #28.
That check was done ad hoc at the time, against two separately-configured server
processes, and not saved as a reusable script. This is that script, generalised:
point it at ONE already-running server twice, once before you add
--spec-type draft-mtp/--model-draft to its ExecStart and once after (restart
between), or at two different ports if you'd rather run both at once. Either way
it is the OPERATOR'S job to control which arm is live; this script only compares
whatever answers at the URL each time it is invoked with --tag.

METHOD, unchanged from pitfall #28: temperature 0, max_tokens capped so a runaway
completion doesn't hide a real divergence in noise, six prompts spanning short
factual, arithmetic and one open-ended completion (the kind of prompt where a
target's logits are most likely to move under a drafter). A run's tokens/response
are saved to --out (JSON); a second run with the same --out and a different --tag
diffs against the first tag found there instead of re-asking.

THIS IS DIVERGENCE DETECTION, NOT A VERDICT. Divergence found does not by itself
mean draft-mtp is unsafe for THIS model -- Gemma-4-26B-A4B diverged too and stayed
in production because the divergence was judged acceptable for a prose role.
Whether a code-and-tool-calling role should accept it is the same judgement call
llama-ornith.service's history already made once; read that unit's comment block
before deciding, don't just look at the pass/fail count here.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.util import module_from_spec, spec_from_file_location

_HTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool-harness.py")  # hermes-tool-harness.py until 2026-10-09
_spec = spec_from_file_location("hermes_tool_harness", _HTH)
h = module_from_spec(_spec)
_spec.loader.exec_module(h)

PROMPTS = [
    "What is 17*23? Answer with just the number.",
    "What is the capital of France? Answer with just the city name.",
    "Complete this sentence with one word: The sky is",
    "Write a haiku about a rainy afternoon in a small coastal town.",
    "List the first five prime numbers, comma-separated, nothing else.",
    "In one sentence, explain what a mutex is for.",
]


def run(base, model, max_tokens):
    out = []
    for p in PROMPTS:
        resp = h.post(base, {
            "model": model,
            "messages": [{"role": "user", "content": p}],
            "temperature": 0,
            "max_tokens": max_tokens,
        })
        text = resp["choices"][0]["message"].get("content", "")
        out.append({"prompt": p, "text": text})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="e.g. http://127.0.0.1:8091/v1")
    ap.add_argument("model")
    ap.add_argument("--tag", required=True, help="label for this run, e.g. no-mtp / mtp-n3")
    ap.add_argument("--out", default="mtp-divergence.json")
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()

    # Warm the slot -- pitfall #27, the first request carries cold-cache noise.
    h.post(a.base, {"model": a.model, "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0, "max_tokens": 4})

    result = run(a.base, a.model, a.max_tokens)
    store = {}
    if os.path.exists(a.out):
        store = json.load(open(a.out))
    store[a.tag] = result
    json.dump(store, open(a.out, "w"), indent=2)

    tags = list(store)
    print(f"Saved run '{a.tag}' ({len(tags)} tag(s) in {a.out}: {tags})")
    if len(tags) < 2:
        print("Run again with a second --tag (the other arm) to compare.")
        return

    a_tag, b_tag = tags[-2], tags[-1]
    identical = 0
    for i, (ra, rb) in enumerate(zip(store[a_tag], store[b_tag])):
        same = ra["text"] == rb["text"]
        identical += same
        status = "SAME" if same else "DIFFERS"
        print(f"  [{i}] {status}: {ra['prompt'][:50]!r}")
        if not same:
            print(f"        {a_tag}: {ra['text'][:100]!r}")
            print(f"        {b_tag}: {rb['text'][:100]!r}")
    print(f"\n{identical}/{len(PROMPTS)} identical between '{a_tag}' and '{b_tag}'")


if __name__ == "__main__":
    main()
