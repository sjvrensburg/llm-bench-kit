#!/usr/bin/env python3
"""Reasoning-leak check: with thinking OFF, does chain-of-thought reach the visible reply?

  python3 leak-check.py http://127.0.0.1:8104/v1 ornith-1.5-122b
  python3 leak-check.py ... --repeats 2 --json results.json

WHY THIS EXISTS. `tool-harness.py` cannot see this failure: Mellum2.1 scored 9/9 on tools and
leaked `</think>` into 9 of 16 chat replies (2026-10-11 bake-off, docs/04). Every `:8092`
candidate gate since carries it: 8 prompts x 2 repeats, thinking off via
`chat_template_kwargs.enable_thinking=false`, count replies whose visible `content` contains a
`<think>` or `</think>` tag, or comes back empty. Pass is 0 leaks and 0 empties.

The sampler is a flag, not hard-wired: copy it from the unit being gated rather than retyping it
(shipped sampler != harness sampler, pitfall #25's lesson). Defaults to the 35B unit's.
Stdlib only, so it runs under the system python3.
"""
import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "What is the capital of Australia, and why isn't it Sydney?",
    "Explain what a mutex is in two sentences.",
    "Give me three names for a small coffee shop that sells only filter coffee.",
    "Convert 72 degrees Fahrenheit to Celsius and show the arithmetic.",
    "Is it safer to cross a road at a zebra crossing or a pelican crossing? Answer briefly.",
    "Write a haiku about load shedding.",
    "What does `git rebase --onto` do? One short paragraph.",
    "Summarise the plot of Romeo and Juliet in under 50 words.",
]


def ask(base, model, prompt, sampler, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        **sampler,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument(
        "--sampler",
        default='{"temperature":0.6,"top_k":20,"top_p":0.95}',
        help="JSON of the sampler the unit serves",
    )
    ap.add_argument("--json", help="append a result record to this file")
    a = ap.parse_args()
    sampler = json.loads(a.sampler)

    leaks = empties = total = 0
    for p in PROMPTS:
        for _ in range(a.repeats):
            total += 1
            text = ask(a.base, a.model, p, sampler, a.max_tokens)
            leaked = "</think>" in text or "<think>" in text
            leaks += leaked
            empties += not text.strip()
            if leaked or not text.strip():
                print(f"  {'LEAK ' if leaked else 'EMPTY'} {p[:50]!r} -> {text[:80]!r}")
    print(f"leak-check: {leaks}/{total} leaked, {empties}/{total} empty "
          f"({'PASS' if leaks == 0 and empties == 0 else 'FAIL'})")
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"check": "leak", "model": a.model, "total": total,
                                "leaks": leaks, "empties": empties, "sampler": sampler}) + "\n")
    return 0 if leaks == 0 and empties == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
