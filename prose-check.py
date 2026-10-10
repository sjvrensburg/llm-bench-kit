#!/usr/bin/env python3
"""
Small, ad hoc prose-quality + terseness-lever check for a text model over the
OpenAI chat completions API. Not a formal harness -- there isn't one in
halo-prep for open-ended prose, only tool-calling (hermes-tool-harness.py)
and raw speed (llama-bench). Sends each prompt twice: once with the model's
default system behavior, once with a plain "be concise" system instruction
(a prompt-level lever, not a template splice -- see halo-prep's Granite
pitfall #30 for why a borrowed jinja template is not attempted here).

Usage:
    python3 prose_check.py <port> [--ctk KEY=VALUE ...] [--label NAME]
"""
import argparse, json, time, urllib.request

PROMPTS = [
    ("explain", "Explain what a Kalman filter is and when you'd actually reach for one."),
    ("summarize", "Summarize this in two sentences: The 1986 Chernobyl disaster resulted from "
     "a flawed reactor design combined with inadequately trained personnel. During a test of "
     "the reactor's ability to cool itself during a power failure, operators disabled key "
     "safety systems, and a sudden power surge led to explosions that destroyed the reactor "
     "core, releasing large amounts of radioactive material into the atmosphere."),
    ("opinion", "What caused the fall of the Western Roman Empire? Give the main factors."),
    ("casual", "What's a good weekend project for someone learning Rust?"),
    ("constrained", "In exactly 3 bullet points, explain why cache invalidation is hard."),
]

TERSE_SYSTEM = "Be concise. Answer in as few words as possible without losing correctness."


def ask(port, prompt, ctk, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": "local", "temperature": 0, "max_tokens": 700, "messages": messages}
    if ctk:
        body["chat_template_kwargs"] = ctk
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    dt = time.time() - t0
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), dt, d["choices"][0].get("finish_reason", "?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--ctk", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    port = a.port
    ctk = dict(kv.split("=", 1) for kv in a.ctk)
    label = a.label or port

    for tag, prompt in PROMPTS:
        for variant, system in [("default", None), ("terse-system", TERSE_SYSTEM)]:
            text, dt, fr = ask(port, prompt, ctk, system)
            words = len(text.split())
            print(f"\n=== {label} :: {tag} :: {variant} ({dt:.1f}s, {words}w, fr={fr}) ===")
            print(text)
