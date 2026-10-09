#!/usr/bin/env python3
# Was scripts/hermes-tool-harness.py in halo-prep (renamed 2026-10-09).
"""Score a candidate Hermes model on tokens-per-completed-turn, not t/s.

  python3 tool-harness.py http://127.0.0.1:8091/v1 laguna-xs-2.1
  python3 tool-harness.py http://127.0.0.1:8095/v1 ornith-1.5-9b --no-think
  python3 tool-harness.py ... --json results.json

WHY THIS FILE EXISTS. docs/04 has now run this comparison twice and BOTH times
recorded that "the original three prompts were not saved" -- so the second run
could not be compared against the first and had to re-run the incumbent too.
The prompts are the experiment. They live here now.

WHAT IT MEASURES, AND WHY NOT t/s. Laguna-XS was beaten on raw throughput by
LFM2.5-2.6B (2.73x prefill, 0.80x decode, a quarter of the memory) and still
won, because LFM2.5's template injects <think> every turn with no toggle and it
spent 2.3x the tokens reaching the same answer. The axis that decides this role
is therefore:

    tokens to COMPLETE the turn  /  decode t/s   =  seconds the user waits

llama-bench cannot see this. It measures the denominator only.

THE PROMPTS. Each carries TWO requirements, per pitfall #25: single-clause
prompts pass in every configuration and prove nothing. The failure mode being
hunted is not a malformed call -- it is a well-formed call whose ARGUMENTS
quietly dropped one of the two requirements. That is why `check` inspects
arguments rather than asserting a tool name. Hermes has terminal access, so a
plausible-but-wrong command is the expensive outcome, not a crash.

Toolset mirrors what Hermes actually exposes on `signal`
(platform_toolsets.signal in ~/.hermes/config.yaml: terminal, file, web).

CONDITIONS. temperature 0. The box MUST be quiet -- docs/04 discarded a whole
run taken at 81% GPU (Laguna measured 18.8-88.9 t/s across three prompts).
Verify with `rocm-smi --showuse` and check every resident slot is idle first.
Three repeats per prompt: pitfall #27 recorded a 515-vs-530 token "effect" on
single shots that vanished entirely on repeat -- that was cold-slot warm-up.
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "terminal",
        "description": "Run a shell command on this machine and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The exact command to run."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path to the file."},
            "max_lines": {"type": "integer", "description": "Read at most this many lines from the start."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"}},
            "required": ["query"]}}},
]


def chk_readfile(call):
    """Both requirements: the path AND the 40-line bound. Either tool may carry them.

    THIS ACCEPTS `terminal` ON PURPOSE, and the reason is a bug this check used
    to have. It asserted `read_file` by name, which contradicts the header above
    ("`check` inspects arguments rather than asserting a tool name") and scored
    a correct answer as a failure. Ornith-1.5-35B-A3B, 2026-08-25, temp 0:

        read_file  {"path": "/etc/fstab", "max_lines": 40}
        terminal   {"command": "head -n 40 /etc/fstab"}

    Both honour both requirements. The model alternated between them across
    repeats, so the old check reported 5/10 and a headline 25/30 for a model
    that never once dropped the bound. `head -n 40 /etc/fstab` is not a worse
    answer than `read_file` with `max_lines: 40`; it is the same answer through
    the other tool Hermes exposes. Scoring it wrong would have retired a
    candidate for the harness's mistake.

    What is NOT relaxed: the bound itself. `head /etc/fstab` (default 10 lines)
    and `cat /etc/fstab` still fail, which is the failure this file exists to
    catch. Only the choice of tool is free.
    """
    a = call["args"]
    if call["name"] == "read_file":
        if a.get("path") != "/etc/fstab":
            return False, f"path={a.get('path')!r}"
        if a.get("max_lines") != 40:
            return False, f"DROPPED max_lines (got {a.get('max_lines')!r})"
        return True, "read_file: path + max_lines both present"
    if call["name"] == "terminal":
        c = a.get("command", "")
        if "/etc/fstab" not in c:
            return False, f"wrong path: {c!r}"
        # head -n 40 / head -40 / sed -n '1,40p'. The bound is matched as
        # (?<!\d)40(?!\d) rather than \b40\b: sed's `1,40p` puts a word
        # character straight after the 40, so \b would reject the very form
        # this branch exists to accept, while `head -n 400` must still fail.
        # `tail` is absent on purpose -- the requirement is the FIRST 40 lines.
        if not re.search(r"\b(head|sed)\b[^|;]*(?<!\d)40(?!\d)", c):
            return False, f"DROPPED the 40-line bound: {c!r}"
        return True, f"terminal: {c}"
    return False, f"called {call['name']}, neither read_file nor terminal"


def chk_userunit(call):
    """Both requirements: the right unit AND --user. This is pitfall #25's exact case."""
    if call["name"] != "terminal":
        return False, f"called {call['name']}, not terminal"
    c = call["args"].get("command", "")
    if "llama-gemma26" not in c:
        return False, f"wrong unit: {c!r}"
    if "--user" not in c:
        return False, f"DROPPED --user (queries system units): {c!r}"
    return True, c


def chk_search(call):
    """Both requirements: a query AND the result bound."""
    if call["name"] != "web_search":
        return False, f"called {call['name']}, not web_search"
    a = call["args"]
    if not a.get("query"):
        return False, "empty query"
    if a.get("max_results") != 3:
        return False, f"DROPPED max_results (got {a.get('max_results')!r})"
    return True, f"{a['query']!r} / max_results=3"


# THE UNIT NAMED IN PROMPT 2 IS DELIBERATELY NOT UPDATED. `llama-gemma26` was
# retired 2026-08-25 (replaced by llama-ornith35 on the same port), so the prompt
# now asks about a unit that is disabled. That is fine and must stay: the unit
# file still exists, `systemctl --user status llama-gemma26` still answers, and
# the thing being tested is whether the model keeps `--user` -- not whether the
# service is running. Renaming it would silently break comparability with every
# result recorded against these prompts, which is the exact failure the header
# above says this file exists to prevent.
PROMPTS = [
    ("read_file+bound",
     "Show me the first 40 lines of /etc/fstab.",
     chk_readfile),
    ("terminal+userflag",
     "Is the llama-gemma26 user service running? It is a systemd user unit, not a system one.",
     chk_userunit),
    ("web_search+bound",
     "Search the web for the llama.cpp release notes for build b10582, and only give me the top 3 results.",
     chk_search),
]


def post(base, body, timeout=300):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def extract(msg):
    out = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            args = {"__unparseable__": raw}
        out.append({"name": fn.get("name"), "args": args})
    return out


def run(base, model, think, repeats, verbose, ctk_extra=None):
    # Warm the slot. Pitfall #27: the first request carries cold-cache noise that
    # reads exactly like a real effect.
    try:
        post(base, {"model": model, "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8, "temperature": 0})
    except Exception as e:
        print(f"  warm-up failed: {e}", file=sys.stderr)

    results = []
    for label, prompt, check in PROMPTS:
        toks, rtoks, secs, oks, notes = [], [], [], [], []
        for _ in range(repeats):
            body = {"model": model, "temperature": 0, "max_tokens": 900,
                    "messages": [{"role": "user", "content": prompt}], "tools": TOOLS}
            if think is False:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            elif think is True:
                body["chat_template_kwargs"] = {"enable_thinking": True}
            if ctk_extra:
                # some templates don't key off enable_thinking at all -- Nex-N2.5-mini
                # (GPT-OSS-style) uses reasoning_effort instead; enable_thinking:false
                # is silently a no-op on it. See memory nex-n25-mini-reasoning-effort-lever.
                body.setdefault("chat_template_kwargs", {}).update(ctk_extra)
            t0 = time.time()
            try:
                d = post(base, body)
            except Exception as e:
                notes.append(f"ERROR {e}")
                oks.append(False)
                continue
            dt = time.time() - t0
            msg = d["choices"][0]["message"]
            calls = extract(msg)
            u = d.get("usage", {})
            toks.append(u.get("completion_tokens", 0))
            rc = msg.get("reasoning_content") or ""
            rtoks.append(len(rc))
            secs.append(dt)
            if calls:
                ok, note = check(calls[0])
            else:
                ok, note = False, "NO TOOL CALL: " + (msg.get("content") or "")[:70]
            oks.append(ok)
            notes.append(note)
        med_t = statistics.median(toks) if toks else 0
        med_s = statistics.median(secs) if secs else 0
        results.append({"prompt": label, "pass": sum(oks), "of": repeats,
                        "tokens": med_t, "reasoning_chars": statistics.median(rtoks) if rtoks else 0,
                        "seconds": round(med_s, 2),
                        "tok_per_s": round(med_t / med_s, 1) if med_s else 0,
                        "notes": notes})
        if verbose:
            print(f"    {label:20} {sum(oks)}/{repeats}  {med_t:4.0f} tok  {med_s:5.2f}s  :: {notes[-1][:80]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-think", action="store_true", help="send enable_thinking:false")
    g.add_argument("--think", action="store_true", help="send enable_thinking:true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", help="append a result record to this file")
    ap.add_argument("--ctk", action="append", default=[], metavar="KEY=VALUE",
                     help="extra chat_template_kwargs entry, e.g. --ctk reasoning_effort=none "
                          "(Nex-N2.5-mini's actual lever; enable_thinking is a no-op on it)")
    a = ap.parse_args()
    think = False if a.no_think else (True if a.think else None)
    ctk_extra = dict(kv.split("=", 1) for kv in a.ctk)

    mode = "thinking OFF" if think is False else ("thinking ON" if think else "template default")
    if ctk_extra:
        mode += f", ctk={ctk_extra}"
    print(f"\n=== {a.model}  ({mode}, temp 0, {a.repeats} repeats) ===")
    res = run(a.base, a.model, think, a.repeats, True, ctk_extra)

    total_tok = sum(r["tokens"] for r in res)
    total_s = sum(r["seconds"] for r in res)
    passed = sum(r["pass"] for r in res)
    of = sum(r["of"] for r in res)
    print(f"    {'TOTAL':20} {passed}/{of}  {total_tok:4.0f} tok  {total_s:5.2f}s"
          f"  ({total_tok/total_s:.1f} tok/s effective)")
    print(f"    --> seconds-to-complete across the three turns: {total_s:.2f}s   <-- the number that decides it")

    if a.json:
        rec = {"model": a.model, "mode": mode, "repeats": a.repeats,
               "pass": passed, "of": of, "tokens": total_tok, "seconds": round(total_s, 2),
               "detail": res}
        with open(a.json, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
