#!/usr/bin/env python3
"""Run tool-harness.py's prompts through `pi -p`, not a bare request.

  python3 pi-tool-harness.py local-ornith/ornith-1.5-9b
  python3 pi-tool-harness.py local-ornith/ornith-1.5-9b --temp shipped --repeats 5
  python3 pi-tool-harness.py ... --thinking medium --json results.json

WHY THIS FILE EXISTS. docs/13 Phase 0's gate is "does Ornith-9B keep 15/15 on
the tool harness through pi?". The Hermes harness cannot answer that: it POSTs a
bare user message with its own three tools and no system prompt, so it measures
the model, never the harness around it. Here the model sees what it will see in
production -- pi's system prompt and pi's tools (`read`, `bash`, `edit`,
`write`) -- plus a `web_search` stub (ext/, same schema as the Hermes harness's)
because pi has no web tool until pi-web-access is installed.

THE PROMPTS AND CHECKS ARE IMPORTED, NOT COPIED. tool-harness.py's header
says the prompts are the experiment and a lost copy made two runs incomparable.
A second copy here would drift the same way. pi's tool names are mapped onto the
names the checks expect: bash -> terminal (same `command` arg), read -> read_file
(`limit` -> `max_lines`). The check still decides; only the spelling is adapted.

THE RUN IS KILLED AT THE FIRST TOOL CALL. The checks only ever looked at the
first call, and pi executes tools for real -- stopping at `toolcall_end` means
nothing past the first call runs and each repeat costs one model turn, not a
whole agent loop.

TEMPERATURE. `--temp 0` (default) matches the Hermes harness and is the
comparable number. `--temp shipped` sends none, so the unit's own `--temp`
applies -- that is what a pi-job will actually get, and memory
shipped-sampler-not-harness-sampler records how far apart those two can be on
this exact model. Report both. temp 0 is set through `samplingParams` in a
throwaway copy of ~/.pi/agent/models.json (PI_CODING_AGENT_DIR), so the live
config is never touched.

CONDITIONS as tool-harness.py: quiet GPU, every resident slot idle.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hth", HERE / "tool-harness.py")  # hermes-tool-harness.py until 2026-10-09
hth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hth)

STUBS = {"hermes": HERE / "pi-harness-ext" / "web-search-stub.ts",
         "pi-web-access": HERE / "pi-harness-ext" / "web-search-piwebaccess.ts",
         "hermes-real": HERE / "pi-harness-ext" / "web-search-hermesreal.ts"}
# The models.json the throwaway agent dir is copied from. Env-overridable so the harness
# runs against a pi install that is not ~/.pi/agent; the default is the live one.
LIVE_MODELS = pathlib.Path(os.environ.get("PI_HARNESS_MODELS_JSON",
                                          pathlib.Path.home() / ".pi" / "agent" / "models.json"))


def map_call(name, args):
    if name == "bash":
        return {"name": "terminal", "args": {"command": args.get("command", "")}}
    if name == "read":
        a = {"path": args.get("path")}
        if "limit" in args:
            a["max_lines"] = args["limit"]
        if args.get("offset") not in (None, 0, 1):
            a["offset"] = args["offset"]  # a non-zero offset is not "the first 40"
            a["max_lines"] = None
        return {"name": "read_file", "args": a}
    if name == "web_search" and "numResults" in args:
        # pi-web-access spells the bound numResults; `queries` counts as a query.
        a = {"query": args.get("query") or " | ".join(args.get("queries") or []),
             "max_results": args["numResults"]}
        return {"name": name, "args": a}
    if name == "web_search" and "limit" in args:
        return {"name": name, "args": {"query": args.get("query"), "max_results": args["limit"]}}
    if name == "web_search" and not args.get("query") and args.get("queries"):
        return {"name": name, "args": {**args, "query": " | ".join(args["queries"])}}
    return {"name": name, "args": args}


def agent_dir(provider_model, temp):
    d = pathlib.Path(tempfile.mkdtemp(prefix="pi-harness-"))
    cfg = json.loads(LIVE_MODELS.read_text())
    if temp != "shipped":
        prov, mid = provider_model.split("/", 1)
        for m in cfg["providers"][prov]["models"]:
            if m["id"] == mid:
                m["samplingParams"] = {"temperature": float(temp)}
    (d / "models.json").write_text(json.dumps(cfg))
    return d


def one(model, prompt, thinking, env, cwd, stub, exts=()):
    cmd = ["pi", "--offline", "--mode", "json", "--no-session", "-ne", "-ns", "-np", "-nc",
           "-e", str(stub)]
    for e in exts:
        cmd += ["-e", e]
    cmd += ["--model", model, "--thinking", thinking, prompt]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, env=env, cwd=cwd)
    call, out_toks, text = None, 0, ""
    try:
        for line in p.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            ame = ev.get("assistantMessageEvent") or {}
            if ev.get("usage"):
                out_toks = ev["usage"].get("output", out_toks)
            if ame.get("type") == "toolcall_end":
                tc = ame["toolCall"]
                call = map_call(tc["name"], tc.get("arguments") or {})
                break
            if ev.get("type") == "message_end" and ev["message"].get("role") == "assistant":
                text = "".join(c.get("text", "") for c in ev["message"]["content"]
                               if c.get("type") == "text")
                if ev["message"].get("stopReason") != "toolUse":
                    break
    finally:
        p.kill()
        p.wait()
    return call, out_toks, time.time() - t0, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="provider/id as in ~/.pi/agent/models.json")
    ap.add_argument("--temp", default="0", help="a number, or 'shipped' for the unit's own")
    ap.add_argument("--thinking", default="off")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--stub", choices=sorted(STUBS), default="hermes",
                    help="web_search schema: the Hermes harness's, or pi-web-access's real one")
    ap.add_argument("-e", "--extension", action="append", default=[],
                    help="extra pi extension to load, e.g. config/pi/extensions/schema-key-order.ts")
    ap.add_argument("--json")
    a = ap.parse_args()

    d = agent_dir(a.model, a.temp)
    env = dict(os.environ, PI_CODING_AGENT_DIR=str(d))
    cwd = tempfile.mkdtemp(prefix="pi-harness-cwd-")  # no AGENTS.md/CLAUDE.md to pick up
    print(f"\n=== pi / {a.model}  (thinking {a.thinking}, temp {a.temp}, {a.stub} stub, {a.repeats} repeats{", +" + ",".join(os.path.basename(e) for e in a.extension) if a.extension else ""}) ===")
    stub = STUBS[a.stub]
    one(a.model, "hi", a.thinking, env, cwd, stub, a.extension)  # warm the slot, pitfall #27

    res = []
    for label, prompt, check in hth.PROMPTS:
        oks, toks, secs, notes = [], [], [], []
        for _ in range(a.repeats):
            call, n, dt, text = one(a.model, prompt, a.thinking, env, cwd, stub, a.extension)
            ok, note = check(call) if call else (False, "NO TOOL CALL: " + text[:70])
            oks.append(ok); toks.append(n); secs.append(dt); notes.append(note)
        med_t, med_s = statistics.median(toks), statistics.median(secs)
        res.append({"prompt": label, "pass": sum(oks), "of": a.repeats, "tokens": med_t,
                    "seconds": round(med_s, 2), "notes": notes})
        print(f"    {label:20} {sum(oks)}/{a.repeats}  {med_t:4.0f} tok  {med_s:5.2f}s")
        for n_ in notes:
            print(f"        {n_[:100]}")
    passed, of = sum(r["pass"] for r in res), sum(r["of"] for r in res)
    total_s = sum(r["seconds"] for r in res)
    print(f"    {'TOTAL':20} {passed}/{of}  {total_s:5.2f}s (median seconds, summed)")
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"harness": "pi", "model": a.model, "thinking": a.thinking,
                                "temp": a.temp, "stub": a.stub, "extensions": a.extension, "pass": passed, "of": of,
                                "seconds": round(total_s, 2), "detail": res}) + "\n")
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(cwd, ignore_errors=True)


if __name__ == "__main__":
    main()
