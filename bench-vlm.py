#!/usr/bin/env python3
"""Measure a llama-server VLM endpoint: flag A/Bs, and multi-image recall at depth.

Phase: none — this is a measurement tool, not a runbook step. It exists because
`llama-bench` cannot see either of the things we needed: the memory a flag saves,
and whether a model can still find an image it was shown 60K tokens ago.

Run (server must already be listening; this script never starts one):
    bench-vlm.py text   --port 8095 --depths 0 32768
    bench-vlm.py vision --port 8095 --images 6 --depth 60000

GTT is read from the kernel counter rather than rocm-smi: rocm-smi reports VRAM,
and on this box the models live in GTT (see pitfall #4 and the README budget).
"""
import argparse, base64, io, json, os, random, sys, time, urllib.request

# card1 is the Radeon 8060S on the EVO-X3; another box may number its GPU differently.
# A wrong path is not fatal: gtt_mib() returns -1 and the memory column reads -1.
GTT = os.environ.get("BENCH_GTT_COUNTER", "/sys/class/drm/card1/device/mem_info_gtt_used")
# --port picks the server; the host was always loopback. Env so a remote box can be measured
# without growing a flag on every subcommand (GTT is then this machine's, not the server's).
HOST = os.environ.get("BENCH_HOST", "127.0.0.1")


def gtt_mib():
    try:
        return int(open(GTT).read().strip()) // (1024 * 1024)
    except Exception:
        return -1


def post(port, path, payload, timeout=1800):
    req = urllib.request.Request(
        f"http://{HOST}:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ntok(port, text):
    return len(post(port, "/tokenize", {"content": text})["tokens"])


def filler(port, target):
    """Random-word filler trimmed to an exact token count by bisection.

    Random rather than repeated: repeated text compresses in the KV cache in ways
    that flatter prefill, which is the thing being measured.
    """
    if target <= 0:
        return ""
    rng = random.Random(1234)
    words = [
        "harbour", "ledger", "quartz", "meridian", "fossil", "tangent", "willow",
        "cobalt", "lantern", "drift", "pylon", "syntax", "marrow", "ember",
    ]
    text = " ".join(rng.choice(words) for _ in range(int(target * 1.6)))
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if ntok(port, text[:mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return text[:lo]


def timed_completion(port, prompt, npred, tag):
    t0 = time.time()
    r = post(port, "/completion", {
        "prompt": prompt, "n_predict": npred, "temperature": 0,
        "cache_prompt": False, "stream": False,
    })
    wall = time.time() - t0
    t = r["timings"]
    return {
        "tag": tag, "wall_s": round(wall, 2),
        "prompt_n": t["prompt_n"], "pp_tps": round(t["prompt_per_second"], 2),
        "pred_n": t["predicted_n"], "tg_tps": round(t["predicted_per_second"], 2),
        "gtt_mib": gtt_mib(),
    }


def make_image(code, panel, size=(448, 448)):
    """One image carrying exactly one fact, large and high-contrast.

    The point is to test retrieval, not OCR: if the model cannot read this it
    cannot read anything, so a miss is a context failure rather than a vision one.
    """
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    try:
        f_big = ImageFont.truetype("/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf", 46)
        f_small = ImageFont.truetype("/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf", 30)
    except Exception:
        f_big = f_small = ImageFont.load_default()
    d.rectangle([8, 8, size[0] - 8, size[1] - 8], outline="black", width=5)
    # Each line is measured and shrunk to fit rather than trusted: an overflowing
    # code renders as a truncated one, and the model then "misreads" text that was
    # never fully drawn -- a harness bug that looks exactly like a vision failure.
    d.text((30, 120), f"PANEL {panel}", fill="black", font=f_small)
    d.text((30, 190), "CODE", fill="black", font=f_small)
    while d.textlength(code, font=f_big) > size[0] - 60 and f_big.size > 12:
        f_big = f_big.font_variant(size=f_big.size - 2)
    d.text((30, 240), code, fill="black", font=f_big)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def chat_sampled(port, messages, npred, sampler, timeout=1800):
    """Same as chat() but at the unit's production sampler rather than greedy.

    Muse's unit serves the model card's temp 1.0 / top_p 0.95 / top_k 64. Token
    counts under a forced temp 0 are not the counts this endpoint actually
    produces, so anything measuring *how much it writes* has to run hot and
    repeat.
    """
    t0 = time.time()
    body = {"messages": messages, "max_tokens": npred, "stream": False}
    body.update(sampler)
    r = post(port, "/v1/chat/completions", body, timeout=timeout)
    wall = time.time() - t0
    m = r["choices"][0]["message"]
    return (m.get("content") or "", m.get("reasoning_content") or "",
            wall, r.get("usage", {}))


def cmd_reasoning(a):
    """A/B Muse's "Reasoning strength" system line at the production sampler.

    The model takes `Reasoning strength: low|medium|high|xhigh` in the SYSTEM
    prompt -- system, not user; as user text it does nothing. Default is verbose,
    so this is a tokens-generated lever rather than a tokens-per-second one, and
    wall-clock is what it moves.
    """
    sampler = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    img = make_image("MAROON-6305", 7)
    vis = [{"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + img}},
           {"type": "text", "text": "What is the CODE on this panel?"}]
    probes = [
        ("vision-code",  vis,                                          "maroon-6305"),
        ("vision-desc",  [vis[0], {"type": "text",
                                   "text": "Describe this image in one sentence."}], None),
        ("text-fact",    "What is the capital of Australia? One word.", "canberra"),
        ("text-short",   "In one sentence, what does a GPU do?",        None),
    ]
    print(f"# sampler {sampler}, {a.repeats} repeats/arm")
    for tag, content, want in probes:
        for arm in ("default", "low"):
            walls, gens, oks = [], [], []
            for _ in range(a.repeats):
                msgs = []
                if arm == "low":
                    msgs.append({"role": "system",
                                 "content": "Reasoning strength: low"})
                msgs.append({"role": "user", "content": content})
                out, reason, wall, usage = chat_sampled(a.port, msgs, a.npred, sampler)
                walls.append(wall)
                gens.append(usage.get("completion_tokens", 0))
                if want:
                    oks.append(want in (out + reason).lower())
            walls.sort(); gens.sort()
            mid = len(walls) // 2
            print(json.dumps({
                "probe": tag, "arm": arm,
                "wall_s_med": round(walls[mid], 2),
                "gen_tok_med": gens[mid],
                "gen_tok_all": gens,
                "correct": f"{sum(oks)}/{len(oks)}" if want else "n/a",
            }))


def chat(port, messages, npred, timeout=1800):
    """Returns (content, reasoning, wall, usage).

    Both fields are returned because these are thinking models: with no
    `--reasoning-format` set, llama-server leaves `content` empty until the model
    closes its think block, so a budget that is merely generous for an answer can
    be far too small for one. Scoring only `content` reports 0/N for a model that
    read every image correctly -- which is exactly what the first run of this
    harness did.
    """
    t0 = time.time()
    r = post(port, "/v1/chat/completions", {
        "messages": messages, "max_tokens": npred, "temperature": 0, "stream": False,
    }, timeout=timeout)
    wall = time.time() - t0
    m = r["choices"][0]["message"]
    return (m.get("content") or "", m.get("reasoning_content") or "",
            wall, r.get("usage", {}))


def cmd_text(a):
    print(f"# idle GTT {gtt_mib()} MiB")
    for d in a.depths:
        pre = filler(a.port, d) if d else ""
        prompt = (pre + "\n\nContinue this text.") if pre else "Write a paragraph about tides."
        print(json.dumps(timed_completion(a.port, prompt, a.npred, f"d{d}")))


def cmd_vision(a):
    rng = random.Random(7)
    words = ["MAROON", "CYPRESS", "OBSIDIAN", "SAFFRON", "TUNDRA", "VELLUM",
             "ZEPHYR", "INDIGO", "GRANITE", "AMBER"]
    codes = [f"{words[i]}-{rng.randint(1000, 9999)}" for i in range(a.images)]

    content = [{"type": "text", "text":
                "You will be shown numbered panels. Remember the code on each."}]
    for i, c in enumerate(codes, 1):
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + make_image(c, i)}})
        content.append({"type": "text", "text": f"(that was panel {i})"})

    # Filler AFTER the images: it pushes them back in the window, which is the
    # thing under test. Filler before them would only test a long prefix.
    if a.depth:
        pad = filler(a.port, a.depth)
        content.append({"type": "text", "text":
                        "Unrelated notes follow; ignore them.\n" + pad})

    print(f"# {a.images} panels, filler target {a.depth} tok, idle GTT {gtt_mib()} MiB")
    base = [{"role": "user", "content": content}]

    hits = 0
    for i, want in enumerate(codes, 1):
        q = base + [{"role": "assistant", "content": "Understood."},
                    {"role": "user", "content":
                     f"What is the CODE on panel {i}? Reply with the code only."}]
        out, reason, wall, usage = chat(a.port, q, a.npred)
        ok = want.lower() in out.lower() or want.lower() in reason.lower()
        hits += ok
        print(json.dumps({"panel": i, "want": want,
                          "got": (out.strip() or reason.strip()[-60:])[:60],
                          "hit": ok, "in_content": want.lower() in out.lower(),
                          "wall_s": round(wall, 1),
                          "prompt_tok": usage.get("prompt_tokens"),
                          "out_tok": usage.get("completion_tokens"),
                          "gtt_mib": gtt_mib()}))
    print(json.dumps({"summary": f"{hits}/{len(codes)}", "images": a.images,
                      "depth": a.depth}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text"); t.set_defaults(fn=cmd_text)
    t.add_argument("--port", type=int, required=True)
    t.add_argument("--depths", type=int, nargs="+", default=[0])
    t.add_argument("--npred", type=int, default=128)
    v = sub.add_parser("vision"); v.set_defaults(fn=cmd_vision)
    v.add_argument("--port", type=int, required=True)
    v.add_argument("--images", type=int, default=6)
    v.add_argument("--depth", type=int, default=0)
    v.add_argument("--npred", type=int, default=768)
    rr = sub.add_parser("reasoning"); rr.set_defaults(fn=cmd_reasoning)
    rr.add_argument("--port", type=int, required=True)
    rr.add_argument("--npred", type=int, default=1024)
    rr.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    a.fn(a)
