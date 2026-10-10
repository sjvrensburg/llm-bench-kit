#!/usr/bin/env python3
"""
Sends every degraded page to a given llama-server (OpenAI-compatible) port
for transcription. Same prompt/methodology as the earlier synthetic-font
proxy (/tmp/degraded-test/transcribe.py) and bench-handwriting.py's operator
prompt in halo-prep, so results are comparable across all three corpora.

Usage:
    python3 handwriting-corpus/transcribe.py <port> <out_dir> [--no-think]

--no-think sends chat_template_kwargs {"reasoning_effort": "none"}, Nex-N2.5-mini's
native lever (GPT-OSS-style, found in its GGUF's jinja template) to force an
empty <think></think> block. Unlike halo-prep's Ornith terse-splice (a
template swap that wrecked Granite 15/15 -> 1/15, CLAUDE.md pitfall #30), this
is the model's *own* documented mechanism, not a borrowed one -- but it still
only gets used on Nex, and only after being verified against Nex's own template.
"""
import base64
import json
import pathlib
import sys
import time
import urllib.request

PROMPT = (
    "Transcribe the text in the attached image of a page, exactly as written. "
    "Provide only the word for word transcription and do not correct, normalise, "
    "or infer meaning. If any text is struck through, transcribe it anyway and "
    "mark it clearly as struck through."
)

DEGRADED_DIR = pathlib.Path(__file__).parent / "degraded"


def ask(port, jpeg_path, max_tokens=4096, no_think=False):
    """
    max_tokens=4096, not the original 800: these are thinking models and with
    no --reasoning-format set, llama-server leaves `content` empty until the
    thinking block finishes (see scripts/bench-vlm.py's chat() docstring in
    halo-prep). At 800 both Nex and Muse burned the whole budget on
    reasoning_content and returned empty content on a large fraction of pages
    -- a harness bug, not a transcription failure. If content is still empty
    at 4096, we fall back to reasoning_content with a marker so nothing is
    silently blank, and finish_reason is returned so truncation is visible.
    """
    jpeg = pathlib.Path(jpeg_path).read_bytes()
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    body = {
        "model": "local",
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    if no_think:
        body["chat_template_kwargs"] = {"reasoning_effort": "none"}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    dt = time.time() - t0
    m = d["choices"][0]["message"]
    finish_reason = d["choices"][0].get("finish_reason", "?")
    content = m.get("content") or ""
    if not content and m.get("reasoning_content"):
        content = "[TRUNCATED-DURING-THINKING, showing reasoning_content]\n" + m["reasoning_content"]
    return content, dt, finish_reason


if __name__ == "__main__":
    port = sys.argv[1]
    outdir = pathlib.Path(sys.argv[2])
    no_think = "--no-think" in sys.argv[3:]
    outdir.mkdir(parents=True, exist_ok=True)
    total_t = 0.0
    files = sorted(DEGRADED_DIR.glob("*.jpg"))
    for i, jpg in enumerate(files, 1):
        text, dt, finish_reason = ask(port, jpg, no_think=no_think)
        total_t += dt
        (outdir / f"{jpg.stem}.txt").write_text(text)
        flag = "" if finish_reason == "stop" else f"  [finish_reason={finish_reason}]"
        print(f"[{i}/{len(files)}] {jpg.stem}: {dt:.1f}s, {len(text)} chars{flag}")
    print(f"\nTOTAL {len(files)} pages, {total_t:.1f}s, {total_t/len(files):.1f}s/page avg")
