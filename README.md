# llm-bench-kit

Harnesses that measure a model behind any OpenAI-compatible (mostly llama-server) endpoint.
Standard library Python unless noted; none of them starts a server. Run them on an **idle
GPU**: a concurrent request on the same box contaminates every number.

Split out of halo-prep, the author's private runbook for an AMD Strix Halo box, on
2026-10-09. The doc references below are to that repo, where the results are recorded;
they are kept as provenance and are not public.

| Harness | Measures | Run | Results in halo-prep |
|---|---|---|---|
| `tool-harness.py` | Tool calling: 5 prompts × N repeats at temp 0, scored on correct calls and tokens/seconds per completed turn, not t/s | `python3 tool-harness.py <base>/v1 <model> [--think\|--no-think] [--json f]` | `docs/04` (every model assessment), pitfall #30 |
| `sampler-check.py` | The same prompts at the sampler the unit actually serves, not temp 0 (imports `tool-harness.py`) | `python3 sampler-check.py <base>/v1 <model> --sampler '<json>' [--kwargs '<json>']` | pitfall #31, `config/llama-ornith.service` |
| `pi-tool-harness.py` | The same prompts through `pi -p`, so the agent's system prompt and tool schemas are in the loop. Needs `pi`; fixtures in `pi-harness-ext/` | `python3 pi-tool-harness.py <provider>/<model> [--temp N\|shipped] [--stub ...]` | `docs/13` Phase 0 |
| `mtp-divergence-check.py` | Whether speculative decoding changes greedy output (it should not) | `python3 mtp-divergence-check.py <base>/v1 <model> --tag no-mtp`, again with `--tag mtp-n3`, then compare | pitfall #28 |
| `slot-contention.py` | Head-of-line blocking: how long a short request waits behind a long one, per `--parallel` setting | `python3 slot-contention.py <base>/v1 <model> --label np1` | `config/llama-ornith.service` (two slots, 2026-09-21), `docs/00-backlog.md` |
| `bench-parallel-depth.py` | Whether `--parallel > 1` helps at 32K–130K depth (four fixed arms, ~10 min) | `BENCH_URL=http://127.0.0.1:8088 python3 bench-parallel-depth.py` | pitfall #34 |
| `code-eval.py` | Code generation: tasks with hidden tests run in a subprocess; `--selftest` checks the reference solutions offline | `python3 code-eval.py <base>/v1 <model> [--think\|--no-think]` | `docs/04` "Flash-Next GSQ-RCO Coder" |
| `bench-vlm.py` | VLM flag A/Bs (prefill/decode/GTT at depth) and multi-image recall at depth. Needs Pillow | `python3 bench-vlm.py text --port 8095 --depths 0 32768` / `vision --port 8095 --images 6 --depth 60000` | `docs/04` vision sections, pitfall #36 |
| `prose-check.py` | Open-ended prose quality and whether a plain "be concise" system prompt shortens answers (5 prompts, each with and without it) | `python3 prose-check.py <port> [--label NAME]` | `docs/04` "Prose quality and terseness" |
| `handwriting-corpus/` | VLM transcription of 112 synthetic handwritten pages (14 texts × 4 checkpoints × biases, then degraded), scored on ground-truth tokens; transcripts from each model run are kept | `python3 handwriting-corpus/transcribe.py <port> <out_dir>`, then `python3 handwriting-corpus/score.py <out_dir>` | `docs/04` "Handwriting/degraded-text corpus" |

## Environment overrides

Defaults are the values these were written against on the EVO-X3, so a run there is unchanged.

| Variable | Used by | Default |
|---|---|---|
| `BENCH_URL` | `bench-parallel-depth.py` | `http://127.0.0.1:8088` |
| `BENCH_OUT` | `bench-parallel-depth.py` | `depth_bench.json` (cwd) |
| `BENCH_HOST` | `bench-vlm.py` | `127.0.0.1` |
| `BENCH_GTT_COUNTER` | `bench-vlm.py` | `/sys/class/drm/card1/device/mem_info_gtt_used` (-1 if absent) |
| `PI_HARNESS_MODELS_JSON` | `pi-tool-harness.py` | `~/.pi/agent/models.json` |

## Renamed files

`tool-harness.py`, `sampler-check.py` and `slot-contention.py` were `scripts/hermes-*.py` in
halo-prep. They were named after the agent they first scored (Hermes, retired 2026-09-24),
not because they contain Hermes code. halo-prep's docs still use the old names. The
`--stub hermes` value of `pi-tool-harness.py` keeps its name: it is a recorded condition.

## Handwriting corpus

`handwriting-corpus/` came from handwriting-bench, a local clone of
[pytorch-handwriting-synthesis-toolkit](https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit)
(MIT), on 2026-10-10. The rendered pages (`clean/`, `degraded/`) are committed, so transcribing and scoring need
nothing else. Regenerating them (`gen_corpus.py`, `render_strike.py`) needs a checkout of that toolkit with its
pretrained `checkpoints/`, with `HANDWRITING_SYNTH_REPO` pointing at it. `prose-check.py` was
`prose_check.py` in the same clone.
