#!/usr/bin/env python3
"""Re-runs transcribe.ask() for a filtered subset (by filename substring)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from transcribe import ask, DEGRADED_DIR

if __name__ == "__main__":
    port = sys.argv[1]
    outdir = pathlib.Path(sys.argv[2])
    substrings = sys.argv[3:]
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in DEGRADED_DIR.glob("*.jpg")
                    if any(s in f.name for s in substrings))
    for i, jpg in enumerate(files, 1):
        text, dt, finish_reason = ask(port, jpg)
        (outdir / f"{jpg.stem}.txt").write_text(text)
        print(f"[{i}/{len(files)}] {jpg.stem}: {dt:.1f}s finish_reason={finish_reason}")
