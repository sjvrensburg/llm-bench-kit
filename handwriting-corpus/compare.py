#!/usr/bin/env python3
"""
Cross-model comparison: scores all three transcript sets, flags any page
where every model scored 0 (or near it) as a likely corpus-generation
failure (illegible synthetic handwriting) rather than a genuine VLM miss,
and reports both the raw and the "clean" (failures excluded) totals.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from score import MANIFEST, score_page

HERE = pathlib.Path(__file__).parent

MODELS = {
    "nex-n25-mini (no-think)": HERE / "transcripts-nex",
    "muse-glimmer": HERE / "transcripts-muse",
    "olmocr-2": HERE / "transcripts-olmocr",
}


def main():
    per_page = {}
    totals = {name: [0, 0] for name in MODELS}

    for entry in MANIFEST:
        stem = pathlib.Path(entry["file"]).stem
        scores = {}
        for name, d in MODELS.items():
            if (d / f"{stem}.txt").exists():
                _, hit, total, _, _ = score_page(entry, d)
                scores[name] = (hit, total)
                totals[name][0] += hit
                totals[name][1] += total
        per_page[stem] = scores

    print("=== Raw totals ===")
    for name, (hit, total) in totals.items():
        print(f"  {name}: {hit}/{total} ({100*hit/total:.1f}%)")

    # a page all three models scored 0 on is a corpus-generation failure
    # (illegible synthetic handwriting), not a genuine transcription miss
    broken_pages = [stem for stem, scores in per_page.items()
                     if all(h == 0 for h, t in scores.values())]

    print(f"\n=== {len(broken_pages)} pages ALL THREE models scored 0/N on"
          f" (likely corpus generation failures, not VLM failures) ===")
    for stem in sorted(broken_pages):
        print(f"  {stem}")

    print("\n=== Clean totals (broken pages excluded) ===")
    clean_totals = {name: [0, 0] for name in MODELS}
    for stem, scores in per_page.items():
        if stem in broken_pages:
            continue
        for name, (hit, total) in scores.items():
            clean_totals[name][0] += hit
            clean_totals[name][1] += total
    for name, (hit, total) in clean_totals.items():
        print(f"  {name}: {hit}/{total} ({100*hit/total:.1f}%)")


if __name__ == "__main__":
    main()
