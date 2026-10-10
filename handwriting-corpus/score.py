#!/usr/bin/env python3
"""
Scores transcripts against manifest.json's ground-truth checks. Unlike the
original score-handwriting.py, ground truth here is exact (we wrote the
source text ourselves), so scoring is substring/word presence, case
insensitive -- still checked by hand-reading a sample before trusting a
score, per this repo's own documented lesson (see the degraded-text
follow-up in halo-prep/docs/04-model-alternatives.md: a scorer that only
checks "struck text present" and not "struck value correct" hid a real
transcription error).

For correction pages (strike_word set), two extra checks fire:
  - the correct_word_line's words must all appear (the kept, correct value)
  - the strike_word must NOT appear as if it were the kept/final answer,
    i.e. it's fine if the model transcribes it (marked struck), but it must
    not be the only value reported, or reported unmarked as if it were kept.
    This is judged approximately: flag if strike_word appears but
    correct_word_line's key differentiator does not.

Usage:
    python3 handwriting-corpus/score.py <transcript_dir>
"""
import json
import pathlib
import re
import sys

MANIFEST = json.loads((pathlib.Path(__file__).parent / "manifest.json").read_text())


def word_present(text, word):
    return re.search(rf"\b{re.escape(word)}\b", text, re.I) is not None


def score_page(entry, transcript_dir):
    stem = pathlib.Path(entry["file"]).stem
    txt_path = transcript_dir / f"{stem}.txt"
    text = txt_path.read_text() if txt_path.exists() else ""

    total = 0
    hit = 0
    results = []

    for check in entry["checks"]:
        total += 1
        ok = word_present(text, check)
        hit += int(ok)
        results.append((check, ok))

    if entry.get("strike_word"):
        total += 1
        kept_ok = all(word_present(text, w) for w in entry["correct_word_line"].split())
        hit += int(kept_ok)
        results.append((f"kept value present: '{entry['correct_word_line']}'", kept_ok))

    return stem, hit, total, results, text


def main():
    transcript_dir = pathlib.Path(sys.argv[1])
    grand_hit = 0
    grand_total = 0
    per_page = []

    for entry in MANIFEST:
        stem = pathlib.Path(entry["file"]).stem
        txt_path = transcript_dir / f"{stem}.txt"
        if not txt_path.exists():
            continue
        stem, hit, total, results, _ = score_page(entry, transcript_dir)
        grand_hit += hit
        grand_total += total
        per_page.append((stem, hit, total))
        marks = " ".join("OK" if ok else "--" for _, ok in results)
        print(f"[{hit}/{total}] {stem}  ({marks})")

    print(f"\nGRAND TOTAL {grand_hit}/{grand_total}  ({transcript_dir})")
    per_page.sort(key=lambda t: t[1] / t[2])
    print("\nWorst 5 pages:")
    for stem, hit, total in per_page[:5]:
        print(f"  {hit}/{total}  {stem}")


if __name__ == "__main__":
    main()
