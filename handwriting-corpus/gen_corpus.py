#!/usr/bin/env python3
"""
Renders every entry in pages.py across several (checkpoint, bias) combos to
build a synthetic handwriting-style corpus larger than the 8 photographed
exam pages this project no longer has copies of. One PNG per (page, style).

Run with the toolkit venv and HANDWRITING_SYNTH_REPO set:
    ~/venvs/handwriting-synth/bin/python handwriting-corpus/gen_corpus.py
"""
import json
import os
import sys

import torch

# The synthesis model lives in X-rayLaser/pytorch-handwriting-synthesis-toolkit (built against
# 899e943); point HANDWRITING_SYNTH_REPO at a checkout with its checkpoints/.
REPO = os.environ.get("HANDWRITING_SYNTH_REPO") or sys.exit(
    "set HANDWRITING_SYNTH_REPO to a pytorch-handwriting-synthesis-toolkit checkout")
sys.path.insert(0, REPO)

from handwriting_synthesis import utils
from handwriting_synthesis.utils import TooLargeImageError
from handwriting_synthesis.sampling import HandwritingSynthesizer
from pages import PAGES

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean")

# (checkpoint dir name, bias) -- 4 checkpoints x 3 biases = 12 distinct
# handwriting "styles" ranging from messy/varied (low bias) to clean (high
# bias), each checkpoint being a different training epoch of the same net.
STYLES = [
    ("Epoch_46", 0.3),
    ("Epoch_46", 1.2),
    ("Epoch_50", 0.5),
    ("Epoch_50", 1.5),
    ("Epoch_52", 0.3),
    ("Epoch_52", 1.0),
    ("Epoch_56", 0.5),
    ("Epoch_56", 1.5),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    manifest = []

    for ckpt, bias in STYLES:
        model_path = os.path.join(REPO, "checkpoints", ckpt)
        print(f"=== loading {ckpt} bias={bias} ===")
        synthesizer = HandwritingSynthesizer.load(model_path, device, bias)

        for page in PAGES:
            out_name = f"{page['id']}__{ckpt}__b{bias}.png"
            out_path = os.path.join(OUT_DIR, out_name)
            if os.path.exists(out_path):
                print(f"  skip (exists) {out_name}")
            else:
                print(f"  rendering {out_name}")
                for attempt in range(5):
                    try:
                        utils.text_to_script(synthesizer, page["text"], out_path, thickness=10)
                        break
                    except TooLargeImageError as e:
                        print(f"    attempt {attempt + 1} diverged ({e}); resampling")
                else:
                    print(f"  GAVE UP on {out_name} after 5 attempts")
                    continue
            manifest.append({
                "file": out_name,
                "page_id": page["id"],
                "checkpoint": ckpt,
                "bias": bias,
                "text": page["text"],
                "checks": page["checks"],
                "strike_word": page.get("strike_word"),
                "correct_word_line": page.get("correct_word_line"),
            })

    with open(os.path.join(os.path.dirname(OUT_DIR), "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} pages + manifest.json")


if __name__ == "__main__":
    main()
