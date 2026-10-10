#!/usr/bin/env python3
"""
Re-renders the two "correction" pages (a wrong value struck through, a
corrected value left legible nearby) with an actual strikethrough line drawn
over the row containing the wrong value -- mirroring the real exam pages'
"STRIKE:" pattern (see /tmp/degraded-test/gen.py from the earlier synthetic
proxy) but on real synthesized handwriting instead of a rendered font.

This duplicates handwriting_synthesis.utils.text_to_script's line-splitting
and merge logic because that function only returns the finished page, not
per-line images -- we need per-line boundaries to know which row to strike.

Run after gen_corpus.py, same venv:
    ~/venvs/handwriting-synth/bin/python handwriting-corpus/render_strike.py
"""
import os
import sys

import random

import torch
from PIL import ImageDraw
import numpy as np

# The synthesis model lives in X-rayLaser/pytorch-handwriting-synthesis-toolkit (built against
# 899e943); point HANDWRITING_SYNTH_REPO at a checkout with its checkpoints/.
REPO = os.environ.get("HANDWRITING_SYNTH_REPO") or sys.exit(
    "set HANDWRITING_SYNTH_REPO to a pytorch-handwriting-synthesis-toolkit checkout")
sys.path.insert(0, REPO)

from handwriting_synthesis import utils, data
from handwriting_synthesis.sampling import HandwritingSynthesizer
from pages import PAGES

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean")

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

STRIKE_PAGES = [p for p in PAGES if p.get("strike_word")]


def render_with_strike(synthesizer, text, strike_word, save_path, thickness=10):
    lines, priming_line = utils.split_into_lines(text)

    c = data.transcriptions_to_tensor(synthesizer.tokenizer, [priming_line])
    priming_x = synthesizer.model.sample_means(context=c, steps=700, stochastic=True)
    priming_x = priming_x.unsqueeze(0)

    strike_line_idx = None
    for i, line in enumerate(lines):
        if strike_word.lower() in line.lower():
            strike_line_idx = i
            break

    pil_images = []
    for i, line in enumerate(lines):
        s = data.transcriptions_to_tensor(synthesizer.tokenizer, [line])
        sample = synthesizer.model.sample_primed(priming_x, c, s, steps=1500)
        points_seq = sample.cpu() * synthesizer.sd + synthesizer.mu
        im = utils.create_strokes_png(points_seq, lines=True, shrink_factor=2,
                                        suppress_errors=False, thickness=thickness)
        im = im.convert("RGB")

        if i == strike_line_idx:
            arr = np.array(im.convert("L"))
            ink_mask = arr < 200
            ink_cols = np.where(ink_mask.any(axis=0))[0]
            x0, x1 = int(ink_cols.min()), int(ink_cols.max())
            # weight by ink density per row, not the bounding box: an
            # ascender-only line (no descenders) skews min/max toward
            # cap-height, landing the strike above the x-height body instead
            # of through it
            row_ink = ink_mask.sum(axis=1).astype(float)
            mid = int(np.average(np.arange(len(row_ink)), weights=row_ink))

            draw = ImageDraw.Draw(im)
            r = random.Random(hash((strike_word, i)) & 0xFFFFFFFF)
            # a real strike is a hand-drawn scribble, not a ruler line: a few
            # overlapping wavy passes with a bit of jitter, not one straight one
            for pass_i in range(2):
                pts = []
                n_pts = 8
                for k in range(n_pts + 1):
                    x = x0 + (x1 - x0) * k / n_pts
                    y = mid + r.uniform(-6, 6) + (pass_i - 0.5) * 3
                    pts.append((x, y))
                draw.line(pts, fill=(20, 20, 30), width=3, joint="curve")

        pil_images.append(im)
        print(f'  line {i}{" [STRUCK]" if i == strike_line_idx else ""}: "{line}"')

    max_width = max(im.width for im in pil_images)
    combined_height = sum(im.height for im in pil_images)
    from PIL import Image
    page = Image.new(mode='RGB', size=(max_width, combined_height), color=(255, 255, 255))
    y = 0
    for im in pil_images:
        page.paste(im, (0, y))
        y += im.height
    page.save(save_path)


def main():
    device = torch.device("cpu")
    for ckpt, bias in STYLES:
        model_path = os.path.join(REPO, "checkpoints", ckpt)
        print(f"=== loading {ckpt} bias={bias} ===")
        synthesizer = HandwritingSynthesizer.load(model_path, device, bias)

        for page in STRIKE_PAGES:
            out_name = f"{page['id']}__{ckpt}__b{bias}.png"
            out_path = os.path.join(OUT_DIR, out_name)
            print(f"rendering {out_name}")
            for attempt in range(5):
                try:
                    render_with_strike(synthesizer, page["text"], page["strike_word"], out_path)
                    break
                except utils.TooLargeImageError as e:
                    print(f"  attempt {attempt + 1} diverged ({e}); resampling")
            else:
                print(f"  GAVE UP on {out_name} after 5 attempts")


if __name__ == "__main__":
    main()
