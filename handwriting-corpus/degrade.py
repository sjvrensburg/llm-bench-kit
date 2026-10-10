#!/usr/bin/env python3
"""
Composites each clean synthesized page onto a paper-colored canvas and
applies the same degradation recipe as the earlier synthetic-font proxy
(/tmp/degraded-test/gen.py): slight rotation, PIL-noise blend, Gaussian
blur, JPEG re-encode at quality 72. Deterministic per file via a seed
derived from its name, so re-runs are reproducible.

Run after gen_corpus.py and render_strike.py:
    ~/venvs/handwriting-synth/bin/python handwriting-corpus/degrade.py
"""
import hashlib
import io
import os
import random

from PIL import Image, ImageFilter

CLEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "degraded")
PAPER = (250, 248, 240)


def seed_for(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def degrade(img, seed):
    r = random.Random(seed)
    margin = 80
    canvas = Image.new("RGB", (img.width + 2 * margin, img.height + 2 * margin), PAPER)
    canvas.paste(img.convert("RGB"), (margin, margin))
    img = canvas

    img = img.rotate(r.uniform(-1.5, 1.5), expand=True, fillcolor=PAPER)
    noise = Image.effect_noise(img.size, 24).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    img = Image.blend(img, noise_rgb, alpha=0.06)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(CLEAN_DIR) if f.endswith(".png"))
    for name in files:
        img = Image.open(os.path.join(CLEAN_DIR, name))
        img = degrade(img, seed=seed_for(name))
        out_name = os.path.splitext(name)[0] + ".jpg"
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=72)
        with open(os.path.join(OUT_DIR, out_name), "wb") as f:
            f.write(buf.getvalue())
    print(f"Degraded {len(files)} pages -> {OUT_DIR}")


if __name__ == "__main__":
    main()
