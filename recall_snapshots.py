"""Generate a 2×5 PNG grid showing recall progression for classic vs curved Hopfield.

Top row: classic Hopfield (γ₀=0), bottom row: curved (γ₀=-2).
Both rows start from the same noisy initial state (same RNG seed).
Columns show 5 evenly spaced snapshots from the Glauber chain.

CLI usage:  python recall_snapshots.py [--size N]   (default 128×128)

Output: fig/recall_snapshots.png
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from curvednet import (
    hebbian_weights,
    load_square_patterns,
    run_curved_glauber,
    snap_to_image,
)

# ── Editable parameters ──────────────────────────────────────────────
GAMMAS = [0.0, -2.0]        # top row, bottom row
BETA = 1.5
NOISE_FRAC = 0.30
N_STEPS = 20_000
SNAPSHOT_INTERVAL = 500
DISPLAY_SCALE = 4
N_SNAPSHOTS = 5              # number of columns
RNG_SEED = 42
# ─────────────────────────────────────────────────────────────────────

LABEL_WIDTH = 80             # pixels reserved for row labels
HEADER_HEIGHT = 30           # pixels reserved for column headers
CELL_PAD = 20                # padding between cells


def pick_evenly_spaced(snapshots, n):
    """Return n evenly spaced items from a list (first and last included)."""
    total = len(snapshots)
    if total <= n:
        return list(snapshots)
    indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    return [snapshots[i] for i in indices]


def snapshot_step_labels(total_snapshots, n):
    """Return step-count labels for the n evenly spaced snapshots."""
    if total_snapshots <= n:
        indices = list(range(total_snapshots))
    else:
        indices = [round(i * (total_snapshots - 1) / (n - 1)) for i in range(n)]
    # Each snapshot index maps to a sweep count: index 0 is the noisy init,
    # subsequent indices correspond to (index * SNAPSHOT_INTERVAL) steps.
    labels = []
    for idx in indices:
        if idx == 0:
            labels.append("init")
        else:
            step = idx * SNAPSHOT_INTERVAL
            if step >= 1000:
                labels.append(f"{step // 1000}k")
            else:
                labels.append(str(step))
    return labels


def try_load_font(size=14):
    """Try to load a TrueType font, fall back to default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def compose_grid(rows, col_labels, row_labels, cell_size):
    """Compose a labeled 2D grid of PIL images into a single image."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    cw, ch = cell_size

    grid_w = LABEL_WIDTH + n_cols * (cw + CELL_PAD) - CELL_PAD
    grid_h = HEADER_HEIGHT + n_rows * (ch + CELL_PAD) - CELL_PAD
    canvas = Image.new("L", (grid_w, grid_h), 255)
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(14)
    small_font = try_load_font(12)

    # Column headers
    for c, label in enumerate(col_labels):
        x = LABEL_WIDTH + c * (cw + CELL_PAD) + cw // 2
        bbox = draw.textbbox((0, 0), label, font=small_font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 6), label, fill=0, font=small_font)

    # Row labels and images
    for r, (row_imgs, label) in enumerate(zip(rows, row_labels)):
        y = HEADER_HEIGHT + r * (ch + CELL_PAD)
        # Row label (centered vertically)
        bbox = draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        tw = bbox[2] - bbox[0]
        draw.text((LABEL_WIDTH - tw - 8, y + (ch - th) // 2), label, fill=0, font=font)

        for c, img in enumerate(row_imgs):
            x = LABEL_WIDTH + c * (cw + CELL_PAD)
            canvas.paste(img, (x, y))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Recall progression snapshot grid")
    parser.add_argument("--size", type=int, default=128, help="Pattern side length")
    args = parser.parse_args()

    patterns, n_side, N = load_square_patterns(n_side=args.size)
    W = hebbian_weights(patterns, N)
    target = patterns[0]

    rows = []
    col_labels = None

    for gamma_0 in GAMMAS:
        rng = np.random.default_rng(RNG_SEED)
        snapshots = run_curved_glauber(
            target, W,
            beta=BETA, gamma_0=gamma_0,
            n_steps=N_STEPS, snapshot_interval=SNAPSHOT_INTERVAL,
            noise_frac=NOISE_FRAC, rng=rng,
        )
        picked = pick_evenly_spaced(snapshots, N_SNAPSHOTS)
        images = [snap_to_image(s, n_side, DISPLAY_SCALE) for s in picked]
        rows.append(images)

        if col_labels is None:
            col_labels = snapshot_step_labels(len(snapshots), N_SNAPSHOTS)

    row_labels = [f"\u03b3\u2080={g:g}" for g in GAMMAS]
    cell_size = (n_side * DISPLAY_SCALE, n_side * DISPLAY_SCALE)
    grid = compose_grid(rows, col_labels, row_labels, cell_size)

    os.makedirs("fig", exist_ok=True)
    out_png = "fig/recall_snapshots.png"
    out_eps = "fig/recall_snapshots.eps"
    grid.save(out_png)
    grid.save(out_eps, lossless=True)
    print(f"Saved {out_png} and {out_eps} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
