"""Compare recall speed across different curvature parameters gamma_0.

Runs the curved Hopfield network from identical noisy initial states for each
gamma_0 value, producing a side-by-side GIF (fig/compare_gamma.gif).

Computation is reused from `curvednet`: `load_square_patterns`,
`hebbian_weights`, `run_cycle`, `snap_to_image`. Only the display-specific
helpers (`_load_font`, `compose_frame`) and the outer multi-gamma / pattern
driver live here, and all display/I-O runs under `__main__`.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from curvednet import (
    hebbian_weights,
    load_square_patterns,
    run_curved_glauber,
    snap_to_image,
)


def _load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def compose_frame(images, gammas, beta, img_size, label_height, gap):
    """Horizontally concatenate images with gamma labels on top, separated by a gap."""
    n = len(gammas)
    total_w = img_size * n + gap * (n - 1)
    total_h = label_height + img_size
    frame = Image.new("L", (total_w, total_h), 255)
    draw = ImageDraw.Draw(frame)
    font = _load_font(32)

    for col, (img, g) in enumerate(zip(images, gammas)):
        x = col * (img_size + gap)
        g_str = f"{g:.0e}" if abs(g) >= 1e4 else f"{g}"
        text = f"\u03b2={beta}, \u03b3\u2032={g_str}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (img_size - tw) / 2, (label_height - th) / 2),
                  text, fill=0, font=font)
        frame.paste(img, (x, label_height))
    return frame


if __name__ == "__main__":
    # --- Configuration ---
    GAMMAS = [0.0, -2.0]
    NOISE_FRAC = 0.30
    N_STEPS = 100000
    SNAPSHOT_INTERVAL = 500
    DISPLAY_SCALE = 4
    BETA = 1.5
    LABEL_HEIGHT = 44  # pixels for the gamma label strip
    GAP = 64           # pixels between columns

    patterns, N_SIDE, N = load_square_patterns()
    IMG_SIZE = N_SIDE * DISPLAY_SCALE
    W = hebbian_weights(patterns, N)

    master_rng = np.random.default_rng(42)
    frames = []

    for pat_idx, pat in enumerate(patterns):
        print(f"Pattern {pat_idx + 1}/{len(patterns)}...")

        # Capture RNG state before noise generation so all gammas get the same noise
        rng_state = master_rng.bit_generator.state

        all_snaps = []
        for g in GAMMAS:
            print(f"  gamma_0 = {g} ...")
            rng_g = np.random.default_rng()
            rng_g.bit_generator.state = rng_state
            snaps = run_curved_glauber(
                pat, W,
                beta=BETA, gamma_0=g,
                n_steps=N_STEPS, snapshot_interval=SNAPSHOT_INTERVAL,
                noise_frac=NOISE_FRAC, rng=rng_g,
            )
            all_snaps.append([snap_to_image(s, N_SIDE, DISPLAY_SCALE) for s in snaps])

        # Advance master RNG past the noise generation (use first gamma's consumption)
        tmp = np.random.default_rng()
        tmp.bit_generator.state = rng_state
        tmp.choice(N, size=int(NOISE_FRAC * N), replace=False)
        master_rng.bit_generator.state = tmp.bit_generator.state

        n_snaps = len(all_snaps[0])
        for t in range(n_snaps):
            col_images = [all_snaps[g_idx][t] for g_idx in range(len(GAMMAS))]
            frames.append(compose_frame(
                col_images, GAMMAS, BETA, IMG_SIZE, LABEL_HEIGHT, GAP,
            ))

    out_path = "fig/compare_gamma.gif"
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"Saved {out_path} ({len(frames)} frames)")
