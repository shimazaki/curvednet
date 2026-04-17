"""Compare recall speed across different curvature parameters gamma_0.

Runs the curved Hopfield network from identical noisy initial states for each
gamma_0 value, producing a side-by-side GIF (fig/compare_gamma.gif).
"""

import glob
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
GAMMAS = [0.0, -2.0]
NOISE_FRAC = 0.30
N_STEPS = 100000
SNAPSHOT_INTERVAL = 500
DISPLAY_SCALE = 4
BETA = 1.5
LABEL_HEIGHT = 44  # pixels for the gamma label strip
GAP = 64           # pixels between columns

# --- Load patterns ---
npy_files = sorted(glob.glob("patterns/*.npy"))
if not npy_files:
    print("No patterns found in patterns/. Run generate_patterns.py first.")
    sys.exit(1)

patterns = [np.load(f).ravel() for f in npy_files]
N_SIDE = int(np.sqrt(len(patterns[0])))
assert N_SIDE * N_SIDE == len(patterns[0]), "patterns must be square"
N = N_SIDE * N_SIDE
IMG_SIZE = N_SIDE * DISPLAY_SCALE
print(f"Loaded {len(patterns)} patterns ({N_SIDE}x{N_SIDE}) from {npy_files}")

# --- Build Hebbian weights (shared across all gamma runs) ---
W = sum(np.outer(xi, xi) for xi in patterns) / N
np.fill_diagonal(W, 0.0)


def snap_to_image(s):
    pixels = ((s.reshape(N_SIDE, N_SIDE) + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(pixels, "L").resize(
        (IMG_SIZE, IMG_SIZE), Image.NEAREST
    )


def run_cycle(pattern, gamma_0, rng_state):
    """Run one recall cycle from a noisy version of pattern.

    rng_state is the RNG state captured *before* generating noise,
    so every gamma_0 starts from the identical noisy initial state.
    """
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state

    state = pattern.copy()
    state[rng.choice(N, size=int(NOISE_FRAC * N), replace=False)] *= -1

    energy = -0.5 * state @ W @ state

    snapshots = [state.copy()]
    for step in range(1, N_STEPS + 1):
        i = rng.integers(N)
        h = W[i] @ state

        denom = 1.0 - gamma_0 * energy / N
        if denom <= 0:
            beta_eff = BETA / 1e-8
        else:
            beta_eff = BETA / denom

        prob_plus = (1.0 + np.tanh(beta_eff * h)) / 2.0
        new_val = 1.0 if rng.random() < prob_plus else -1.0

        if new_val != state[i]:
            delta = new_val - state[i]
            energy -= delta * h
            state[i] = new_val

        if step % SNAPSHOT_INTERVAL == 0:
            snapshots.append(state.copy())

    return [snap_to_image(s) for s in snapshots]


def _load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def compose_frame(images, gammas):
    """Horizontally concatenate images with gamma labels on top, separated by a gap."""
    n = len(gammas)
    total_w = IMG_SIZE * n + GAP * (n - 1)
    total_h = LABEL_HEIGHT + IMG_SIZE
    frame = Image.new("L", (total_w, total_h), 255)
    draw = ImageDraw.Draw(frame)
    font = _load_font(32)

    for col, (img, g) in enumerate(zip(images, gammas)):
        x = col * (IMG_SIZE + GAP)
        # Draw centered label
        g_str = f"{g:.0e}" if abs(g) >= 1e4 else f"{g}"
        text = f"\u03b2={BETA}, \u03b3\u2032={g_str}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (IMG_SIZE - tw) / 2, (LABEL_HEIGHT - th) / 2),
                  text, fill=0, font=font)
        # Paste image below label
        frame.paste(img, (x, LABEL_HEIGHT))
    return frame


# --- Run comparisons ---
master_rng = np.random.default_rng(42)
frames = []

for pat_idx, pat in enumerate(patterns):
    print(f"Pattern {pat_idx + 1}/{len(patterns)}...")

    # Capture RNG state before noise generation so all gammas get the same noise
    rng_state = master_rng.bit_generator.state

    # Run recall for each gamma from the identical noisy starting point
    all_snaps = []
    for g in GAMMAS:
        print(f"  gamma_0 = {g} ...")
        snaps = run_cycle(pat, g, rng_state)
        all_snaps.append(snaps)

    # Advance master RNG past the noise generation (use first gamma's consumption)
    tmp = np.random.default_rng()
    tmp.bit_generator.state = rng_state
    tmp.choice(N, size=int(NOISE_FRAC * N), replace=False)
    master_rng.bit_generator.state = tmp.bit_generator.state

    # Compose side-by-side frames
    n_snaps = len(all_snaps[0])
    for t in range(n_snaps):
        col_images = [all_snaps[g_idx][t] for g_idx in range(len(GAMMAS))]
        frames.append(compose_frame(col_images, GAMMAS))

out_path = "fig/compare_gamma.gif"
frames[0].save(
    out_path, save_all=True, append_images=frames[1:], duration=100, loop=0
)
print(f"Saved {out_path} ({len(frames)} frames)")
