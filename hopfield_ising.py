"""Noisy Hopfield network with tanh activation (finite temperature)."""

import glob
import sys

import numpy as np
from PIL import Image

# --- Load patterns from .npy files ---
npy_files = sorted(glob.glob("patterns/*.npy"))
if not npy_files:
    print("No patterns found in patterns/. Run generate_patterns.py first.")
    sys.exit(1)

patterns = [np.load(f).ravel() for f in npy_files]
N_SIDE = int(np.sqrt(len(patterns[0])))
assert N_SIDE * N_SIDE == len(patterns[0]), "patterns must be square"
N = N_SIDE * N_SIDE
print(f"Loaded {len(patterns)} patterns ({N_SIDE}x{N_SIDE}) from {npy_files}")

# --- Parameters ---
NOISE_FRAC = 0.30
N_STEPS = 80000
SNAPSHOT_INTERVAL = 500
DISPLAY_SCALE = 8
beta = 3.0  # inverse temperature: beta->inf recovers sign(), beta->0 is random
RNG = np.random.default_rng(42)

# --- Build Hebbian weights ---
W = sum(np.outer(xi, xi) for xi in patterns) / N
np.fill_diagonal(W, 0.0)


def snap_to_image(s):
    pixels = ((s.reshape(N_SIDE, N_SIDE) + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(pixels, "L").resize(
        (N_SIDE * DISPLAY_SCALE, N_SIDE * DISPLAY_SCALE), Image.NEAREST
    )


def run_cycle(pattern):
    """Run one recall cycle from a noisy version of the given pattern."""
    state = pattern.copy()
    state[RNG.choice(N, size=int(NOISE_FRAC * N), replace=False)] *= -1
    snapshots = [state.copy()]
    for step in range(1, N_STEPS + 1):
        i = RNG.integers(N)
        h = W[i] @ state
        prob_plus = (1.0 + np.tanh(beta * h)) / 2.0
        state[i] = 1.0 if RNG.random() < prob_plus else -1.0
        if step % SNAPSHOT_INTERVAL == 0:
            snapshots.append(state.copy())
    return [snap_to_image(s) for s in snapshots]


# --- Run one recall cycle per loaded pattern ---
frames = []
for pat in patterns:
    frames.extend(run_cycle(pat))

out_path = "hopfield_recall.gif"
frames[0].save(
    out_path, save_all=True, append_images=frames[1:], duration=100, loop=0
)
print(f"Saved {out_path} ({len(frames)} frames)")
