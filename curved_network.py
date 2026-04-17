"""Curved Hopfield network (Aguilera et al., Nat Commun 2025).

Higher-order interactions via a deformation parameter gamma_0 that makes
the effective inverse temperature state-dependent:

    beta'(x) = beta / [1 - gamma_0 * E(x) / N]_+

For gamma_0 < 0: self-regulating annealing accelerates memory retrieval.
For gamma_0 = 0: recovers the standard Hopfield/Ising model.
For gamma_0 > 0: decelerates dynamics, more robust retrieval.
"""

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
beta = 3.0          # inverse temperature
gamma_0 = -1.2      # curvature: <0 accelerates, 0 = standard, >0 decelerates
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

    # Initial energy: E = -0.5 * x^T W x
    energy = -0.5 * state @ W @ state

    snapshots = [state.copy()]
    for step in range(1, N_STEPS + 1):
        i = RNG.integers(N)
        h = W[i] @ state

        # Effective inverse temperature (Eq. 9, 10 of Aguilera et al.)
        denom = 1.0 - gamma_0 * energy / N
        if denom <= 0:
            beta_eff = beta / 1e-8  # clamp
        else:
            beta_eff = beta / denom

        prob_plus = (1.0 + np.tanh(beta_eff * h)) / 2.0
        new_val = 1.0 if RNG.random() < prob_plus else -1.0

        # Incremental energy update
        if new_val != state[i]:
            delta = new_val - state[i]
            energy -= delta * h
            state[i] = new_val

        if step % SNAPSHOT_INTERVAL == 0:
            snapshots.append(state.copy())
    return [snap_to_image(s) for s in snapshots]


# --- Run one recall cycle per loaded pattern ---
frames = []
for pat in patterns:
    frames.extend(run_cycle(pat))

out_path = "curved_recall.gif"
frames[0].save(
    out_path, save_all=True, append_images=frames[1:], duration=100, loop=0
)
print(f"Saved {out_path} ({len(frames)} frames, gamma_0={gamma_0}, beta={beta})")
