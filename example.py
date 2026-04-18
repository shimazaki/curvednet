"""Curved Hopfield recall (Aguilera et al. 2025) - standalone example.

End-to-end pipeline: load binary patterns, build Hebbian weights, for each
pattern start from a noisy copy and run a curved-Glauber Markov chain with the
exact deformed conditional (paper S2.5), snapshot the state, and save an
animated GIF. Reproduces fig/curved_recall.gif from curvednet.py.
"""
import numpy as np
from PIL import Image

from generate_patterns import load_patterns

# --- Parameters ------------------------------------------------------------
BETA = 3.0               # inverse temperature
GAMMA_0 = -1.2           # curvature; < 0 accelerates recall
NOISE_FRAC = 0.30        # fraction of spins flipped in the noisy init
N_STEPS = 80000          # single-spin updates per pattern
SNAPSHOT_INTERVAL = 500  # frame every this many flips
DISPLAY_SCALE = 8        # nearest-neighbor upscaling for the GIF
SEED = 42

# --- Load patterns + Hebbian weights --------------------------------------
patterns = load_patterns()
N = patterns[0].size
N_SIDE = int(np.sqrt(N))
W = sum(np.outer(p, p) for p in patterns) / N
np.fill_diagonal(W, 0.0)
GAMMA_U = GAMMA_0 / (N * BETA)   # raw gamma entering the deformed exponential


def prob_plus(h, s_i, energy):
    """P(s_i = +1 | rest); exact deformed conditional (paper S2.5)."""
    denom = 1.0 - GAMMA_0 * energy / N
    beta_eff = BETA / (denom if denom > 0 else 1e-8)
    z = -2.0 * beta_eff * s_i * h
    inner = 1.0 + GAMMA_U * z
    if inner <= 0.0:
        prob_stay = 1.0 if GAMMA_U > 0.0 else 0.0
    else:
        log_expg = np.log(inner) / GAMMA_U
        prob_stay = 0.0 if log_expg > 700.0 else 1.0 / (1.0 + np.exp(log_expg))
    return prob_stay if s_i > 0 else 1.0 - prob_stay


# --- Run recall from a noisy copy of each stored pattern ------------------
rng = np.random.default_rng(SEED)
frames = []
for pattern in patterns:
    state = pattern.copy()
    state[rng.choice(N, size=int(NOISE_FRAC * N), replace=False)] *= -1
    energy = -0.5 * state @ W @ state
    snapshots = [state.copy()]
    for step in range(1, N_STEPS + 1):
        i = rng.integers(N)
        h = W[i] @ state
        new_val = 1.0 if rng.random() < prob_plus(h, state[i], energy) else -1.0
        if new_val != state[i]:
            energy -= (new_val - state[i]) * h
            state[i] = new_val
        if step % SNAPSHOT_INTERVAL == 0:
            snapshots.append(state.copy())
    for s in snapshots:
        pixels = ((s.reshape(N_SIDE, N_SIDE) + 1) / 2 * 255).astype(np.uint8)
        frames.append(Image.fromarray(pixels, "L").resize(
            (N_SIDE * DISPLAY_SCALE, N_SIDE * DISPLAY_SCALE), Image.NEAREST))

# --- Save animated GIF ----------------------------------------------------
frames[0].save("fig/curved_recall.gif", save_all=True,
               append_images=frames[1:], duration=100, loop=0)
print(f"Saved fig/curved_recall.gif ({len(frames)} frames)")
