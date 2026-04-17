"""Hopfield network simulation storing two binary patterns with animated GIF output."""

import numpy as np
from PIL import Image

# --- Parameters ---
N_SIDE = 32
N = N_SIDE * N_SIDE
NOISE_FRAC = 0.30
N_STEPS = 20000
SNAPSHOT_INTERVAL = 500
DISPLAY_SCALE = 8
RNG = np.random.default_rng(42)


def make_cross(n):
    pat = -np.ones((n, n))
    t = max(n // 8, 1)
    c = n // 2
    pat[c - t : c + t + 1, :] = 1
    pat[:, c - t : c + t + 1] = 1
    return pat


def make_square(n):
    pat = -np.ones((n, n))
    m = n // 6
    t = max(n // 16, 1)
    pat[m : m + t, m : n - m] = 1
    pat[n - m - t : n - m, m : n - m] = 1
    pat[m : n - m, m : m + t] = 1
    pat[m : n - m, n - m - t : n - m] = 1
    return pat


# --- Build patterns and Hebbian weights ---
patterns = [make_cross(N_SIDE).ravel(), make_square(N_SIDE).ravel()]

W = sum(np.outer(xi, xi) for xi in patterns) / N
np.fill_diagonal(W, 0.0)

# --- Noisy initial state (corrupt pattern 0) ---
state = patterns[0].copy()
state[RNG.choice(N, size=int(NOISE_FRAC * N), replace=False)] *= -1

# --- Run asynchronous dynamics and record snapshots ---
snapshots = [state.copy()]
for step in range(1, N_STEPS + 1):
    i = RNG.integers(N)
    state[i] = np.sign(W[i] @ state) or 1.0
    if step % SNAPSHOT_INTERVAL == 0:
        snapshots.append(state.copy())


def snap_to_image(s):
    pixels = ((s.reshape(N_SIDE, N_SIDE) + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(pixels, "L").resize(
        (N_SIDE * DISPLAY_SCALE, N_SIDE * DISPLAY_SCALE), Image.NEAREST
    )


# --- Build animated GIF ---
frames = [snap_to_image(s) for s in snapshots]
out_path = "fig/hopfield_recall.gif"
frames[0].save(
    out_path, save_all=True, append_images=frames[1:], duration=300, loop=0
)
print(f"Saved {out_path} ({len(frames)} frames)")
