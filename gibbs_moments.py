"""Gibbs sampler for the curved Hopfield model.

Runs single-spin-flip dynamics with the gamma-deformed conditional

    p_gamma(x_k | x_\\k) = 1 / (1 + exp_gamma(-2 beta'(x) x_k h_k)),

where exp_gamma(z) = [1 + gamma z]_+^(1/gamma) is the gamma-exponential
and beta'(x) = beta / [1 - gamma_0 * E(x) / N]_+ is the state-dependent
effective inverse temperature. This recovers the standard Gibbs sigmoid
(1 + tanh(beta'·h))/2 as gamma_0 -> 0 and matches Eq. (8) of Aguilera et
al. (Nat. Commun. 2025).

Accumulates equilibrium moments

    eta_i  = <s_i>
    eta_ij = <s_i s_j>

from samples collected after a burn-in. Patterns are downsampled from the
128x128 .npy files in data/ to N_SIDE x N_SIDE so that eta_ij fits
comfortably in memory (N_SIDE=32 -> ~4 MB float32).
"""

import glob
import os
import sys

import numpy as np
from PIL import Image

from curvednet import prob_stay_gamma

# --- Configuration ---
GAMMA_0 = -0.3          # curvature gamma' (override via CLI: python gibbs_moments.py -0.3)
BETA = 1.5              # inverse temperature (matches compare_gamma.py)
N_SIDE = 32             # downsample target
N_SWEEPS = 3000         # total sweeps (1 sweep = N single-spin updates)
BURN_IN = 500           # sweeps discarded before accumulating moments
SAMPLE_INTERVAL = 1     # sweeps between moment updates (thinning)
SEED = 42

OUT_DIR = "results"
if len(sys.argv) > 1:
    GAMMA_0 = float(sys.argv[1])
OUT_PATH = os.path.join(OUT_DIR, f"gibbs_moments_g{GAMMA_0:+.2f}.npz")


def load_and_downsample(path: str, n_side: int) -> np.ndarray:
    """Load a binary {-1,+1} pattern and downsample to n_side x n_side."""
    arr = np.load(path)
    side = int(np.sqrt(arr.size))
    assert side * side == arr.size, f"{path} is not square"
    img = Image.fromarray(((arr.reshape(side, side) + 1) / 2 * 255).astype(np.uint8), "L")
    img = img.resize((n_side, n_side), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    threshold = np.median(pixels)
    return np.where(pixels >= threshold, 1.0, -1.0).ravel()


def main() -> None:
    npy_files = sorted(glob.glob("data/*.npy"))
    if not npy_files:
        print("No patterns found in data/. Run generate_patterns.py first.")
        sys.exit(1)

    patterns = [load_and_downsample(f, N_SIDE) for f in npy_files]
    N = N_SIDE * N_SIDE
    print(f"Loaded {len(patterns)} patterns (downsampled to {N_SIDE}x{N_SIDE}) from {npy_files}")

    W = sum(np.outer(xi, xi) for xi in patterns) / N
    np.fill_diagonal(W, 0.0)

    rng = np.random.default_rng(SEED)
    state = rng.choice([-1.0, 1.0], size=N)
    energy = -0.5 * state @ W @ state

    eta_i = np.zeros(N, dtype=np.float64)
    eta_ij = np.zeros((N, N), dtype=np.float32)
    n_samples = 0

    gamma_u = GAMMA_0 / (N * BETA)  # unscaled gamma entering exp_gamma

    print(f"Sampling: gamma_0={GAMMA_0}, beta={BETA}, sweeps={N_SWEEPS}, burn_in={BURN_IN}")
    for sweep in range(N_SWEEPS):
        for _ in range(N):
            i = rng.integers(N)
            h = W[i] @ state
            denom = 1.0 - GAMMA_0 * energy / N
            beta_eff = BETA / max(denom, 1e-8)
            prob_stay = prob_stay_gamma(-2.0 * beta_eff * state[i] * h, gamma_u)
            if rng.random() >= prob_stay:
                delta = -2.0 * state[i]
                energy -= delta * h
                state[i] = -state[i]

        if sweep >= BURN_IN and (sweep - BURN_IN) % SAMPLE_INTERVAL == 0:
            eta_i += state
            eta_ij += np.outer(state, state).astype(np.float32)
            n_samples += 1

        if (sweep + 1) % 100 == 0:
            print(f"  sweep {sweep + 1}/{N_SWEEPS}  samples={n_samples}  E={energy:.2f}")

    eta_i /= n_samples
    eta_ij /= n_samples

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        OUT_PATH,
        eta_i=eta_i,
        eta_ij=eta_ij,
        gamma_0=GAMMA_0,
        beta=BETA,
        n_samples=n_samples,
        n_sweeps=N_SWEEPS,
        burn_in=BURN_IN,
        n_side=N_SIDE,
    )
    print(f"Saved {OUT_PATH} (n_samples={n_samples}, gamma_0={GAMMA_0}, beta={BETA})")


if __name__ == "__main__":
    main()
