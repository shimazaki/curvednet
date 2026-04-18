"""Gibbs sampler for the curved Hopfield model.

Drives `curvednet.iter_curved_glauber` as a single long Markov chain (one
snapshot per sweep of N single-spin updates) and stream-accumulates
equilibrium moments

    eta_i  = <s_i>
    eta_ij = <s_i s_j>

from samples collected after a burn-in. The `ACTIVATION` constant selects
the exact deformed conditional (paper S2.5, default) or the large-N
approximation (S2.7). Patterns are downsampled from the 128x128 .npy files
in data/ to N_SIDE x N_SIDE so that eta_ij fits comfortably in memory
(N_SIDE=32 -> ~4 MB float32).
"""

import glob
import os
import sys

import numpy as np
from PIL import Image

# --- Parse --binary flag before positional args ---
BINARY = "--binary" in sys.argv
_argv = [a for a in sys.argv[1:] if a != "--binary"]

from curvednet import hebbian_weights
if BINARY:
    from curvednet_binary import (
        ising_to_binary,
        state_ising_to_binary,
        iter_curved_glauber as binary_iter,
    )
else:
    from curvednet import iter_curved_glauber

# --- Configuration ---
GAMMA_0 = -0.3          # curvature gamma' (override via CLI: python gibbs_moments.py -0.3)
BETA = 1.5              # inverse temperature (matches compare_gamma.py)
N_SIDE = 32             # downsample target
N_SWEEPS = 3000         # total sweeps (1 sweep = N single-spin updates)
BURN_IN = 500           # sweeps discarded before accumulating moments
SAMPLE_INTERVAL = 1     # sweeps between moment updates (thinning)
SEED = 42
ACTIVATION = "exact"    # "exact" (paper S2.5) or "approx" (S2.7)

OUT_DIR = "results"
if _argv:
    GAMMA_0 = float(_argv[0])
_suffix = "_binary" if BINARY else ""
OUT_PATH = os.path.join(OUT_DIR, f"gibbs_moments{_suffix}_g{GAMMA_0:+.2f}.npz")


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

    if BINARY:
        W_s = hebbian_weights(patterns, N)
        W, b = ising_to_binary(W_s)
        patterns = [state_ising_to_binary(p) for p in patterns]
    else:
        W = hebbian_weights(patterns, N)
        b = None

    rng = np.random.default_rng(SEED)
    if BINARY:
        initial_state = rng.choice([0.0, 1.0], size=N)
    else:
        initial_state = rng.choice([-1.0, 1.0], size=N)

    eta_i = np.zeros(N, dtype=np.float64)
    eta_ij = np.zeros((N, N), dtype=np.float32)
    n_samples = 0

    mode = "binary {0,1}" if BINARY else "Ising {-1,+1}"
    print(f"Sampling ({mode}): gamma_0={GAMMA_0}, beta={BETA}, sweeps={N_SWEEPS}, "
          f"burn_in={BURN_IN}, activation={ACTIVATION}")
    if BINARY:
        it = binary_iter(
            initial_state, W, b,
            beta=BETA, gamma_0=GAMMA_0,
            n_steps=N * N_SWEEPS, snapshot_interval=N,
            noise_frac=0.0, rng=rng, activation=ACTIVATION,
        )
    else:
        it = iter_curved_glauber(
            initial_state, W,
            beta=BETA, gamma_0=GAMMA_0,
            n_steps=N * N_SWEEPS, snapshot_interval=N,
            noise_frac=0.0, rng=rng, activation=ACTIVATION,
        )
    next(it)  # discard the pre-sweep initial snapshot
    for sweep_idx, s in enumerate(it, start=1):
        if sweep_idx > BURN_IN and (sweep_idx - BURN_IN - 1) % SAMPLE_INTERVAL == 0:
            eta_i += s
            eta_ij += np.outer(s, s).astype(np.float32)
            n_samples += 1
        if sweep_idx % 100 == 0:
            print(f"  sweep {sweep_idx}/{N_SWEEPS}  samples={n_samples}")

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
