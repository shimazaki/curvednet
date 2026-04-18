"""Gibbs sampler for the curved Hopfield model.

Drives `curvednet.iter_curved_glauber` as a single long Markov chain (one
snapshot per sweep of N single-spin updates) and stream-accumulates
equilibrium moments

    eta_i  = <s_i>
    eta_ij = <s_i s_j>

from samples collected after a burn-in. The `ACTIVATION` constant selects
the exact deformed conditional (paper S2.5, default) or the large-N
approximation (S2.7). Patterns are converted from cached JPEGs at
N_SIDE x N_SIDE so that eta_ij fits comfortably in memory
(N_SIDE=32 -> ~4 MB float32).
"""

import os
import sys

import numpy as np

# --- Parse flags before positional args ---
BINARY = "--binary" in sys.argv
_argv = [a for a in sys.argv[1:] if a != "--binary"]
_size_argv = []
_samples_argv = []
for _flag, _store in [("--size", "_size"), ("--samples", "_samples")]:
    for _i, _a in enumerate(_argv):
        if _a == _flag and _i + 1 < len(_argv):
            if _flag == "--size":
                _size_argv = [_argv[_i + 1]]
            else:
                _samples_argv = [_argv[_i + 1]]
            _argv = _argv[:_i] + _argv[_i + 2:]
            break

from curvednet import hebbian_weights
from generate_patterns import load_patterns
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
N_SIDE = int(_size_argv[0]) if _size_argv else 32  # image side length (--size N)
N_SAMPLES = int(_samples_argv[0]) if _samples_argv else 2500  # desired post-burn-in samples
BURN_IN = 500           # sweeps discarded before accumulating moments
N_SWEEPS = N_SAMPLES + BURN_IN  # total sweeps (1 sweep = N single-spin updates)
SAMPLE_INTERVAL = 1     # sweeps between moment updates (thinning)
SEED = 42
ACTIVATION = "exact"    # "exact" (paper S2.5) or "approx" (S2.7)

OUT_DIR = "results"
if _argv:
    GAMMA_0 = float(_argv[0])
_suffix = "_binary" if BINARY else ""
OUT_PATH = os.path.join(OUT_DIR, f"gibbs_moments{_suffix}_size{N_SIDE}_g{GAMMA_0:+.2f}_s{N_SAMPLES}.npz")


def main() -> None:
    patterns = load_patterns(n_side=N_SIDE)
    N = N_SIDE * N_SIDE
    print(f"Loaded {len(patterns)} patterns ({N_SIDE}x{N_SIDE})")

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
    save_dict = dict(
        eta_i=eta_i,
        eta_ij=eta_ij,
        W=W,
        gamma_0=GAMMA_0,
        beta=BETA,
        n_samples=n_samples,
        n_sweeps=N_SWEEPS,
        burn_in=BURN_IN,
        n_side=N_SIDE,
    )
    if b is not None:
        save_dict["b"] = b
    np.savez(OUT_PATH, **save_dict)
    print(f"Saved {OUT_PATH} (n_samples={n_samples}, gamma_0={GAMMA_0}, beta={BETA})")


if __name__ == "__main__":
    main()
