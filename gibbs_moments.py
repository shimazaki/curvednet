"""Gibbs sampler for the curved Hopfield model.

Cycles through all stored memory patterns, running one Markov-chain segment
per pattern (initialized near that pattern with 30% noise).  Each segment
yields one snapshot per sweep of N single-spin updates, and moments

    eta_i  = <s_i>
    eta_ij = <s_i s_j>

are stream-accumulated across all segments with no burn-in.
The `ACTIVATION` constant selects the exact deformed conditional
(paper S2.5, default) or the large-N approximation (S2.7). Patterns are
converted from cached JPEGs at N_SIDE x N_SIDE so that eta_ij fits
comfortably in memory (N_SIDE=32 -> ~4 MB float32).  The ``--patterns N``
flag controls how many patterns (alphabetically sorted) are encoded into
the Hebbian weight matrix (default 2 = all available).
"""

import os
import sys

import numpy as np

# --- Parse flags before positional args ---
BINARY = "--binary" in sys.argv
_argv = [a for a in sys.argv[1:] if a != "--binary"]
_size_argv = []
_samples_argv = []
_patterns_argv = []
for _flag, _store in [("--size", "_size"), ("--samples", "_samples"), ("--patterns", "_patterns")]:
    for _i, _a in enumerate(_argv):
        if _a == _flag and _i + 1 < len(_argv):
            if _flag == "--size":
                _size_argv = [_argv[_i + 1]]
            elif _flag == "--samples":
                _samples_argv = [_argv[_i + 1]]
            else:
                _patterns_argv = [_argv[_i + 1]]
            _argv = _argv[:_i] + _argv[_i + 2:]
            break

from curvednet import hebbian_weights
import generate_patterns
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
N_PATTERNS = int(_patterns_argv[0]) if _patterns_argv else 2  # number of patterns to encode
N_SWEEPS = N_SAMPLES    # total sweeps (1 sweep = N single-spin updates)
SAMPLE_INTERVAL = 1     # sweeps between moment updates (thinning)
SEED = 42
ACTIVATION = "exact"    # "exact" (paper S2.5) or "approx" (S2.7)

OUT_DIR = "results"
if _argv:
    GAMMA_0 = float(_argv[0])
_suffix = "_binary" if BINARY else ""
OUT_PATH = os.path.join(OUT_DIR, f"gibbs_moments{_suffix}_size{N_SIDE}_p{N_PATTERNS}_g{GAMMA_0:+.2f}_s{N_SAMPLES}.npz")


def main() -> None:
    all_patterns = load_patterns(n_side=N_SIDE)
    patterns = all_patterns[:N_PATTERNS]
    N = N_SIDE * N_SIDE
    n_image = len(patterns)
    pattern_names = sorted(generate_patterns.SOURCES.keys())[:N_PATTERNS]

    if N_PATTERNS > n_image:
        fill_rng = np.random.default_rng(seed=12345)
        for _ in range(N_PATTERNS - n_image):
            patterns.append(fill_rng.choice([-1.0, 1.0], size=N))
        print(f"Using {N_PATTERNS} patterns ({n_image} image + {N_PATTERNS - n_image} random, {N_SIDE}x{N_SIDE})")
    else:
        print(f"Using {n_image} of {len(all_patterns)} patterns ({N_SIDE}x{N_SIDE}): {', '.join(pattern_names)}")

    if BINARY:
        W_s = hebbian_weights(patterns, N)
        W, b = ising_to_binary(W_s)
        patterns = [state_ising_to_binary(p) for p in patterns]
    else:
        W = hebbian_weights(patterns, N)
        b = None

    rng = np.random.default_rng(SEED)

    eta_i = np.zeros(N, dtype=np.float64)
    eta_ij = np.zeros((N, N), dtype=np.float32)
    pK = np.zeros(N + 1, dtype=np.int64)  # population spike count histogram
    n_samples = 0

    sweeps_per_seg = N_SWEEPS // N_PATTERNS
    noise_frac = 0.30

    mode = "binary {0,1}" if BINARY else "Ising {-1,+1}"
    print(f"Sampling ({mode}): gamma_0={GAMMA_0}, beta={BETA}, "
          f"sweeps={N_SWEEPS} ({sweeps_per_seg}/pattern x {N_PATTERNS}), "
          f"activation={ACTIVATION}")

    for p_idx, pat in enumerate(patterns):
        init = pat.copy()
        if BINARY:
            it = binary_iter(
                init, W, b,
                beta=BETA, gamma_0=GAMMA_0,
                n_steps=N * sweeps_per_seg, snapshot_interval=N,
                noise_frac=noise_frac, rng=rng, activation=ACTIVATION,
            )
        else:
            it = iter_curved_glauber(
                init, W,
                beta=BETA, gamma_0=GAMMA_0,
                n_steps=N * sweeps_per_seg, snapshot_interval=N,
                noise_frac=noise_frac, rng=rng, activation=ACTIVATION,
            )
        next(it)  # discard the pre-sweep initial snapshot
        for sweep_idx, s in enumerate(it, start=1):
            if (sweep_idx - 1) % SAMPLE_INTERVAL == 0:
                eta_i += s
                eta_ij += np.outer(s, s).astype(np.float32)
                if BINARY:
                    k = int(s.sum())
                else:
                    k = int((s + 1).sum()) // 2
                pK[k] += 1
                n_samples += 1
            if sweep_idx % 100 == 0:
                print(f"  pattern {p_idx + 1}/{N_PATTERNS}  "
                      f"sweep {sweep_idx}/{sweeps_per_seg}  samples={n_samples}")

    eta_i /= n_samples
    eta_ij /= n_samples

    os.makedirs(OUT_DIR, exist_ok=True)
    save_dict = dict(
        eta_i=eta_i,
        eta_ij=eta_ij,
        pK=pK,
        W=W,
        gamma_0=GAMMA_0,
        beta=BETA,
        n_samples=n_samples,
        n_sweeps=N_SWEEPS,
        n_side=N_SIDE,
    )
    if b is not None:
        save_dict["b"] = b
    np.savez(OUT_PATH, **save_dict)
    print(f"Saved {OUT_PATH} (n_samples={n_samples}, gamma_0={GAMMA_0}, beta={BETA})")


if __name__ == "__main__":
    main()
