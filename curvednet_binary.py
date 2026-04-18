"""Curved Hopfield network for {0, 1} binary neurons.

Parallel to `curvednet.py` (which uses {-1, +1} Ising spins). All functions
follow the same API but operate in the binary encoding. Translation functions
`ising_to_binary` and `binary_to_ising` convert parameters between the two
encodings so both produce the same distribution.

Energy:  E(x) = -0.5 x^T W x - b^T x

With the mapping s = 2x - 1 and Ising energy E_s = -0.5 s^T W_s s:
    W_x = 4 W_s,   b_i = -2 sum_j W_s[i,j]

Module-level exports are importable with no side effects. Display + GIF
output runs only under `if __name__ == "__main__":`.
"""

import sys

import numpy as np
from PIL import Image

from generate_patterns import load_patterns


# --- Ising <-> Binary translation -------------------------------------------


def ising_to_binary(W_s):
    """Convert Ising weights to equivalent binary (W_x, b_x).

    Given W_s from the {-1,+1} model, returns (W_x, b_x) such that
    both models define the same probability distribution (up to a constant).
    """
    W_x = 4.0 * W_s
    b_x = -2.0 * W_s.sum(axis=1)
    return W_x, b_x


def binary_to_ising(W_x, b_x):
    """Convert binary (W_x, b_x) to equivalent Ising weights W_s.

    Returns W_s. Raises ValueError if b_x is inconsistent with the
    Ising model (which has no bias freedom -- b must equal -0.5 * W_x @ 1).
    """
    W_s = W_x / 4.0
    expected_b = -0.5 * W_x.sum(axis=1)
    if not np.allclose(b_x, expected_b):
        raise ValueError("b_x is not consistent with a zero-bias Ising model")
    return W_s


def state_ising_to_binary(s):
    """Convert {-1, +1} state to {0, 1}."""
    return (s + 1.0) / 2.0


def state_binary_to_ising(x):
    """Convert {0, 1} state to {-1, +1}."""
    return 2.0 * x - 1.0


# --- Core binary functions ---------------------------------------------------


def load_square_patterns(n_side=128, data_dir="data"):
    """Load square {0,1} patterns from cached JPEGs. Returns (patterns, N_SIDE, N).

    Source JPEGs are converted to {-1,+1} then mapped to {0,1}.
    """
    ising_patterns = load_patterns(n_side=n_side, data_dir=data_dir)
    patterns = [state_ising_to_binary(p) for p in ising_patterns]
    N = n_side * n_side
    print(f"Loaded {len(patterns)} patterns ({n_side}x{n_side})")
    return patterns, n_side, N


def hebbian_weights(patterns, N):
    """Hebbian weight matrix and bias for {0,1} patterns.

    W = (1/N) sum_k (x_k - mean)(x_k - mean)^T with zero diagonal.
    b = zeros(N).
    """
    patterns_arr = np.array(patterns)
    mean = patterns_arr.mean(axis=0)
    W = sum(np.outer(xi - mean, xi - mean) for xi in patterns) / N
    np.fill_diagonal(W, 0.0)
    b = np.zeros(N)
    return W, b


def snap_to_image(state, n_side, display_scale):
    """Render a {0,1} state vector as an upscaled grayscale PIL image."""
    pixels = (state.reshape(n_side, n_side) * 255).astype(np.uint8)
    return Image.fromarray(pixels, "L").resize(
        (n_side * display_scale, n_side * display_scale), Image.NEAREST
    )


def prob_stay_gamma(z, gamma):
    """1 / (1 + exp_gamma(z)) with exp_gamma(z) = [1 + gamma*z]_+^(1/gamma).

    Identical to the Ising version -- encoding-agnostic.
    """
    if abs(gamma) < 1e-12:
        return 1.0 / (1.0 + np.exp(z))
    inner = 1.0 + gamma * z
    if inner <= 0.0:
        return 1.0 if gamma > 0.0 else 0.0
    log_expg = np.log(inner) / gamma
    if log_expg > 700.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(log_expg))


def activation_approx(h, *, beta, gamma_0, energy, N, x_i=None):
    """Approximate (large-N) activation p(x_i=1|rest) = sigma(beta_eff * h).

    For binary neurons the local field already carries the right scale,
    so the activation is sigma(beta_eff * h) rather than sigma(2*beta_eff*h)
    as in the Ising version.
    """
    if gamma_0 == 0.0:
        return 1.0 / (1.0 + np.exp(-beta * h))
    denom = 1.0 - gamma_0 * energy / N
    beta_eff = beta / (denom if denom > 0 else 1e-8)
    return 1.0 / (1.0 + np.exp(-beta_eff * h))


def activation_exact(h, *, beta, gamma_0, energy, N, x_i):
    """Exact deformed conditional p(x_i=1|rest) for binary neurons.

    Analogous to paper eq. S2.5 but in {0,1} encoding. The energy change
    for flipping x_i is delta_E = -(2*x_i - 1) * h, so the 'stay' argument
    to prob_stay_gamma is z = -beta_eff * (2*x_i - 1) * h.
    """
    if gamma_0 == 0.0:
        return 1.0 / (1.0 + np.exp(-beta * h))
    denom = 1.0 - gamma_0 * energy / N
    beta_eff = beta / (denom if denom > 0 else 1e-8)
    gamma_u = gamma_0 / (N * beta)
    z = -beta_eff * (2.0 * x_i - 1.0) * h
    prob_stay = prob_stay_gamma(z, gamma_u)
    return prob_stay if x_i > 0.5 else 1.0 - prob_stay


_ACTIVATIONS = {"exact": activation_exact, "approx": activation_approx}


def iter_curved_glauber(pattern, W, b, *, beta, gamma_0, n_steps,
                        snapshot_interval, noise_frac, rng, activation="exact"):
    """Generator of curved-Glauber state snapshots for {0,1} binary neurons.

    Yields a fresh `state.copy()` first for the post-noise initial state,
    then once per step that is a multiple of `snapshot_interval`.
    """
    try:
        act_fn = _ACTIVATIONS[activation]
    except KeyError as exc:
        raise ValueError(
            f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
        ) from exc

    N = pattern.size
    state = pattern.copy()
    # Noise: flip 0<->1
    flip_idx = rng.choice(N, size=int(noise_frac * N), replace=False)
    state[flip_idx] = 1.0 - state[flip_idx]

    # Initial energy: E = -0.5 x^T W x - b^T x
    energy = -0.5 * state @ W @ state - b @ state

    yield state.copy()
    for step in range(1, n_steps + 1):
        i = rng.integers(N)
        h = W[i] @ state + b[i]

        prob_one = act_fn(
            h, beta=beta, gamma_0=gamma_0, energy=energy, N=N, x_i=state[i],
        )
        new_val = 1.0 if rng.random() < prob_one else 0.0

        # Incremental energy update
        if new_val != state[i]:
            delta = new_val - state[i]
            energy -= delta * h
            state[i] = new_val

        if step % snapshot_interval == 0:
            yield state.copy()


def run_curved_glauber(pattern, W, b, *, beta, gamma_0, n_steps,
                       snapshot_interval, noise_frac, rng, activation="exact"):
    """Run a curved-Glauber Markov chain for {0, 1} binary neurons.

    Returns a list of state snapshots. Thin wrapper around `iter_curved_glauber`.
    """
    return list(iter_curved_glauber(
        pattern, W, b, beta=beta, gamma_0=gamma_0, n_steps=n_steps,
        snapshot_interval=snapshot_interval, noise_frac=noise_frac, rng=rng,
        activation=activation,
    ))


if __name__ == "__main__":
    # --- Parameters ---
    NOISE_FRAC = 0.30
    N_STEPS = 80000
    SNAPSHOT_INTERVAL = 500
    DISPLAY_SCALE = 8
    beta = 3.0
    gamma_0 = -1.2
    RNG = np.random.default_rng(42)

    from curvednet import (
        load_square_patterns as ising_load,
        hebbian_weights as ising_hebbian,
    )

    ising_patterns, N_SIDE, N = ising_load()
    W_s = ising_hebbian(ising_patterns, N)
    W, b = ising_to_binary(W_s)
    patterns = [state_ising_to_binary(p) for p in ising_patterns]

    frames = []
    for pat in patterns:
        snapshots = run_curved_glauber(
            pat, W, b,
            beta=beta, gamma_0=gamma_0,
            n_steps=N_STEPS, snapshot_interval=SNAPSHOT_INTERVAL,
            noise_frac=NOISE_FRAC, rng=RNG,
        )
        frames.extend(snap_to_image(s, N_SIDE, DISPLAY_SCALE) for s in snapshots)

    out_path = "fig/curved_recall_binary.gif"
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"Saved {out_path} ({len(frames)} frames, gamma_0={gamma_0}, beta={beta})")
