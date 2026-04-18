"""Curved Hopfield network (Aguilera et al., Nat Commun 2025).

Higher-order interactions via a deformation parameter gamma_0 that makes
the effective inverse temperature state-dependent:

    beta'(x) = beta / [1 - gamma_0 * E(x) / N]_+

For gamma_0 < 0: self-regulating annealing accelerates memory retrieval.
For gamma_0 = 0: recovers the standard Hopfield/Ising model.
For gamma_0 > 0: decelerates dynamics, more robust retrieval.

Two single-spin activation rules are provided:
- `activation_exact`  — exact deformed conditional (paper eq. S2.5).
- `activation_approx` — large-N Glauber approximation (paper eq. S2.7).
`run_curved_glauber(..., activation="exact")` dispatches between them;
the default is the exact rule.

Module-level exports (`load_square_patterns`, `hebbian_weights`,
`run_curved_glauber`, `iter_curved_glauber`, `snap_to_image`,
`activation_exact`, `activation_approx`, `prob_stay_gamma`) are importable
with no side effects. `iter_curved_glauber` is a generator variant that
yields state snapshots one at a time so callers can stream-accumulate
moments without materialising the full trajectory; `run_curved_glauber` is
a thin `list(iter_curved_glauber(...))` wrapper. Display + GIF output runs
only under `if __name__ == "__main__":`.
"""

import sys

import numpy as np
from PIL import Image

from generate_patterns import load_patterns


def load_square_patterns(n_side=128, data_dir="data"):
    """Load square {-1,+1} patterns from cached JPEGs. Returns (patterns, N_SIDE, N)."""
    patterns = load_patterns(n_side=n_side, data_dir=data_dir)
    N = n_side * n_side
    print(f"Loaded {len(patterns)} patterns ({n_side}x{n_side})")
    return patterns, n_side, N


def hebbian_weights(patterns, N):
    """Hebbian weight matrix W = (1/N) sum_k x_k x_k^T with zero diagonal."""
    W = sum(np.outer(xi, xi) for xi in patterns) / N
    np.fill_diagonal(W, 0.0)
    return W


def snap_to_image(state, n_side, display_scale):
    """Render a +-1 state vector as an upscaled grayscale PIL image."""
    pixels = ((state.reshape(n_side, n_side) + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(pixels, "L").resize(
        (n_side * display_scale, n_side * display_scale), Image.NEAREST
    )


def prob_stay_gamma(z, gamma):
    """1 / (1 + exp_gamma(z)) with exp_gamma(z) = [1 + gamma*z]_+^(1/gamma).

    The [.]_+ clipping forbids transitions into states with p_gamma(x) = 0:
    inner <= 0 maps to exp_gamma -> 0 (gamma > 0) or +inf (gamma < 0).
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


def activation_approx(h, *, beta, gamma_0, energy, N, s_i=None):
    """Approximate (large-N) activation p(s_i=+1|rest) = sigma(2*beta_eff*h).

    Paper eq. S2.7: beta_eff = beta / [1 - gamma_0 * E / N]_+. No s_i
    dependence (accepted for a uniform signature, unused).
    """
    if gamma_0 == 0.0:
        return 0.5 * (1.0 + np.tanh(beta * h))
    denom = 1.0 - gamma_0 * energy / N
    beta_eff = beta / (denom if denom > 0 else 1e-8)
    return 0.5 * (1.0 + np.tanh(beta_eff * h))


def activation_exact(h, *, beta, gamma_0, energy, N, s_i):
    """Exact deformed conditional p(s_i=+1|rest), paper eq. S2.5.

    Evaluated as the 'prob-stay' logistic of the deformed exponential with
    beta' = beta / [1 - gamma_0 * E / N]_+ at the current state, then mapped
    to p(s_i=+1) by flipping when the current spin is -1. Reduces to
    sigma(2*beta*h) at gamma_0 = 0.
    """
    if gamma_0 == 0.0:
        return 0.5 * (1.0 + np.tanh(beta * h))
    denom = 1.0 - gamma_0 * energy / N
    beta_eff = beta / (denom if denom > 0 else 1e-8)
    gamma_u = gamma_0 / (N * beta)
    prob_stay = prob_stay_gamma(-2.0 * beta_eff * s_i * h, gamma_u)
    return prob_stay if s_i > 0 else 1.0 - prob_stay


_ACTIVATIONS = {"exact": activation_exact, "approx": activation_approx}


def iter_curved_glauber(pattern, W, *, beta, gamma_0, n_steps, snapshot_interval,
                        noise_frac, rng, activation="exact"):
    """Generator of curved-Glauber state snapshots.

    Yields a fresh `state.copy()` first for the post-noise initial state,
    then once per step that is a multiple of `snapshot_interval`. Peak
    memory is O(N) (the one state array plus the per-yield copy), which
    lets callers stream-accumulate moments without materialising the full
    trajectory.

    See `run_curved_glauber` for the semantics of `activation`.
    """
    try:
        act_fn = _ACTIVATIONS[activation]
    except KeyError as exc:
        raise ValueError(
            f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
        ) from exc

    N = pattern.size
    state = pattern.copy()
    state[rng.choice(N, size=int(noise_frac * N), replace=False)] *= -1

    # Initial energy: E = -0.5 * x^T W x
    energy = -0.5 * state @ W @ state

    yield state.copy()
    for step in range(1, n_steps + 1):
        i = rng.integers(N)
        h = W[i] @ state

        prob_plus = act_fn(
            h, beta=beta, gamma_0=gamma_0, energy=energy, N=N, s_i=state[i],
        )
        new_val = 1.0 if rng.random() < prob_plus else -1.0

        # Incremental energy update
        if new_val != state[i]:
            delta = new_val - state[i]
            energy -= delta * h
            state[i] = new_val

        if step % snapshot_interval == 0:
            yield state.copy()


def run_curved_glauber(pattern, W, *, beta, gamma_0, n_steps, snapshot_interval,
                       noise_frac, rng, activation="exact"):
    """Run a curved-Glauber Markov chain from a noisy version of `pattern`.

    Single-spin-flip updates of an Ising state in {-1, +1}^N using one of two
    conditional rules:

    - activation="exact"  (default) — eq. S2.5, logistic of the deformed
      exponential.
    - activation="approx"           — eq. S2.7, sigma(2*beta_eff*h); large-N
      limit of the exact rule.

    Returns a list of raw state snapshots (numpy arrays) captured every
    `snapshot_interval` steps. Thin wrapper around `iter_curved_glauber`.
    """
    return list(iter_curved_glauber(
        pattern, W, beta=beta, gamma_0=gamma_0, n_steps=n_steps,
        snapshot_interval=snapshot_interval, noise_frac=noise_frac, rng=rng,
        activation=activation,
    ))


if __name__ == "__main__":
    # --- Parameters ---
    NOISE_FRAC = 0.30
    N_STEPS = 80000
    SNAPSHOT_INTERVAL = 500
    DISPLAY_SCALE = 8
    beta = 3.0          # inverse temperature
    gamma_0 = -1.2      # curvature: <0 accelerates, 0 = standard, >0 decelerates
    RNG = np.random.default_rng(42)

    patterns, N_SIDE, N = load_square_patterns()
    W = hebbian_weights(patterns, N)

    frames = []
    for pat in patterns:
        snapshots = run_curved_glauber(
            pat, W,
            beta=beta, gamma_0=gamma_0,
            n_steps=N_STEPS, snapshot_interval=SNAPSHOT_INTERVAL,
            noise_frac=NOISE_FRAC, rng=RNG,
        )
        frames.extend(snap_to_image(s, N_SIDE, DISPLAY_SCALE) for s in snapshots)

    out_path = "fig/curved_recall.gif"
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"Saved {out_path} ({len(frames)} frames, gamma_0={gamma_0}, beta={beta})")
