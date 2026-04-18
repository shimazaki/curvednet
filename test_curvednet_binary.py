"""Tests for curvednet_binary.py (binary {0,1} encoding + Ising<->Binary translation)."""

import numpy as np
import pytest

from curvednet_binary import (
    activation_approx,
    activation_exact,
    binary_to_ising,
    hebbian_weights,
    ising_to_binary,
    iter_curved_glauber,
    prob_stay_gamma,
    run_curved_glauber,
    snap_to_image,
    state_binary_to_ising,
    state_ising_to_binary,
)


# --- State conversion -------------------------------------------------------


def test_state_roundtrip_ising_binary():
    s = np.array([-1.0, 1.0, -1.0, 1.0])
    assert np.allclose(state_binary_to_ising(state_ising_to_binary(s)), s)


def test_state_roundtrip_binary_ising():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.allclose(state_ising_to_binary(state_binary_to_ising(x)), x)


# --- ising_to_binary / binary_to_ising --------------------------------------


def test_translation_roundtrip():
    """binary_to_ising(ising_to_binary(W_s)) recovers original W_s."""
    rng = np.random.default_rng(42)
    N = 8
    W_s = rng.standard_normal((N, N))
    W_s = (W_s + W_s.T) / 2
    np.fill_diagonal(W_s, 0.0)

    W_x, b_x = ising_to_binary(W_s)
    W_s_recovered = binary_to_ising(W_x, b_x)
    assert np.allclose(W_s_recovered, W_s)


def test_binary_to_ising_rejects_inconsistent_bias():
    """If b_x doesn't satisfy the Ising constraint, raise ValueError."""
    N = 4
    W_x = np.ones((N, N))
    np.fill_diagonal(W_x, 0.0)
    b_x = np.array([999.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not consistent"):
        binary_to_ising(W_x, b_x)


def test_ising_to_binary_weight_scale():
    """W_x = 4 * W_s."""
    W_s = np.array([[0.0, 1.0], [1.0, 0.0]])
    W_x, _ = ising_to_binary(W_s)
    assert np.allclose(W_x, 4.0 * W_s)


def test_ising_to_binary_bias():
    """b_i = -2 * sum_j W_s[i,j]."""
    W_s = np.array([[0.0, 0.5, -0.3],
                     [0.5, 0.0, 0.2],
                     [-0.3, 0.2, 0.0]])
    _, b_x = ising_to_binary(W_s)
    expected = -2.0 * W_s.sum(axis=1)
    assert np.allclose(b_x, expected)


# --- prob_stay_gamma ---------------------------------------------------------


@pytest.mark.parametrize("z", [-2.5, -0.1, 0.0, 0.7, 3.0])
def test_prob_stay_gamma_zero_is_sigmoid(z):
    assert prob_stay_gamma(z, 0.0) == pytest.approx(1.0 / (1.0 + np.exp(z)))


@pytest.mark.parametrize("gamma", [-0.3, -0.05, 0.05, 0.3])
def test_prob_stay_gamma_in_unit_interval(gamma):
    for z in np.linspace(-4.0, 4.0, 21):
        p = prob_stay_gamma(z, gamma)
        assert 0.0 <= p <= 1.0


# --- activation_approx ------------------------------------------------------


@pytest.mark.parametrize("h", [-1.2, -0.3, 0.0, 0.4, 1.5])
def test_activation_approx_gamma_zero_is_sigmoid(h):
    beta = 1.5
    expected = 1.0 / (1.0 + np.exp(-beta * h))
    assert activation_approx(h, beta=beta, gamma_0=0.0, energy=-100.0, N=1024) == \
        pytest.approx(expected)


def test_activation_approx_half_at_zero_field():
    assert activation_approx(0.0, beta=2.0, gamma_0=-0.5, energy=-200.0, N=1024) == \
        pytest.approx(0.5)


def test_activation_approx_monotonic_in_h():
    kwargs = dict(beta=1.5, gamma_0=-0.3, energy=-100.0, N=1024)
    p_lo = activation_approx(-0.5, **kwargs)
    p_mid = activation_approx(0.0, **kwargs)
    p_hi = activation_approx(0.5, **kwargs)
    assert p_lo < p_mid < p_hi


# --- activation_exact --------------------------------------------------------


@pytest.mark.parametrize("h", [-1.0, 0.0, 0.7])
@pytest.mark.parametrize("x_i", [0.0, 1.0])
def test_activation_exact_gamma_zero_is_sigmoid(h, x_i):
    """At gamma=0 the exact conditional reduces to sigma(beta*h), independent of x_i."""
    beta = 1.5
    expected = 1.0 / (1.0 + np.exp(-beta * h))
    assert activation_exact(h, beta=beta, gamma_0=0.0, energy=-100.0, N=1024, x_i=x_i) == \
        pytest.approx(expected)


@pytest.mark.parametrize("x_i", [0.0, 1.0])
def test_activation_exact_half_at_zero_field(x_i):
    assert activation_exact(0.0, beta=2.0, gamma_0=-0.5, energy=-200.0, N=1024, x_i=x_i) == \
        pytest.approx(0.5)


def test_activation_exact_agrees_with_approx_at_large_N():
    N = 100000
    h = 0.4
    p_ex = activation_exact(h, beta=1.5, gamma_0=-0.3, energy=-50000.0, N=N, x_i=1.0)
    p_ap = activation_approx(h, beta=1.5, gamma_0=-0.3, energy=-50000.0, N=N)
    assert abs(p_ex - p_ap) < 1e-4


def test_activation_exact_differs_from_approx_at_small_N():
    N = 100
    h = 0.3
    p_ex = activation_exact(h, beta=1.5, gamma_0=-0.5, energy=-50.0, N=N, x_i=1.0)
    p_ap = activation_approx(h, beta=1.5, gamma_0=-0.5, energy=-50.0, N=N)
    assert abs(p_ex - p_ap) > 1e-4


def test_activation_exact_self_consistency():
    """p(x=1|rest) same whether queried from x_i=0 or x_i=1.

    When x_i flips 1->0, the energy changes by delta_E = h (since delta=-1).
    So E_0 = E_1 + h.
    """
    h = 0.5
    E_at_1 = -100.0
    E_at_0 = E_at_1 + h  # energy when x_i=0 (flipped from 1: delta=-1, dE=-(-1)*h=h)
    kwargs = dict(beta=1.5, gamma_0=-0.3, N=1024)
    p_from_1 = activation_exact(h, energy=E_at_1, x_i=1.0, **kwargs)
    p_from_0 = activation_exact(h, energy=E_at_0, x_i=0.0, **kwargs)
    assert p_from_1 == pytest.approx(p_from_0)


# --- hebbian_weights ---------------------------------------------------------


def test_hebbian_zero_diagonal():
    rng = np.random.default_rng(0)
    patterns = [rng.choice([0.0, 1.0], size=8) for _ in range(3)]
    W, b = hebbian_weights(patterns, N=8)
    assert np.all(np.diag(W) == 0.0)


def test_hebbian_symmetric():
    rng = np.random.default_rng(1)
    patterns = [rng.choice([0.0, 1.0], size=10) for _ in range(2)]
    W, b = hebbian_weights(patterns, N=10)
    assert np.allclose(W, W.T)


def test_hebbian_bias_is_zero():
    rng = np.random.default_rng(2)
    patterns = [rng.choice([0.0, 1.0], size=8) for _ in range(3)]
    _, b = hebbian_weights(patterns, N=8)
    assert np.allclose(b, 0.0)


# --- run_curved_glauber ------------------------------------------------------


def _toy_run(activation, *, gamma_0=0.0, seed=0):
    N = 16
    pattern = np.random.default_rng(99).choice([0.0, 1.0], size=N)
    patterns = [pattern]
    W, b = hebbian_weights(patterns, N)
    rng = np.random.default_rng(seed)
    return run_curved_glauber(
        pattern, W, b, beta=1.0, gamma_0=gamma_0,
        n_steps=200, snapshot_interval=50, noise_frac=0.3, rng=rng,
        activation=activation,
    )


def test_run_curved_glauber_state_stays_in_01():
    for act in ("exact", "approx"):
        snaps = _toy_run(act)
        for s in snaps:
            assert set(np.unique(s).tolist()) <= {0.0, 1.0}


def test_run_curved_glauber_snapshot_count():
    snaps = _toy_run("exact")
    # initial snapshot + 200/50 = 4 captured along the way
    assert len(snaps) == 5


def test_run_curved_glauber_invalid_activation():
    with pytest.raises(ValueError, match="activation must be"):
        _toy_run("bogus")


def test_run_curved_glauber_exact_and_approx_agree_at_gamma_zero():
    """With gamma_0=0 both rules reduce to the same sigmoid, so with the same
    seed the Markov chains must trace identical trajectories."""
    snaps_ex = _toy_run("exact", gamma_0=0.0, seed=7)
    snaps_ap = _toy_run("approx", gamma_0=0.0, seed=7)
    for a, b in zip(snaps_ex, snaps_ap):
        assert np.array_equal(a, b)


def test_iter_and_run_equivalent():
    N = 16
    pattern = np.random.default_rng(99).choice([0.0, 1.0], size=N)
    W, b = hebbian_weights([pattern], N)

    rng_list = np.random.default_rng(123)
    snaps_list = run_curved_glauber(
        pattern, W, b, beta=1.0, gamma_0=-0.3, n_steps=200,
        snapshot_interval=50, noise_frac=0.3, rng=rng_list, activation="exact",
    )

    rng_iter = np.random.default_rng(123)
    snaps_iter = list(iter_curved_glauber(
        pattern, W, b, beta=1.0, gamma_0=-0.3, n_steps=200,
        snapshot_interval=50, noise_frac=0.3, rng=rng_iter, activation="exact",
    ))

    assert len(snaps_list) == len(snaps_iter)
    for a, b_ in zip(snaps_list, snaps_iter):
        assert np.array_equal(a, b_)


# --- Equivalence: Ising <-> Binary models produce same dynamics --------------


def test_ising_binary_equivalence_trajectory():
    """Run both models from equivalent initial states with the same RNG seed
    and translated parameters -> identical trajectory after state mapping."""
    from curvednet import (
        activation_exact as ising_activation_exact,
        hebbian_weights as ising_hebbian,
        run_curved_glauber as ising_run,
    )

    N = 16
    rng_pat = np.random.default_rng(77)
    s_pattern = rng_pat.choice([-1.0, 1.0], size=N)
    x_pattern = state_ising_to_binary(s_pattern)

    W_s = ising_hebbian([s_pattern], N)
    W_x, b_x = ising_to_binary(W_s)

    beta, gamma_0 = 1.0, 0.0  # gamma_0=0 so both activations are identical sigmoid
    seed = 42

    rng_ising = np.random.default_rng(seed)
    snaps_ising = ising_run(
        s_pattern, W_s, beta=beta, gamma_0=gamma_0,
        n_steps=200, snapshot_interval=50, noise_frac=0.3, rng=rng_ising,
        activation="exact",
    )

    rng_binary = np.random.default_rng(seed)
    snaps_binary = run_curved_glauber(
        x_pattern, W_x, b_x, beta=beta, gamma_0=gamma_0,
        n_steps=200, snapshot_interval=50, noise_frac=0.3, rng=rng_binary,
        activation="exact",
    )

    for s_snap, x_snap in zip(snaps_ising, snaps_binary):
        x_from_s = state_ising_to_binary(s_snap)
        assert np.allclose(x_from_s, x_snap), \
            f"Ising->binary state mismatch: {x_from_s} vs {x_snap}"


# --- snap_to_image -----------------------------------------------------------


def test_snap_to_image_size_and_mode():
    state = np.random.default_rng(0).choice([0.0, 1.0], size=16)
    img = snap_to_image(state, n_side=4, display_scale=3)
    assert img.size == (12, 12)
    assert img.mode == "L"
