"""Tests for curvednet.py (activation functions, Hebbian weights, Glauber step)."""

import numpy as np
import pytest

from curvednet import (
    activation_approx,
    activation_exact,
    hebbian_weights,
    prob_stay_gamma,
    run_curved_glauber,
    snap_to_image,
)


# --- prob_stay_gamma ---------------------------------------------------------


@pytest.mark.parametrize("z", [-2.5, -0.1, 0.0, 0.7, 3.0])
def test_prob_stay_gamma_zero_is_sigmoid(z):
    """At gamma = 0, prob_stay reduces to the ordinary sigmoid 1/(1+e^z)."""
    assert prob_stay_gamma(z, 0.0) == pytest.approx(1.0 / (1.0 + np.exp(z)))


def test_prob_stay_gamma_positive_boundary():
    """Positive gamma with 1+gamma*z <= 0: exp_gamma -> 0, prob_stay -> 1."""
    assert prob_stay_gamma(-5.0, gamma=0.5) == 1.0


def test_prob_stay_gamma_negative_boundary():
    """Negative gamma with 1+gamma*z <= 0: density forbidden, prob_stay -> 0."""
    assert prob_stay_gamma(5.0, gamma=-0.5) == 0.0


@pytest.mark.parametrize("gamma", [-0.3, -0.05, 0.05, 0.3])
def test_prob_stay_gamma_in_unit_interval(gamma):
    for z in np.linspace(-4.0, 4.0, 21):
        p = prob_stay_gamma(z, gamma)
        assert 0.0 <= p <= 1.0


# --- activation_approx (paper S2.7) -----------------------------------------


@pytest.mark.parametrize("h", [-1.2, -0.3, 0.0, 0.4, 1.5])
def test_activation_approx_gamma_zero_is_sigmoid(h):
    beta = 1.5
    assert activation_approx(h, beta=beta, gamma_0=0.0, energy=-100.0, N=1024) == \
        pytest.approx(0.5 * (1.0 + np.tanh(beta * h)))


def test_activation_approx_half_at_zero_field():
    assert activation_approx(0.0, beta=2.0, gamma_0=-0.5, energy=-200.0, N=1024) == \
        pytest.approx(0.5)


def test_activation_approx_monotonic_in_h():
    kwargs = dict(beta=1.5, gamma_0=-0.3, energy=-100.0, N=1024)
    p_lo = activation_approx(-0.5, **kwargs)
    p_mid = activation_approx(0.0, **kwargs)
    p_hi = activation_approx(0.5, **kwargs)
    assert p_lo < p_mid < p_hi


# --- activation_exact (paper S2.5) ------------------------------------------


@pytest.mark.parametrize("h", [-1.0, 0.0, 0.7])
@pytest.mark.parametrize("s_i", [-1.0, 1.0])
def test_activation_exact_gamma_zero_is_sigmoid(h, s_i):
    """At gamma = 0 the exact conditional reduces to sigma(2*beta*h), independent of s_i."""
    beta = 1.5
    assert activation_exact(h, beta=beta, gamma_0=0.0, energy=-100.0, N=1024, s_i=s_i) == \
        pytest.approx(0.5 * (1.0 + np.tanh(beta * h)))


@pytest.mark.parametrize("s_i", [-1.0, 1.0])
def test_activation_exact_half_at_zero_field(s_i):
    assert activation_exact(0.0, beta=2.0, gamma_0=-0.5, energy=-200.0, N=1024, s_i=s_i) == \
        pytest.approx(0.5)


def test_activation_exact_agrees_with_approx_at_large_N():
    """Exact - approx = O(1/N); at large N they should be very close."""
    N = 100000
    h = 0.4
    p_ex = activation_exact(h, beta=1.5, gamma_0=-0.3, energy=-50000.0, N=N, s_i=1.0)
    p_ap = activation_approx(h, beta=1.5, gamma_0=-0.3, energy=-50000.0, N=N)
    assert abs(p_ex - p_ap) < 1e-4


def test_activation_exact_differs_from_approx_at_small_N():
    """At small N and nontrivial curvature, the two formulas must give different probs."""
    N = 100
    h = 0.3
    p_ex = activation_exact(h, beta=1.5, gamma_0=-0.5, energy=-50.0, N=N, s_i=1.0)
    p_ap = activation_approx(h, beta=1.5, gamma_0=-0.5, energy=-50.0, N=N)
    assert abs(p_ex - p_ap) > 1e-4


def test_activation_exact_self_consistency():
    """p(s=+1|rest) should not depend on which current state we express it from.

    If we query from state with s_i=+1 (energy E_+) vs state with s_i=-1
    (energy E_- = E_+ + 2h), both must return the same p(s=+1|rest).
    """
    h = 0.5
    E_plus = -100.0
    E_minus = E_plus + 2.0 * h
    kwargs = dict(beta=1.5, gamma_0=-0.3, N=1024)
    p_from_plus = activation_exact(h, energy=E_plus, s_i=1.0, **kwargs)
    p_from_minus = activation_exact(h, energy=E_minus, s_i=-1.0, **kwargs)
    assert p_from_plus == pytest.approx(p_from_minus)


# --- hebbian_weights --------------------------------------------------------


def test_hebbian_zero_diagonal():
    rng = np.random.default_rng(0)
    patterns = [rng.choice([-1.0, 1.0], size=8) for _ in range(3)]
    W = hebbian_weights(patterns, N=8)
    assert np.all(np.diag(W) == 0.0)


def test_hebbian_symmetric():
    rng = np.random.default_rng(1)
    patterns = [rng.choice([-1.0, 1.0], size=10) for _ in range(2)]
    W = hebbian_weights(patterns, N=10)
    assert np.allclose(W, W.T)


# --- run_curved_glauber -----------------------------------------------------


def _toy_run(activation, *, gamma_0=0.0, seed=0):
    N = 16
    pattern = np.random.default_rng(99).choice([-1.0, 1.0], size=N)
    patterns = [pattern]
    W = hebbian_weights(patterns, N)
    rng = np.random.default_rng(seed)
    return run_curved_glauber(
        pattern, W, beta=1.0, gamma_0=gamma_0,
        n_steps=200, snapshot_interval=50, noise_frac=0.3, rng=rng,
        activation=activation,
    )


def test_run_curved_glauber_state_stays_in_pm1():
    for act in ("exact", "approx"):
        snaps = _toy_run(act)
        for s in snaps:
            assert set(np.unique(s).tolist()) <= {-1.0, 1.0}


def test_run_curved_glauber_snapshot_count():
    snaps = _toy_run("exact")
    # initial snapshot + 200/50 = 4 captured along the way
    assert len(snaps) == 5


def test_run_curved_glauber_invalid_activation():
    with pytest.raises(ValueError, match="activation must be"):
        _toy_run("bogus")


def test_run_curved_glauber_exact_and_approx_agree_at_gamma_zero():
    """With gamma_0 = 0 both rules reduce to the same sigmoid, so with the same
    seed the Markov chains must trace identical trajectories."""
    snaps_ex = _toy_run("exact", gamma_0=0.0, seed=7)
    snaps_ap = _toy_run("approx", gamma_0=0.0, seed=7)
    for a, b in zip(snaps_ex, snaps_ap):
        assert np.array_equal(a, b)


# --- snap_to_image ----------------------------------------------------------


def test_snap_to_image_size_and_mode():
    state = np.random.default_rng(0).choice([-1.0, 1.0], size=16)
    img = snap_to_image(state, n_side=4, display_scale=3)
    assert img.size == (12, 12)
    assert img.mode == "L"
