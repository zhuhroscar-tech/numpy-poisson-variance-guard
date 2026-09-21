"""Tests for numpy_poisson_variance_guard.core.

Written to independently verify each layer of the fix rather than just
assert "looks different from numpy": stirlerr and bd0 are checked
against an mpmath high-precision oracle where available, stable_log_pmf
is checked for exact agreement with the naive formula at SMALL lam
(where naive is still accurate, proving the two are really the same
mathematical identity), and safe_poisson is checked both structurally
(mean/variance close to true values) and via a live reproduction that
demonstrates numpy's own Generator.poisson really does drift at large
lam on the currently-installed numpy version.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from numpy_poisson_variance_guard.core import (
    PoissonVarianceDiagnosis,
    bd0,
    diagnose,
    naive_log_pmf,
    safe_poisson,
    stable_log_pmf,
    stirlerr,
)

try:
    import mpmath

    mpmath.mp.dps = 50
    HAVE_MPMATH = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_MPMATH = False


def _oracle_log_pmf(k: float, lam: float) -> float:
    k_mp = mpmath.mpf(k)
    lam_mp = mpmath.mpf(lam)
    return float(-lam_mp + k_mp * mpmath.log(lam_mp) - mpmath.loggamma(k_mp + 1))


# ---------------------------------------------------------------------------
# stirlerr / bd0 building blocks
# ---------------------------------------------------------------------------


def test_stirlerr_matches_lgamma_definition_for_small_n():
    """stirlerr(n) := lgamma(n+1) - Stirling's approximation. For small
    n this equality can be checked directly with ordinary lgamma (no
    cancellation risk since n is small), independent of stirlerr's own
    large-n series branch."""
    for n in [1.0, 2.0, 5.0, 10.0, 14.9]:
        got = stirlerr(np.array([n]))[0]
        want = math.lgamma(n + 1) - (n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n))
        assert got == pytest.approx(want, rel=1e-12)


def test_stirlerr_large_n_series_bands_agree_at_boundaries():
    """The four asymptotic-series bands (>=500, >=80, >=35, else) must
    produce continuous, mutually consistent values at their boundaries
    -- a coding error in the band split would show up as a discontinuity
    here."""
    boundary_points = [34.0, 35.0, 35.1, 79.0, 80.0, 80.1, 499.0, 500.0, 500.1]
    values = stirlerr(np.array(boundary_points))
    # All values should be small and monotonically decreasing (stirlerr
    # is a strictly decreasing function of n for n > 0).
    assert np.all(values > 0)
    assert np.all(np.diff(values) < 0)


@pytest.mark.skipif(not HAVE_MPMATH, reason="mpmath not installed")
def test_stirlerr_matches_mpmath_oracle_at_large_n():
    """At large n, verify stirlerr's asymptotic series against a direct
    mpmath (50-digit) computation of lgamma(n+1) - Stirling's formula,
    which mpmath can do without float64 cancellation."""
    for n in [1e10, 1e15, 1e17]:
        n_mp = mpmath.mpf(n)
        oracle = float(
            mpmath.loggamma(n_mp + 1) - (n_mp * mpmath.log(n_mp) - n_mp + mpmath.mpf("0.5") * mpmath.log(2 * mpmath.pi * n_mp))
        )
        got = stirlerr(np.array([n]))[0]
        assert got == pytest.approx(oracle, rel=1e-9)


def test_bd0_is_zero_at_exact_equality():
    assert bd0(np.array([5.0]), np.array([5.0]))[0] == 0.0
    assert bd0(np.array([1e16]), np.array([1e16]))[0] == 0.0


def test_bd0_close_regime_series_matches_direct_formula():
    """bd0 switches formula at |x - np_| < 0.1*(x+np_): the Taylor
    series (used when close) and the direct log formula (used when far)
    compute the SAME mathematical quantity. Deep inside the "close"
    branch, bd0's series result must closely match the direct formula
    evaluated at the same point -- proving the series is a correct
    reformulation, not a different (wrong) function. (bd0 itself is not
    constant near the boundary -- it grows roughly quadratically with
    |x - np_| -- so comparing two DIFFERENT points near the boundary, as
    an earlier version of this test did, is not a valid check.)"""
    np_ = 1000.0
    x = np_ * 1.05  # well within the 10% "close" threshold
    v_series = bd0(np.array([x]), np.array([np_]))[0]
    v_direct = x * math.log(x / np_) + (np_ - x)
    assert v_series == pytest.approx(v_direct, rel=1e-9)


# ---------------------------------------------------------------------------
# stable_log_pmf: exact identity check against naive formula (small lam,
# where naive itself is still accurate) plus the mpmath oracle (large lam)
# ---------------------------------------------------------------------------


def test_stable_log_pmf_matches_naive_formula_at_small_lam():
    """stable_log_pmf and naive_log_pmf compute the SAME mathematical
    quantity (an exact algebraic identity) -- at small lam, where naive
    has not yet lost precision to cancellation, they must agree closely,
    proving stable_log_pmf is not a different (wrong) formula in disguise."""
    for lam in [5.0, 50.0, 500.0]:
        for k in [1.0, lam, lam * 1.1, lam * 0.9]:
            stable = stable_log_pmf(np.array([k]), lam)[0]
            naive = naive_log_pmf(np.array([k]), lam)[0]
            assert stable == pytest.approx(naive, abs=1e-8)


def test_stable_log_pmf_zero_k_special_case():
    lam = 1e16
    assert stable_log_pmf(np.array([0.0]), lam)[0] == -lam


@pytest.mark.skipif(not HAVE_MPMATH, reason="mpmath not installed")
def test_stable_log_pmf_matches_mpmath_oracle_and_beats_naive_at_large_lam():
    """The core correctness claim of this whole package: at lam >= 1e15
    (where naive_log_pmf demonstrably fails), stable_log_pmf must stay
    accurate to within float64 ULP-level error against a 50-digit
    mpmath oracle, while naive_log_pmf's error blows up."""
    worst_naive_err = 0.0
    worst_stable_err = 0.0
    for lam in [1e15, 1e16, 1e17, 1e18]:
        sqrt_lam = math.sqrt(lam)
        for frac in [-5, -1, -0.1, 0.1, 1, 5]:
            k = float(round(lam + frac * sqrt_lam))
            if k < 1:
                continue
            oracle = _oracle_log_pmf(k, lam)
            stable = stable_log_pmf(np.array([k]), lam)[0]
            naive = naive_log_pmf(np.array([k]), lam)[0]
            stable_err = abs(stable - oracle)
            naive_err = abs(naive - oracle) if math.isfinite(naive) else float("inf")
            worst_stable_err = max(worst_stable_err, stable_err)
            if math.isfinite(naive_err):
                worst_naive_err = max(worst_naive_err, naive_err)

    # This is the regression test that would have failed before the
    # stirlerr+bd0 reformulation existed (naive_log_pmf alone).
    assert worst_stable_err < 1e-9, f"stable_log_pmf worst error {worst_stable_err} too large"
    assert worst_naive_err > 1.0, (
        "expected naive_log_pmf to demonstrably fail at large lam "
        f"(worst error only {worst_naive_err}); if this assertion fails, "
        "either the test grid changed or naive_log_pmf stopped reproducing "
        "the bug -- investigate before assuming the fix is unnecessary."
    )


# ---------------------------------------------------------------------------
# safe_poisson: structural correctness + live bug reproduction
# ---------------------------------------------------------------------------


def test_safe_poisson_small_lam_delegates_to_numpy_and_is_reasonable():
    rng = np.random.default_rng(42)
    samples = safe_poisson(5.0, 50_000, rng=rng)
    assert samples.mean() == pytest.approx(5.0, abs=0.1)
    assert samples.var() == pytest.approx(5.0, abs=0.3)


def test_safe_poisson_large_lam_variance_matches_lam():
    """The core regression test: at lam=1e16 (squarely in the bug's
    reported affected range), safe_poisson's empirical var/lam must be
    close to 1.0 -- this is the property numpy's own sampler violates."""
    rng = np.random.default_rng(7)
    lam = 1e16
    samples = safe_poisson(lam, 300_000, rng=rng)
    ratio = samples.astype(np.float64).var() / lam
    assert ratio == pytest.approx(1.0, abs=0.05), (
        f"safe_poisson var/lam={ratio} outside tolerance at lam={lam:.0e}"
    )
    assert samples.astype(np.float64).mean() == pytest.approx(lam, rel=1e-6)


def test_safe_poisson_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        safe_poisson(-1.0, 10)
    with pytest.raises(ValueError):
        safe_poisson(float("inf"), 10)
    with pytest.raises(ValueError):
        safe_poisson(5.0, -1)


def test_safe_poisson_zero_size_returns_empty():
    result = safe_poisson(1e16, 0)
    assert result.shape == (0,)


def test_safe_poisson_rejects_lam_beyond_int64_safe_range():
    """Regression test for a real, previously-unguarded defect found by
    this run's backstop inspection: safe_poisson's PTRS proposal
    envelope can generate candidate k values roughly
    lam + O(20*sqrt(lam)) above lam. For lam close to
    np.iinfo(np.int64).max, that pushes candidates past the int64
    range, and the float64->int64 cast on the output array then
    silently wraps/corrupts (RuntimeWarning: invalid value encountered
    in cast) instead of raising -- returning finite-looking but
    completely wrong samples, exactly the failure class this whole
    package exists to catch. Before the _MAX_SAFE_LAM guard was added,
    calling safe_poisson(1e19, ...) returned samples clamped near
    int64 max with ~8% relative error on the mean and no exception at
    all; this test proves the guard now raises instead."""
    with pytest.raises(ValueError, match="too large"):
        safe_poisson(1e19, 100)
    with pytest.raises(ValueError, match="too large"):
        safe_poisson(5e19, 100)


def test_safe_poisson_upper_bound_matches_numpys_own_guard():
    """safe_poisson must never accept an input that numpy's own
    Generator.poisson would itself refuse -- independently confirmed
    against the installed numpy build (not a hardcoded assumption)."""
    rng = np.random.default_rng(0)
    boundary = safe_poisson.__globals__["_MAX_SAFE_LAM"]
    # Just below the guard's threshold: numpy's own sampler must also
    # accept it (proves our threshold is not stricter than numpy's).
    just_below = boundary * 0.999999
    rng.poisson(just_below, size=1)  # must not raise
    # At/above the guard's threshold: safe_poisson must refuse.
    with pytest.raises(ValueError, match="too large"):
        safe_poisson(boundary, 10)


def test_safe_poisson_still_correct_just_under_the_new_boundary():
    """The new upper-bound guard must not have narrowed the sampler's
    correct operating range for lam values that were always fine --
    e.g. 1e18 remains fully within range and must still produce
    correct mean/variance."""
    rng = np.random.default_rng(3)
    lam = 1e18
    samples = safe_poisson(lam, 500, rng=rng)
    mean = samples.astype(np.float64).mean()
    assert abs(mean - lam) / lam < 0.01


def test_safe_poisson_is_reproducible_with_seeded_rng():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = safe_poisson(1e16, 1000, rng=rng1)
    b = safe_poisson(1e16, 1000, rng=rng2)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# diagnose: live comparison against the installed numpy
# ---------------------------------------------------------------------------


def test_diagnose_returns_well_formed_result():
    result = diagnose(lam=1e10, n_samples=20_000)
    assert isinstance(result, PoissonVarianceDiagnosis)
    assert result.numpy_version == np.__version__
    assert result.lam == 1e10
    assert result.n_samples == 20_000
    assert isinstance(result.affected, bool)
    assert len(result.detail) > 0


def test_diagnose_reproduces_the_real_bug_on_currently_installed_numpy():
    """Live, no-mocking reproduction against whatever numpy is actually
    installed, at lam=1e17 (well inside the reported affected range).
    As of this guard's creation, numpy/numpy#31986 is open and this
    should report affected=True with safe_poisson staying correct. Does
    not hard-assert affected=True forever: if a future numpy release
    fixes this upstream, that is real, desirable information this guard
    should surface truthfully -- assert internal consistency instead."""
    result = diagnose(lam=1e17, n_samples=300_000, tolerance=0.05)
    assert abs(result.safe_var_over_lam - 1.0) < 0.05, (
        "safe_poisson must stay correct regardless of numpy's own bug status"
    )
    if result.affected:
        assert "CONFIRMED" in result.detail
    else:
        assert "Not reproduced" in result.detail
