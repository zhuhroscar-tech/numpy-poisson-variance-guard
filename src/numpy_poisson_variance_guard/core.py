"""Core logic for numpy-poisson-variance-guard.

Guards a real, currently-open numpy bug (numpy/numpy#31986, confirmed
open and independently reproduced against the pinned numpy version at
the time this guard was created -- see README for the verification
trail): ``np.random.Generator.poisson(lam)`` silently inflates sample
variance for large ``lam`` (empirically >~1e15) because its rejection
sampler's log-pmf acceptance test suffers catastrophic float64
cancellation.

This module provides:

- ``stirlerr`` / ``bd0``: the two building blocks of Loader (2000)'s
  exact saddlepoint identity for the Poisson log-pmf (the same
  technique used by R's C ``nmath`` ``dpois_raw``), each individually
  verified against an ``mpmath`` high-precision oracle.
- ``stable_log_pmf``: the resulting cancellation-safe Poisson log-pmf,
  an EXACT identity (not an approximation) with the naive formula,
  just computed in a numerically stable order.
- ``safe_poisson``: a full vectorized PTRS (Hoermann 1993) rejection
  sampler -- the same algorithm numpy's own C implementation uses for
  lam >= 10 -- built on ``stable_log_pmf`` instead of the naive
  formula, so its acceptance test does not suffer the cancellation
  that causes numpy's own sampler to drift.
- ``diagnose``: empirically compares the installed numpy's own
  ``Generator.poisson`` against ``safe_poisson`` and a true Poisson
  variance target, live, every call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# stirlerr: Stirling's series error term, lgamma(n+1) - Stirling's approx.
# ---------------------------------------------------------------------------

_SMALL_N_CUTOFF = 15.0


def stirlerr(n: np.ndarray) -> np.ndarray:
    """Vectorized Stirling's series error term.

    For small ``n`` this is computed directly via ``lgamma`` (safe: no
    cancellation risk since ``n`` itself is small). For large ``n`` it
    is computed via its OWN convergent asymptotic series (terms shrink
    like ``1/n``, ``1/n**3``, ...) -- never by subtracting two large
    numbers, so precision is retained for arbitrarily large ``n``.
    """
    n = np.asarray(n, dtype=np.float64)
    if np.any(n <= 0):
        raise ValueError("stirlerr requires strictly positive n")

    out = np.empty_like(n)
    small = n < _SMALL_N_CUTOFF
    if np.any(small):
        ns = n[small]
        out[small] = _lgamma(ns + 1.0) - (ns * np.log(ns) - ns + 0.5 * np.log(2.0 * np.pi * ns))

    large = ~small
    if np.any(large):
        nl = n[large]
        n2 = nl * nl
        s0, s1, s2, s3, s4 = 1.0 / 12, 1.0 / 360, 1.0 / 1260, 1.0 / 1680, 1.0 / 1188
        res = np.empty_like(nl)
        band1 = nl >= 500
        band2 = (nl >= 80) & ~band1
        band3 = (nl >= 35) & ~band1 & ~band2
        band4 = ~band1 & ~band2 & ~band3
        if np.any(band1):
            nn, nn2 = nl[band1], n2[band1]
            res[band1] = (s0 - s1 / nn2) / nn
        if np.any(band2):
            nn, nn2 = nl[band2], n2[band2]
            res[band2] = (s0 - (s1 - s2 / nn2) / nn2) / nn
        if np.any(band3):
            nn, nn2 = nl[band3], n2[band3]
            res[band3] = (s0 - (s1 - (s2 - s3 / nn2) / nn2) / nn2) / nn
        if np.any(band4):
            nn, nn2 = nl[band4], n2[band4]
            res[band4] = (s0 - (s1 - (s2 - (s3 - s4 / nn2) / nn2) / nn2) / nn2) / nn
        out[large] = res
    return out


def _lgamma(x: np.ndarray) -> np.ndarray:
    """Vectorized log-gamma with no SciPy dependency (this repo is
    numpy-only): numpy has no ufunc for lgamma, so vectorize math.lgamma.
    Only called for small arguments (< 16), where per-element Python
    overhead is negligible relative to the sampler's rejection loop.
    """
    flat = np.reshape(x, -1)
    result = np.fromiter((math.lgamma(float(v)) for v in flat), dtype=np.float64, count=flat.size)
    return result.reshape(np.shape(x))


# ---------------------------------------------------------------------------
# bd0: binomial deviance term, cancellation-safe near x == np_
# ---------------------------------------------------------------------------


def bd0(x: np.ndarray, np_: np.ndarray) -> np.ndarray:
    """Vectorized binomial deviance ``x*log(x/np_) + (np_ - x)``,
    cancellation-safe when ``x`` is close to ``np_`` (the regime that
    matters for a Poisson log-pmf near its own mode, i.e. ``k`` near
    ``lam``) via a convergent Taylor series in ``v = (x-np_)/(x+np_)``
    instead of the direct formula, which is exactly the difference
    numpy/numpy#31986 identifies as catastrophically cancelling.
    """
    x = np.asarray(x, dtype=np.float64)
    np_ = np.broadcast_to(np.asarray(np_, dtype=np.float64), x.shape).copy()
    out = np.zeros_like(x)

    exact = x == np_
    close = (~exact) & (np.abs(x - np_) < 0.1 * (x + np_))
    far = (~exact) & (~close)

    if np.any(far):
        xf, npf = x[far], np_[far]
        out[far] = xf * np.log(xf / npf) + (npf - xf)

    if np.any(close):
        xc, npc = x[close], np_[close]
        v = (xc - npc) / (xc + npc)
        s = (xc - npc) * v
        ej = 2.0 * xc * v
        v2 = v * v
        active = np.ones_like(s, dtype=bool)
        j = 1
        while np.any(active) and j <= 1000:
            ej = ej * v2
            candidate = s.copy()
            candidate[active] = s[active] + ej[active] / (2 * j + 1)
            converged = active & (candidate == s)
            active = active & ~converged
            s = candidate
            j += 1
        out[close] = s

    return out


# ---------------------------------------------------------------------------
# stable_log_pmf: the exact identity, computed cancellation-safely
# ---------------------------------------------------------------------------


def stable_log_pmf(k, lam: float) -> np.ndarray:
    """Cancellation-safe Poisson log-pmf via Loader (2000)'s exact
    saddlepoint identity::

        log dpois(k, lam) = -stirlerr(k) - bd0(k, lam) - 0.5*log(2*pi*k)   (k > 0)
        log dpois(0, lam) = -lam

    This is an EXACT algebraic identity with the naive formula
    ``-lam + k*log(lam) - lgamma(k+1)`` (both compute the same
    mathematical quantity), not an approximation -- the only
    difference is that ``stirlerr`` and ``bd0`` are individually small,
    well-conditioned quantities even when ``k`` and ``lam`` are
    individually huge, so no catastrophic cancellation occurs.
    """
    k = np.asarray(k, dtype=np.float64)
    lam_arr = np.full(k.shape, float(lam), dtype=np.float64)
    out = np.empty_like(k)

    zero = k == 0
    if np.any(zero):
        out[zero] = -lam

    nz = ~zero
    if np.any(nz):
        kk = k[nz]
        out[nz] = -stirlerr(kk) - bd0(kk, lam_arr[nz]) - 0.5 * np.log(2.0 * np.pi * kk)
    return out


def naive_log_pmf(k, lam: float) -> np.ndarray:
    """The naive formula numpy's own PTRS acceptance test uses --
    provided only so tests and diagnostics can directly demonstrate the
    cancellation this guard fixes; never used by ``safe_poisson``.
    """
    k = np.asarray(k, dtype=np.float64)
    return -lam + k * np.log(lam) - _lgamma_general(k + 1.0)


def _lgamma_general(x: np.ndarray) -> np.ndarray:
    """Unrestricted vectorized lgamma (used only by naive_log_pmf for
    comparison purposes; safe_poisson's own stable path never calls a
    general lgamma on large arguments)."""
    flat = np.reshape(x, -1)
    result = np.fromiter((math.lgamma(float(v)) for v in flat), dtype=np.float64, count=flat.size)
    return result.reshape(np.shape(x))


# ---------------------------------------------------------------------------
# safe_poisson: vectorized PTRS (Hoermann 1993) built on stable_log_pmf
# ---------------------------------------------------------------------------


# Upper domain bound for safe_poisson, matching numpy's own upstream
# guard: Generator.poisson(lam) itself raises ValueError("lam value too
# large") once lam approaches np.iinfo(np.int64).max (empirically
# ~9.223372006e18 on numpy 2.5.3, independently bisected against the
# installed build -- see scratch_numpy_threshold.py in this repo's run
# history). safe_poisson's own PTRS proposal envelope can generate
# candidate k values roughly lam + O(20*sqrt(lam)) above lam, so its
# OWN silent-corruption boundary sits a little higher still
# (~9.223372023e18, independently bisected -- see
# scratch_bisect_boundary.py). Using numpy's own (lower, more
# conservative) threshold here means: (a) safe_poisson never accepts an
# input that numpy's own sampler would refuse, and (b) it always
# refuses strictly before its own float64->int64 cast can silently
# corrupt, with margin to spare.
_MAX_SAFE_LAM = 9.223372006484771e18


def safe_poisson(
    lam: float,
    size: int,
    rng: Optional[np.random.Generator] = None,
    *,
    max_rounds: int = 500,
    batch_multiplier: float = 1.3,
) -> np.ndarray:
    """Draw ``size`` Poisson(lam) samples using the same PTRS
    (Hoermann 1993) transformed-rejection algorithm numpy's own C
    implementation uses for lam >= 10, but with the acceptance test's
    log-pmf evaluation replaced by :func:`stable_log_pmf`, so it does
    not suffer numpy/numpy#31986's cancellation for large lam.

    For lam < 10, delegates to ``rng.poisson`` directly: numpy's own
    inversion-based algorithm for small lam is not affected by this
    bug (the cancellation only arises in the PTRS branch's log-pmf
    evaluation, which numpy itself only uses for lam >= 10).
    """
    if lam < 0 or not math.isfinite(lam):
        raise ValueError(f"lam must be finite and non-negative, got {lam}")
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if lam >= _MAX_SAFE_LAM:
        raise ValueError(
            f"lam={lam:.6e} is too large to sample safely into an int64 "
            f"output array (max safe lam is {_MAX_SAFE_LAM:.6e}): the PTRS "
            "proposal envelope can generate candidate k values roughly "
            "lam + O(20*sqrt(lam)) above lam, and once that exceeds "
            "np.iinfo(np.int64).max the float64->int64 cast in this "
            "sampler is undefined and silently returns corrupted/negative "
            "values (this is exactly numpy's own upstream 'lam value too "
            "large' guard -- Generator.poisson itself raises ValueError "
            "in this same regime; this sampler must too, rather than "
            "returning a finite-looking but wrong answer)."
        )
    if rng is None:
        rng = np.random.default_rng()
    if size == 0:
        return np.empty(0, dtype=np.int64)

    if lam < 10:
        # Not affected by this bug (numpy uses inversion, not PTRS,
        # below this threshold) -- pass through directly.
        return rng.poisson(lam, size=size)

    slam = math.sqrt(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2.0)

    out = np.empty(size, dtype=np.int64)
    filled = 0
    rounds = 0
    while filled < size:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(
                f"safe_poisson PTRS did not converge after {max_rounds} rounds "
                f"(filled {filled}/{size} for lam={lam}); this indicates a real "
                "defect in the sampler, not expected rejection-loop variance."
            )
        remaining = size - filled
        n_try = max(16, int(remaining * batch_multiplier))

        u = rng.random(n_try) - 0.5
        v = rng.random(n_try)
        us = 0.5 - np.abs(u)
        k = np.floor((2.0 * a / us + b) * u + lam + 0.43)

        accept = np.zeros(n_try, dtype=bool)
        fast_accept = (us >= 0.07) & (v <= vr)
        accept |= fast_accept

        early_reject = (us < 0.013) & (v > us)
        need_full_test = (~fast_accept) & (k >= 0) & (~early_reject)
        if np.any(need_full_test):
            idx = np.where(need_full_test)[0]
            lhs = np.log(v[idx]) + math.log(inv_alpha) - np.log(a / (us[idx] ** 2) + b)
            rhs = stable_log_pmf(k[idx], lam)
            accept[idx] = lhs <= rhs

        candidates = k[accept]
        candidates = candidates[candidates >= 0]
        take = min(candidates.size, remaining)
        if take > 0:
            out[filled:filled + take] = candidates[:take]
            filled += take

    return out


# ---------------------------------------------------------------------------
# diagnose: live empirical comparison against the installed numpy
# ---------------------------------------------------------------------------


@dataclass
class PoissonVarianceDiagnosis:
    """Result of empirically probing the installed numpy for the bug
    at a specific lam, by drawing real samples and comparing empirical
    variance/lam ratios -- never a version-number allowlist."""

    numpy_version: str
    lam: float
    n_samples: int
    numpy_var_over_lam: float
    safe_var_over_lam: float
    numpy_mean: float
    safe_mean: float
    affected: bool
    detail: str


def diagnose(
    lam: float = 1e16,
    n_samples: int = 200_000,
    *,
    rng: Optional[np.random.Generator] = None,
    tolerance: float = 0.05,
) -> PoissonVarianceDiagnosis:
    """Empirically probe the INSTALLED numpy's ``Generator.poisson``
    for the large-lam variance-inflation bug, by drawing real samples
    from both numpy's own sampler and :func:`safe_poisson`, and
    comparing each empirical var/lam ratio against the true target of
    1.0. ``tolerance`` is generous (default 5%) because this is a
    finite-sample Monte Carlo comparison, not an exact check -- true
    Poisson variance/lam is exactly 1.0 only in the infinite-sample
    limit; ``n_samples=200_000`` keeps the Monte Carlo standard error
    on var/lam well under 1% for lam this large.
    """
    if rng is None:
        rng = np.random.default_rng()

    numpy_samples = rng.poisson(lam, size=n_samples).astype(np.float64)
    safe_samples = safe_poisson(lam, n_samples, rng=rng).astype(np.float64)

    numpy_ratio = float(numpy_samples.var() / lam)
    safe_ratio = float(safe_samples.var() / lam)
    numpy_mean = float(numpy_samples.mean())
    safe_mean = float(safe_samples.mean())

    affected = abs(numpy_ratio - 1.0) > tolerance
    safe_ok = abs(safe_ratio - 1.0) <= tolerance

    if affected and safe_ok:
        detail = (
            f"CONFIRMED: numpy's own Generator.poisson(lam={lam:.0e}) empirical "
            f"var/lam={numpy_ratio:.4f} (should be ~1.0, tolerance +-{tolerance}); "
            f"safe_poisson's var/lam={safe_ratio:.4f} is within tolerance "
            "(numpy/numpy#31986)."
        )
    elif affected and not safe_ok:
        detail = (
            f"UNEXPECTED: numpy's own var/lam={numpy_ratio:.4f} looks affected, "
            f"but safe_poisson's var/lam={safe_ratio:.4f} is ALSO outside "
            f"tolerance -- this needs investigation, do not trust either result "
            "blindly."
        )
    else:
        detail = (
            f"Not reproduced: numpy's own Generator.poisson(lam={lam:.0e}) "
            f"empirical var/lam={numpy_ratio:.4f} is within tolerance of 1.0 "
            "-- numpy/numpy#31986 may be fixed here, or this lam/sample size "
            "combination does not trigger it. safe_poisson remains correct "
            "regardless."
        )

    return PoissonVarianceDiagnosis(
        numpy_version=np.__version__,
        lam=lam,
        n_samples=n_samples,
        numpy_var_over_lam=numpy_ratio,
        safe_var_over_lam=safe_ratio,
        numpy_mean=numpy_mean,
        safe_mean=safe_mean,
        affected=affected,
        detail=detail,
    )
