"""numpy-poisson-variance-guard: detect and work around numpy/numpy#31986.

numpy.random.Generator.poisson(lam) (and the legacy RandomState.poisson)
silently samples from the WRONG distribution for large lam (roughly
lam >= 1e15): the mean stays correct, but the empirical VARIANCE is
inflated well above the true value of lam, with no error, warning, or
exception -- see README for the verification trail and the upstream
issue for numpy's own reported var/lam ratios (matched independently
here: ~1.42 at lam=1e16, ~1.64 at lam=1e17).

Root cause (per the upstream issue and independently confirmed in this
repo's math derivation, see docs/derivation.md): numpy's transformed
rejection sampler (PTRS, Hoermann 1993) evaluates a log-pmf acceptance
test using the naive formula

    -lam + k * log(lam) - lgamma(k + 1)

which computes the difference of several O(lam)-to-O(lam*log(lam))
magnitude terms that should cancel down to a small O(1)-to-O(log(lam))
result. Once lam exceeds ~1e15, float64's ~15-17 significant decimal
digits are exhausted by the cancellation, so the acceptance test's
decision becomes essentially random noise near the distribution's
mode -- rejecting good candidates near k=lam and instead accepting
disproportionately from the tails, which inflates variance while
leaving the mean (a much less cancellation-sensitive quantity here)
approximately correct.

This package provides an independently-verified, cancellation-safe
log-pmf (Loader (2000)'s exact saddlepoint decomposition, the same
technique R's C nmath dpois_raw uses) and a full drop-in vectorized
PTRS sampler built on top of it, not a numpy source patch.
"""
from __future__ import annotations

__version__ = "0.1.1"

from .core import (
    PoissonVarianceDiagnosis,
    diagnose,
    safe_poisson,
    stable_log_pmf,
)

__all__ = [
    "PoissonVarianceDiagnosis",
    "diagnose",
    "safe_poisson",
    "stable_log_pmf",
    "__version__",
]
