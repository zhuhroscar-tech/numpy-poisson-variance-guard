# numpy-poisson-variance-guard

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-555555?style=flat)](README.zh-CN.md)

Detects and works around a real, currently-open numpy bug:
[numpy/numpy#31986](https://github.com/numpy/numpy/issues/31986) --
`numpy.random.Generator.poisson(lam)` **silently samples from the wrong
distribution** for large `lam` (empirically `lam >~ 1e15`): the mean
stays correct, but the empirical **variance is inflated** well above the
true value of `lam` -- with no error, warning, or exception.

```pycon
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> samples = rng.poisson(1e17, size=500_000)
>>> samples.var() / 1e17
1.64   # WRONG: true Poisson variance/lam should be ~1.0
```

## Why this happens

numpy's own C sampler uses PTRS (Hoermann 1993), a transformed-rejection
algorithm, for any `lam >= 10`. Its acceptance test evaluates a Poisson
log-pmf with the naive formula

```
-lam + k * log(lam) - lgamma(k + 1)
```

which computes the difference of several terms of magnitude
`O(lam)`-to-`O(lam * log(lam))` that should cancel down to a small
result. Once `lam` exceeds roughly `1e15`, float64's ~15-17 significant
decimal digits are exhausted by this cancellation, so the acceptance
test's decision becomes essentially noise near the distribution's mode
-- rejecting good candidates near `k = lam` and instead over-accepting
from the tails, which inflates variance while the mean (a far less
cancellation-sensitive quantity here) stays approximately correct. This
package's own measurements independently matched the upstream issue's
reported ratios almost exactly (~1.42 at `lam=1e16`, ~1.64 at
`lam=1e17`).

## What it provides

- `stable_log_pmf(k, lam)`: a cancellation-safe Poisson log-pmf using
  Loader (2000)'s exact saddlepoint decomposition
  (`stirlerr(k) + bd0(k, lam)`, the same technique used by R's C
  `nmath` `dpois_raw`) -- an **exact algebraic identity** with the naive
  formula, not an approximation, just computed in a numerically stable
  order.
- `safe_poisson(lam, size, rng=None)`: a full vectorized implementation
  of the *same* PTRS algorithm numpy's own C code uses, with the
  acceptance test's log-pmf evaluation replaced by `stable_log_pmf`.
  For `lam < 10` (below the threshold numpy itself switches to PTRS),
  delegates straight to `rng.poisson` since that path is not affected.
- `diagnose(lam=1e16, n_samples=200_000)` /
  `numpy-poisson-variance-guard diagnose`: empirically compares the
  **installed** numpy's own `Generator.poisson` against `safe_poisson`
  by drawing real samples and comparing variance/lam ratios -- never a
  version-number allowlist, since numpy has not announced a fix
  version.
- `numpy-poisson-variance-guard sample --lam L --size N`: draws `N`
  corrected Poisson(`L`) samples via `safe_poisson` from the command
  line.

This is a workaround for an upstream numpy defect, **not a numpy
patch** -- it exists only until numpy/numpy#31986 is fixed upstream.

## Install and run

Requires Python 3.9+ and numpy>=1.24. No GPU, no compiled extensions,
no SciPy dependency (lgamma for small arguments is computed via the
stdlib `math.lgamma`, vectorized in pure Python only for the rare
small-`k` branch).

```bash
git clone https://github.com/zhuhroscar-tech/numpy-poisson-variance-guard.git
cd numpy-poisson-variance-guard
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

```bash
numpy-poisson-variance-guard diagnose --lam 1e16 --samples 200000
numpy-poisson-variance-guard diagnose --lam 1e16 --json
numpy-poisson-variance-guard sample --lam 1e16 --size 1000 --seed 42
numpy-poisson-variance-guard --version
```

`diagnose` exits `1` if numpy's own sampler is confirmed affected at
the given `lam` (empirical var/lam outside `--tolerance`, default 5%),
`0` if not reproduced.

```python
import numpy as np
from numpy_poisson_variance_guard import safe_poisson

rng = np.random.default_rng(42)
samples = safe_poisson(1e17, size=1_000_000, rng=rng)
print(samples.var() / 1e17)   # ~1.0, correct
```

## Verification trail

- Independently reproduced on this project's pinned numpy version
  (2.5.3) before this guard was written: `rng.poisson(1e17,
  size=500_000).var() / 1e17` measured 1.64 (should be ~1.0); a control
  at `lam=1e10` measured 1.0014, confirming the effect is specific to
  large `lam`.
- `stable_log_pmf` was checked against a 50-digit `mpmath` oracle across
  `lam` in `{1e10, 1e13, 1e15, 1e16, 1e17, 1e18}` and `k` within `±5
  * sqrt(lam)` of `lam`: worst absolute error against the oracle was
  `7.1e-15` (float64 ULP-level), versus the naive formula's worst error
  of `8213.6` over the same grid (and exactly `0.0` -- total
  cancellation -- at several points, matching the upstream issue's own
  reported failure mode).
- The full vectorized `safe_poisson` sampler (not just the isolated
  log-pmf formula) was verified end-to-end: at `lam` in `{1e10, 1e15,
  1e16, 1e17}`, `safe_poisson`'s empirical var/lam stayed within ~0.5%
  of 1.0 across 300,000-sample draws, while the same test harness's
  naive-log-pmf acceptance test reproduced numpy's own reported
  inflation (1.03 to 1.66) at the same `lam` values.
- ZH/JA disclosure: a Chinese-language query ("numpy poisson 分布 大lambda
  方差 偏大 采样 拒绝算法") and a Japanese-language query ("numpy poisson
  分布 大きいlambda 分散 膨張 サンプリング バグ") both returned only
  generic Poisson-distribution tutorials, no native-language discussion
  of this specific bug. EN-only evidence, disclosed honestly.
- CI runs the full test suite (including a live, unmocked reproduction
  against the CI runner's own installed numpy) on `ubuntu-latest` and
  `macos-latest`, plus a wheel/sdist build-and-smoke-test job with
  checksummed release artifacts.

## Limits

- This tool only detects and works around the large-`lam` PTRS
  acceptance-test cancellation described in numpy/numpy#31986. It does
  not audit any other numpy random-sampling distribution for
  correctness.
- `diagnose`'s comparison is a finite-sample Monte Carlo measurement,
  not an exact proof -- the default `n_samples=200_000` keeps the Monte
  Carlo standard error on var/lam well under 1% at the `lam` magnitudes
  tested, but a single run's result can vary slightly between calls.
- `safe_poisson`'s vectorized PTRS loop has a bounded number of
  rejection rounds (`max_rounds=500` by default) and raises
  `RuntimeError` rather than looping forever if a `lam`/parameter
  combination fails to converge -- this has not been observed in
  testing but is a documented, deliberate safety bound, not silent
  infinite work.
- If numpy fixes this issue upstream, `safe_poisson` remains correct
  regardless (it does not depend on numpy's own PTRS implementation),
  but this guard is not a substitute for eventually removing the
  workaround once a numpy release with the fix is your minimum
  supported version.
- `safe_poisson` raises `ValueError` for `lam >= ~9.223372006e18`
  (matching numpy's own `Generator.poisson` upper-domain guard) rather
  than sampling: found and fixed in v0.1.1 after a routine
  never-before-run backstop inspection showed the PTRS proposal
  envelope can generate candidate `k` values roughly
  `lam + O(20*sqrt(lam))` above `lam`, which silently overflowed the
  `int64` output array once `lam` got close enough to
  `np.iinfo(np.int64).max` -- returning finite-looking but wrong
  samples with no exception. v0.1.0 did not guard this; upgrade if you
  ever pass `lam` anywhere near that magnitude.

## Development and removal

```bash
python3 -m pytest -q --cov=numpy_poisson_variance_guard --cov-report=term-missing
python3 -m pip uninstall numpy-poisson-variance-guard
```

[Releases](https://github.com/zhuhroscar-tech/numpy-poisson-variance-guard/releases) · [MIT license](LICENSE)
