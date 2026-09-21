# numpy-poisson-variance-guard

This repository has moved into the consolidated [`numpy-correctness-guards`](https://github.com/zhuhroscar-tech/numpy-correctness-guards) package.

Use the shared CLI/API there instead:

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/numpy-correctness-guards.git
numpy-guard run poisson-variance detect --lam 1e16 --samples 200000 --json
numpy-guard run poisson-variance sample --lam 1e16 --size 1000 --seed 42
```

```python
from numpy_correctness_guards.guards.poisson_variance import diagnose, safe_poisson

result = diagnose(lam=1e16, n_samples=200_000)
samples = safe_poisson(1e16, 1000)
```

The original implementation, tests, and CLI behavior were migrated into `numpy-correctness-guards` and verified there before this repository was archived.

This repository is retained only as a historical pointer. Please open new issues or pull requests on the consolidated package.
