# numpy-poisson-variance-guard

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-555555?style=flat)](README.zh-CN.md)

检测并规避一个真实存在、目前仍未修复的 numpy 缺陷：
[numpy/numpy#31986](https://github.com/numpy/numpy/issues/31986) ——
`numpy.random.Generator.poisson(lam)` 在 `lam` 较大时（经验值约
`lam >~ 1e15`）会**悄悄从错误的分布中采样**：均值仍然正确，但经验
**方差被明显放大**，远高于 `lam` 的真实值——没有任何报错或警告。

```pycon
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> samples = rng.poisson(1e17, size=500_000)
>>> samples.var() / 1e17
1.64   # 错误：真实泊松分布的方差/lam 应约为 1.0
```

## 缺陷成因

对于任意 `lam >= 10`，numpy 自身的 C 采样器使用 PTRS（Hoermann 1993）
变换拒绝算法。其接受检验使用朴素公式计算泊松对数概率质量函数：

```
-lam + k * log(lam) - lgamma(k + 1)
```

该式计算的是若干量级为 `O(lam)` 到 `O(lam * log(lam))`
的项之间的差值，这些项本应几乎完全抵消为一个很小的结果。一旦 `lam`
超过约 `1e15`，float64 约 15-17 位的十进制有效数字就会被这种相消消耗殆尽，
导致接受检验在分布众数附近的判断基本沦为噪声——在 `k = lam`
附近本应接受的候选值被拒绝，反而过多地从分布尾部接受，从而使方差被
放大，而均值（在此处对相消远不敏感）仍大致保持正确。本项目自己独立
测得的比值与上游 issue 报告的数值几乎完全吻合（`lam=1e16` 时约
1.42，`lam=1e17` 时约 1.64）。

## 提供的功能

- `stable_log_pmf(k, lam)`：使用 Loader (2000) 的精确鞍点分解
  （`stirlerr(k) + bd0(k, lam)`，与 R 语言 C 语言实现 `nmath` 中
  `dpois_raw` 所用技术相同）计算的、对相消问题免疫的泊松对数概率质量
  函数——这与朴素公式是**精确的代数恒等式**，而非近似，只是计算顺序在
  数值上更稳定。
- `safe_poisson(lam, size, rng=None)`：完整的向量化实现，采用与
  numpy 自身 C 代码**相同**的 PTRS 算法，只是将接受检验中的对数概率
  质量函数计算替换为 `stable_log_pmf`。对于 `lam < 10`（低于 numpy
  自身切换到 PTRS 的阈值），直接委托给 `rng.poisson`，因为该路径不受
  此缺陷影响。
- `diagnose(lam=1e16, n_samples=200_000)` /
  `numpy-poisson-variance-guard diagnose`：通过实际抽样比较**已安装**
  numpy 自身的 `Generator.poisson` 与 `safe_poisson` 的方差/lam
  比值——绝不依赖版本号白名单，因为 numpy 尚未宣布修复版本。
- `numpy-poisson-variance-guard sample --lam L --size N`：在命令行
  通过 `safe_poisson` 抽取 `N` 个经过修正的泊松（`L`）样本。

这是针对上游 numpy 缺陷的**规避方案，而非对 numpy 的补丁**——它的存在
仅仅是为了在 numpy/numpy#31986 被上游修复之前提供保护。

## 安装与运行

需要 Python 3.9+ 和 numpy>=1.24。无 GPU 需求，无编译扩展，也不依赖
SciPy（小参数下的 lgamma 通过标准库 `math.lgamma` 计算，仅在极少数
小 `k` 分支中以纯 Python 方式向量化）。

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

若在给定 `lam` 下确认 numpy 自身的采样器受影响（经验方差/lam
超出 `--tolerance`，默认 5%），`diagnose` 退出码为 `1`；未复现时为 `0`。

```python
import numpy as np
from numpy_poisson_variance_guard import safe_poisson

rng = np.random.default_rng(42)
samples = safe_poisson(1e17, size=1_000_000, rng=rng)
print(samples.var() / 1e17)   # ~1.0，正确
```

## 验证记录

- 在编写本工具之前，已在本项目锁定的 numpy 版本（2.5.3）上独立复现：
  `rng.poisson(1e17, size=500_000).var() / 1e17` 测得 1.64（应约为
  1.0）；在 `lam=1e10` 处的对照组测得 1.0014，证实该效应确实特定于
  较大的 `lam`。
- `stable_log_pmf` 已针对 50 位精度的 `mpmath` 参照实现，在
  `lam ∈ {1e10, 1e13, 1e15, 1e16, 1e17, 1e18}`、`k` 位于 `lam ± 5 *
  sqrt(lam)` 范围内进行了校验：相对参照实现的最大绝对误差为
  `7.1e-15`（float64 机器精度量级），而朴素公式在同一网格上的最大
  误差达到 `8213.6`（在若干点上甚至精确为 `0.0`——完全相消——与上游
  issue 报告的失效模式一致）。
- 完整的向量化 `safe_poisson` 采样器（而不仅仅是孤立的对数概率质量
  函数公式）经过了端到端验证：在 `lam ∈ {1e10, 1e15, 1e16, 1e17}`
  下，`safe_poisson` 在 300,000 个样本的抽样中，经验方差/lam
  始终保持在 1.0 的约 0.5% 范围内；而同一测试框架中使用朴素对数概率
  质量函数的接受检验则复现了 numpy 自身报告的方差膨胀（在相同
  `lam` 值下为 1.03 至 1.66）。
- 中/日文披露：中文查询（"numpy poisson 分布 大lambda 方差 偏大 采样
  拒绝算法"）和日文查询（"numpy poisson 分布 大きいlambda 分散 膨張
  サンプリング バグ"）均只返回了通用的泊松分布教程，未发现关于此
  具体缺陷的母语社区讨论。如实披露为仅有英文证据。
- CI 在 `ubuntu-latest` 和 `macos-latest` 上运行完整测试套件
  （包括针对 CI 运行器自身已安装 numpy 的真实、未打桩的复现测试），
  并附带 wheel/sdist 构建与冒烟测试任务及校验和发布产物。

## 局限性

- 本工具仅检测并规避 numpy/numpy#31986 所述的大 `lam` PTRS
  接受检验相消问题，不审计 numpy 其他任何随机采样分布的正确性。
- `diagnose` 的比较是有限样本的蒙特卡洛测量，而非精确证明——默认
  `n_samples=200_000` 能将测试所用 `lam` 量级下方差/lam
  的蒙特卡洛标准误差控制在远低于 1% 的水平，但单次运行的结果在不同
  调用之间可能会有轻微波动。
- `safe_poisson` 的向量化 PTRS 循环有一个有界的拒绝轮数上限
  （默认 `max_rounds=500`），若某个 `lam`/参数组合未能收敛，会抛出
  `RuntimeError` 而不是无限循环——测试中尚未观察到此情况，但这是
  一个明确设计的安全边界，而非静默的无限计算。
- 如果 numpy 上游修复了此问题，`safe_poisson`
  仍然保持正确（它并不依赖 numpy 自身的 PTRS 实现），但这并不能
  替代在你的最低支持 numpy 版本已包含修复后最终移除该规避方案。

## 开发与卸载

```bash
python3 -m pytest -q --cov=numpy_poisson_variance_guard --cov-report=term-missing
python3 -m pip uninstall numpy-poisson-variance-guard
```

[Releases](https://github.com/zhuhroscar-tech/numpy-poisson-variance-guard/releases) · [MIT 许可证](LICENSE)
