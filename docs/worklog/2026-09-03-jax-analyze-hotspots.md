# JAX 分析路径：哪些能默认切、哪些不能

**结论先行**: 默认 `make analyze` **只**把 GRU Jacobian 切到 JAX。完整序列 forward 虽然快 ~2.5x，但 LIF 阈值穿越会把 latency 拧偏 20%，不能当默认。

## 实测（RTX 5060 Ti + WSL2，2026-09-03）

| 路径 | torch | jax | 倍数 | 默认？ |
|---|---|---|---|---|
| GRU Jacobian N=32/100, H=64 | — | — | **~20–21x** | ✅ `jacfwd`；`max |ΔJ| ~1e-4` |
| fused JAX jac+eig | jac ~0.004s | 0.33–1.17s | **更慢** | ❌ eig 留 `torch.linalg.eigvals` |
| forward B=8 T=2400 | 3.59s | 1.73s | 2.08x | ❌ 见下 |
| internals / lesion | ~3.7s | ~1.7s | ~2.1x | ❌ 同上 |
| fingerprint 100 ragged | 0.044s | 1.92s | **0.02x** | ❌ 已有 worklog |
| autoreg 200×T=1 | 0.92s | 0.03s | 28x **假** | ❌ JAX 没有 `states=`；每次是全新 T=1 |

## 为什么 forward 快但不能默认

随机权重、短序列 (T≤100) 时 `spk_disagree=0`，`y` / gates ~4e-4。T=600 起 Heaviside `v > v_th` 在 fp32 漂移下翻转 **0.05%** 的 spike。真实 `best_model.pth` + 96 trial：

| 报告量 | 相对差 |
|---|---|
| MSE intact / lesion | 0.0005–0.007% |
| mean `g_gru` post-stim | 0.008% |
| mean firing rate | 0.024% |
| **peak velocity** | **2.0%** |
| **latency (argmax frame)** | **20.7%**（仅 52/96 trial 一致） |
| spike flips | 0.56% |

均值稳、极值不稳。dynamics / lesion / integration / psychophysics 报告的是 kinematics 极值，不能静默换后端。

## 怎么避免

- `make analyze` = hybrid（jacobian `--backend jax`，其余 PyTorch）
- `make analyze-torch` = 旧全 torch 聚合（jacobian `--backend torch`）
- `JAXEvalWrapper` 是 **opt-in**（`nsmor/analysis/jax_eval.py`），`states=` 直接 TypeError
- 不要把 fused JAX eig 或 fingerprint JAX 设为默认

## 相关

- [JAX 加速的真实边界](2026-09-03-jax-perf-boundary.md)
- [vmap 零填充](2026-09-03-jax-vmap-padding-trap.md)
- [JAX/PyTorch 数值对齐](2026-09-03-jax-torch-parity-checklist.md)
