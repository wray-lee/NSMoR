# vmap 零填充静默污染变长序列归约

**严重度**: BLOCKER（静默错误，无异常无警告）
**发现于**: `nsmor/analysis/gating_cluster_jax.py` 首版实现的代码审查

## 现象

`compute_fingerprints` 为了用 `jax.vmap` 批量处理 trial，把变长 gate 序列零填充到统一 `max_T`，然后 vmap 调用 fingerprint 提取。

16 维 fingerprint 中 **14 维被污染**，只有 index 5/11 的 entropy 幸免（因为事后单独修补过）。

T=30 填充到 max_T=80 的实测偏差：

| 特征 | 正确值 | 填充后 | 说明 |
|---|---|---|---|
| mean | 0.39 | 0.15 | 被 50 个零稀释 |
| min | 真实最小值 | 0.0 | 填充的零成了最小值 |
| dominant_frac | 0.37 | 0.14 | 分母变成 80 |
| std | — | 虚高 | 零与真实值的方差 |
| correlation | — | 错误 | 相关性被填充段拉平 |
| argmax/argmin 时刻 | — | 偏移 | 索引位置错位 |

与 PyTorch 参考实现的最大特征差异达 **1.46**。

## 为什么没被测试抓住

`test_fingerprint_parity_with_pytorch` 只用了**单个固定长度 T=80 的 trial**，而 T=80 恰好等于 vmap 的 `max_T`,所以那次调用根本没有填充。测试通过，bug 完好无损地躺在生产路径上。

同时容差设成 `atol=0.05, rtol=0.05`,即便有偏差也可能被吞掉,而独立的 `fingerprint_jax` 函数（不走 vmap）本来能做到 `max diff = 0.0` 的完全一致。

## 根因

`vmap` 要求所有 batch 元素形状一致，这与"归约操作只应作用于有效帧"天然冲突。填充是为了满足 vmap 的形状约束，但 `mean`/`min`/`std`/`argmax` 这类归约**没有 length 概念**,填充值会无差别参与计算。

模型 forward 里的同类问题被正确处理了（用 `mask` 门控每个时间步的状态更新），但分析代码里的归约操作漏掉了这一层。

## 怎么避免

**写 vmap 之前先问：这个函数里有归约操作吗？**

- 有归约 + 变长输入 → 必须把 `lengths` 一起传进去，所有 `mean`/`sum`/`min`/`max`/`std`/`argmax` 都要 mask 到有效帧
- 不想传 lengths → 放弃 vmap，逐样本调用（本次采用，代价见 [性能边界](2026-09-03-jax-perf-boundary.md)）
- 想兼顾 → 按长度分桶，同长度组内 vmap，散长走逐样本兜底

**变长数据的 parity 测试必须包含至少两个不同长度**，且其中一个不能等于 `max_T`。等长测试对填充 bug 完全免疫,是假通过。本次修复后测试覆盖 T=30/50/80/120,容差收紧到 `1e-4`。

**容差要按"应该能达到的精度"设，不是按"当前能过的精度"设。** 纯数值迁移（同样的算术、同样的 float32）应该做到 `1e-6~1e-7`;设成 5% 等于放弃了回归检测能力。

## 相关

- 修复 commit: `809c518`
- 性能取舍: [JAX 加速的真实边界](2026-09-03-jax-perf-boundary.md)
- 同类陷阱: [JAX/PyTorch 数值对齐清单](2026-09-03-jax-torch-parity-checklist.md) 中的 LIF carry state mask 门控
