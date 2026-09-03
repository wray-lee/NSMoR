# JAX 加速的真实边界

**结论先行**: JAX/XLA 不是万能加速器。小规模归约操作上，per-call dispatch 开销会让它**比纯 NumPy 慢一个数量级**。

## 实测数据（本机 RTX 5060 Ti + WSL2）

### 训练流水线：真加速

| 实现 | 每 epoch 耗时 | 显存 |
|---|---|---|
| PyTorch FP32 | ~131 秒 | ~3.9 GB |
| JAX + `lax.scan` | ~64 秒 | ~1.5 GB |

**2.0-2.6x 加速**。收益来源是 `lax.scan` 把 2400 步 LIF 时间循环融合成单一 XLA kernel，消除了 PyTorch eager 模式下每步 7 个小算子的调度开销和显存往返。

注意首个 epoch 因 JIT 编译会慢 3-5x，加速要从第 2 个 epoch 起算。

### fingerprint 提取：慢 8 倍

| 实现 | 100 trials × T=200 |
|---|---|
| PyTorch（纯 NumPy 路径） | 0.026 秒 |
| JAX 逐 trial 调用 | 0.229 秒 |

**慢 8.8x**。开销来自每次调用的 JAX dispatch + NumPy↔JAX 数组转换 + adapter 实例化。这些 16 维小归约的实际计算量微不足道，全被调用开销盖住了。

（本可以用 vmap 批量化,但那版实现[静默算错了 14/16 个特征](2026-09-03-jax-vmap-padding-trap.md)，正确性优先。）

### MC dropout：真加速

`uq_jax.py` 用 `jax.vmap` over 独立 PRNG key,把 n_samples 次 dropout forward 合成单次 vmapped 调用。这里 vmap 是安全的（每个 sample 形状一致，无变长归约），且单次 forward 计算量足够大，dispatch 开销可忽略。

## 判断准则

用不用 JAX，看**单次调用的计算量 vs dispatch 开销**：

| 场景 | 用 JAX？ | 理由 |
|---|---|---|
| 长序列递归（`lax.scan`） | ✅ | 融合数千步，收益巨大 |
| 大 batch 模型 forward | ✅ | 计算密集，dispatch 可忽略 |
| 高阶导（jacobian/hessian） | ✅ | `jacfwd`/`jacrev` 比 PyTorch 高效 |
| 同形状样本并行（vmap） | ✅ | 前提：无变长归约 |
| 小张量归约（16 维统计量） | ❌ | dispatch 开销 >> 计算量,NumPy 更快 |
| 逐样本 Python 循环调用 | ❌ | 每次都付 dispatch 成本 |

**经验法则**: 单次调用如果在 CPU 上用 NumPy 都不到 1 毫秒，那 JAX 大概率会更慢。

## 关于 OpenXLA / torchax 方案的评估

曾评估过 `torch.compile(backend="openxla")` 和 `torchax` 作为进一步优化路径，结论是**当前不适用**：

- **OpenXLA backend 主要面向 TPU 和数据中心 GPU**（A100/H100）。消费级 NVIDIA 卡上 `torch.compile` 实际走的是 Inductor/Triton，不是 XLA。
- **LIF 的 sequential scan 本身无法并行化**（每步依赖上一步膜电位），这是 SNN 的固有瓶颈，换编译器后端不解决。声称的 3-8x 在此场景不现实。
- **已有原生 JAX 实现**（`nsmor/jax/`）,再用 torchax 包一层是多此一举，且 torchax API 尚不稳定。

真正还有价值的方向：StableHLO 导出（为论文可复现性），但优先级低于模型结果验证。

## 相关

- vmap 正确性陷阱: [vmap 零填充](2026-09-03-jax-vmap-padding-trap.md)
- docstring 中已记录 fingerprint 路径的取舍: `nsmor/analysis/gating_cluster_jax.py`
