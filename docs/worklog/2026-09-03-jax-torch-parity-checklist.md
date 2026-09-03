# JAX/PyTorch 数值对齐清单

从 JAX 训练流水线审查中提炼的逐项对齐清单。每项都曾作为 BLOCKER 或 MAJOR 被发现。

## LIF 物理参数

| 参数 | PyTorch (`model_nsmor_core.py`) | 必须一致 |
|---|---|---|
| `v_clamp_max` | `3.0 * v_threshold` (line 436) | ✅ JAX 首版用了 5.0,导致膜电位溢出产生虚假 spike |
| `i_syn_clamp` | `5.0 * v_threshold` (line 442) | ✅ JAX 首版用了 10.0 |
| `alpha = exp(-dt/tau)` | 全部 tau 参数 | 必须用相同的 dt_ms |
| TBPTT truncation | `lax.stop_gradient` vs `detach()` | 语义对齐（截断位置、窗口大小） |

**教训**: 安全余量参数（clamp threshold）不能"放宽"，它们是动力学仿真的一部分。5x→3x 不是保守，是**正确**。

## 状态更新的 mask 门控

所有递归状态在 padded 帧必须**冻结**（保持上一帧的值），不能让 LayerNorm bias 或其他活性值在 padding 区间积累。

| 状态 | PyTorch 行为 | JAX 首版 | 修复 |
|---|---|---|---|
| `h_gru` | mask 门控 ✅ | mask 门控 ✅ | — |
| `v_reset` | mask 门控 ✅ | ❌ 未门控 | `jnp.where(m_2d > 0.5, new, prev)` |
| `i_syn` | mask 门控 ✅ | ❌ 未门控 | 同上 |
| `w_adapt` | mask 门控 ✅ | ❌ 未门控 | 同上 |
| `ref` | mask 门控 ✅ | ❌ 未门控 | 同上 |
| `rel_ref` | mask 门控 ✅ | ❌ 未门控 | 同上 |
| `spk_hist` | mask 门控 ✅ | ❌ 未门控 | 同上 |

**教训**: 如果 PyTorch forward 中 GRU state 做了 mask 门控,所有其他 carry state 也必须做。首版只看到 GRU 的 mask,漏掉了 LIF 的 6 个状态。

## 权重双向转换

| 方向 | 陷阱 |
|---|---|
| PyTorch → JAX | Dense kernel 需要转置（PyTorch 是 `(out, in)`, Flax 是 `(in, out)`） |
| JAX → PyTorch | 必须导出**所有**参数,包括可选的 neuromod `gain_scale`/`gain_bias`。首版 `to_torch_state_dict` 遗漏了这两个,`strict=True` 加载直接报错 |
| 类型注解 | `to_torch_state_dict` 返回 `Dict[str, torch.Tensor]` 但 `torch` 只在函数体内导入。用 `Dict[str, Any]` 或 module-level import |

**验证方法**: round-trip 测试——PyTorch → JAX → PyTorch → `load_state_dict(strict=True)`。首版测试没开 `strict=True`,漏掉了缺失 key。

## lif_potentials 语义

| 框架 | 导出什么 |
|---|---|
| PyTorch | post-reset: `(v_new - spk_mask * v_th) * mask` |
| JAX 首版 | pre-reset: `v_new * mask` |

这影响下游的 dynamics 分析和 Jacobian 计算。必须对齐到 PyTorch 的 post-reset 语义。

## 损失函数

| 问题 | 详情 |
|---|---|
| `lambda_reg` 硬编码 | JAX 首版在 train.py 中 `lambda_reg = config.loss.lambda_energy` 后又被 `lambda_reg_val = 0.01` 覆盖。用户配置完全被忽略 |
| `sensory_noise_std` 未传递 | 模型初始化时未从 config 读取,训练时静默跳过随机共振噪声注入 |
| `anchor_frames` 未传递 | 数据加载器虽然实现了 anchor-aligned cropping,但训练入口未传参,效果等于没实现 |

**教训**: 实现了功能但没在入口接线 = 没实现。审查时要追溯"config 里的每个参数是否真的流到了使用点"。

## 快速检查清单

新增 JAX 迁移时,对照以下列表:

- [ ] clamp/threshold 参数与 PyTorch 完全一致（不要"放宽"）
- [ ] 所有 carry state 在 padded 帧 mask 门控（不只是 GRU）
- [ ] 权重转换 round-trip + `strict=True`
- [ ] config 中的每个相关参数都从入口传到使用点
- [ ] 变长数据的测试包含**至少两个不同长度**
- [ ] parity 容差按"应该能达到的精度"设（float32 归约通常 1e-6）

## 相关

- vmap 陷阱: [vmap 零填充](2026-09-03-jax-vmap-padding-trap.md)
- 性能边界: [JAX 加速的真实边界](2026-09-03-jax-perf-boundary.md)
