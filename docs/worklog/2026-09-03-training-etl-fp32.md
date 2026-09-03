# 训练流水线：ETL 切换与 FP16 NaN

## ELT → ETL：5x 加速

### 现象

Lazy loading (ELT) 模式下每 batch 需要 40 秒,GPU 利用率仅 15%。瓶颈在 WSL2 的 9p 文件系统——每个 trial 按需从磁盘加载 .csv,I/O 延迟远超 GPU 计算时间。

1440 trials × 150 epochs 预估需要 **15+ 小时**,后续数据扩到 5000+ trials 完全无法训练。

### 修复

创建 `scripts/convert_metadata_to_etl.py`,一次性将所有 trial 预加载到内存并保存为单个 `.pt` 文件（138 MB）。

```
data/processed/nsmor_metadata_3cond_v2.pt  →  data/processed/nsmor_dataset_3cond_v2.pt
(lazy specs, 2.4 MB)                          (pre-loaded arrays, 138 MB)
```

切换后每 batch 从 40 秒降到 7-10 秒。GPU 利用率从 15% 升到 92%。

### 转换过程中的 4 个连环 bug

| 序号 | 错误 | 根因 | 修复 |
|---|---|---|---|
| 1 | `'list' object has no attribute 'sum'` | `lengths` 存为 Python list | 改为 `np.array(lengths, dtype=np.int64)` |
| 2 | `invalid literal for int(): 'PRE_ACTIVE'` | labels 存为字符串 | 用 `label_encoder` 转为 int |
| 3 | `broadcast (4,) into (2400,0)` | X_seq 只有 4 列物理特征 | 创建 8-D 数组,cols 0:4 填物理特征,4:8 留零 |
| 4 | `MCMC columns already contain valid simplex` | lazy loader 已填好 8 列 | 只复制 cols 0:4,让 NSMoRDataset._fill_priors 填 4:8 |

每个 bug 都是格式兼容问题,错误信息明确,修复直接。关键在于 ETL 产物的 schema 要与 `NSMoRDataset.__init__` 期望的格式**精确对齐**——8-D X_seq,cols 4:8 为零（由 dataset 类自己填 MCMC prior）。

## FP16 混合精度 → FP32

### 现象

AMP (FP16) 模式下,每 epoch 的 9 个 batch 中有 **5 个产生 NaN loss**,1 个产生 NaN gradient。只有 3 个 batch 正常完成。

```
Epoch 0 batch 0: non-finite loss=nan — skipping step
Epoch 0 batch 1: non-finite gradient before clipping — skipping step
Epoch 0 batch 2: non-finite loss=nan — skipping step
...
Epoch 0 skipped steps: 5 non-finite loss, 1 non-finite grad
```

### 根因

FP16 的动态范围（6e-8 ~ 65504）对 LIF 膜电位动力学太窄：

- 膜电位 leak factor `alpha=0.9587` 接近 1.0,累积的指数衰减在 FP16 下精度丢失
- 突触电流 IIR 滤波器 `alpha_syn * i_syn + (1-alpha_syn) * input` 在 FP16 下产生灾难性消去
- surrogate gradient `sigmoid(4.0 * (v - v_th))` 在 v 远离 v_th 时梯度为零,FP16 下更容易下溢

### 修复

`scripts/train.py` line 1781: `use_amp = False`

禁用 AMP 后:
- 零 NaN（150 epochs 全程无异常）
- 速度几乎无损（模型仅 34K 参数,FP16 的 Tensor Core 加速对小模型不明显）
- 内存无影响（batch_size=128 仍在 GPU 显存安全范围内）

### 启示

**SNN 的生物物理参数（alpha, tau, threshold）对浮点精度敏感**。FP16 适合大 Transformer,不适合膜电位动力学。如果需要混合精度,考虑只对 GRU 和 linear 层用 FP16,LIF 路径保持 FP32——但这增加工程复杂度,收益微小。

## 相关

- ETL 产物的 schema: `scripts/convert_metadata_to_etl.py`
- JAX 训练流水线（原生 FP32）: `nsmor/jax/train.py`
