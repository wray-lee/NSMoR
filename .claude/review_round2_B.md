# Review Round 2 — Reviewer B（计算神经科学 / 数学动力学 / 计算统计学）

**[is_accepted: FALSE]**

**REJECT**

审查范围：working tree 全部未提交改动（`D:\Projects\NSMoR\config\default.yaml`、`D:\Projects\NSMoR\nsmor\config_parser.py`、`D:\Projects\NSMoR\nsmor\model_nsmor_core.py`、`D:\Projects\NSMoR\nsmor\mcmc_module.py`、`D:\Projects\NSMoR\nsmor\pipeline\labeling.py`、`D:\Projects\NSMoR\nsmor\analysis\uq.py`、`D:\Projects\NSMoR\nsmor\analysis\gating_cluster.py`、`D:\Projects\NSMoR\scripts\prepare_data.py`、`D:\Projects\NSMoR\scripts\simulate_psychophysics.py`、`D:\Projects\NSMoR\scripts\simulate_lesion.py`、`D:\Projects\NSMoR\scripts\analyze_jacobian.py`、`D:\Projects\NSMoR\scripts\train.py`、`tests/`）。

Round-1 三条 BLOCKER 的修复方向均正确且已落地（dt_ms 物理化换算、MCMC OOF cross-fitting、psychophysics 配对推断 + Holm-Bonferroni + Cohen's d_z）。工程诚意可辨。**但本轮修复在数学上引入了一个新的致命错误，另有两处修复不彻底导致原缺陷以新形态复发。**

---

## BLOCKER

### B-1（数学/统计学 · 本轮新引入）：gating bootstrap 稳定性评估在两个不一致的特征空间之间比较聚类

- **位置**：`D:\Projects\NSMoR\nsmor\analysis\gating_cluster.py`（本轮 diff，OOB 评估段，约 :608–640）。
- **问题**：每个 resample 内用 `boot_scaler.fit_transform(fingerprints[idx])` 训练 `kmeans_boot` —— 其质心位于 boot-scaler 空间；随后却对 OOB 样本重新拟合一个**不同的** scaler（`oob_scaler.fit_transform(fingerprints[idx_oob])`），再把 boot 空间的质心用于预测 oob 空间中的点（`kmeans_boot.predict(oob_scaled)`）。StandardScaler 是逐特征仿射变换，两个 scaler 的 center/scale 一般不同，质心坐标在不同空间不可迁移：该 predict 输出的标签接近随机分配。更糟的是 ARI 的另一侧 `kmeans_oob.fit_predict(oob_scaled)` 与被比较一侧使用不同表示 —— 这个 ARI 既不度量重采样稳定性也不度量任何可解释量。
- **对比**：修复前（全量单一 scaler）至少空间一致，只是有轻微乐观偏差；修复后是数学上无效的估计，且带着 "Reviewer B MAJOR-3 fix" 注释出现，比原缺陷更具迷惑性。
- **附注**：`labels_boot` 与 `labels_oob_true` 使用不同扰动 init seed 引入 init 方差的方向正确，label-switching 下 ARI 不受 seed 差异影响，这点没有问题；问题只在特征空间不一致。
- **修复方向**：OOB 点必须用 `boot_scaler.transform(fingerprints[idx_oob])`（复用训练侧 scaler，禁止 refit）投影后再 predict。"每 resample 重估 scaler" 的正确语义仅适用于 fit 侧。

### B-2（计算统计学 · cross-fitting 折划分忽略组依赖）：snapshot 层 StratifiedKFold 仍存在 session 级泄漏

- **位置**：`D:\Projects\NSMoR\nsmor\mcmc_module.py:349` 附近（`StratifiedKFold(n_splits=n_folds, shuffle=True, ...)`）；调用方 `D:\Projects\NSMoR\scripts\prepare_data.py:637-646` 未传入任何分组变量。
- **问题**：同一 session（同动物、同日、同一记录系统）的 trials 高度相关；snapshot 层 shuffle 使同一 session 的样本同时出现在 fold 的训练侧与测试侧，session 级信息（基线运动统计、增益状态、噪声水平）仍可泄漏进 "held-out" 先验。Round-1 BLOCKER-2 只被半修复：样本级标签泄漏消除了，组级泄漏仍在。主模型的性能、router gating 与 lesion/psychophysics 结论的可信度依旧受损。
- **修复方向**：按 session_id 做 `StratifiedGroupKFold`（或至少 `GroupKFold`）；这要求 prepare_data 把 session 归属传入 cross-fitting —— 当前函数签名根本不含分组变量。

---

## MAJOR

### M-1（工程/统计 · 潜在 NameError + 条件间样本构成漂移）

- **位置**：`D:\Projects\NSMoR\scripts\simulate_psychophysics.py`（本轮 diff 推断段与 JSON 段）。
- (a) `p_values` 仅在 `if 0.0 in latency_arrays:` 分支内定义；summary JSON 构造中 `"inference": {...} if p_values else {...}` 在 σ=0 不属于 CLI `--noise_levels` 时直接 NameError 崩溃。
- (b) NaN 排除按各条件独立执行：pre-stimulus-peak 的 trial 在 σ=0 被排除、在高噪声下可能保留（或反之），不同条件的有效刺激集不同，配对推断 "identical stimulus set" 的前提被静默破坏。应改为以 clean 条件的排除集合为锚做统一过滤，并报告各条件排除率。
- (c) 日志行 `logger.info("Latency: %.1f ± %.1f ms (n=%d)", mean_lat, sem_lat, len(latencies))` 打印的是含 NaN 的总数而非 `valid.size`，与 `latency_stats` 中修正后的 n 自相矛盾。

### M-2（数学动力学 · jacobian 残差门限与归因报告缺陷）

- **位置**：`D:\Projects\NSMoR\scripts\analyze_jacobian.py`（本轮 diff：`FP_RESIDUAL_THRESHOLD` 定义处约 :131-141、attractor 验证段约 :590-620、frozen-input control 约 :700-727）。
- (a) `FP_RESIDUAL_THRESHOLD = 0.1` 的辩护是纯手挥（"hidden states are O(1)"）：GRU hidden 逐分量受 tanh ∈(−1,1) 约束不等于 L2 范数阈值 0.1 合理；应由数据驱动分位数（候选残差分布的肘点）或与自发轨迹漂移速率对比标定。
- (b) attractor 验证中 `n_verified/n_tested` 跨 epoch **累积计数**，却在循环内以 per-epoch 名义打日志——第二个 epoch 起日志数字是累计值，报告直接误导。
- (c) attractor 验证只测每 epoch 前 10 个状态且通过与否不影响谱计算——验证与结论仍是两条平行线，只是多了几行日志；应将验证结果写入 JSON 输出并与谱结论绑定。
- (d) frozen-input control 把所有 epoch 的 e_sensory 取 pooled median 作公共输入，但被评估的状态未必是该冻结映射的准不动点——残差门是在各自原始输入下通过的，换输入后门失效；此对照的谱同样不可做稳定性解读，代码未对该对照重复残差校验。

### M-3（统计/部署 · OOF 先验的推理时程序未定义）

- **位置**：`D:\Projects\NSMoR\scripts\prepare_data.py:637-646,808-812`。
- 生成的 `fold_models` 被丢弃，数据集仅存字符串 provenance `"oof_5fold_stratified_cv"`。下游 simulate_psychophysics / simulate_lesion 对新数据需要先验时无任何定义良好的生成器（ensemble？refit 全量？），等于把 Round-1 的泄漏以另一种形式留给下一个脚本作者重新引入。`fold_models` 必须随数据集持久化，并文档化推理时先验生成协议（推荐：fold ensemble 平均后归一化）。

---

## MINOR

### m-1：默认值双源不一致
`config/default.yaml` 设 `lif_rel_refract_steps: 2`，但 `nsmor/config_parser.py` 数据类默认仍为 0 —— YAML 运行与编程式构造行为不一致，应统一。

### m-2：旧 checkpoint 加载时 dt_ms 静默取默认
`nsmor/model_utils._extract_model_params` 对缺失 `dt_ms` 键静默取 inspect 默认 10.0；若模型实际以其他采样率训练则 Round-1 的单位语义错误再次复活。加载时应 warning 或要求 checkpoint config 显式携带 dt_ms。

### m-3：labeling 新参数为裸魔法数
`nsmor/pipeline/labeling.py:_check_sustained_speed` 新增 `min_fraction=0.5`、`max_latency_ms=200.0` 未暴露到配置也未给出文献依据；作为 "sustained" 语义修复方向正确，但参数敏感性应在 tests/docstring 中说明。

### m-4（正面确认）
`nsmor/analysis/uq.py` circular block bootstrap 实现正确（模回绕消除边界偏倚），lesion 脚本改用 block_size=5 合理；`holm_bonferroni` step-down 修复符合 Wright (1992) 语义。这两处予以肯定。

---

## 结论与修复优先级

B-1 是本轮新写入的数学硬伤；B-2 表明 cross-fitting 尚未达到其声称的保证。任一 BLOCKER 均足以否决。优先级：

1. **B-1**：OOB 必须用 boot_scaler.transform，禁止 refit；
2. **B-2**：cross-fitting 按 session 分组划分；
3. M-1(a) NameError 修复；(b)(c) 条件间统一样本集与日志修正；
4. M-2 阈值数据驱动标定、per-epoch 日志归因、frozen-input 对照补残差校验；
5. M-3 持久化 fold_models 并文档化推理协议；
6. MINOR 各项。

— Reviewer B
