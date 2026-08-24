# Review Round 1 — Reviewer B（计算神经科学 / 数学动力学 / 计算统计学）

**[is_accepted: FALSE]**

**REJECT**

审查范围：`nsmor/model_nsmor_core.py`、`nsmor/loss.py`、`nsmor/config.py`、`nsmor/config_parser.py`、`config/default.yaml`、`nsmor/analysis/{uq,dynamics,gating_cluster}.py`、`scripts/{analyze_jacobian,simulate_lesion,simulate_psychophysics,train,prepare_data}.py`、`nsmor/{mcmc_module,data_extractor,nsmor_dataloader}.py`、`nsmor/pipeline/{labeling,kinematics}.py`、`tests/`。

代码工程质量可接受（形状断言、try/finally 状态恢复、Holm-Bonferroni 工具链等），但从生物物理学、数学动力学与统计严谨性三个维度均存在实质性缺陷。以下按 BLOCKER / MAJOR / MINOR 分级。

---

## BLOCKER-1（生物物理学 · 致命）：LIF 时间常数单位语义混乱 ——"ms" 标称与"帧"实际不符

- **位置**：`D:\Projects\NSMoR\config\default.yaml:31-38`；`D:\Projects\NSMoR\nsmor\model_nsmor_core.py:403-411`（`alpha_syn = exp(-1/tau_syn)`）；`D:\Projects\NSMoR\nsmor\config.py:52`（`frame_interval_ms = 10.0`，100 Hz）。
- **问题**：YAML 注释声称 `lif_tau_syn=5.0 (ms)`、"dt≈1ms → alpha_syn ≈ 0.82"，但数据管道真实 dt = 10 ms，而实现中 tau 以**帧数**为单位。实际等效时间常数为：
  - `lif_tau_syn=5` → 实际 **50 ms** 突触低通；
  - `lif_tau_w=100` → 实际 **1 s** 适应时间常数。
- **生物学后果**：蟋蟀逃逸回路（GI→TTM/CoLa）的逃逸反应潜伏期本身仅 ~50–100 ms。50 ms 的突触滤波会把模型声称要编码的 "fast, event-driven sensory transients" 在到达阈值前抹平。同一物理量在 YAML 注释、config.py 与模型实现中使用了两套不一致的时间基准——这是单位语义错误，不是参数调优问题。
- **附加**：`abs_refract_steps`/`rel_refract_steps` 默认为 0，即默认配置下 LIF 无不应期，理论上可每帧发放（上限 100 Hz 且无 Na⁺ 失活约束），与 "five biologically grounded mechanisms" 的自我表述相悖。
- **修复方向**：在 config 中显式携带 `dt_ms` 并统一换算（`alpha = exp(-dt/tau_ms)`），所有 tau 以物理时间单位声明；默认启用合理的不应期参数。

## BLOCKER-2（计算统计学 · 致命）：MCMC 先验标签泄漏，污染全部分析

- **位置**：`D:\Projects\NSMoR\scripts\prepare_data.py:634-647`（`train_mcmc(snapshots, snapshot_labels)` 后立即 `predict_proba(snapshots)` 于同一样本）；`D:\Projects\NSMoR\nsmor\mcmc_module.py:229-273`（训练函数无任何 split 参数）。
- **问题**：MCMC 先验生成器在**全部 trials 的 snapshots + ground-truth labels** 上训练，再对**同一批 snapshots** 推理生成先验，写入数据集供 NSMoR 主模型作为输入的第 5–8 维。
- **后果**：
  1. 输入特征含对同一 trial 标签过拟合的 softmax 输出 —— 标签泄漏直接进入模型输入；
  2. 主模型的预测性能、router gating 分析、lesion 实验、心理物理学曲线全部建立在被污染特征上，"Router 学到贝叶斯因果推断" 类解释不可信；
  3. `simulate_psychophysics.py:78-117` 自行重切"最后 20% 验证集"，但 MCMC 先验早在 prepare_data 阶段已在这些样本的标签上训练过 —— 划分形同虚设。
- **修复方向**：对 MCMC 训练做 out-of-fold（K-fold cross-fitting）预测后再写入数据集；或严格按 trial 划分 train/test 后仅在测试段用 held-out 模型推理。修复后所有下游产物必须全部重跑。

## BLOCKER-3（计算统计学）：心理物理学实验无推断检验、无效应量、无多重比较校正

- **位置**：`D:\Projects\NSMoR\scripts\simulate_psychophysics.py:452-460`（只报 mean ± SEM）；`:238-254`（latency 提取）。
- **问题**：
  1. 四个噪声水平 σ ∈ {0,5,15,30}° 之间没有任何配对检验、效应量或 Holm/FDR 校正 —— 而 lesion 脚本已具备完整工具链（`nsmor/analysis/uq.py` 的 `holm_bonferroni` + paired Cohen's d），标准明明就在仓库里；
  2. visual_angle 通道未做归一化/SNR 论证即注入高斯噪声，σ 的物理意义（度 vs 信号量级）未界定；
  3. `extract_latency_to_peak` 用全局 argmax 且 `max(0.0, ...)` 把刺激前峰值截为 0，制造在零点堆积的质量分布 —— SEM 与任何参数检验对该分布不成立；
  4. 仅 abort 于 n_ttc0 == 0，未报告最小有效样本量。
- **修复方向**：跨噪声水平补配对 Wilcoxon/t-test + Holm 校正 + paired Cohen's d；latency 改用刺激后窗口内 argmax 并报告分布（非仅均值±SEM）；给出 SNR 定义。

---

## MAJOR-1（数学动力学）：Jacobian 谱分析的 slow-point 方法缺乏不动点校验

- **位置**：`D:\Projects\NSMoR\scripts\analyze_jacobian.py:267-304`（`_find_slow_point`）、`:377-419`（单帧输入冻结）。
- **问题**：
  (a) 用最小化 ‖h_{t+1}−h_t‖₂ 在 ±5 帧内选点，但**从不验证**所选 h_slow 是否满足准不动点残差 ‖GRU(x,h)−h‖ 足够小；`FixedPointAdapter.test_attractor_convergence` 存在却未被主分析管线调用。
  (b) 对非对称 GRU Jacobian，特征值只在渐近线性化意义下刻画收敛；远离不动点的轨迹点上的谱不能直接支撑 "line attractor / 连续积分器" 结论。
  (c) 三个 epoch 各自冻结不同帧的输入 x_slow —— 实际比较的是"不同输入下的不同映射"，却被解释为"同一系统在不同行为阶段的状态依赖动力学"，输入依赖性与状态依赖性未被分离对照。
- **修复方向**：加入残差阈值校验并调用 test_attractor_convergence；报告固定输入下的状态扫描（同一 x 多个 h）与固定 h 下输入扫描的分离结果。

## MAJOR-2（数学/工程）：Phase 8 中两类数学对象混排

- **位置**：`D:\Projects\NSMoR\nsmor\analysis\dynamics.py:523-651`（`compute_full_system_jacobian`，H×F 非方阵 → SVD 奇异值）vs `analyze_jacobian.py` Panel 输出（GRU Jacobian 特征值 + 单位圆）。
- **问题**：docstring 已诚实警告 surrogate-gradient Jacobian 不能用于定量稳定性分析（dynamics.py:537-558），但主图叙事把奇异值分析与特征值单位圆并置在同一 "Phase 8" 结论框架下，读者无法区分两类谱的适用范围。
- **修复方向**：图注与 JSON 输出显式区分 "GRU-pathway eigenvalues (exact)" 与 "full-system singular values (surrogate)"，后者不得用于稳定性结论。

## MAJOR-3（统计学）：gating cluster 的 bootstrap 稳定性评估有偏

- **位置**：`D:\Projects\NSMoR\nsmor\analysis\gating_cluster.py:580-600`。
- **问题**：bootstrap 重采样后在**全量 scaled fingerprints** 上 fit KMeans 再算 ARI 一致性；标准做法应在每个 resample 内重新估计（含 scaler）。且 silhouette 选 k 与 bootstrap 稳定性共用同一份数据（选择后评估，无嵌套），k=4 的"无监督选择"带乐观偏差。`n_bootstrap=100` 对 ARI 区间偏少（惯例 ≥500）。
- **修复方向**：每个 resample 内重新 StandardScaler + fit；k 选择与稳定性评估嵌套或至少用独立重采样。

---

## MINOR 条目

4. **MINOR-A（labeling 语义错误）**：`D:\Projects\NSMoR\nsmor\pipeline\labeling.py:57-84` `_check_sustained_speed` 文档写 "sustained for duration"，实现是窗口内 `np.max(...) > threshold` —— 单个瞬时尖峰即可判 ESCAPE。"sustained" 语义完全未实现，改变四类 ground truth 构成。修复：改为持续超阈帧数 ≥ duration/dt 判定。
5. **MINOR-B（trial 匹配脆弱假设）**：`D:\Projects\NSMoR\scripts\simulate_psychophysics.py:150-193` `find_multisensory_ttc0` 假设 events 文件顺序 = 数据集顺序再切"最后20%"；prepare_data 会丢弃无效 trial（baseline < 500 ms 等），索引错位时静默给错误的 trial 打条件标签。修复：在 prepare_data 导出 per-trial 元数据并在分析脚本按键匹配。
6. **MINOR-C（block-bootstrap 边界偏倚）**：`D:\Projects\NSMoR\nsmor\analysis\uq.py:43-59` 已在 docstring 承认截断边界偏倚但未修；circular bootstrap（Politis & Romano 1992）实现成本极低，建议直接替换。
7. **MINOR-D（jerk mask 未在 BioDecisionLoss 外层定义清晰）**：`nsmor/loss.py:189-197` router 正则在 jerk_mask 上归一化，mask 稀疏时 N→clamp(min=1)，正则强度随 mask 密度隐式变化，建议在文档中明确该耦合。

---

## 结论

BLOCKER-1/2/3 任一条足以否决。优先级排序：

1. 修复 MCMC 标签泄漏（cross-fitting）并**全量重跑**下游所有产物；
2. 统一 LIF tau 的 dt 语义（config 显式携带 dt_ms）；
3. psychophysics 补齐配对推断 + Holm 校应 + latency 分布处理；
4. slow-point 分析加入不动点残差校验并接入 test_attractor_convergence；
5. labeling sustained 判据改为真正的持续超阈时长判定；
6. 其余 MINOR 按上表修复。

— Reviewer B
