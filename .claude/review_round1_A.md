# Reviewer A 审查报告 — Round 1

**判定: REJECT** (`[is_accepted: FALSE]`)

审查范围：`nsmor/model_nsmor_core.py`（全文 2154 行）、`nsmor/loss.py`、`scripts/train.py`（全文 1341 行）、`nsmor/analysis/{dynamics,uq,gating_cluster}.py`、`scripts/{analyze_jacobian,simulate_lesion,simulate_psychophysics}.py`、`config/default.yaml`、`nsmor/nsmor_dataloader.py`、`nsmor/model_utils.py`。

---

## BLOCKER（必须重构，禁止进入测试环节）

### BLOCKER-1【生物物理学 + 数学动力学】LIF 时间常数单位制混乱：配置以 ms 语义写入，模型按 dt 步数解释

- **位置**：
  - `D:\Projects\NSMoR\config\default.yaml:29-37` — `lif_tau_syn: 5.0`（注释 "tau_syn=5ms at dt≈1ms → alpha_syn ≈ 0.82"）、`lif_tau_w: 100.0`（注释 "tau_w=100ms"）
  - `D:\Projects\NSMoR\nsmor\config_parser.py:44,49` — 字段注释明示 "(dt units; 0=disabled)"
  - `D:\Projects\NSMoR\nsmor\model_nsmor_core.py:403,419,447-448` — `alpha = exp(-1/tau)` 直接以配置值计算，无任何单位换算
- **问题**：全仓库不存在 ms→步 的换算层。实际数据帧间隔 dt = 10 ms，因此：
  - `tau_syn = 5.0` 步 = **50 ms** 突触滤波（声称 5 ms），比声明慢 10 倍；
  - `tau_w = 100.0` 步 = **1000 ms** 适应性衰减——蟋蟀逃逸回路在风刺激后数百毫秒内完成决策，秒级适应电流在该时序尺度上无生理意义；
  - 更换采样率时所有"生物常数"将静默漂移，模型不可复现。
- **修复方向**：在 `LIFCell.__init__` 强制要求显式 `dt_ms` 参数，所有 tau 以物理时间（ms）传入并在内部换算为 `exp(-dt/tau)`；配置文件与 dataclass 注释统一为 ms；加断言防止 dt 未定义时的静默回退。

### BLOCKER-2【数学动力学 + 统计学】analyze_jacobian.py 的 GRU 特征值分析：非驻留态求谱 + 假设/实现自相矛盾 + 输入重构失真

- **位置**：
  - `D:\Projects\NSMoR\scripts\analyze_jacobian.py:8`（docstring "Target: Label.PREWALK trials"）vs `analyze_jacobian.py:590` 及 main 默认参数 `target_class = Label.ESCAPE.value`
  - `D:\Projects\NSMoR\scripts\analyze_jacobian.py:296-303`（`_find_slow_point` ±5 帧窗口最小动能搜索）
  - `D:\Projects\NSMoR\nsmor\analysis\dynamics.py:380-410`（`test_attractor_convergence` 含不动点残差检验——但主分析路径未调用）
  - `D:\Projects\NSMoR\scripts\analyze_jacobian.py:386-391` — 用 `model.sensory_encoder(sensory_x)` 重算 GRU 输入
- **问题**：
  1. `_find_slow_point` 只是在瞬态轨迹里挑漂移最小的帧，得到的 (h*, x*) 从未被验证为固定点（残差检验存在却不在 `compute_eigenvalues_at_epochs` 主路径中执行）。对非不动点处的 J 求谱并宣称 "|λ|≈1 证明 line attractor"，是动力学系统分析中的无效推断。
  2. docstring 与默认参数矛盾：ESCAPE 是短爆发行为，其 "sustained (+1000 ms)" epoch 本身就不存在持续行走，用它的谱去证明持续积分器，假设与数据完全脱节。
  3. FrontendEncoder 启用 dendritic IIR 时 `_dendritic_state` 是跨 forward 泄漏的模块缓存（`model_nsmor_core.py:1208`），批量 eval 重算的滤波历史与原始 forward 不一致，x_t 并非 GRU 当时真正接收的输入——"Exact input reconstruction"（Task 2）名不副实。
- **修复方向**：主路径强制调用固定点残差检验（超阈值样本剔除或单独标注）；修正 target_class 默认值与文档一致；dendritic 状态需按序列重放或禁用后重算 e_sensory。

---

## MAJOR（结论可信度受损，修复前相应结果不得引用）

### MAJOR-1【计算统计学】lesion 统计：配对设计与 i.i.d. bootstrap 矛盾 + Holm step-down 拒绝规则缺失 + 数值病理静默吞没

- **位置**：
  - `D:\Projects\NSMoR\scripts\simulate_lesion.py:856`（`bootstrap_ci(mse_arr, np.mean, n_bootstrap=1000)`，未传 `block_size`）
  - `D:\Projects\NSMoR\scripts\simulate_lesion.py:892-898` — NaN p 值一律替换为 1.0
  - `D:\Projects\NSMoR\nsmor\analysis\uq.py:150-151` — 零方差时 Cohen's d 返回 0.0
  - `D:\Projects\NSMoR\nsmor\analysis\uq.py:184-191` — Holm 实现
- **问题**：
  1. 配对 t 检验 / 配对 Cohen's d 要求 trial 一一对应，但 CI 用 i.i.d. bootstrap；trial 顺序即会话顺序，MSE 序列有时序相关时 CI 系统性偏窄。代码自己实现了 Künsch block-bootstrap（uq.py:44-101）却在唯一消费者处不用。
  2. `holm_bonferroni` 将 adjusted p 单调化后统一以 `adjusted_p < alpha` 判定，未实现 Holm 步降法的拒绝联动（rank 1 不拒绝时后续必须全部不拒绝）；小 m 下会过度拒绝。正确做法是按步降规则逐级判定，而非对调整后 p 统一比较。
  3. NaN→1.0、零方差 d→0.0 把数值病理伪装成"无效应"，应显式报告退化样本数（n_degenerate）而非静默替换。
- **修复方向**：CI 与检验使用同一配对结构（paired bootstrap over trials 或 block-bootstrap）；重写 Holm 判定循环；所有退化分支计入日志与输出 JSON。

### MAJOR-2【生物物理学 + 统计学】心理物理学实验的"贝叶斯可靠性"叙事缺乏机制支撑 + latency 地板效应

- **位置**：`D:\Projects\NSMoR\scripts\simulate_psychophysics.py:196-215`（噪声注入）、`:252`（latency 定义）
- **问题**：
  1. 仅 visual 通道加噪，MCMC prior 列（X[:,:,4:7])保持不变——先验不随证据可靠性退化而更新。router 对含噪 e_sensory 的被动响应与贝叶斯最优 cue combination 无任何机制联系，图题与假设（"Bayesian re-weighting of sensory evidence"）过度声明。
  2. `latency = max(0, peak_frame - STIM_ONSET_FRAME) * dt` 把峰值出现在刺激前的 trial 全部截断为 0，人工地板效应系统性低估高 σ 下的 latency 均值偏移——恰恰是被测量的效应方向。
- **修复方向**：要么让 prior 随 σ 退化（重跑 MCMC 或按可靠性缩放），要么把声明降格为"路由门控对输入噪声的敏感性"；latency 改为仅统计 post-onset 峰值 trial 并报告剔除比例。

### MAJOR-3【计算统计学】gating_cluster.py 聚类稳定性判据的样本量悖论与初始化方差不敏感

- **位置**：`D:\Projects\NSMoR\nsmor\analysis\gating_cluster.py:597-640`
- **问题**：
  1. `N < max(k_range)+5` 时 stability 静默置 0.0 但继续跑 silhouette 选 k——小 N 下稳定性评估本无意义，应报错或降级声明，而非给出看似完整的分数。
  2. bootstrap 子采样稳定性使用固定 `random_state` 且 KMeans `n_init=10` 固定 seed：测到的是 resample 方差，初始化方差被人为消除，稳定性指数被高估。
- **修复方向**：小 N 分支显式抛错或在输出中标记 `stability: null`；stability 评估中对 KMeans 初始化 seed 做扰动。

---

## MINOR（不影响判定，建议修复）

### MINOR-1
`D:\Projects\NSMoR\scripts\simulate_lesion.py:236-241` — `global_idx = batch_idx * B + i` 在最后一个不满 batch 时仍成立（shuffle=False），但依赖 DataLoader 不做 drop_last 与不 shuffle 的隐式约定；若有人改 shuffle=True，标签将整体错位且无断言报警。建议在 dataset 层直接携带 label 进 collate 返回值。

### MINOR-2
`D:\Projects\NSMoR\nsmor\model_nsmor_core.py:733-784` — 侧抑制 spike history 存于普通属性 `_spike_history`（TBPTT-1），跨 batch 泄漏防护依赖 `init_state()` 被调用的路径唯一性（`:1544`）。当调用方传入自定义 `lif_state0` 时 `init_state()` 不执行、spike history 不重置——自回归模式（`states=` 分支 :1400-1406）已处理，但直接调 `_run_lif_path(lif_state0=...)` 的第三方代码会继承上一序列的抑制史。建议把 spike history 并入 state tuple 强制传递。

### MINOR-3
`D:\Projects\NSMoR\nsmor\analysis\uq.py:108-110` — percentile 法 CI 未报告 bootstrap 标准误与 BCa 修正选项；对有偏统计量（如比率型 p_hat）percentile CI 覆盖率不足。建议补充 BCa 或至少 basic 反转法选项。

### MINOR-4
`D:\Projects\NSMoR\scripts\train.py:1142-1148` — Phase 2 新建 optimizer 后，`CosineAnnealingLR`（:1057）仍在旧 optimizer 上创建（scheduler 在 phase transition 分支之后构建于新 optimizer 之前……实际顺序是 scheduler 构建于 :1057，phase transition 发生于 epoch 循环内 ：1024-1048，此时 scheduler 引用的是旧 optimizer）。Phase 2 起 `scheduler.step()` 操作的是已被丢弃的 optimizer，LR 退火对新参数组失效（恒为 base_lr）。当前因 eta_min 差异小而不致命，但两阶段长训练下 LIF 组 lr=0.3×base 无退火会放大后期振荡。

---

## 结论

BLOCKER-1（单位制）与 BLOCKER-2（特征值分析的动力学-统计逻辑）属于必须重构的科学硬伤；MAJOR-1 必须修复后任何 lesion 显著性结论才可发表。工程防御性注释密度很高，但注释不能替代正确性。重构方向已逐条给出；按要求不代写修正代码。
