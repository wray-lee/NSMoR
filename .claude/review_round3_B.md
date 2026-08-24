# Reviewer B — Round 3 审查报告（双盲，聚焦生物完备性与统计严谨）

**判定：REJECT**

审查对象：工作树中相对 HEAD (37ee1fd) 的全部未提交改动（21 个文件，+1522/-203），即 Developer 对 Round-1/Round-2 审查意见的重构提交。

总体评价：Round-2 的修复方向正确——物理时间基（`dt_ms`）、跨拟合 OOF 先验、循环块自助法、Holm 步降链接、provenance 版本戳均是实质性改进。但提交物中存在一处证明代码从未被执行的必现崩溃，以及若干未闭合的统计学漏洞。以下按严重度分级。

---

## CRITICAL-1：analyze_jacobian.py 存在必然触发的 NameError —— 本轮改动从未端到端运行

- **位置**：`D:\Projects\NSMoR\scripts\analyze_jacobian.py:807`
- **证据**：frozen-input 控制分支中 `keep_mask = res_frozen < fp_threshold`。经 AST 级验证，`fp_threshold` 在函数 `compute_eigenvalues_at_epochs`（定义于 :620）内部既无赋值也不是形参；其唯一赋值点在 :575，属于另一个函数 `extract_gru_states_at_epochs` 的局部作用域。
- **后果**：任何真实运行中只要 `epoch_data` 非空且任一 epoch 有候选状态，:807 立即抛出 `NameError`。这不是边界情况，是 GRU-only 主分析路径上的必现崩溃。
- **推论**：Round-2 对 Jacobian 分析的全部"修复"（残差门控校准、吸引子验证、冻结输入控制、JSON 导出）没有一次成功执行的记录。在一个以科学严谨为目标的审查闭环中提交未经运行的统计管线，本身就是对流程的否定。
- **要求**：修复该变量传递（例如把 `fp_threshold` 作为参数传入或返回），并由 Tester 提供该脚本全流程真实运行的日志作为本轮放行前置条件。

## CRITICAL-2：固定点残差门控的"数据驱动校准"存在自指性与跨 epoch 选择偏倚

- **位置**：`D:\Projects\NSMoR\scripts\analyze_jacobian.py:143-180`（`_calibrate_fp_threshold`）、:570-590（池化校准与应用）
- **问题 A（自指性）**：Kneedle 式肘点对单调凸的排序残差曲线**总是**返回某个值；当状态空间根本没有慢流形时（恰是该检验要排除的情形），只要肘点侥幸低于硬编码 sanity cap 0.5，门控照样放行瞬态谱并为其背书。cap=0.5 本身亦未经论证：64 维 tanh 有界隐状态下逐步 L2 残差 0.49 意味着每步状态位移接近幅值上限，与"准定点"毫无关系。该门控无法证伪其前提。
- **问题 B（选择偏倚，统计学核心缺陷）**：阈值由**全体 epoch 混合分布**池化校准后逐 epoch 应用——本征残差较大的 epoch（如 Sustained）被不成比例地剔除，各 epoch 接受率不同；随后在**经过差异化筛选后的子样本**之间比较谱统计量。这是教科书式的 collider / 选择偏倚：跨 epoch 谱差异可能完全是门控筛选的产物而非动力学差异，而该谱比较正是本分析的核心假设检验。冻结输入控制的再门控（:790-820）没有纠正这一点——它只是用同一有偏阈值做了第二次筛选。
- **要求**：(i) 按 epoch 分别报告门控前后的完整残差分布与接受率；(ii) 对阈值做敏感性分析（如肘点 ±25% 区间内谱结论是否稳定）；(iii) 在报告中显式声明谱比较条件于共同筛选程序及其局限。

## CRITICAL-3：跨拟合先验仍留有三条未闭合的泄漏/失配通道

- **位置**：`D:\Projects\NSMoR\nsmor\mcmc_module.py`（`train_mcmc_cross_fitted`）、`D:\Projects\NSMoR\scripts\prepare_data.py:645-690`
- **(a) 组可行性零校验**：仅当 `groups is None` 时检查每类样本数 ≥ n_folds；代码注释自己承认 `StratifiedGroupKFold` 需要每类出现在足够多的 session 中，却没有实现任何断言。若某行为类只存在于少数 session（n_folds=5 时部分折的训练侧缺失该类），OOF 先验对应列退化为近常数——sklearn 只发 warning 不报错，管线静默产出退化的 "held-out" 特征。这与 provenance guard"宁可拒绝也不静默"的自我标榜直接矛盾。要求：对每类 × 每折的训练侧类别覆盖做硬断言。
- **(b) NSMoR train/val 划分叠加于 OOF 先验之上的残余泄漏**：试验 i（落入 NSMoR 验证集）的先验由未见过 i 的折模型产生，但该模型见过其他验证集试验的标签；这些兄弟验证试验与 i 共享 session 层级相关结构，验证性能因此被轻度抬高。严谨方案是嵌套划分（外层切 NSMoR train/val，内层仅在 NSMoR-train 内做跨拟合，验证集先验用折模型集成预测）。当前文档与代码均未讨论此层。
- **(c) 训练-服务分布失配**：训练输入是单折 OOF 概率（高方差），推理协议却是折模型集成的均值概率（方差压缩、校准特性不同）。模型学到的先验-行为映射是在前者分布上拟合的，部署时喂后者，无任何一致性检验或校准迁移分析。要求至少报告两种先验分布的差异统计量。

## MAJOR-1：单位制改革半途而废——不应期仍锁死在帧单位，复活了 BLOCKER-1 要消灭的病灶

- **位置**：`D:\Projects\NSMoR\nsmor\config_parser.py:52-58`、`D:\Projects\NSMoR\config\default.yaml:63`
- **问题**：`lif_rel_refract_steps: 2`（帧）。所有 tau 已改为物理 ms 并声称"改变采样率不再能静默重标定生物物理"，但相对不应期仍是帧单位：dt_ms 从 10 改为 1 时，20 ms 阈值恢复变成 2 ms——这正是 Round-1 BLOCKER-1 定性的同一类静默重标定，只是换了参数。注释中的辩解（abs 不应期 ~2 ms 小于一帧故不可表示）只对 abs 成立；rel 完全可以也应该像 tau 一样以 ms 声明、内部经 `exp(-dt/tau)` 类转换。同一配置两套时间单位并存，是对本轮核心修复的自我拆台。

## MAJOR-2：心理物理学推断条件于单一噪声实现，且用正态性预检验选统计量

- **位置**：`D:\Projects\NSMoR\scripts\simulate_psychophysics.py`（推断段，约 :530-660）
- **问题 A**：全部 σ 水平共用同一个 `--seed`，各水平的噪声实现仅差一个标度因子。共同随机数配对设计本身可辩护（降低配对方差），但推论随之条件于这一个噪声模式：p 值与 d_z 刻画的是"给定该实现的刺激集 × 噪声模式"，噪声实现方差完全未被量化。要求多 seed 重复（seed 作为随机因子纳入）或在 JSON scope 中明确声明推断范围。
- **问题 B**：Shapiro-Wilk 显著 → Wilcoxon、否则 → t-test 的预检验门控本身会膨胀第一类错误（pre-test 两难），是基础统计常识；且 Wilcoxon 检验的是差值分布对称性/中位数而报告 Cohen's d_z，范式混搭。固定一种检验（配对设计下推荐直接 Wilcoxon + Hodges-Lehmann 或直接 t + d_z）即可。

---

## MINOR（合并陈述，均需处理但不阻塞重构方向）

1. **BCa 边界处理**（`D:\Projects\NSMoR\nsmor\analysis\uq.py`，bootstrap_ci BCa 分支）：`prop_below ∈ {0,1}` 时 z0 强制为 0——prop_below=0 恰是极端偏置信号，置零等于假装无偏；调整分母 `1 − a(z0+zα)` 越过奇异点时应告警而非静默裁剪百分位到 [0,1]。
2. **循环块自助法的平稳性假设**（`D:\Projects\NSMoR\scripts\simulate_lesion.py`）：Politis-Romano 前提是序列平稳；session 内动物状态漂移即违反。"block_size 减半/加倍 CI 变化 <10%"的敏感性声明没有任何持久化证据（无输出文件、无测试）。要么落盘敏感性结果，要么删除该声明。
3. **吸引子验证样本量**（analyze_jacobian.py:664-674）：每 epoch 仅 n=min(10,N)，报告 fraction 无二项置信区间；n=10 不足以支撑任何关于"验证比例"的定量表述。至少加 Wilson 区间并扩大 n。
4. **labeling 参数收编不彻底**（`D:\Projects\NSMoR\nsmor\pipeline\labeling.py`）：`anchor_min_frames=2` 仍为函数内魔数，与 Round-2 m-3"持续判据收编进配置"的声明不一致；PREWALK 预刺激检查复用 `response_max_latency_ms`（刺激-响应潜伏期语义）作为自发行走的锚点搜索窗，应在配置层面拆分为两个独立参数。
5. **dt_ms 缺失仅 warning**（`D:\Projects\NSMoR\nsmor\model_utils.py:135` 附近）：既然 provenance guard 已保证 v2.0 戳存在，v2.0 checkpoint 缺 dt_ms 就是数据损坏，应为硬错误；warning 与你们自己的拒绝哲学不一致。
6. **侧向抑制回退常数未入配置**（`model_nsmor_core.py`）：`_inhib_tau_ms = max(tau_syn, 50.0)` 的 50 ms 回退仍是魔数。

---

## 重构要求（放行前置条件）

1. 修复 CRITICAL-1 并提供 analyze_jacobian 全流程真实运行日志（Tester 出具）。
2. 重新设计残差门控的报告与敏感性分析以消除跨 epoch 选择偏倚（CRITICAL-2）。
3. 补齐组交叉拟合可行性断言、嵌套划分方案论证与训练-服务先验分布一致性检验（CRITICAL-3）。
4. rel_refract 以 ms 声明并内部转换（MAJOR-1）。
5. 心理物理学多噪声 seed 重复或显式限定推断范围（MAJOR-2）。

以上完成前，任何下游产物（虚拟病变 CI、心理物理学图表、Jacobian 谱图）一律不予采信。
