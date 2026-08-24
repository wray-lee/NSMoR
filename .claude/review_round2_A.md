# Reviewer A 复审报告 — Round 2（针对 Round 1 修复的独立双盲复审）

**[is_accepted: FALSE]**

**REJECT**

审查方式：通读全部未提交 diff（14 文件，+823/-170），并对关键修复逐一运行了可执行验证（单位换算数值、Holm 步降判定、circular block-bootstrap、5-fold cross-fitting 全流程、dendritic 状态重置、labeling 判据边界、`pytest tests/` 全套 111 项通过）。

先肯定：Round 1 的多数 BLOCKER/MAJOR 在实现层面已正确修复——`_decay_per_step` 换算经我实测正确（alpha_syn=exp(-10/5)=0.1353）、Holm 实现通过我的步降联动测试（rank-1 拒绝失败时后续全部不拒绝，判定正确）、circular bootstrap 环绕索引无误、cross-fitting 在合成数据上 OOF 行概率合法（行和误差 1e-7，随机特征上 acc≈0.24 证明无泄漏）、dendritic 状态同形状重复 forward 与 fresh model 输出一致。工程质量显著提升。但以下问题仍构成否决级缺陷。

---

## CRITICAL-A【计算统计学 · 致命】：修复只落在代码层，磁盘上的全部科学产物仍是污染版本，且无任何版本防护阻止新旧语义静默混用

- 我实测 `D:\Projects\NSMoR\runs\default\best_model.pth`（epoch 240）：checkpoint config 中**没有 `dt_ms` 键**，且 `lif_rel_refract_steps=0`。`load_model_from_checkpoint` 会以默认值 `dt_ms=10.0` 重建模型——该模型训练时 `lif_tau_syn=5.0` 按**帧单位**解释（等效 50 ms 滤波，alpha≈0.82），加载后按新语义解释为 **5 ms**（alpha=0.135）。buffer 是 `persistent=False`，`load_state_dict(strict=True)` 静默通过。也就是说：**仓库中每一个旧 checkpoint，经任何分析脚本加载后，运行的都不是它训练时的生物物理系统**，无警告、无断言、无日志。
- 同理，`data/processed/nsmor_dataset.pt` 是用**旧泄漏版 MCMC 先验 + 旧 np.max 标注**生成的——我检查过，其中没有 `mcmc_prior_provenance` 字段。代码里的 cross-fitting 修得再漂亮，当前数据集的每一列先验仍然是对同一 trial 标签过拟合的输出。
- Round 1 已明确要求"修复后所有下游产物必须全部重跑"。没有任何证据表明重跑发生。
- **修复方向**：
  1. 在 checkpoint 保存/加载路径写入并校验 `pipeline_version`/`tau_semantics` 溯源键，检测到旧语义 checkpoint 时显式报错或告警；
  2. 用新代码重新执行 prepare_data → train → 全部分析脚本；
  3. 数据集落盘时持久化 provenance 字段并在加载端断言。

## CRITICAL-B【数学 · 致命】：gating_cluster.py bootstrap 稳定性修复引入了新的尺度失配

- 位置：`D:\Projects\NSMoR\nsmor\analysis\gating_cluster.py:619-643`。
- `kmeans_boot` 在 `boot_scaler.fit_transform(fingerprints[idx])` 的坐标系中拟合质心；但预测 OOB 时构造了一个**仅在 OOB 子样本上重新 fit 的 `oob_scaler`** 再喂给 `kmeans_boot.predict(oob_scaled)`。两个 StandardScaler 的均值/方差不同——质心位于 boot 坐标系，输入却是 OOB 坐标系。这不是"resample 内重新估计"，是把两套不可交换坐标系的聚类做 ARI。ARI 被系统性压低（不稳定性被高估），方向虽与旧偏差相反，但同样是测量伪影而非稳定性。`kmeans_oob` 那条腿用 `oob_scaled` 自己拟合是对的，错在 `kmeans_boot.predict(oob_scaled)` 这一步。
- **修复方向**：OOB 预测必须用 `boot_scaler.transform(fingerprints[idx_oob])`（与质心同一坐标系）；`labels_oob_true` 一侧可保留 OOB 内独立 scaler+KMeans。

## MAJOR-C【生物物理学/标注语义】：sustained 判据的锚定窗口会跨过刺激 onset，把 ESCAPE 反应帧计入 PREWALK 证据

- 位置：`D:\Projects\NSMoR\nsmor\pipeline\labeling.py:101-113`。
- 我构造了反例并实测：pre-stim 检查窗口为 `[stim-1000, stim+∞)` 内由锚点延伸的 `[anchor, anchor+1000 ms)`。当锚点位于搜索窗末端（如 onset 前 190 ms 的单帧传感器毛刺）且真实 pre-walk 仅 ~40 帧（400 ms）时，onset 后 1060–1190 ms 的 **ESCAPE 逃逸奔跑帧落入持续窗**，把本应判 ESCAPE 的 trial 翻转为 PREWALK（实测翻转成功：40 帧真实 pre-stim 行走 + 1 帧毛刺 + 14 帧 escape 外溢 = 55/100 ≥ 50%，若无外溢则为 41/100 < 50%）。ground truth 标签被刺激后响应污染——这正是本轮要修的那类语义错误的残余形式。
- **修复方向**：PREWALK 的 pre-stim 检查中，持续窗必须在 `stimulus_onset_ms` 处截断（或等价地要求锚点+duration 整体位于 onset 之前）；单帧毛刺作为锚点还应要求锚点本身满足最小连续性（如连续 ≥2 帧超阈）。

## MAJOR-D【计算统计学】：psychophysics 推断段自相矛盾——刚在 lesion 里废除的两个坏模式在这里原样复活

- 位置：`D:\Projects\NSMoR\scripts\simulate_psychophysics.py:567` 与 `:530` 附近。
  1. `if np.isnan(p_raw): p_raw = 1.0` —— 数值病理伪装成"无效应"，且该 1.0 **留在 Holm 家族内稀释校正**。Reviewer A 在 lesion 脚本上否决的正是这个模式，同一个 PR 里一处改成排除并计数、另一处照旧吞掉。
  2. `sd_diff <= 1e-12 → d_z = 0.0` 静默归零——零方差配对差是退化样本，应计入 `n_degenerate` 并从检验家族剔除，与 lesion 侧处理保持一致。
  3. `inject_visual_noise` 用裸 `torch.randn`，全程无种子——配对设计的 p 值不可复现；至少应在 main 入口固定并记录 generator。
  4. 精心定义的 per-condition `snr_db` 只进了 logger，summary JSON 里只有 `snr_definition` 字符串而没有各 σ 的 SNR 数值——统计产物不自足。
- **修复方向**：NaN/零方差一律剔除出家族并计数上报；噪声注入绑定显式种子写入 JSON；per-σ SNR 数值落盘。

## MINOR（记录在案）

- **MINOR-E**：`D:\Projects\NSMoR\scripts\analyze_jacobian.py:599-615` —— `n_verified/n_tested` 在 epoch 循环外初始化、循环内累加，每个 epoch 打印的 "%d/%d" 实为累计值却冠以单 epoch 名义；且 `min(10, Nv)` 取前 10 个状态无随机抽样，存在按批次排序的选择偏倚。frozen-input 对照结果只进日志不进 JSON/图。
- **MINOR-F**：`simulate_lesion.py` 的 `block_size=5` 硬编码，未从数据的自相关衰减尺度估计或做敏感性说明。
- **MINOR-J**：`mcmc_module.train_mcmc_cross_fitted` 返回的 `fold_models` 在 `prepare_data.py` 中被丢弃——docstring 承诺的"推理期 ensemble"没有落地路径。
- **MINOR-K**：`bootstrap_ci` 的 percentile 法仍未提供 BCa/basic 选项（Round 1 MINOR-3 未动）。

---

## 结论

代码层的修复质量比上一轮有明显提升（换算、Holm、circular bootstrap、cross-fitting 均经我实测通过），但：

1. CRITICAL-A 使本轮所有现存产物在科学上仍然无效——修好的引擎接在污染的燃料上；
2. CRITICAL-B 是修复过程中引入的新数学错误；
3. MAJOR-C 有我构造的实证反例；
4. MAJOR-D 与同 PR 内已采纳的标准直接矛盾。

四项全部解决并完成全量重跑后，方可进入下一轮审查。

— Reviewer A（Round 2）
