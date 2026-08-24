# Reviewer A 复审报告 — Round 3（针对 Round 2 修复的独立双盲复审）

**[is_accepted: FALSE]**

**REJECT**

审查方式：通读全部 working-tree diff（21 文件，+1522/-203），对关键修复逐一执行可执行验证（BCa 与 Holm 步降数值实测、`_check_sustained_speed` 构造性反例、elbow 校准器合成分布探针、产物溯源字段与时间戳审计、`pytest tests/` 全套 111 项通过）。

先承认进步：Round-2 的四项否决级缺陷在代码层全部得到正面回应且方向正确——

1. **CRITICAL-A（污染产物无防护）**：`nsmor/checkpoint.py:_require_pipeline_version` + `nsmor/config.py:PIPELINE_SEMANTICS_VERSION="2.0"` 溯源门已落地；我实测旧 `runs/default/best_model.pth`（无版本键）会被加载路径拒绝，新 `data/processed/nsmor_dataset.pt`（2026-08-24 重生成）携带 `pipeline_semantics_version="2.0"`、`mcmc_prior_provenance="oof_5fold_session_grouped_cv"` 与 5 个 fold models——B-2/M-3 同时闭环。全部 8 个数据集消费脚本均接入 `validate_dataset_provenance`。
2. **CRITICAL-B / B-1（bootstrap 特征空间失配）**：OOB 现在用 `boot_scaler.transform` 投影（`gating_cluster.py:613-621`），坐标系统一，修复正确。
3. **MAJOR-C（sustained 窗跨 onset）**：`hard_end_ms` 截断 + 锚点连续性 ≥2 帧。我构造三个边界案例实测：940 ms 真实 pre-walk 含 60 ms 亚阈凹陷 → True（凹陷容忍正确）；仅 100 ms 贴 onset 短爆发 → False；500 ms 连续 pre-walk → False（后者见 M-A-2 讨论）。Round-2 的逃逸帧污染路径已封死。
4. **MAJOR-D（psychophysics 推断段）**：p_values/effect_sizes 无条件绑定（NameError 消除）、NaN p 与零方差配对差剔除出 Holm 家族并计数（与 lesion 侧标准一致，我实测 Holm step-down 联动判定正确：{.001,.03,.04} → c 被联动压制）、噪声注入绑定显式 Generator 并落盘 `noise_seed`、per-σ SNR 数值进 JSON。
5. MINOR 全部处理：m-1 默认值统一（`lif_rel_refract_steps=2` 双源一致）、m-2 缺 dt_ms 告警、m-3 参数入配置并给出 Bean 2007 生理学辩护、MINOR-K 补上 BCa（我实测 lognormal 偏态样本上 BCa 区间正确覆盖点估计且与 percentile 有合理差异）、MINOR-E per-epoch 归因 + 随机抽样 + JSON 落盘、M-2c/d attractor 验证与 frozen-input 对照补残差再门控。

工程质量较上一轮显著提升。但以下缺陷仍构成否决。

---

## BLOCKER

### BLK-3A【计算统计学 · 致命】：固定点残差门的 "Kneedle elbow" 校准在凸形排序曲线上数学失效，接受数由采样噪声决定

- **位置**：`D:\Projects\NSMoR\scripts\analyze_jacobian.py:146-180`（`_calibrate_fp_threshold`）与 :573-587（调用点）。
- **问题**：残差的**排序曲线单调上升且通常为凸**（大量低残差候选 + 少量高残差瞬态）。对凸曲线，端点弦位于曲线上方或附近，`argmax(r_sorted - chord)` 落在曲线左端而非真正的双峰谷底。我用三组合成分布实测：
  - **双峰分布**（60 个真准不动点 resid~N(0.02,0.01) + 40 个瞬态 N(0.30,0.05)，即该函数 docstring 设想的理想场景）：elbow = **0.0071**，60 个真 stationary 候选只接受 **1 个**——门把要找的东西几乎全部扔掉；
  - **纯均匀噪声**（200 个无任何流形结构的残差）：三次重复分别接受 28、186、133 个——同一无结构总体，接受比例从 14% 摆到 93%，完全由采样噪声决定；
  - 单峰 lognormal：接受 0 个。
  即：这个门在有慢流形时系统性拒绝真不动点，在无慢流形时随机放行大半瞬态。Round-2 M-2a 要求的"数据驱动标定"被实现成了一个**比原硬编码 0.1 更不可辩护的估计器**。所有下游特征值谱的入选集合由这个失效估计器决定——谱分析的科学结论建立在沙上。
- **修复方向**：对凸单调曲线做 elbow 必须先凹化（对 `log r_sorted` 或二阶差分/最大二阶跌落 `max(r[i+1]-r[i])` 定位双峰间隙），或直接用双峰混合模型（1-D 两分量 GMM 以 BIC 选模）取后验分界；同时保留 sanity cap。无论选哪种，必须在合成双峰+单峰基准上展示接受率分离度作为单元测试——本轮的实现连 docstring 自设的理想场景都无法通过。

### BLK-3B【计算统计学 · 致命】：重跑后的"科学产物"是一套 PREWALK 类为零、R²=0.19 的失败训练——当前磁盘上的证据链不足以支撑任何下游结论

- **位置**：`data/processed/nsmor_dataset.pt`（2026-08-24 04:57）与 `runs/test_r2_final*/best_model.pth`（2026-08-24 05:16）。
- **事实链**：(a) 新数据集中 `Label.PREWALK` 计数为 **0/360**（ESCAPE 78、PRE_ACTIVE 189、NO_RESPONSE 93）；(b) 新 checkpoint 的 val R²=**0.188**、MSE=5.11，而旧污染管线是 R²=0.466；(c) 主力目录 `runs/default/best_model.pth` 仍是 **2026-07-29 的 pre-2.0 checkpoint**（无版本键，加载即 RuntimeError）；(d) `results/` 下全部图与 JSON 停留在 2026-07-29——没有任何一个用新语义生成的分析产物存在。
- **问题分三层**：
  1. **标注判据改严后一类行为整体消失**却无人过问。PREWALK=0 不是"更严谨"，是新的 sustained 判据（锚点连续性 + 50% 分数 + 200 ms latency 上限）与真实动物行走统计不相容的直接信号。要么判据参数错（如 pre-walk 检查窗起点 `onset-1000` 处锚点搜索带内动物恰逢停顿），要么数据本身 pre-stim 行走不足 500 ms——两种情况都必须先诊断并报告，而不是静默产出三类分类问题。jacobian 分析默认 `--target_class 1`（PREWALK），在该数据上将得到**零个状态**、必然 abort——默认实验路径已死。
  2. **性能崩塌未解释**：R² 从 0.47 → 0.19 可能是泄漏移除后的真实泛化水平（这本身重要），也可能是重跑超参不匹配。报告必须明确区分"诚实去泄漏的代价"与"回归"。
  3. **20 epoch 的 test run 冒充重训**：`test_r2_final4` 只有 epoch_10/epoch_20，而正式协议是 300 epochs。没有一次完整训练发生。
- **修复方向**：诊断 PREWALK=0（输出各判据阶段的淘汰瀑布：多少 trial 死于锚点连续性、死于 hard_end 截断、死于 min_fraction），要么修正判据、要么在报告中声明数据不支持四类标注并把全部分析降为三类；随后以完整 epoch 数重训并重跑全部下游脚本；`runs/default` 旧产物必须删除或归档到明确的 legacy 目录，防止 best_model.pth 路径惯性复用。

---

## MAJOR

### MAJ-3A【生物物理学/统计学】：PREWALK 判据的锚点搜索窗设计使"pre-stim 行走"的判定依赖刺激前恰好 200 ms 内的运动状态

- **位置**：`nsmor/pipeline/labeling.py`（`classify_response` 中 pre-stim 检查，start_ms=onset−1000）。
- **问题**：pre-stim 检查以 `start_ms=onset−1000` 为**搜索窗起点**，锚点必须落在 `[onset−1000, onset−800)` 内（受 max_latency_ms=200 与 hard_end 截断约束）。一只在 onset 前 700 ms 就开始持续行走的动物，其锚点不在搜索带内 → 判 False（我实测 Case2：onset 前 500 ms 起连续行走至 onset → **False**）。也就是说：**越早开始行走的动物越不可能被判 PREWALK**——这与"pre-stimulus walking"的行为学定义正好相反。PREWALK=0 很可能主要由此产生（BLK-3B 的机制解释）。正确的语义应是：在整个 `[onset−1000, onset)` 窗内验证 ≥50% 帧超阈，而不是把 1 s 窗错误地当作 latency 搜索带来用——latency 语义属于 post-stim ESCAPE 检查，机械地复制给 pre-stim 检查是概念错位。
- **修复方向**：pre-stim 检查改为窗分数判据（`mean(abs_v[win] > threshold) >= min_fraction` over `[onset−1000, onset)`），去掉 latency 锚定；或以窗首为锚但允许锚点搜索覆盖整个 pre-stim 窗。

### MAJ-3B【工程/统计】：frozen-input 对照的残差门复用了原始空间的校准阈值

- **位置**：`scripts/analyze_jacobian.py:807`（`keep_mask = res_frozen < fp_threshold`）。
- **问题**：`fp_threshold` 由各状态**自身输入**下的残差分布校准；frozen-input 对照中残差在公共 median 输入下重算，其分布位置整体不同（对照日志自己就在打印 "median residual"）。用一个分布的肘部阈值去门控另一个分布，两侧通过率不可比——对照的幸存集合与主分析的幸存集合之间没有可比的解释基础。此外若 `_calibrate_fp_threshold` 在候选为空时从未执行（`if n_candidates:` 不成立），`fp_threshold` 成为未绑定名，:807 处 NameError（虽然该路径下 epoch_data 为空、循环不进入，属潜伏而非必现）。
- **修复方向**：frozen-input 残差单独校准（同一估计器、独立分布），或在 JSON 中同时报告两侧完整残差分布而非只报通过数。

### MAJ-3C【统计学】：StratifiedGroupKFold 在 22 个 session 上划分 5 折，fold 间组数与类构成不平衡未检验

- **位置**：`scripts/prepare_data.py:657-668`。
- **问题**：22 sessions / 5 folds ≈ 每 fold 4–5 组；StratifiedGroupKFold 不保证 fold 间类比例平衡，也不保证每折训练侧含全部四类（某类若集中在少数 session，可能整类缺席某折训练侧，该折模型对该类先验输出退化为常数列）。代码无任何折后诊断（每折类直方图、OOF 先验熵下限检查）。360 个 trial、78/0/189/93 的类分布在 4–5 组一折的粒度下方差很大。
- **修复方向**：落盘每折 train/test 的 (session 数, 类计数)；加 OOF 先验质量断言（任一类的 OOF 先验方差不得低于阈值，否则告警）；考虑 n_folds=4 或留一组法（LOCO：22 组留一太贵，可用 GroupKFold with shuffle 重试多次取最优平衡划分并记录选择依据）。

---

## MINOR

### m-3a
`scripts/simulate_psychophysics.py`：`inject_visual_noise` 的 Generator 按 σ 循环内每次重建并以同一种子播种——σ=5/15/30 各自消耗独立的随机流，配对设计中不同 σ 的噪声实现相互独立，这是对的；但 σ 扫描顺序耦合了 `args.seed` 相同前缀流，建议在 JSON 中额外记录每个 σ 的实际派生种子以便逐条件复现。（轻微）

### m-3b
`tests/test_gating_cluster.py` 对 stability None 的三处放宽使 `test_stability_high_for_well_separated` 在 score=None 时静默跳过断言——"well-separated 数据应测得高稳定性"的保护意图被架空。应在该测试中显式要求 N 足够大时 stability 非 None。

### m-3c
`analyze_jacobian.py` 的 attractor 验证 `n_check=min(10,Nv)` 样本量仍过小：10 个状态的二项比例 95% CI 半宽 ≥±0.31，`fraction` 字段的精度声明超出其信息含量。JSON 中至少附 Wilson 区间。

### m-3d（正面确认）
uq.py BCa 实现正确（z0 退化保护、加速度分母退化回退 bias-corrected、percentile clip）；circular bootstrap 模回绕正确；Holm step-down 联动经实测符合 Wright (1992)。checkpoint provenance 门在旧 checkpoint 上确实抛 RuntimeError（实测）。dendritic state 每次 forward 显式置 None/恢复，批间泄漏路径关闭。这些予以肯定。

---

## 结论与修复优先级

两处 BLOCKER 均足以否决：

1. **BLK-3A**：替换 elbow 估计器（凹化/GMM），并用合成基准测试锁定行为；
2. **BLK-3B**：诊断 PREWALK=0 淘汰瀑布 → 修判据（见 MAJ-3A）→ 完整重训 → 重跑全部分析 → 清理 runs/default 旧产物；
3. MAJ-3A 是 BLK-3B 的机制根因，优先于其他 MAJOR;
4. MAJ-3B/3C 与 MINOR 随下一轮一并提交。

代码层的修复纪律在本轮有实质进步（溯源门、cross-fitting 分组、BCa、Holm 联动均经我实测通过），但"修好了引擎"仍然不等于"产出了科学"。当磁盘上的唯一一套产物是一个三类问题上的 20-epoch R²=0.19 模型时，本轮审查只能得出"尚无可评审的科学结论"的判断。

— Reviewer A（Round 3）
