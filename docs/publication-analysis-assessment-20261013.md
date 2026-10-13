# NSMoR：现有数据范围内的投稿证据与分析补充

日期：2026-10-13。范围：现有行为/刺激数据和已训练模型；不增加神经记录、其它传感器、刺激类型或目标维度。新增分析在看到前期结果之后制定，均为 **EXPLORATORY / descriptive**，不扩大冻结的 phase-2 48-slot 家族，不修改冻结评分器、训练配置和数据。

## 1. 判断与证据边界

当前适合定位为“行为预测与计算充分性”的研究原型。现有结果尚不足以支撑蟋蟀神经机制、跨动物泛化或顶刊级快/慢通路分工的强结论。补图的目的在于检验和清楚呈现证据，而不是把算法内部状态作为实测神经活动。

现有文档确认：2304 个带标签试次，64 个记录前缀，52/12 train/validation 前缀，1872/432 个试次；animal identity unverified。记录前缀与刺激条件完全嵌套，验证集缺 wind_only；scalar speed 是既定目标。模型选择用过 nested_outer_validation，不能将它称为 untouched test。旧 A1/A2 的 MSE 低于历史基线，但均不优于一步 persistence；这是旧阶段结果，不能代替正在完成的 R0–R3 三种随机种子结果。phase-2 在审计开始时完成 10/12，正式 48-slot 评分尚未运行。

证据：`docs/realdata-preliminary-conclusions-20261009.md`，`docs/realdata-exploratory-step1-20261010.md`，`docs/realdata-phase2-protocol-20261010.md`，`docs/realdata-phase3-protocol-20261010.md`，`docs/behavior-exploratory-20261010.md`。版本化数字以生成结果的输入 SHA 为准。

不能把一百万帧当作独立样本，也不能把三个优化随机种子当作三个动物。相同个体是否产生多个前缀尚未核实，因此以下前缀重采样区间仅是条件性敏感性描述，不能解决未验证的动物独立性。

## 2. 本次新增方法

### A. 数据支持与选择敏感性

从已存在的小型 model-free behavior summary 和冻结 nested-prior artifact 读取前缀、split、条件和试次计数，不加载整个 2.7 GB corpus。输出每个条件的训练/验证试次与前缀数、D1 stationarity 资格筛选、eligible 内 responders/nonresponders，以及既有 5 与 10 cm/s 资格阈值敏感性。所有分母写入 CSV/JSON，零支持明确显示为零；不以 whole-trial label 充当 grooming/resting/walking 注释。

这组图主要检验：模型在什么条件上有评估支持，模型无关的视觉响应有多少，筛选规则改变了多少试次。条件比较仍然是 between-recording groups；不作条件因果解释。

### B. 固定模型的记录删除敏感性

读取冻结 phase-2 family JSON 的 paired prefix means。记第 j 个前缀的 trial-mean error difference 为 d_j = candidate MSE - comparator MSE，则新描述量为 D_prefix = mean_j(d_j)。它是 equal-prefix 的估计对象，不等同于原评分器的 frame-pooled skill 或 trial-weighted mean delta。

依次删除每个记录前缀，计算 D_(-j)，输出全部删除估计、最小/最大值、最影响结果的前缀和方向是否稳定。**不重训练，故这不是 leave-one-prefix-out cross-validation，也不是新泛化证据。** 负值表示 candidate error 较低；零表示打平。

若输出前缀 bootstrap 区间，必须以 replacement 保留重复抽中的前缀权重，写清 bootstrap estimator 和“假定记录前缀独立/可交换”的限定。区间为 pointwise/descriptive，不作同时覆盖或新增显著性声明。只有一个前缀时删除估计 unavailable；缺失值和不可用 family slot 不当作负结果。冻结 Holm p、trial d_z 原样保留，新增方法不产生 p 值。

### C. 预测价值、时间尺度与优化随机性

直接绘制冻结四个 primary scopes 的 `1 - pooled MSE_model / pooled MSE_persistence`，并显示 h5/h25/h125 的 full/escape 预测增益。一步 observed history 与多步 feedback rollout 分开。horizon persistence 必须是 frozen scorer 的 held-at-start comparator，不混入 pure lag-h 的旧 ceiling。

随机种子图使用原产物的 per-seed trial-mean MSE，并标明 unweighted trial mean；不是 frame-pooled MSE，也不是生物学重复。R1 的 persistence skip 是模型定义的一部分；其增益只说明 residual head 是否超过已加入的 persistence。

### D. 通路与自发活动的必要描述分析

phase-2 protocol §8/8.1 要求每个 arm × seed 的 onset/sustained readout lesion、250 ms baseline / 3 SD departure pathway timing，以及 prestim predicted/observed escape rates。新增 runner 要保存实际 checkpoint/config/数据/prior SHA，验证 seed、target、dt 和 split，并将 R2 zero_input_channels 作用于每个实际分析输入。

只改变 g_lif/g_gru 的实验叫 **readout replacement/knockout**，不能称整条通路活动已被关闭。保留模型自然读出作为同试次对照，使用相同窗口/分母，逐试次/前缀保存差值。解释属于 model intervention。

时序应同时保存每条通路未响应比例和共同未删失子集的 paired latency difference。不同删失子集的 marginal medians 不能构成总体先后顺序；相等必须显示 tie。响应累计概率图保留未响应试次作为分母，避免只画 responders。实际观察窗口截断与未达到 departure threshold 的原因分别记录。

prestim frame false alarm 的分母只含 observed speed < 10 cm/s 的帧。预测 bout trial rate 与“观察无 bout 时的 false alarm”不是同一指标，要分别命名；两个连续 threshold frames 的 run 必须发生在未压缩时间序列，mask 中断 run。独立 prestim 行为注释缺失时，仅报告 speed/activity proxy 分布，并明确 grooming/resting/walking control unavailable。

## 3. 图的用途与状态

| 图 | 回答的问题 | 数据来源与门控 |
| --- | --- | --- |
| Fig 1 / Supplement: evidence_support | split、条件支持和筛选是否足够？ | 现成 behavior summary + nested prior；现在即可生成 |
| Fig 2: primary_skill + prefix_robustness | 超过 persistence 的增益有多大，是否被单个记录驱动？ | 12/12 后冻结 48-slot scorer JSON；不挑显著 cell |
| Fig 3: horizon_skill | 增益能否超过一步惯性并持续到更长 horizon？ | 同一冻结结果；full/escape、held-at-start comparator |
| Fig 4: readout lesion + pathway timing | onset/sustained 是否有分工，是否存在稳定先后？ | 每 arm×seed 次级 runner；同时展示 censoring/支持数 |
| Supplement: seed variation, prestim activity | 训练随机性及自发活动是否解释现象？ | 原始 per-seed summaries + 次级明细；不合并为 biological n |
| Dynamics panel | 是否存在可检验的模型内部局部动力学？ | 旧 fixed-map gate 失败；当前保持 withheld，不绘制无效 spectrum |

每个新图导出 PDF/SVG 和 PNG，附数值 CSV、机器可读 JSON、源产物和源码 SHA，写清单位、样本/记录分母、估计对象、exploratory 标记。新的图不会替换旧图或旧证据目录。

## 4. 仍需决定能否成立的项目

1. **预测基础**：等待冻结 48-slot 评分。phase-3 gate 按原规则读取 onset 与 h5 的 skill/d_z/Holm；不能因为新 exploratory 图好看就改 gate。
2. **可识别性**：原始录制元数据如果能确认 animal ID，可在新版本数据中恢复正确 cluster；这是同类数据元数据补全，不要求其它数据模态。不能从 filename 的 `cricket_001` 自动认定同一动物。条件与记录完全嵌套时，mixed-effects 随机截距也不能识别条件与记录效应。
3. **真正泛化**：现有验证集已用于选择。若剩余 22 只属于尚未处理的同类型数据，应先确认个体与条件，再为未来版本固定 untouched evaluation；不得回写当前冻结 corpus，也不得将同一旧验证集事后升级为 test。本次不启动新训练。
4. **时钟和响应定义**：`docs/clock-recovery-r4.md` 的 scientific timing acceptance 仍 pending。下一步应对实际有依据的时钟重建方案进行 paired 事件归属/response-window sensitivity；没有证据的 ±ms 不能宣称 operational tolerance。
5. **机制对照**：parameter/compute-matched 单通路或对称通路对照、真实 cross-condition prediction，是未来机制证据的必要条件。现成 phase-3 protocol 已限定预算/门控；本次不私自启动那套训练，不凭 lesion 图宣称神经分工。
6. **新动力学**：slow points 或 trajectory-local response 可成为新 exploratory 分析，但必须验证执行状态、输入、finite perturbation fidelity 与窗口敏感性；只做 GRU block 不能叫 full-system stability，LIF hard spikes 的 surrogate derivative 也不能自动当作真实物理 Jacobian。本次不绕过旧 gate。

## 5. 方法依据

- Saravanan, Berman & Sober (2020). *Application of the hierarchical bootstrap to multi-level data in neuroscience*. Neurons, Behavior, Data Analysis, and Theory; arXiv:2007.07797. 用于明确层级非独立性；本次前缀均值 bootstrap 不是该文完整多层 hierarchical bootstrap。
- Kapoor & Narayanan. *Leakage and the Reproducibility Crisis in ML-based Science*. arXiv:2207.07048. 用于区分验证选择与真正测试，以及完整数据/选择流程的可复现性。
- Sussillo & Barak (2013). *Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks*. Neural Computation 25:626–649; DOI 10.1162/NECO_a_00409. 用于说明 fixed/slow points 与局部线性化是模型计算解释工具，需验证慢点/线性近似，不能把无效固定点谱当生物学证据。

未声称完成系统性 novelty 文献综述、动物身份核验、时钟科学验收或发表等级评审。新增代码需经过双独立 ACCEPT 与 WSL 回归，产物状态在交付记录中单独标明。

## 6. 新颖性定位的初步检索

近期原始研究已经讨论蟋蟀气流逃逸的感觉计算和神经输出。例如 Clémençon 等的 *Perception of aerodynamical looming stimuli*（Current Biology，2025；PubMed PMID 41187755）结合行为与气流敏感中间神经元的活动分析；Chida、Shidara 与 Ogawa 的 *Neural circuit architecture and directional information processing of airflow stimuli in the cricket brain*（Journal of Neurophysiology 134:1214–1231，2025，DOI 10.1152/jn.00254.2025）直接测量气流响应神经元。这里只核对了题录和摘要，不把这些实验当作 NSMoR 的验证数据。

因此，NSMoR 不应把“蟋蟀气流逃逸可建模”或“多感觉刺激改变行为”单独当作机制创新。现有数据下，更可信的候选贡献是：明确超越行为惯性的多时间尺度预测、跨记录稳定的行为签名，以及通过可证伪的模型干预比较不同计算解释。是否达到这一贡献，要由尚未完成的冻结评分和必要机制对照决定。未完成的完整 novelty review 不因这两个例子而视为完成。

### 交付时的两个额外限定

现有 behavior summary 没有保存源 corpus/prior 的 SHA；支持图可以核验该 summary 的计数/前缀映射与 nested artifact 的 split 结构一致，并保存两个输入文件的实际 SHA，但不能倒推出 summary 已与该 corpus 字节绑定。本次不为补这个记录而在健康训练期间重新加载全语料。

冻结评分器没有保存逐试次 ID、frame counts、persistence vectors 或有符号残差。记录删除分析因此仅针对已保存的 prefix-mean delta；不计算伪造的 frame-pooled 删除 skill。新的次级 runner 从对齐预测直接保存自然输出的 window residual bias/variance，与“跨种子预测方差”明确区分。当前稳健性图使用删除范围，不额外输出 bootstrap CI。


