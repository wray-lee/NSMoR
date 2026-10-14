# 多感官整合与 Mixture-of-Recursions：重要解释转折记录

日期：2026-10-14（JST）  
版本：v2；按用户指出的原始 MoR 论文纠正定义。  
状态：工作假设、定义核对与论断边界；不是已验证的动物机制结论。  
目的：保存本次重要解释转折，供后续分析、图表、结论写作和方案审查使用。

## 1. 先纠正 MoR 的含义

本项目所讨论的核心 MoR 指 **Mixture-of-Recursions**：Bae 等（2025），*Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*，arXiv **2507.10524**，后发表于 NeurIPS 2025。论文由 KAIST、Mila、Google 相关机构和 Université de Montréal 的研究者共同署名。

原文的核心是：**共享计算块在递归深度轴上重复使用，并由路由器为不同输入 token 自适应分配递归次数。**不是仅仅混合两条 recurrent 通路的输出。

原文 §2.2.1 描述两类路由：expert-choice 在每个递归深度选择继续计算的 token；token-choice 起始时为 token 选择递归深度。论文还包含配合递归计算的 KV 缓存策略。它是人工 Recursive Transformer 的计算框架，并没有验证蟋蟀神经系统采用同样机制。

**纠正此前解释：**把仓库类名 `MoRRouter`、LIF/GRU 通路混合、ACT 停止策略和原始 MoR 当作同一个机制，会误导结论。ACT 是另一种自适应计算方法，不能代替原始 MoR 论文的定义；NSMoR 的 ACT 实现也不能未经对照就称为原文算法的完整复现。

## 2. 本次真正的解释转折

当前更值得检验的行为假设是：**风刺激主导逃逸启动，视觉调节威胁准备、触发阈值或运动增益的不对称整合。**这是暂定方向，尚未在本语料中辨认出具体机制。

**视觉调制不能否定 Mixture-of-Recursions；视觉调制完全可能通过自适应递归深度实现。**视觉调制描述发生了什么行为变化，Mixture-of-Recursions 提供一种候选计算实现，两者不是互斥选项。

同样，**“LIF 快 / GRU 慢”或“LIF 起始 / GRU 持续”的分工没有得到支持，也不能推出自适应递归不存在或存在概率低。**通路分工和递归深度是不同假设。此前把两类证据连在一起评价 MoR，论断范围不成立。

新的研究问题是：**视觉是否改变内部表征及递归深度分配，并由此解释风诱发行为中的额外结构；还是较简单的阈值、增益或连续积累已经足够？**当前没有可靠的 MoR 存在概率或候选机制概率排序。

## 3. 当前证据与其限度

| 证据 | 当前可以说 | 仍不能说 |
| --- | --- | --- |
| D1：单风组响应者时延中位数 120 ms，多感官组 84 ms；prefix 中位时延的组间均值差 38.4 ms | 当前定义下，多感官组呈现更快的响应 | 视觉因果性地缩短神经计算时间；差异由 MoR 造成 |
| D1 visual_only：497 个 eligible trials 中 8 个碰撞后 responder | 该资格门与触发参考下，碰撞后的阈值跨越较少 | 视觉不能触发逃逸、不参与整合；碰撞前不存在视觉准备 |
| 旧三个模型：替换 GRU 读出在起始和持续窗口都更有害；RMS（均方根）响应先后随模型变化 | 旧模型未支持稳定的特定快慢通路分工 | 否定自适应递归；判定动物没有 MoR |
| 本轮 R0–R3 的 12 个 checkpoint：`refinement_mode=off`，`time_consuming_mode=off` | 本轮评估预测地基；没有启用这些递归模块 | 把其好坏解释为 Mixture-of-Recursions 的有效性检验 |
| 本次记录时正式 48 槽评分、新 12 模型 secondary 分析和图表验收未完成 | 保存待检验假设，并等待实际结果 | 本轮已经验证或否定双通路 / MoR |

12 个 checkpoint 的递归配置已从实际保存的 `config.model` 独立读取，并逐一核对其完整 SHA-256 与 exact12 preflight 一致；未载入语料、未做 forward、未训练。审计回执：[exact12 recursion modes](.scratch/phase2-runtime-audit-20261014/exact12-recursion-modes-audit-20261014-v1.json)。这是一项工程事实核对，不是机制验证。

行为边界必须保留：condition 与 recording prefix 完全嵌套，动物身份未核验；组间比较不能消除 session/个体混杂。D1 visual_only 以碰撞为触发参考，响应搜索严格在其后，资格门覆盖触发前 250 ms。因此 8/497 不能概括所有视觉诱发行为，也不能忽略可能的碰撞前处理。

旧模型 RMS 偏离时序并伴随较高删失，不是测得的神经通路时延。读出替换不等同于关闭动物的通路。旧三个模型的 secondary 结果也不能代替新 12 个模型的实际结果。

来源：[行为分析](docs/behavior-exploratory-20261010.md)、[旧模型探索分析](docs/realdata-exploratory-step1-20261010.md)、[Phase-2 协议](docs/realdata-phase2-protocol-20261010.md)。行为数值沿用已有记录，本文件未新增原始数据分析。

## 4. 原论文、仓库组件与动物机制分开命名

| 对象 | 实际含义 | 与原始 MoR 的关系 |
| --- | --- | --- |
| Bae 等的 Mixture-of-Recursions | 共享递归计算块、输入 token 相关的递归深度分配及配套缓存 | 本讨论的定义来源 |
| 仓库的 `MoRRouter` 类 | 逐时间步生成 LIF/GRU 的两个融合权重 | 历史类名；此模块自身没有实现递归深度分配 |
| `AdaptiveLatentRefinement` | 融合后共享更新块、ACT 累积停止及输出混合，已停止的有效行不再计算 | 具有共享迭代与自适应深度的相关思想，但不是原文 expert/token-choice 路由的同一实现 |
| `depth_delay` / `accumulate` | 将内部处理映射到模型网格上的输出延迟 | 额外的行为时长建模；原文并未给出蟋蟀深度→神经时延映射 |
| 蟋蟀的多感官加工 | 需要通过行为证据与可辨识对照检验的生物过程 | 不能由人工网络的组件名或成功拟合直接确认 |

仓库定义来源：[模型核心](nsmor/model_nsmor_core.py)、[ADR 0008](docs/adr/0008-adaptive-latent-recursion.md)、[ADR 0009](docs/adr/0009-time-consuming-recursion.md)。本文件只纠正解释与命名范围，没有修改核心代码、配置或 checkpoint。

## 5. 视觉调制如何可能涉及自适应递归

以下是从共享计算与可变深度思想提出的候选假设，不是已观察到的动物事实：

1. 视觉先形成威胁/准备表征。风刺激到来时，联合表征影响深度路由，改变共享计算块还需执行多少次。
2. 一致且明确的线索，可能使部分输入以较浅深度完成处理；不确定的输入可能需要进一步递归细化。因此更快的多感官反应与自适应递归可以相容。
3. 视觉调节也可能只改变表征值或输出增益，递归深度仍然不变。这会支持调制，但不能单独支持自适应深度。

不能预先固定“多感官一定更多递归”或“更多递归必然对应更长动物反应时”。原始 MoR 的 token 深度、NSMoR 的每帧隐变量深度和动物反应时是三个不同量；它们之间的映射必须验证。

这里应问的是“递归深度的变化是否提供额外解释”，而不是把“调制”和“递归”当作对立机制。单纯发现多感官更快，既不确认也不排除 Mixture-of-Recursions。

## 6. 后续什么证据可以支持或削弱递归解释

既定顺序仍是冻结 48 槽评分 → 新 12 模型 secondary 分析与图表验收 → 按授权和协议执行后续计划。Phase-2 的基础门槛不等于 MoR 检验。

针对递归，真正相关的对照是共享块在固定深度与自适应深度下的表现，以及参数、信息可用性和执行预算的明确匹配。深度、输出时序及行为误差需分别记录；“权重改变”不等同于“深度改变”，“深度与时延相关”也不等于动物递归机制得到确认。新增确认性比较不能在看过结果后混入冻结 family。

当前 Phase-3 协议承认，`depth_delay` 与连续 `accumulate` 可能产生相近的行为结果。即使计划内检验有利，最多支持“模型加工时长与当前行为数据相容，递归可作为一种离散近似”，不能写成“蟋蟀具有 Mixture-of-Recursions”。没有检出差异也不证明候选等价或机制不存在。

对称 GRU+GRU / LIF+LIF 模式在当前协议中尚不存在、未预算、未安排运行。即使未来实施，这类对照直接检验通路异质性；仍需要另外针对递归深度的对照，不能把两类检验合并解释。

来源：[Phase-3 协议 §3、§6、§12、§13](docs/realdata-phase3-protocol-20261010.md)。

## 7. 原始研究与适用范围

- **定义来源：Bae 等（2025）**，*Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*。arXiv：2507.10524；核对版本：v3；NeurIPS 2025 正式稿。§2.1 共享参数、§2.2.1 expert-choice/token-choice、§2.2.2 KV 缓存。该论文是人工语言模型研究，不是动物机制验证。
- Kanou, Matsuyama & Takuwa（2014），*Effects of Visual Information on Wind-Evoked Escape Behavior of the Cricket, Gryllus bimaculatus*，*Zoological Science*。DOI：10.2108/zs130218；PMID：25186926。环境中的视觉庇护所图像调节风诱发逃逸转向；没有证明本项目 looming 刺激通过递归调节阈值。
- Lu, Fukutomi, Shidara & Ogawa（2023），*Persistence of auditory modulation of wind-induced escape behavior in crickets*，*Frontiers in Physiology*。DOI：10.3389/fphys.2023.1153913；PMID：37250114。先行听觉背景能持续调节风诱发逃逸策略；不能直接移植为本项目的视觉递归机制。
- Graves（2016），*Adaptive Computation Time for Recurrent Neural Networks*，arXiv：1603.08983。仅作为仓库 ACT 实现的相关计算背景；它不是用户所指的 Mixture-of-Recursions 原论文。

## 8. 保存的核心判断

**当前更值得检验的是不对称、情境和时序依赖的多感官调节。该现象与 Mixture-of-Recursions 兼容，但没有识别递归机制。稳定快慢通路分工缺乏支持，不能推出递归存在可能性低；尤其本轮 Phase-2 的递归模块关闭，不能据其结果否定 MoR。**

这是重要的解释转折与定义纠正。后续需要评估自适应递归深度是否增加可辨识的解释，而不能用通路名称、读出权重或行为加速替代检验。

本记录不改变冻结数据、方法、统计 family、阈值或预算，也不构成训练启动指令。下一轮仍等用户一次“开始”；该次授权覆盖整个既定计划，不逐阶段或逐 run 再次询问。
