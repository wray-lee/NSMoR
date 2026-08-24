# Round-3 执行阶段交付报告（Orchestrator 接管完成）

日期: 2026-08-24
执行者: team-lead（orchestrator 直接接管；developer_r3 / developer_r3b 均因上游 API 不稳定失联）

## 背景与接管原因

Developer 角色连续 3 个实例（developer_r3、developer_r3b）在 watchdog 心跳无回复、
无产物落盘的状态下失联。剩余任务为纯机械执行阶段，由 orchestrator 直接完成。

## T1. 遗留状态核查

- developer_r3 已完成大部分 Round-3 代码修复（git diff ~22 文件）
- 数据集曾以 v2.0 判据生成（10:03）；随后发现 `nsmor/config.py` 的
  PIPELINE_SEMANTICS_VERSION 已按 Reviewer A BLK-3B 分支重排修复升至 **2.1**
- **版本守卫正确拦截**：v2.1 代码拒绝加载 v2.0 checkpoint —— 溯源系统按设计工作

## 关键事件链（诚实记录）

1. 12:09 完成第一次训练（当时数据集为 v2.0，checkpoint 带 v2.0 溯源键，R²=0.3765）
2. 12:23 developer_r3 死前已触发 v2.1 数据集重生成（prep_v21.log）：
   - 新判据（escape-first 分支重排）：ESCAPE=204, NO_RESPONSE=129, PREWALK=11, PRE_ACTIVE=52
   - **PREWALK=0 塌缩问题被 BLK-3B 根因修复解决**（11 trials 恢复）
   - 敏感性: x0.75→PREWALK=15, x1.25→PREWALK=8（阈值稳健）
   - 注意: 日志中"PREWALK count is ZERO"警告是 prepare_data.py 的 key 大小写 bug
     （`funnel.get("n_Prewalk")` 应为 `"n_PREWALK"`），已修复；实际分布健康
3. 版本守卫拒绝 v2.0 checkpoint → 用 v2.1 数据集完整重训

## T2. 训练（v2.1 数据集）

- 协议: config/default.yaml 完整 150 epochs，无偏离
- 旧 checkpoint 归档: runs/legacy_pre2.0/best_model.pth（2026-07-29 污染管线产物）
- 结果: best val_loss=0.561125
- **最终指标: MSE=2.4622 RMSE=1.5691 MAE=0.3911 R²=0.3655**
- 对比声明: 旧泄漏管线 R²=0.466。去泄漏后的诚实泛化水平 R²≈0.37，
  差值即 session 级泄漏的通胀幅度——预期代价而非失败。
- 日志: .claude/train_r3.log（v2.0 弃用）、.claude/train_r3b.log（v2.1 正式）

## T3. analyze_jacobian.py 端到端验证 ✅

此前必现 NameError。本轮真实运行 EXIT=0，且各修复组件均实证生效：
- GMM+BIC 门控: transient epoch threshold=0.0741，接受 7/11，拒绝 4
- sanity cap 生效: sustained epoch 校准边界 0.375 > cap 0.3 → 诚实拒绝并记录
  （修复: 补回缺失常量 FP_RESIDUAL_THRESHOLD_CAP=0.3 定义，原代码引用未定义名）
- frozen-input 对照独立再门控（对照残余分布单峰 → 光谱如实扣留）
- PREWALK 类仅 11 trials 且 early epoch 无候选 → no_candidates 丢弃路径正常
- 产物: results/jacobian_spectrum.png + jacobian_spectrum.json（13:06）

## T4. 下游全量重跑 ✅（全部使用 v2.1 checkpoint + v2.1 数据集）

| 脚本 | 状态 | 产物时间 |
|---|---|---|
| analyze_dynamics.py | OK | 13:07 |
| analyze_jacobian.py | OK | 13:06 |
| analyze_gating.py | OK | 13:10 |
| analyze_integration.py | OK | 13:10 |
| simulate_lesion.py | OK | 13:12 |
| simulate_psychophysics.py | OK | 13:12 |

results/ 全部产物时间戳已从 Jul-29 更新至 Aug-24。日志: .claude/downstream_r3.log

## T5. 测试 ✅

`python -m pytest tests/ -q` → **114 passed**（基线 111 + 本轮新增 3），187s。

## 自检清单

- [x] 数据集 pipeline_semantics_version=2.1，provenance=oof_5fold_session_grouped_cv
- [x] fold models + train/serve 一致性诊断随数据集落盘
- [x] 四类计数健康: ESCAPE=204, NO_RESPONSE=129(快照过滤后93), PREWALK=11, PRE_ACTIVE=52
- [x] PREWALK 塌缩根因修复（BLK-3B 分支重排）经数据实证有效
- [x] labeling waterfall 日志 key bug 修复（n_Prewalk→n_PREWALK）
- [x] FP_RESIDUAL_THRESHOLD_CAP 未定义名修复
- [x] 版本守卫双向验证（v2.0 ckpt 被 v2.1 代码正确拒绝）
- [x] 旧污染 checkpoint 归档 runs/legacy_pre2.0/
- [x] 全部下游分析基于新语义重跑
- [x] 114 tests passed
- [ ] git commit — 留给 Tester/发布流程

## 遗留事项（不阻塞本轮交付）

1. PREWALK n=11 极小，jacobian early-epoch 无候选属数据属性；
   sustained-epoch 线吸引子主张在当前数据下无法支撑（cap 诚实拒绝），
   论文表述需相应收敛或扩充该类样本。
2. simulate_psychophysics 的 M-1（配对 NaN 排除）、M-2/M-3 等 Round-2 B 报告
   条目已在 developer_r3 代码修复中处理，本轮端到端运行未再触发崩溃；
   统计学层面的最终裁决交由下一轮双盲审查确认。
