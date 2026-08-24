# Tester Round-3 验收报告（test_round3_report.md）

日期: 2026-08-24
执行者: tester_r3
结论: **技术验证通过（V1–V5 全序列零异常）；发布闸门（V6 commit/push）暂缓，等待 team-lead 两项批复**

---

## V1. 跨环境数据重置 ✅

- `data/raw/` 清空后自 `/mnt/d/Projects/bak/` 重拷 44 个 legacy CSV
  （原始 schema: sys_time/ard_time/dx/dy/dz/stim_state）。
- `make load` EXIT=0：22 份轨迹重构 + 22 份事件流格式化。
- 说明：转换后文件与 bak 字节不同属设计内行为
  （`scripts/pre_load_adapt.py` 做 legacy→canonical 转换）；
  确定性核验改用标注漏斗计数比对。

## V2. 端到端管道 ✅

- `make data` EXIT=0。v2.1 漏斗计数与 prep_v21 基准完全一致：
  - ESCAPE=204 / NO_RESPONSE=129 / PREWALK=11 / PRE_ACTIVE=52
  - 敏感性 x0.75→PREWALK=15, x1.25→PREWALK=8
- 数据集溯源双键实测读取确认：
  - `pipeline_semantics_version = "2.1"`
  - `mcmc_prior_provenance = "oof_5fold_session_grouped_cv"`
- 折诊断落盘：每折 (session 数, 类直方图)；OOF 先验方差下限检查通过；
  train-vs-serve KS/var 统计落盘（B-CRIT-3c）。

## V3. 冒烟训练 ✅

- `python scripts/train.py --config config/default.yaml --epochs 1 --output_dir runs/tester_smoke` EXIT=0
- 无 NaN/Inf；spike_rate=0.04（阈值 0.5）正常；
  val_loss=5.242（1 epoch 预期量级）。

## V4. 物理分析抽检 ✅（含一次并发事故的发现与解决）

- **第一次运行崩溃**：`analyze_jacobian.py` 在
  `nsmor/analysis/dynamics.py:427` 抛出
  `RuntimeError: cudnn RNN backward can only be called in training mode`。
  该文件不在 Round-3 diff 内（上次改动 2026-06-29），属潜伏 bug 被
  v2.1 checkpoint 首次真实端到端触发——Reviewer B Round-3 CRITICAL-1
  "修复从未真实运行" 的判断在这一点上被事后证实。
- **并发覆盖事故**：验证期间磁盘 `runs/default/best_model.pth` 被
  developer_r3b 并发训练覆盖（13:13, epoch=137, val_loss=0.561125,
  v2.1 键齐全——已实测核验）。team-lead 已知情并通知。
- `nsmor/analysis/dynamics.py` 被并发补上 train()-switch 修复
  （+23/-13, try/finally 保证模式恢复）；复跑 test_biophysics 35 passed。
- **修复后重跑** `analyze_jacobian.py --checkpoint runs/default/best_model.pth`
  EXIT=0（14:04）：
  - GMM+BIC 校准 transient threshold=0.0828（ΔBIC>10）
  - frozen-input 对照独立再校准：5/7 通过（MAJ-3B 闭环实证）
  - Wilson CI 落盘 JSON；无 NaN/Inf
  - 诚实拒绝路径正常（early 仅 2 candidates < 4 拒绝校准）

## V5. 全量下游 + 回归 ✅

六脚本全部基于当前磁盘状态（r3b epoch137 checkpoint + v2.1 数据集）
真实重跑，EXIT 全 0，results/ 时间戳更新至 14:04–14:07：

| 脚本 | EXIT | 关键数值 |
|---|---|---|
| analyze_dynamics | 0 | PCA 三分量正常 |
| analyze_jacobian | 0 | 见 V4 |
| analyze_gating | 0 | k_opt=5 无空簇 {167,87,80,16,10} |
| analyze_integration | 0 | 九条件 n=36/72 一致 |
| simulate_lesion | 0 | MSE CI 有限；block_sensitivity.json 落盘 |
| simulate_psychophysics | 0 | Wilcoxon+HL 一致；Holm 校正正常 |

数值稳定性扫描：全部日志无 NaN/Inf/除零。

`python -m pytest tests/ -q` → **114 passed**（两轮复验一致，70s）。
新增用例覆盖本轮新机制：rel_refract ms 换算、legacy steps→ms 迁移、
responder-first 分支顺序、walking-non-responder→PRE_ACTIVE、
stability=None 架空防护。

## Reviewer 意见代码层核销（抽样实证）

| 意见 | 证据位置 | 状态 |
|---|---|---|
| A-BLK-3A / B-CRIT-2 | `_calibrate_fp_threshold` GMM(log)+ΔBIC>10+cap0.3+±25% 敏感性 (analyze_jacobian.py:159-280) | ✅ |
| A-BLK-3B / MAJ-3A | labeling 窗分数判据 + hard_end 截断 + anchor_min_frames 入配置 | ✅ |
| B-CRIT-1 | dynamics.py cuDNN train()-switch（本轮 Tester 实测触发并验证修复） | ✅ |
| B-CRIT-3a | mcmc_module.py:425-468 每类×每折硬断言 | ✅ |
| B-CRIT-3c | prepare_data.py:802-837 KS/var 落盘 | ✅ |
| B-CRIT-4b | Shapiro 预检废除，固定 Wilcoxon+HL | ✅ |
| B-MAJ-1 | rel_refract_ms 全链路（yaml/parser/model/test） | ✅ |
| MINOR 全项 | BCa z0 极端值告警、inhib_tau_ms 入配置、dt_ms 缺失硬错误等 | ✅ |

## ⚠ V6 发布闸门 — 暂缓，等待两项批复

### 裁决事项 (1)：产物-报告错位

事实链：
- dev_round3_report.md 声称的训练产物（R²=0.3655, val_loss=0.561125,
  12:23 train_r3b）在验证期间被 developer_r3b 的并发训练覆盖。
- 当前磁盘 `runs/default/best_model.pth`（13:13 落盘）实测：
  epoch=137, val_loss=0.561125, pipeline_semantics_version="2.1"。
  注意：dev 报告中的 val_loss 数值与 r3b 产物相同（0.561125），但
  epoch 与落盘时间不同——两个训练高度相似但非同一文件，无法断言
  指标等价。
- results/ 下全部下游产物是本 Tester 基于**当前磁盘 checkpoint**
  于 14:04–14:07 重新生成的，与 dev 报告中 13:06–13:12 的旧产物集
  不是同一批文件。
- 因此 commit message 的"验证结果"段只能声明当前可复现状态：
  v2.1 数据集 + r3b epoch137 checkpoint + 本 Tester 全套下游重跑
  （六脚本 EXIT=0、114 tests passed）。不得沿用 dev 报告的
  R²=0.3655 声明（该数字对应的模型文件已不存在于磁盘）。

### 裁决事项 (2)：未跟踪文件范围

`.claude/` 未被 .gitignore 覆盖
   （.gitignore 仅含 data/raw、data/processed、runs、results 等），
   当前 `git status` 未跟踪文件共 **39 个**，其中 `.claude/` 下 37 个：
   - 报告类（建议纳入）：dev_round2/3_report.md、review_round1-3_{A,B}.md、
     test_round3_report.md
   - 日志类（体积大、复现价值低，建议排除或归档）：
     train_r3.log、train_r3b.log、train_v21.log、downstream_r3{,b}.log、
     prep_v21.log、jacobian_run_log.txt、jacobian_v21b_log.txt、
     tester_*.log（11 个）、diag_*_out.txt（5 个）、ds_inspect_result.txt
   - 临时脚本类：diag_counterfactual/funnel/prestats/window.py、
     inspect_ds.py、run_inspect.sh
   需 team-lead 明确：(a) 全部纳入 / (b) .gitignore 排除 .claude/ /
   (c) 白名单（如仅报告类）。

另：工作树代码改动为 **23 文件 +2705/-290**（含本轮并发修复的
nsmor/analysis/dynamics.py），全部为已修改状态，无新增源码文件。

批复后即执行 squash-commit（规范 message + Approved-by footer）并 push。

## 产物清单（本Tester生成）

- `.claude/tester_prep.log` — make data 全量日志
- `.claude/tester_smoke.log` — 1-epoch 冒烟
- `.claude/tester_jacobian{,2,3}.log` — jacobian 三次运行（崩溃→复现→通过）
- `.claude/tester_dynamics.log`, `tester_analyze_*.log`, `tester_simulate_*.log`
- `.claude/tester_pytest.log`, `tester_pytest_final.log` — 114 passed ×2
