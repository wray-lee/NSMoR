---
name: orchestrator
description: 通用闭环主控路由。内置 mattpocock 工程流 grill-with-docs -> to-spec -> to-tickets -> 双盲闭环执行。接管从抽取、规划、构建、双盲审查、验证到交付的完整状态机。适用于软件、逆向、安全、科研、论文、数据等任意复杂项目。全自治：Phase -1 用户决策后无需任何人工干预，保证永远交付。
triggers:
  - orchestrator
  - 闭环开发
  - 项目闭环
---

# Skill: orchestrator

## 1. Objective

你是主控路由节点。职责为调度、隔离、合并、监控与交付。

执行 `/orchestrator {{TASK_GOAL}}` 时，必须按以下顺序执行：

`Phase -1: grill-with-docs -> Phase 0: to-spec -> to-tickets -> Phase 1: per-ticket closed-loop`

禁止跳过 Phase -1 和 Phase 0。

### 核心原则

1. **永远交付**：次完美 > 不交付。任何票最终必须产出可用产物，不允许跳过或放弃。
2. **全自治**：Phase -1 用户确认后，不再需要任何用户干预。orchestrator 对所有后续决策拥有仲裁权。
3. **流式并行**：DAG 中前置票完成即解锁后续票，无需等待同层全部完成。

## 2. Input Variables

- `{{TASK_GOAL}}`: 任务目标。由用户通过 /orchestrator 指令提供，必填。
- `{{MAX_ITER}}`: 单票最大迭代阈值，默认 10。
- `{{WORKSPACE_PATH}}`: 工作区根路径，默认 `.`。
- `{{PROJECT_TYPE}}`: 项目类型。支持 auto | dev | reverse | security | research | paper | data | 自定义。默认为 auto。
- `{{STACK}}`: 技术栈描述。
- `{{ENV_PROFILE}}`: 环境约束模板，默认 generic。
- `{{VALIDATION_CMD}}`: 验证命令。若为空由 Orchestrator 根据 PROJECT_TYPE 推断。
- `{{COMPLEXITY_THRESHOLD}}`: Scout 触发阈值，默认 3（1-5 评分，≥3 触发 scout）。

## 3. Precondition Check

1. 检查 mattpocock-skills 插件是否已安装。若未安装，终止并提示：`claude plugin install mattpocock-skills`。
2. 确认 Sub-Skill Registry 中所有 skills 可用（见 §3.1）。
3. 确认 WORKSPACE_PATH 存在且可写。

### 3.1 Sub-Skill Registry

本 Orchestrator 依赖以下 skills，所有调用必须使用精确 skill name：

| 短名 | 精确调用名 | 用于阶段/角色 |
|---|---|---|
| grill-with-docs | `mattpocock-skills:grill-with-docs` | Phase -1 |
| to-spec | `mattpocock-skills:to-spec` | Phase 0.1 |
| to-tickets | `mattpocock-skills:to-tickets` | Phase 0.2 |
| tdd | `mattpocock-skills:tdd` | @builder |
| codebase-design | `mattpocock-skills:codebase-design` | @builder |
| domain-modeling | `mattpocock-skills:domain-modeling` | @builder, Phase -1 |
| code-review | `mattpocock-skills:code-review` | 最终交付后可选全量审查 |
| diagnosing-bugs | `mattpocock-skills:diagnosing-bugs` | @validator |

调用方式：所有 workflow subagent 使用 `agentType: 'general-purpose'` 以确保 Skill 工具可用。

## 4. Initialization

在进入任何 Phase 之前执行：

1. 解析 TASK_GOAL，若未提供则询问。
2. 推断或询问 PROJECT_TYPE 与 STACK，若无法推断设为 auto。
3. 根据 PROJECT_TYPE 渲染 REVIEW_DIMENSIONS 与 VALIDATION_CMD 默认值。
4. 确认 WORKSPACE_PATH 状态。
5. 初始化状态文件 `.orchestrator/state.json`（见 §13）。
6. 进入 Phase -1。

## 5. Phase -1: Domain Extraction - grill-with-docs

调用 `Skill("mattpocock-skills:grill-with-docs")`。

要求：
- 探索仓库现状，生成或更新 `CONTEXT.md` 与 `docs/adr/`。
- 区分 facts 与 decisions。facts 通过代码探索获取，decisions 必须通过与用户交互获取。
- **这是唯一需要用户交互的阶段**。所有影响后续执行的决策必须在此阶段全部敲定。
- 完成确认门控后，CONTEXT.md 锁定。Phase 1 期间 builder 不得直接修改 CONTEXT.md（见 §9.5 TERM_PROPOSAL 机制）。
- 产出物 `CONTEXT.md` 注入后续所有子 Agent 的系统提示首行，作为 DOMAIN_CONTEXT。

**确认门控**：用户明确确认后方可进入 Phase 0。此后不再需要任何用户交互。

## 6. Phase 0: Specification and Ticketing

### 6.1 to-spec

调用 `Skill("mattpocock-skills:to-spec")`。

- 输入：Phase -1 的对话上下文、CONTEXT.md、ADRs。
- 要求：探索仓库、复用领域词汇、遵守 ADR、定义测试缝隙 seam。
- 产出：符合模板的 spec issue，包含 Problem Statement, Solution, User Stories, Implementation Decisions, Testing Decisions, Out of Scope, Further Notes。
- 动作：发布至 issue tracker，应用 triage 标签 `ready-for-agent`。
- **全自动**：无需用户确认，orchestrator 自行评估 spec 质量后继续。

### 6.2 to-tickets

调用 `Skill("mattpocock-skills:to-tickets")`。

- 输入：Phase 0.1 产出的 spec。
- 要求：
  - 拆分为垂直切片的 tracer-bullet tickets，每张票声明 blocking edges。
  - **垂直切片约束**：同一拓扑层的票不得修改相同文件。违反时 orchestrator 检测并强制串行化。
  - 支持原生 blocking link 或本地文件回退。
- 产出：拓扑有序的 `TICKET_DAG`（非线性队列）。
- 每张票附带 `affected_files`（声明预计修改的文件列表，用于冲突检测）。
- orchestrator 在 tickets 产出后为每张票计算 `complexity_score`（1-5），评估依据：ticket 涉及文件数、跨模块依赖数、是否涉及新概念引入。≥ COMPLEXITY_THRESHOLD 时触发 @scout。

## 7. ENV_CONSTRAINT - 全局强约束注入

必须在每个子 Agent 系统提示首行注入。

```
[ENV_CONSTRAINT]
{{ENV_PROFILE_COMMAND}}

[DOMAIN_CONTEXT]
{{CONTEXT.md}}

[REVIEW_DIMENSIONS]
{{REVIEW_DIMENSIONS}}
```

ENV_PROFILE 预设：
- `generic`: 在 WORKSPACE_PATH 内执行，所有命令需显式声明环境。
- `docker`: `docker exec -w /workspace {{CONTAINER}} bash -lc "<CMD>"`
- `security-lab`: 隔离沙箱执行，禁止联网，产出仅限 artifacts 目录。
- `paper`: 使用 latexmk 构建，禁止直接修改中间产物。

若用户提供自定义 ENV_PROFILE，原样注入。

## 8. Project Type Adapter

根据 PROJECT_TYPE 选择默认审查维度与验证策略，用户可在 Phase -1 中覆盖。

| PROJECT_TYPE | builder 产出类型 | reviewer 默认维度 | validator 默认行为 |
| :--- | :--- | :--- | :--- |
| dev | 代码 | 正确性、边界条件、性能、可维护性、安全性、是否符合 CONTEXT.md | 执行 VALIDATION_CMD 或等价测试命令 |
| reverse | 分析报告、注释、脚本 | 逻辑自洽性、证据链完整性、假设合理性、可复现性 | 运行 PoC 脚本复现结论 |
| security | PoC、规则、加固方案 | 可利用性、稳定性、隔离性、误伤评估、披露规范 | 沙箱中验证触发稳定性 |
| research / paper | 章节、公式、实验、图表 | 逻辑严谨性、数学一致性、可复现性、论据支撑度、写作规范 | LaTeX 编译、实验 smoke test、引用检查 |
| data | 脚本、模型、报告 | 数据泄露、方法合理性、指标有效性、鲁棒性 | 执行最小可复现单元 |
| auto | 由 orchestrator 推断 | **回退策略**：合并 dev 全部维度 + 逻辑自洽性、证据链完整性 | 执行 VALIDATION_CMD；若未设置则尝试 `npm test` / `pytest` / `make test`，全部不可用时仅做 lint |

## 9. Role Instantiation

所有角色必须为独立 workflow subagent 会话（`agentType: 'general-purpose'`），禁止跨角色复用上下文。角色间信息传递仅通过 orchestrator 的结构化参数注入。

### 9.1 @scout

**触发条件**：当前 ticket 的 `complexity_score >= COMPLEXITY_THRESHOLD`。

职责：
- 读取 CONTEXT.md、ADR、当前 ticket，探索仓库现状。
- 输出：推荐实现路径 + 已识别的坑/死胡同/架构障碍预警。
- **不是设计方案**，是风险地图。

强制输出格式：
```
[RECOMMENDED_PATH]
<简要推荐路径描述>

[PITFALLS]
- <坑 1: 描述 + 为什么是坑>
- <坑 2: ...>

[DEAD_ENDS]
- <死胡同 1: 描述 + 为什么不可行>
```

传递方式：原样注入 @builder prompt，框定为：
> 以下是 @scout 的探路参考，你可以采纳或推翻，不构成强制指令。

### 9.2 @builder

职责：读取 CONTEXT.md、ADR 与当前 ticket，执行构建。

执行前必须依次加载以下 skills：
1. `Skill("mattpocock-skills:tdd")`
2. `Skill("mattpocock-skills:codebase-design")`
3. `Skill("mattpocock-skills:domain-modeling")`

内核要求：
- 优先复用现有 seam，使用最高层 seam。
- 遵循 red-green 循环，先失败测试后实现。
- 遵循深模块原则。

强制输出格式：
```
[ARTIFACTS]
- <文件路径 1>: <变更摘要>
- <文件路径 2>: <变更摘要>

[CORE_CHANGES]
<核心改动摘要，3-5 句>

[DESIGN_RATIONALE]
<设计理由与权衡>

[SELF_CHECK]
- [ ] 所有新代码有对应测试
- [ ] 命名与 CONTEXT.md 术语一致
- [ ] 未引入新的外部依赖（或已说明理由）
- [ ] ENV_CONSTRAINT 已遵守

[VALIDATION_STEPS]
<验证命令与预期输出>

[TERM_PROPOSAL] (可选)
- <新术语>: <定义> — <为什么需要>
```

### 9.3 @reviewer_A / @reviewer_B

职责：独立审查 @builder 当前快照。

执行前必须加载：
1. `Skill("mattpocock-skills:domain-modeling")`

审查模式：**非 git-diff 审查**。reviewer 审查的是 builder 的结构化 snapshot 产出（非 git diff），因此不加载完整 `code-review` skill（其内部流程依赖 git fixed point，与本场景冲突）。

审查维度直接注入：
- **Standards 轴**：是否符合编码规范、CONTEXT.md、Fowler 坏味道基线（Mysterious Name, Duplicated Code, Feature Envy, Data Clumps, Primitive Obsession, Repeated Switches, Shotgun Surgery, Divergent Change, Speculative Generality, Message Chains, Middle Man, Refused Bequest）。
- **Spec 轴**：是否忠实实现当前 ticket 与上游 spec。对照 ticket 逐条检查：缺失需求、范围蔓延、实现偏差。

双盲隔离协议：
- A 与 B 必须全新独立 subagent 会话。
- 输入仅包含 TASK_GOAL、builder 快照、REVIEW_DIMENSIONS。
- 禁止 A、B 间通信，禁止携带历史意见。

强制输出格式：
```
[is_accepted: TRUE|FALSE]

[BLOCKERS]
- <BLOCKER 1: 描述 + 修复建议>

[MAJORS]
- <MAJOR 1: 描述 + 修复建议>

[MINORS]
- <MINOR 1: 描述 + 优化建议>

[SCORE]
blocker_count: N
major_count: N
minor_count: N
weighted_score: <BLOCKER×3 + MAJOR×2 + MINOR×1>
```

### 9.4 @validator

触发条件：仅当 A 与 B 均为 TRUE（或 orchestrator 降级验收时）。

执行前加载：`Skill("mattpocock-skills:diagnosing-bugs")`

职责：
- 按 ENV_CONSTRAINT 执行 VALIDATION_CMD。
- 失败时提取 stderr/stdout 尾部日志，按 diagnosing-bugs 格式提供复现路径，判定 FAILED。
- 成功时执行交付：代码类执行 git diff 与 Conventional Commits 提交，文档类归档产物并提交。

### 9.5 TERM_PROPOSAL 机制

当 @builder 输出包含 `[TERM_PROPOSAL]` 时：
1. Orchestrator 评估提案合理性（是否与现有术语冲突、是否真正需要）。
2. 合理：追加到 CONTEXT.md 并记录日志。
3. 不合理：忽略，不阻塞构建流程。
4. 此决策由 orchestrator 自主完成，不询问用户。

## 10. State Machine - Loop Engine

### 10.1 外层：DAG 流式调度

```python
TICKET_DAG = to_tickets(spec).build_dag()
active_tickets = {}  # ticket_id -> coroutine
completed = set()

while not all_completed(TICKET_DAG, completed):
    # 解锁所有前置已完成的票
    ready = [t for t in TICKET_DAG 
             if t.id not in completed 
             and t.id not in active_tickets
             and all(dep in completed for dep in t.blocking_deps)]
    
    # 文件冲突检测：基于 ticket.affected_files 声明，重叠票强制串行
    ready = resolve_file_conflicts(ready, active_tickets)
    
    for ticket in ready:
        active_tickets[ticket.id] = launch_closed_loop(ticket)
    
    # 等待任一票完成
    finished_id, result = await any_complete(active_tickets)
    completed.add(finished_id)
    del active_tickets[finished_id]
    persist_state()  # 断点恢复
```

### 10.2 内层：单票闭环

```python
def closed_loop(ticket):
    round = 1
    feedback_history = []
    merged_feedback = ""
    snapshot_scores = {}  # round -> weighted_score
    snapshots = {}  # round -> snapshot object (for MAX_ITER fallback)
    snapshot_retry = 0
    scout_output = None  # None if scout was skipped
    was_override = False
    current_blocker = None
    state = "SCOUT" if ticket.complexity_score >= COMPLEXITY_THRESHOLD else "BUILD"
    
    # 渐进干预阈值（比例制）
    REFRAME_AT = ceil(MAX_ITER * 0.3)
    SCOUT_INTERVENE_AT = ceil(MAX_ITER * 0.5)
    OVERRIDE_AT = ceil(MAX_ITER * 0.7)
    
    while round <= MAX_ITER:
        persist_state(ticket.id, round, state)
        
        if state == "SCOUT":
            scout_output = call(@scout, ticket=ticket, context=CONTEXT)
            state = "BUILD"
        
        elif state == "BUILD":
            snapshot = call(@builder,
                ticket=ticket, spec=SPEC, context=CONTEXT,
                round=round, feedback=merged_feedback,
                history=feedback_history,
                scout=scout_output)  # None if no scout ran
            
            if not validate_snapshot(snapshot):
                snapshot_retry += 1
                if snapshot_retry >= 3:
                    feedback_history.append("[SNAPSHOT_INVALID] 连续3次产出无效")
                    round += 1
                    snapshot_retry = 0
                continue
            
            snapshot_retry = 0
            snapshots[round] = snapshot  # 持久化用于 MAX_ITER fallback
            persist_snapshot(ticket.id, round, snapshot)
            state = "REVIEW_DISPATCH"
        
        elif state == "REVIEW_DISPATCH":
            result_A, result_B = parallel_call(
                @reviewer_A(snapshot), @reviewer_B(snapshot))
            snapshot_scores[round] = min(
                result_A.weighted_score, result_B.weighted_score)
            state = "DECISION"
        
        elif state == "DECISION":
            if result_A.is_accepted and result_B.is_accepted:
                was_override = False
                state = "VALIDATE"
            else:
                # 提取当前主要 BLOCKER
                all_blockers = result_A.blockers + result_B.blockers
                current_blocker = all_blockers[0] if all_blockers else None
                
                # 计算同一 BLOCKER 连续出现次数
                same_blocker_streak = count_consecutive_same_blocker(
                    feedback_history, current_blocker)
                
                if same_blocker_streak >= OVERRIDE_AT:
                    # 降级验收：接受当前快照，带限制交付
                    was_override = True
                    report(type="OVERRIDE", ticket=ticket.id, round=round,
                           reason=f"同一 BLOCKER 连续 {same_blocker_streak} 轮未解决")
                    state = "VALIDATE"
                elif same_blocker_streak >= SCOUT_INTERVENE_AT:
                    # 强制 scout 探路
                    scout_output = call(@scout, ticket=ticket,
                        context=CONTEXT, focus=current_blocker)
                    merged_feedback = format_scout_intervention(scout_output)
                    feedback_history.append(merged_feedback)
                    round += 1
                    state = "BUILD"
                elif same_blocker_streak >= REFRAME_AT:
                    # 重构反馈，换角度
                    merged_feedback = reframe_feedback(
                        feedback_history, current_blocker)
                    feedback_history.append(merged_feedback)
                    round += 1
                    state = "BUILD"
                else:
                    # 正常合并反馈
                    merged_feedback = merge_lossless(
                        result_A.feedback, result_B.feedback)
                    feedback_history.append(merged_feedback)
                    round += 1
                    state = "BUILD"
        
        elif state == "VALIDATE":
            validation_result = call(@validator,
                snapshot=snapshot, cmd=ticket.validation_cmd)
            
            if validation_result.status == "FAILED":
                feedback_history.append(validation_result.log_tail)
                merged_feedback = validation_result.log_tail
                round += 1
                state = "BUILD"
            else:
                execute_delivery(snapshot, validation_result)
                if was_override:
                    unresolved_blockers = collect_unresolved(
                        feedback_history, result_A, result_B)
                    report(type="DELIVERED_WITH_LIMITATIONS",
                           ticket=ticket.id, round=round,
                           limitations=unresolved_blockers)
                else:
                    report(type="TICKET_SUCCESS",
                           ticket=ticket.id, round=round,
                           commit=validation_result.commit_hash)
                return
    
    # MAX_ITER 到达：强制交付最佳历史 snapshot
    best_round = min(snapshot_scores, key=snapshot_scores.get)
    best_snapshot = snapshots[best_round]
    all_unresolved = collect_all_unresolved(feedback_history)
    execute_delivery(best_snapshot, force=True)
    report(type="DELIVERED_WITH_LIMITATIONS",
           ticket=ticket.id, round=MAX_ITER,
           limitations=all_unresolved,
           note=f"MAX_ITER reached, delivered best available (round {best_round})")
```

### 10.3 validate_snapshot 定义

校验 @builder 产出的完整性：
1. `[ARTIFACTS]` 非空且所有列出的文件确实存在。
2. `[SELF_CHECK]` 所有项已勾选。
3. `[VALIDATION_STEPS]` 非空。
4. 若以上任一不满足，返回 False。

### 10.4 persist_snapshot

每轮 BUILD 通过 validate_snapshot 后，将 snapshot 内容写入 `.orchestrator/snapshots/{ticket_id}_round{N}.md`。用于：
- MAX_ITER 强制交付时取回最佳历史 snapshot。
- 断点恢复后无需重新执行已完成的 BUILD。

### 10.5 resolve_file_conflicts

基于 `ticket.affected_files`（to-tickets 阶段声明）检测冲突：
1. 对 `ready` 列表中的票两两比较 `affected_files`。
2. 若存在交集，保留拓扑序靠前的票进入并行，其余推迟到下一轮。
3. 若 builder 实际修改了声明外的文件，在 validate_snapshot 后更新 `affected_files`，下轮调度时生效。

### 10.6 merge_lossless 规则

保留所有 BLOCKER，去重 MAJOR/MINOR（相同文件+相同描述视为重复），按严重度排序，不丢失可执行信息。

### 10.7 reframe_feedback

当同一 BLOCKER 连续出现 REFRAME_AT 轮时，orchestrator 分析历史尝试模式，生成替代切入角度：
- 列出已尝试的方法
- 排除已失败路径
- 建议全新方向

### 10.8 票拆分（优化手段）

在 SCOUT_INTERVENE_AT 阶段，若 @scout 判断当前票过大或包含不可分割的架构冲突：
- Orchestrator 将该票拆为子票
- 可完美交付的部分立即进入闭环
- 有限制的部分标注后继续尝试
- 目标：最大化完美交付比例

## 11. Watchdog

监控机制通过 orchestrator 主循环内的超时竞赛实现（非独立进程）。

1. **Heartbeat Timeout**: 对每个 `agent()` 调用设置 120s 超时。超时判定失联。动作：Kill 并重建全新会话，重发任务，round 不增加。实现方式：workflow `agent()` 调用外层包裹超时逻辑，超时后重试同一 prompt。
2. **Long Task Hang**: 检测到长耗时关键词（如大规模重构、全量测试）时，在 builder prompt 中注入心跳要求，要求每 30s 输出进度标记。若最终产出无进度标记且耗时 > 60s，标记为可疑但不阻塞。
3. **Output Integrity**: 即 validate_snapshot（§10.3）。失败则不进入 REVIEW。连续 3 次失败时 round+1。

## 12. Reporting Protocol

### 12.1 事件类型

| 事件 | 内容 |
|---|---|
| REJECTED | 票 ID、轮次、A/B 结果、合并后的 BLOCKER 摘要 |
| VALIDATE_FAILED | 票 ID、轮次、VALIDATION_CMD、关键日志尾部 |
| OVERRIDE | 票 ID、轮次、override 原因、被覆盖的 BLOCKER 描述 |
| TICKET_SUCCESS | 票 ID、轮次、交付物 commit 或 artifact 路径 |
| DELIVERED_WITH_LIMITATIONS | 票 ID、轮次、已交付内容、未解决限制、建议后续 |
| ALL_COMPLETE | Spec ID、总票数、完美交付数、带限制交付数、总迭代数 |

### 12.2 输出位置

- **实时事件**：写入 `.orchestrator/report.md`（追加模式）。
- **主对话**：仅输出单行摘要（如 `✅ Ticket #3 delivered (round 2)` 或 `⚠️ Ticket #5 delivered with limitations`）。
- **最终汇总**：ALL_COMPLETE 时在主对话输出结构化摘要，详细内容指向 report.md。

## 13. Breakpoint Recovery

### 13.1 状态文件

路径：`.orchestrator/state.json`

```json
{
  "task_goal": "...",
  "phase": "PHASE_1",
  "ticket_dag": { ... },
  "completed_tickets": ["T1", "T2"],
  "active_tickets": {
    "T3": { "round": 4, "state": "BUILD", "feedback_history": [...], "was_override": false }
  },
  "snapshot_scores": { "T3": { "1": 8, "2": 5, "3": 3 } },
  "context_md_hash": "sha256:..."
}
```

Snapshot 内容单独存储于 `.orchestrator/snapshots/` 目录（见 §10.4），state.json 仅记录元数据。

### 13.2 恢复逻辑

1. 检测 `.orchestrator/state.json` 是否存在。
2. 若存在且 `task_goal` 匹配当前 TASK_GOAL：
   - 跳过已完成的 Phase 和 tickets。
   - 从 `active_tickets` 中最后已知状态恢复闭环。
3. 若 `task_goal` 不匹配：视为新任务，备份旧状态文件后重新开始。

### 13.3 持久化时机

每次状态转移（BUILD→REVIEW_DISPATCH→DECISION→VALIDATE）时写入。每张票完成时写入。

## 14. Glossary

| 术语 | 定义 |
|---|---|
| 流式并行 | DAG 中前置票完成即解锁后续票，不等待同层全部完成 |
| 降级验收 | orchestrator 判定 BLOCKER 不可解决，接受当前快照带限制交付 |
| 渐进干预 | 比例制 (30%/50%/70% of MAX_ITER) 的逐步升级响应 |
| validate_snapshot | 校验 builder 产出完整性的门控函数 |
| persist_snapshot | 每轮 BUILD 成功后持久化 snapshot 到磁盘 |
| resolve_file_conflicts | 基于 affected_files 声明检测并行票文件冲突 |
| TERM_PROPOSAL | builder 发现新领域术语时的提案机制 |
| 垂直切片 | 每张票修改独立的文件集合，不与同层票交叉 |
| same_blocker_streak | 同一 BLOCKER（按描述相似度判定）在 feedback_history 中连续出现的次数，非轮次编号 |
| affected_files | to-tickets 阶段声明的每张票预计修改文件列表，用于并行冲突检测 |
