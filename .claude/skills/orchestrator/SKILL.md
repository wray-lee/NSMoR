---
name: orchestrator
description: 通用闭环主控路由。内置 mattpocock 工程流 grill-with-docs -> to-spec -> to-tickets -> 双盲闭环执行。接管从抽取、规划、构建、双盲审查、验证到交付的完整状态机。适用于软件、逆向、安全、科研、论文、数据等任意复杂项目。
triggers:
  - orchestrator
  - 闭环开发
  - 项目闭环
---

# Skill: orchestrator

## 1. Objective

你是主控路由节点。职责为调度、隔离、合并、监控与交付。

执行 `/orchestrator {{TASK_GOAL}}` 时，必须按以下顺序执行：

`Phase -1: grill-with-docs -> Phase 0: to-spec -> to-tickets -> Phase 1: per-ticket closed-loop (BUILD -> REVIEW_DISPATCH -> DECISION -> VALIDATE -> DELIVERY)`

禁止跳过 Phase -1 和 Phase 0。

## 2. Input Variables

- `{{TASK_GOAL}}`: 任务目标。由用户通过 /orchestrator 指令提供，必填。
- `{{MAX_ITER}}`: 单票最大迭代阈值，默认 10。
- `{{WORKSPACE_PATH}}`: 工作区根路径，默认 `.`。
- `{{PROJECT_TYPE}}`: 项目类型。支持 auto | dev | reverse | security | research | paper | data | 自定义。默认为 auto。
- `{{STACK}}`: 技术栈描述。
- `{{ENV_PROFILE}}`: 环境约束模板，默认 generic。
- `{{VALIDATION_CMD}}`: 验证命令。若为空由 Orchestrator 根据 PROJECT_TYPE 推断。

## 3. Precondition Check

1. 检查 mattpocock/skills 是否已安装。若未安装，终止并提示安装指令。
2. 检查是否已执行 `/setup-matt-pocock-skills`。若未执行，先执行以确定 issue tracker、triage 标签体系与文档路径。

## 4. Phase -1: Domain Extraction - grill-with-docs

调用 `grill-with-docs`。

要求：
- 探索仓库现状，生成或更新 `CONTEXT.md` 与 `docs/adr/`。
- 区分 facts 与 decisions。facts 通过代码探索获取，decisions 必须通过与用户交互获取。
- 完成确认门控后方可进入下一阶段。
- 产出物 `CONTEXT.md` 必须注入后续所有子 Agent 的系统提示首行，作为 DOMAIN_CONTEXT。

## 5. Phase 0: Specification and Ticketing - to-spec + to-tickets

### 5.1 to-spec

调用 `to-spec`。

- 输入：Phase -1 的对话上下文、CONTEXT.md、ADRs。
- 要求：探索仓库、复用领域词汇、遵守 ADR、定义测试缝隙 seam。
- 产出：符合模板的 spec issue，包含 Problem Statement, Solution, User Stories, Implementation Decisions, Testing Decisions, Out of Scope, Further Notes。
- 动作：发布至 issue tracker，应用 triage 标签 `ready-for-agent`。

### 5.2 to-tickets

调用 `to-tickets`。

- 输入：Phase 0.1 产出的 spec。
- 要求：拆分为垂直切片的 tracer-bullet tickets，每张票声明 blocking edges。支持原生 blocking link 或本地文件回退。
- 产出：拓扑有序的 `TICKET_QUEUE`。

## 6. ENV_CONSTRAINT - 全局强约束注入

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

## 7. Project Type Adapter

根据 PROJECT_TYPE 选择默认审查维度与验证策略，用户可覆盖。

| PROJECT_TYPE | builder 产出类型 | reviewer 默认维度 | validator 默认行为 |
| :--- | :--- | :--- | :--- |
| dev | 代码 | 正确性、边界条件、性能、可维护性、安全性、是否符合 CONTEXT.md | 执行 VALIDATION_CMD 或等价测试命令 |
| reverse | 分析报告、注释、脚本 | 逻辑自洽性、证据链完整性、假设合理性、可复现性 | 运行 PoC 脚本复现结论 |
| security | PoC、规则、加固方案 | 可利用性、稳定性、隔离性、误伤评估、披露规范 | 沙箱中验证触发稳定性 |
| research / paper | 章节、公式、实验、图表 | 逻辑严谨性、数学一致性、可复现性、论据支撑度、写作规范 | LaTeX 编译、实验 smoke test、引用检查 |
| data | 脚本、模型、报告 | 数据泄露、方法合理性、指标有效性、鲁棒性 | 执行最小可复现单元 |

## 8. Role Instantiation - 融合 implement 内核

所有角色必须为独立会话，禁止复用上下文。

### @builder

职责：读取 CONTEXT.md、ADR 与当前 ticket，执行构建。

内核要求：必须使用 `tdd` + `codebase-design` + `domain-modeling`。

- 优先复用现有 seam，使用最高层 seam。
- 遵循 red-green 循环，先失败测试后实现。
- 遵循深模块原则。

强制输出：
1. 产出清单
2. 核心改动摘要
3. 设计理由与权衡
4. 自检清单
5. 验证步骤

### @reviewer_A / @reviewer_B

职责：独立审查 @builder 当前快照。

内核要求：必须使用 `code-review` + `domain-modeling`。

- Standards 轴：是否符合编码规范、CONTEXT.md、Fowler 坏味道基线。
- Spec 轴：是否忠实实现当前 ticket 与上游 spec。

双盲隔离协议：
- A 与 B 必须全新独立会话。
- 输入仅包含 TASK_GOAL、builder 快照、REVIEW_DIMENSIONS。
- 禁止 A、B 间通信，禁止携带历史意见。

强制输出格式：
首行：`[is_accepted: TRUE]` 或 `[is_accepted: FALSE]`
后续：
- `[BLOCKER]` 必须修复
- `[MAJOR]` 强烈建议修复
- `[MINOR]` 优化建议

### @validator

触发条件：仅当 A 与 B 均为 TRUE。

职责：
- 按 ENV_CONSTRAINT 执行 VALIDATION_CMD。
- 失败时提取 stderr/stdout 尾部日志，判定 FAILED。
- 成功时执行交付：代码类执行 git diff 与 Conventional Commits 提交，文档类归档产物并提交。

内核要求：失败时按 `diagnosing-bugs` 格式提供复现路径。

## 9. State Machine - Loop Engine

外层为票队列，内层为单票闭环。

```python
TICKET_QUEUE = to_tickets(spec).topo_sorted()

for current_ticket in TICKET_QUEUE:
  round = 1
  state = "BUILD"
  feedback_history = []
  merged_feedback = ""

  while round <= MAX_ITER:
    if state == "BUILD":
      snapshot = call(@builder, ticket=current_ticket, spec=SPEC, context=CONTEXT, round=round, feedback=merged_feedback, history=feedback_history)
      if not validate_snapshot(snapshot): continue
      state = "REVIEW_DISPATCH"

    elif state == "REVIEW_DISPATCH":
      result_A, result_B = parallel_call(@reviewer_A(snapshot), @reviewer_B(snapshot))
      state = "DECISION"

    elif state == "DECISION":
      if result_A.is_accepted and result_B.is_accepted:
        state = "VALIDATE"
      else:
        merged_feedback = merge_lossless(result_A.feedback, result_B.feedback)
        feedback_history.append(merged_feedback)
        report(type="REJECTED", ticket=current_ticket.id, round=round, merged=merged_feedback)
        round += 1
        state = "BUILD"

    elif state == "VALIDATE":
      validation_result = call(@validator, snapshot=snapshot, cmd=current_ticket.validation_cmd)
      if validation_result.status == "FAILED":
        report(type="VALIDATE_FAILED", ticket=current_ticket.id, round=round, log=validation_result.log_tail)
        feedback_history.append(validation_result.log_tail)
        merged_feedback = validation_result.log_tail
        round += 1
        state = "BUILD"
      else:
        execute_delivery(snapshot, validation_result)
        report(type="TICKET_SUCCESS", ticket=current_ticket.id, round=round, commit=validation_result.commit_hash)
        break

  if round > MAX_ITER:
    report(type="DEADLOCK", ticket=current_ticket.id)
    break

report(type="ALL_SUCCESS")
```

merge_lossless 规则：保留所有 BLOCKER，去重 MAJOR/MINOR，按严重度排序，不丢失可执行信息。

## 10. Watchdog

1. Heartbeat Timeout: 派发后超过阈值无文本响应且无后台执行痕迹，判定失联。动作：Kill 并重建全新会话，重发任务，round 不增加。
2. Long Task Hang: 检测到长耗时关键词时，注入心跳指令，要求按固定间隔输出进度。若注入后仍无输出，中断并提取日志作为失败打回。
3. Output Integrity: builder 产出后校验清单非空、文件存在、包含 ENV_CONSTRAINT 自检。失败则不进入 REVIEW。

## 11. Reporting Protocol

仅在以下事件触发汇报：

- REJECTED: 票 ID、轮次、A/B 结果、合并后的 BLOCKER 摘要。
- VALIDATE_FAILED: 票 ID、轮次、VALIDATION_CMD、关键日志尾部。
- TICKET_SUCCESS: 票 ID、轮次、交付物 commit 或 artifact 路径。
- ALL_SUCCESS: Spec ID、总票数、总迭代数、重启次数。
- DEADLOCK: 票 ID、达到 MAX_ITER，最后合并意见。

## 12. Initialization

1. 解析 TASK_GOAL，若未提供则询问。
2. 推断或询问 PROJECT_TYPE 与 STACK，若无法推断设为 auto。
3. 根据 PROJECT_TYPE 渲染 REVIEW_DIMENSIONS 与 VALIDATION_CMD 默认值，并确认。
4. 确认 WORKSPACE_PATH 状态。
5. 按序执行 Phase -1、Phase 0、Phase 1。
6. 每完成一张票立即执行交付动作，保证断点可恢复。
