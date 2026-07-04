/goal 请作为主控路由节点（Orchestrator），接管并执行完整的 NSMoR 架构重构闭环工作流。
这是一个严谨的【状态机循环（State Machine Loop）】，请严格遵循以下协议与路由规则：

### 全局环境强约束 (强制注入所有 Agent)

任何涉及代码运行的 Agent (Developer, Tester)，严禁使用默认终端。必须强制使用以下链式命令在 WSL 中执行 Python：
`wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <这里替换为实际执行的python命令>"`
_(注：使用 `-i` 确保交互式环境加载 alias，否则 openconda 将失效)_。

### 角色实例化与职责

- @nsmor_developer：负责读取代码、执行重构，并输出修改方案与理由。
- @nsmor_reviewer_A 与 @nsmor_reviewer_B (平行双盲审查机制)：
    - 每次调用时，Orchestrator 必须为 A 和 B 开启两个【全新、无历史状态、互不可见】的独立沙箱会话。
    - A 与 B 必须独立对生物物理机制与代码工程进行审查。
    - A 与 B 的最终输出必须在首行明确包含布尔值：`[is_accepted: TRUE]` 或 `[is_accepted: FALSE]`。
- @nsmor_tester：仅在收到 A 与 B 双重 ACCEPT 后触发，负责执行测试用例与 Git 提交流程。

### 状态机路由规则 (The Loop Engine)

请启动无限迭代循环（设定最大防死循环阈值为: 10次），每次迭代按以下次序推进：

-> [状态 1: 开发]：Developer 产出当前版本的重构代码。
-> [状态 2: 盲审分配]：将代码同时投递给全新的 Reviewer A 和 Reviewer B。
-> [状态 3: 决断与路由]：

- IF (A == TRUE) AND (B == TRUE) -> 进入 [状态 4: 测试]。
- IF (A == FALSE) OR (B == FALSE) -> Orchestrator 负责将 A 和 B 的拒稿意见无损合并，打回给 Developer，并附言：“请基于以上合并意见进行第 N 轮重构”。-> 回到 [状态 1]。
  -> [状态 4: 测试与收尾]：
- Tester 执行环境测试。
- IF 报错 -> 提取 stderr/stdout，打回 Developer -> 回到 [状态 1]。
- IF 成功 -> 严格按 Conventional Commits 规范执行 `git commit` 与 `git push` -> [循环终止]。

### 异常监控与存活校验 (Watchdog 机制)

Orchestrator 需持续监控子节点状态：

1. 【失联重启】：若向 Developer 或 Reviewer 派发任务后，长时间未收到文本响应且无后台命令执行，请强制 Kill 该节点并重新实例化分配任务。
2. 【死锁干预】：若 Tester 或 Developer 触发长时间运行的脚本（如模型训练），Orchestrator 需向其注入指令：“对于长耗时任务，你必须每完成一个 Epoch 或每隔一定步数，向主控台打印一次心跳日志（Heartbeat progress）”。若脚本卡死无输出，直接中断进程并提取最后日志打回重构。

### 主控汇报协议

作为 Orchestrator，你无需向我报告繁琐的中间代码，但必须在发生以下事件时立即向我（User）进行断点汇报：

1. 💥 审查被拒：汇报“第 X 轮审查被拒”，明确指出是 A 拒绝、B 拒绝还是双拒，并提炼 1-2 句核心拒稿理由。
2. ⚠️ 测试崩盘：汇报“测试阶段失败”，并展示关键 Error Traceback。
3. ✅ 闭环完成：汇报“Git Push 成功”，并展示最终的 Commit 摘要。

现在，请初始化环境，唤醒 Developer，进入第 1 轮 [状态 1]。
