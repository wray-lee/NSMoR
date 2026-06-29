/goal 解决现在训练不稳定的问题。**调用subagents在主会话监听**完成NSMoR 架构重构闭环。强制执行“平行双盲评审”机制：

- @nsmor_developer：读取代码并输出重构方案。
- @nsmor_reviewer_A 与 @nsmor_reviewer_B：每次调用必须实例化为两个全新、无历史状态的独立 agent。两者均需对生物物理机制与工程实现进行全盘审查。

## 条件约束

- 如果一个agent超过30min没有响应，且没有在执行程序，则需要对这个agent执行重启。
- 如果一个agent在执行程序，超过4h没有响应，应当重启并开启一个新的agent实例，告知这个agent需要每5min输出一次进度。

## 状态路由规则：

1. Developer 生成代码 -> 同时平行移交至 Reviewer A 与 Reviewer B 进行独立盲审（A与B互不可见）。
2. A 与 B 各自输出布尔值 `is_accepted` 及具体审查意见。
3. 状态判定：
    - 只有当 A 且 B 均输出 ACCEPT 时：将代码移交 @nsmor_tester。
    - 若任意一方或双方输出 REJECT：将 A 和 B 的所有审查意见合并汇总，一次性打回给 Developer。
4. Developer 基于汇总意见进行下一轮重构 -> 产出后再次交由全新的 A 和 B 平行盲审。
5. Tester 执行失败 -> 提取 error log 打回给 Developer。
6. Tester 执行通过 -> 执行 Git commit 并 push。

请在后台接管此控制流。一旦发生审查被拒（需输出是谁拒绝及核心理由）、测试失败或最终成功，立即向我汇报。

> 告知所有agent,python环境要调用**wsl**的zsh,运行alias指令`openconda`后激活conda环境，`conda activate torch` 来调用python环境
