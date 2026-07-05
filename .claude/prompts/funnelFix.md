请接管后台工作流，命令子智能体修复 NSMoR 训练脚本 (`scripts/train.py`) 中的“梯度死锁与 AMP 溢出（Gradient Deadlock）”致命 Bug。

【环境约束】
必须使用链式命令执行测试：`wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <命令>"`

【Bug 物理与数学背景】
在“混合漏斗架构”进入第二阶段（Phase 2）时，前端网络（Frontend）被冻结，但其参数上仍残留了第一阶段的“幽灵梯度”。由于 Phase 1 的 MSE 梯度极小，AMP (FP16) 积累了巨大的 Scale Factor。进入 Phase 2 后，复杂的生物学惩罚算子产生剧烈梯度，瞬间导致 FP16 下溢出为 `Inf`。
致命漏洞在于：代码中的全局 `clip_grad_norm_` 对**全模型参数（包含被冻结的残留幽灵梯度）**求了范数，导致总范数为 Inf，从而将所有梯度乘零化为 NaN。致使模型陷入 `train_loss=0.000000` 且连续跳过 Batch 的死循环。

【@nsmor_developer 重构执行清单】
请严格在 `scripts/train.py` 中落实以下 3 处外科手术式修改：

1. **阶段切换点的彻底清洗**：在 Phase 1 → Phase 2 的过渡代码块中（冻结 frontend 之前），强制调用 `model.zero_grad(set_to_none=True)` 抹除所有残余梯度。
2. **裁剪域与异常检测的精确隔离**：在 `train_one_epoch` 中：
    - 将常规的 `optimizer.zero_grad()` 替换为 `model.zero_grad(set_to_none=True)`。
    - 提取出活跃参数：`trainable_params = [p for p in model.parameters() if p.requires_grad]`
    - 强制将 `clip_grad_norm_` 和后续的 `has_nan_grad` 检查的遍历对象，从 `model.parameters()` 替换为 `trainable_params`。
3. **修复指标统计泄漏**：在计算并打印 `lif_grad_norm` 的逻辑中，必须增加 `if p.requires_grad and p.grad is not None:` 判断，绝不允许读取已被冻结参数的梯度属性。

【@nsmor_reviewer_A 与 @nsmor_reviewer_B 盲审基准】

- 独立审查 `train_one_epoch` 函数。
- 致命拦截点：如果发现 `clip_grad_norm_` 依然作用于全量 `model.parameters()`，或者没有使用 `set_to_none=True` 来释放显存中的旧梯度，必须首行输出 **REJECT** 并打回重写。

【@nsmor_tester 验收闸门】

- 必须构建一个跨越两阶段的极速测试用例：如运行 `python scripts/train.py --epochs 3 --phase1_epochs 1` 或类似的小轮次参数，确保必定触发 Phase 1 -> Phase 2 切换。
- 如果在切换后的 Batch 中再次发生无限期的 `loss=0.0` 或死锁，提取错误日志打回。
- 测试跑通后，生成包含 "fix(train): resolve phase transition gradient deadlock by filtering trainable params" 摘要的 Conventional Commit 并 Push。
