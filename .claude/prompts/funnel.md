请接管后台，将 NSMoR 架构重构为“混合漏斗架构（Hybrid Funnel）”。

【环境约束】
所有需运行代码的节点，必须使用此链式命令：
`wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <实际执行的命令>"`

【重构核心目标】

1. model_nsmor_core.py：拆分为 `FrontendEncoder` 与 `BioDecisionCore`。两者间必须插入 `.detach()` 彻底切断梯度回传。
2. loss.py：分离损失，前端常规拟合，后端承担物理与 ATP 能量代谢惩罚。
3. scripts/train.py：实现两阶段训练（阶段一仅更新前端；阶段二彻底冻结前端，仅更新后端）。

【平行审查与执行流】

1. 唤起 @nsmor_developer 执行开发。
2. 将代码同时分发给两个全新的独立沙箱 @nsmor_reviewer_A 与 @nsmor_reviewer_B 执行双盲审。
    - 审查焦点：梯度是否真正切断？物理约束是否被错误施加于前端？
    - 规则：只有当 A 与 B 同时给出 ACCEPT 才可放行；否则汇总拒稿意见打回 Developer 强制重构。
3. 盲审通过后，唤起 @nsmor_tester 执行 `python scripts/train.py --epochs 2`，确保两阶段无 OOM/NaN，最后按 Conventional Commits 规范执行 Git commit。

请在后台自动推进此循环。如果脚本执行卡死超 30 分钟请强制打断。仅在发生“审查被拒”、“测试报错”或“最终提交成功”时向我汇报断点。
