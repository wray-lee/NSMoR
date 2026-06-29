/goal

解决当前 NSMoR 训练严重不稳定与 LIF 通路梯度塌陷问题。

## 问题上下文与物理诊断

先前的基础超参数调整（LR=5e-4, weight_decay=1e-4）已在当前配置中生效，但未能阻止训练崩溃。
日志暴露的核心病灶如下：

1. **LIF 梯度断崖式塌陷**：Epoch 0 时 LIF/non-LIF 梯度比为 0.45（健康），但到 Epoch 10 跌至 0.12，Epoch 20 仅剩 0.04（LIF=0.0306, non_LIF=0.8636）。
2. **极度代偿性震荡**：由于 LIF 停止学习，后置网络或 GRU 强行拟合，导致 `train_loss` 与 `val_loss` 在 1.4 到 5.8 之间发生毁灭性震荡，完全无法收敛。
3. **脉冲死区**：`V_mean` 停滞在 0.64 左右（远离阈值 1.00），大量神经元掉出替代梯度（Surrogate Gradient）的有效更新窗口。

之前的训练 log 位于 `.pytest_cache/log`。请重点排查 `nsmor/model_nsmor_core.py` 中的替代梯度宽度/形状、LIF 参数初始化状态，以及 `nsmor/loss.py` 中 `lambda_reg` 是否引发了路由门的硬切换震荡。

## 执行机制：平行双盲评审重构闭环

**调用 subagents 在主会话监听**并完成 NSMoR 架构重构闭环。强制执行“平行双盲评审”机制：

- **@nsmor_developer**：读取代码、分析上述病灶并输出重构方案与补丁。
- **@nsmor_reviewer_A** 与 **@nsmor_reviewer_B**：每次调用必须实例化为两个全新、无历史状态的独立 agent。两者均需对生物物理机制（神经动力学合理性）与工程实现（梯度流连续性、防 NaN 处理）进行全盘审查。

## 异常与超时约束

- 如果一个 agent 超过 30min 没有响应，且没有在执行程序，则需要对该 agent 执行重启。
- 如果一个 agent 在执行程序，超过 4h 没有响应，应当重启并开启一个新的 agent 实例，告知这个 agent 需要每 5min 输出一次进度。

## 环境与环境状态要求

> 告知所有 agent：Python 环境必须调用 **wsl** 的 zsh，运行 alias 指令 `openconda` 后激活 conda 环境，执行 `conda activate torch` 来调用 Python 运行环境。

## 状态路由规则

1. Developer 生成代码 -> 同时平行移交至 Reviewer A 与 Reviewer B 进行独立盲审（A 与 B 互不可见）。
2. A 与 B 各自输出布尔值 `is_accepted` 及具体审查意见。
3. 状态判定：
    - 只有当 A 且 B 均输出 ACCEPT 时：将代码移交 @nsmor_tester。
    - 若任意一方或双方输出 REJECT：将 A 和 B 的所有审查意见合并汇总，一次性打回给 Developer。
4. Developer 基于汇总意见进行下一轮重构 -> 产出后再次交由全新的 A 和 B 平行盲审。
5. Tester 执行 `make modeltest` 或其他测试脚本。若执行失败 -> 提取 error log 打回给 Developer。
6. Tester 执行通过 -> 执行 Git commit 并 push。

请在后台接管此控制流。一旦发生审查被拒（需输出是谁拒绝及核心理由）、测试失败或最终成功，立即向我汇报。
