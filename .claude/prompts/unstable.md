/goal

解决现在

```
脉冲神经网络（SNN）典型的梯度消失与层间更新失衡。

数据中暴露出三个致命特征：

LIF 层梯度塌陷 (Gradient Vanishing)

日志显示，从 Epoch 10 到 Epoch 30，LIF 与 non-LIF 层的梯度范数比值（ratio）从 0.12 暴跌至 0.02。

到 Epoch 30 时，LIF 的梯度仅剩 0.0219，而 non-LIF（通常是全连接/输出层）高达 0.9998。这意味着作为特征提取核心的脉冲层已经基本停止学习。

非脉冲层的代偿性震荡

因为前置的 LIF 层不更新，后置的 non-LIF 层在试图用随机的脉冲输出强行拟合标签。这种“头重脚轻”的参数更新直接导致了 Loss 的剧烈横跳（如 Epoch 26 到 27，Val Loss 从 2.76 直接飙升至 6.68）。

神经元处于梯度死区

平均膜电位（V_mean）始终停滞在 0.64-0.65 附近，距离阈值（1.00）较远。

约 5% 的发放率（spike_rate）虽然符合生物学上的稀疏编码特征，但结合极低的梯度来看，说明绝大多数神经元的膜电位未进入替代梯度（Surrogate Gradient）的有效激活窗口。
```

导致的训练不稳定问题

之前的训练log 位于 `.pytest_cache\log`

**调用subagents在主会话监听**完成NSMoR 架构重构闭环。强制执行“平行双盲评审”机制：

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
