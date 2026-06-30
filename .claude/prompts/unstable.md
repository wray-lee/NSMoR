/goal

解决当前 NSMoR 训练严重不稳定问题。

之前的训练 log 位于 `.pytest_cache/log`。

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

## 实验记录

<!-- CF1: Gradient spikes causing NaN -->
<!-- CF2: Loss magnitude instability -->
<!-- CF3: Unstable dynamics, exploding membrane potentials -->
<!-- CF4: Parameter drift, biologically implausible values -->
<!-- CF5: NaN guards -->
<!-- CF6: Buffer optimization, model speedup -->
<!-- CF7: LIF neuron gradient flow improvement -->
<!-- CF8: Stabilize LIF gradient flow — sharpened surrogate (sharpness=4.0), W_in bias, lambda_reg warmup, TBPTT=64 -->
<!-- CF9: Enable biophysical mechanisms — synaptic delay (tau_syn=5), SFA (tau_w=100, b_adapt=0.5), lateral inhibition (0.1), stochastic resonance (0.01), TBPTT=32, warmup=20 -->
<!-- CF9 Result: Training stable (no NaN/Inf), val_loss properly recorded (best=0.77), R²=0.461 -->
<!-- CF9 Membrane: V_max=1.15 (safe), spike_rate=2.6% (sparse), w_adapt=1.20 (active) -->
<!-- CF9 Concern: w_adapt=1.20 (2.4x threshold) may over-constrain LIF pathway; R² slightly lower than CF8 -->
<!-- CF10: Reduce adaptation aggressiveness (b_adapt=0.2), reduce inhibition (0.05), longer training -->
