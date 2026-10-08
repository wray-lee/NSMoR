# NSMoR — AI Context Harness & Engineering Guidelines

## Multi-Agent Protocol & AGENTS.md (CRITICAL)
- **Harness Specification**: Read `AGENTS.md` and `HARNESS.md` for full details on multi-agent execution rules, double-blind peer review protocols, and state machine loops.
- **Git Author / Committer**: MUST be `wray-lee <i@wray7.top>` (verified GitHub primary email).
- **Git Constraint**: Never commit under unverified emails or secondary aliases. Always verify `git config user.email` returns `i@wray7.top` before creating commits.
- **⚠️ Email Redaction Hazard**: Claude Code 环境会将邮箱地址脱敏为 `[EMAIL_REDACTED]`。若 `git config user.email` 被环境或 agent 意外写入字面量 `[EMAIL_REDACTED]`，后续所有 commit 将无法关联 GitHub 账户（无头像、不计入 contribution）。**每次 session 开始时必须验证**：`git config user.email` 输出的是真实邮箱 `i@wray7.top` 而非 `[EMAIL_REDACTED]`。如不正确，立即执行 `git config user.email 'i@wray7.top'`。
- **WSL Execution Environment**: All python/pytest/bash operations run in WSL Zsh with `t` conda activate alias.
- **Workflow 定点失败重试（CRITICAL）**: Workflow 内部 subagent 失败时，只重启或补跑指定的失败 subagent；不得停止或重启整个 Workflow、重跑成功项或打断正常运行的 subagent。先用 journal 定位失败项并保留成功结果；若无单项重试接口，用只含失败项的最小补跑 Workflow，再接入必要的后续步骤。不得仅凭 `resume` 调用声称缓存已复用，必须以实际执行记录确认。
- **Claude Pro 并行 Agent 熔断与模型分级规则（CRITICAL）**:
  - **背景与风控**: Claude Pro 属于交互式（interactive）配额体系，无法承受高阶模型大并发（实测 12 个 Opus 5.5 xhigh 并行 agent 会在 17.5 分钟内耗尽 90% 的 5 小时配额并触发 44 分钟 429 Rate Limit）。
  - **派发 >= 4 并行 subagent 前强制检查**:
    1. **模型分级（Tiering）**: 有明确 rubric / checklist / 可验证探针的 segment 或 worker 任务，**必须**使用 Sonnet 或 subagent 默认模型（Sonnet / subagent 默认模型不设并发限制），优先 subagent 默认模型，若困难部分分配sonnet；仅在需要跨分段裁决冲突、给出最终结论的 synthesis 节点允许使用顶级模型（Opus/Fable）。**严禁全员 Opus/Fable + xhigh 并行**。
    2. **顶级模型并发上限**: 单个 Pro 环境下，Opus / Fable 顶级模型并行数 **<= 3**。
    3. **并发例外**: Sonnet / subagent 默认模型不设置并发上限

---

## Project Architecture

NSMoR (Biological Mixture-of-Recursions) models **cricket multi-sensory integration** using a dual-pathway recurrent neural network:

- **LIF Pathway:** Leaky Integrate-and-Fire spiking neuron for fast, event-driven sensory transients.
- **GRU Pathway:** Gated Recurrent Unit for smooth, continuous temporal integration.
- **MoR Router:** Learned representational-routing gate that blends LIF and GRU outputs per timestep. It produces per-step softmax weights ``[g_lif, g_gru]`` over the two pathway outputs; it is not a causal-inference estimator.

Designed for **white-box dynamical systems analysis**: expose routing gates, membrane potentials, spike events, and GRU hidden states for fixed-point and Jacobian analysis.

---

## Mandatory Engineering & Coding Standards

1. **Strict Type Hinting:** All function signatures must have complete type annotations.
2. **Tensor Shape Assertions:** Every `forward()` pass and state transformation must include explicit shape assertions:
    ```python
    assert tensor.shape == (B, T, H), f"Expected (B={B}, T={T}, H={H}), got {tensor.shape}"
    ```
3. **Modular Design & Controlled Changes:** Core mathematical code (`nsmor/model_nsmor_core.py`, `nsmor/loss.py`) may change within a user-authorized optimization or correctness scope. Follow the normative protocol in `nsmor/BOUNDARY.md`: two independent `ACCEPT` reviews, focused and complete regression tests, numerical safety checks, and backward-compatibility evidence. Existing task authorization covers necessary core changes without repeated approval.
4. **Statistical Rigor:** Multi-condition comparisons must calculate effect sizes (Cohen's $d$) and adjusted p-values (FDR/Bonferroni).
5. **Code Style:** PEP 8 (88-char limit), `from __future__ import annotations`, Google-style docstrings.

---

## Modification & Boundary Permissions

| Directory / Module | Status | Permission & Boundary File | Notes |
| :--- | :--- | :--- | :--- |
| `model_nsmor_core.py`, `loss.py` | **Controlled** | 🔐 `nsmor/BOUNDARY.md` | Core changes within user-authorized scope require two independent reviews and complete tests |
| Other `nsmor/` root infrastructure | **Protected** | `nsmor/BOUNDARY.md` | Canonical loaders and data/checkpoint contracts need separately scoped authorization |
| `nsmor/pipeline/` | **Extend** | 🔓 `nsmor/pipeline/BOUNDARY.md` | Data ingestion, feature extraction, dataloader factory |
| `nsmor/analysis/` | **Sandbox** | 🔓 `nsmor/analysis/BOUNDARY.md` | Fixed-point analysis, Jacobians, dynamical manifolds, UQ |
| `scripts/` | **Editable** | — | Training, simulation, analysis scripts |
| `tests/` | **Editable** | — | Pytest suite & integration fixtures |
| `config/` | **Editable** | — | Model & dataset hyperparameter YAML files |

---

## AI Agent Workflow Protocol

```
Developer (nsmor_developer) ── proposal ──> Reviewer #2 (nsmor_reviewer)
      ▲                                                │
      │                                       ACCEPT   │   REJECT (Critique)
      └────────────────────────────────────────────────┴───────┐
                                                               ▼
Release Commit <── Pass Gate ── Tester (nsmor_tester) <── Fix Proposal
```

1. **`nsmor_developer`**: Implements code, ensuring shape assertions, numerical safety, and biological plausibility. Submit proposal to `nsmor_reviewer`.
2. **`nsmor_reviewer`**: Conducts double-blind audit across (1) Biological Plausibility, (2) Mathematical Dynamics, (3) Statistical Rigor. Emits `**ACCEPT**` or `**REJECT**`.
3. **`nsmor_tester`**: Executes data pipeline smoke test, 1-epoch train test, `pytest` suite, numerical `NaN`/`Inf` sweep, and performs Git rebase/commit/push with `Approved-by: Reviewer #2`.

---

## AI Directives — Critical Constraints

1. **Do not rewrite core modules wholesale** when asked to build analysis scripts, training pipelines, or testing infrastructure. `nsmor/model_nsmor_core.py` and `nsmor/loss.py` may be modified when the task explicitly scopes a core change and the controlled-change protocol is followed.
2. **Do not rewrite the loss module wholesale** when asked to add new features or analysis tools. `nsmor/loss.py` may be modified for a declared optimization or correctness change, with its own focused tests and independent review.
3. **ALWAYS check `BOUNDARY.md` files** in subdirectories before modifying code:
    - `nsmor/BOUNDARY.md` — controlled core-change protocol and contracts
    - `nsmor/pipeline/BOUNDARY.md` — Data pipeline (safe to extend)
    - `nsmor/analysis/BOUNDARY.md` — Analysis sandbox (free to modify)
4. **ALWAYS preserve tensor shape assertions** when refactoring. Do not remove `assert` statements in `forward()` methods.
5. **ALWAYS maintain backward compatibility** when extending modules. Existing imports must continue to work.

### When Building New Analysis Tools

1. Create new files in `nsmor/analysis/` — do NOT add to `nsmor/` root.
2. Import from the controlled core modules — do NOT copy code:
    ```python
    from nsmor.model_nsmor_core import NSMoRCore
    from nsmor.loss import BioJointLoss
    ```
3. Respect the I/O contracts defined in `BOUNDARY.md` files.
4. Add shape assertions to all new functions.

---

## Agent skills

### Issue tracker

Issues live as GitHub issues on wray-lee/BioMoR. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context repo: one `CONTEXT.md` + `docs/adr/` at root. See `docs/agents/domain.md`.

---

## Quick Execution Commands

```bash
# WSL Zsh Torch Environment Setup
t  # alias for conda activate torch

# Run training
python scripts/train.py --config config/default.yaml

# Run test suite
pytest tests/ -v

# Run smoke tests
python -m nsmor.model_nsmor_core
python -m nsmor.loss
python -m nsmor.analysis.dynamics
```
