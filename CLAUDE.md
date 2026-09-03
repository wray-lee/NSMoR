# NSMoR — AI Context Harness & Engineering Guidelines

## Multi-Agent Protocol & AGENTS.md (CRITICAL)
- **Harness Specification**: Read `AGENTS.md` and `HARNESS.md` for full details on multi-agent execution rules, double-blind peer review protocols, and state machine loops.
- **Git Author / Committer**: MUST be `wray-lee <i@wray7.top>` (verified GitHub primary email).
- **Git Constraint**: Never commit under unverified emails or secondary aliases. Always verify `git config user.email` returns `i@wray7.top` before creating commits.
- **⚠️ Email Redaction Hazard**: Claude Code 环境会将邮箱地址脱敏为 `[EMAIL_REDACTED]`。若 `git config user.email` 被环境或 agent 意外写入字面量 `[EMAIL_REDACTED]`，后续所有 commit 将无法关联 GitHub 账户（无头像、不计入 contribution）。**每次 session 开始时必须验证**：`git config user.email` 输出的是真实邮箱 `i@wray7.top` 而非 `[EMAIL_REDACTED]`。如不正确，立即执行 `git config user.email 'i@wray7.top'`。
- **WSL Execution Environment**: All python/pytest/bash operations run in WSL Zsh with `t` conda activate alias.

---

## Project Architecture

NSMoR (Biological Mixture-of-Recursions) models **cricket multi-sensory integration** using a dual-pathway recurrent neural network:

- **LIF Pathway:** Leaky Integrate-and-Fire spiking neuron for fast, event-driven sensory transients.
- **GRU Pathway:** Gated Recurrent Unit for smooth, continuous temporal integration.
- **MoR Router:** Learned causal inference gate that blends LIF and GRU outputs per timestep.

Designed for **white-box dynamical systems analysis**: expose routing gates, membrane potentials, spike events, and GRU hidden states for fixed-point and Jacobian analysis.

---

## Mandatory Engineering & Coding Standards

1. **Strict Type Hinting:** All function signatures must have complete type annotations.
2. **Tensor Shape Assertions:** Every `forward()` pass and state transformation must include explicit shape assertions:
    ```python
    assert tensor.shape == (B, T, H), f"Expected (B={B}, T={T}, H={H}), got {tensor.shape}"
    ```
3. **Modular Design & Immutability:** Core mathematical code (`nsmor/model_nsmor_core.py`, `nsmor/loss.py`) is **frozen**. Modifications require explicit user override.
4. **Statistical Rigor:** Multi-condition comparisons must calculate effect sizes (Cohen's $d$) and adjusted p-values (FDR/Bonferroni).
5. **Code Style:** PEP 8 (88-char limit), `from __future__ import annotations`, Google-style docstrings.

---

## Modification & Boundary Permissions

| Directory / Module | Status | Permission & Boundary File | Notes |
| :--- | :--- | :--- | :--- |
| `nsmor/` (Root Core) | **Frozen** | 🔒 `nsmor/BOUNDARY.md` | Frozen core architecture & loss — requires explicit user override |
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

1. **NEVER rewrite `nsmor/model_nsmor_core.py`** when asked to build analysis scripts, training pipelines, or testing infrastructure. This module is stable and frozen.
2. **NEVER rewrite `nsmor/loss.py`** when asked to add new features or analysis tools. The loss function is mathematically verified and frozen.
3. **ALWAYS check `BOUNDARY.md` files** in subdirectories before modifying code:
    - `nsmor/BOUNDARY.md` — Frozen core (requires explicit override)
    - `nsmor/pipeline/BOUNDARY.md` — Data pipeline (safe to extend)
    - `nsmor/analysis/BOUNDARY.md` — Analysis sandbox (free to modify)
4. **ALWAYS preserve tensor shape assertions** when refactoring. Do not remove `assert` statements in `forward()` methods.
5. **ALWAYS maintain backward compatibility** when extending modules. Existing imports must continue to work.

### When Building New Analysis Tools

1. Create new files in `nsmor/analysis/` — do NOT add to `nsmor/` root.
2. Import from frozen core modules — do NOT copy code:
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
