# AGENTS.md — AI Agent Harness Operating Protocol

This document defines the multi-agent operating rules, roles, permissions, execution constraints, and state-machine protocols for all AI agents working on the NSMoR repository.

---

## 1. Core Operating Environment & Constraints

- **Execution Environment**: All bash commands, Python scripts, PyTorch operations, and test runs MUST execute via WSL (Zsh environment using `t` conda activate alias).
- **Git Identity Verification**: All commits MUST strictly use:
  - `user.name`: `wray-lee`
  - `user.email`: `i@wray7.top`
  - *Do NOT create commits under any secondary alias or unverified email.*
- **Pronoun Policy**: Neutral default (they/them) MUST be used for any unspecified individual.

---

## 2. Multi-Agent Role Architecture

The NSMoR harness defines three specialized roles (plus an optional Orchestrator coordinator) for multi-agent development and quality assurance:

| Agent Role | Subagent Slug | Primary Responsibility | Permissions & Scope |
| :--- | :--- | :--- | :--- |
| **Developer** | `nsmor_developer` | Refactoring, feature implementation, PyTorch modeling, analysis tools. | Read/Write code in editable zones (`scripts/`, `nsmor/analysis/`, `nsmor/pipeline/`). **Core frozen code (`nsmor/model_nsmor_core.py`, `nsmor/loss.py`) requires user override.** |
| **Reviewer (#2)** | `nsmor_reviewer` | Double-blind peer review across biological plausibility, mathematical stability, and statistical rigor. | Read-only code access. Must issue explicit `**ACCEPT**` or `**REJECT**` with structured critiques. *Prohibited from writing fix code directly.* |
| **Tester** | `nsmor_tester` | Physical smoke testing, pipeline reset, numerical stability interception, regression testing, and Git release. | Run pipeline/tests (`pytest`, `train.py --epochs 1`, analysis scripts), perform data resets, execute Git commit/push upon zero-error gate pass. |
| **Orchestrator** | `orchestrator` (skill) | Loop engine coordinating DEV→REVIEW→TEST→COMMIT state machine. | Dispatches tasks, merges feedback, enforces watchdog timeouts. See `.claude/skills/orchestrator/SKILL.md`. |

---

## 3. Mandatory State-Machine Workflow

Every substantive code modification or refactoring must traverse the strict 4-stage state machine. The Orchestrator skill (`.claude/skills/orchestrator/SKILL.md`) automates this loop with watchdog timeouts, heartbeat monitoring, and breakpoint reports to the user.

**Parallel Double-Blind Review**: The Orchestrator dispatches the developer's code snapshot to **two independent reviewer instances** (`nsmor_reviewer_A` and `nsmor_reviewer_B`) in isolated sessions. Both must return `ACCEPT` for the workflow to proceed. This prevents single-point-of-failure review bias.

Every substantive code modification or refactoring must traverse the strict 4-stage state machine:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  1. DEVELOP     │ ────> │  2. REVIEW      │ ────> │  3. TEST        │ ────> │  4. COMMIT      │
│ (nsmor_developer)│       │ (nsmor_reviewer)│       │  (nsmor_tester) │       │  (nsmor_tester) │
└─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
        ▲                         │                         │
        │                         │ REJECT                  │ FAIL
        └─────────────────────────┴─────────────────────────┘
```

1. **Phase 1: Development (`nsmor_developer`)**
   - Implements code following PyTorch shape assertion requirements (`assert tensor.shape == (...)`).
   - Ensures mathematical, statistical, and biological plausibility.
   - Generates a Proposal Report detailing [Motivation, Implementation, Rationale].

2. **Phase 2: Double-Blind Review (`nsmor_reviewer`)**
   - Evaluates proposal and diff against three axes:
     - *Biological Plausibility* (refractory periods, ATP consumption, metabolic costs).
     - *Mathematical Dynamics* (Jacobian stability, non-differentiable manifold continuity).
     - *Statistical Rigor* (FDR/Bonferroni corrections, effect sizes, sample sufficiency).
   - Returns `**REJECT**` (with actionable flaws, sending work back to Phase 1) or `**ACCEPT**`.

3. **Phase 3: Integration & Testing (`nsmor_tester`)**
   - Executes WSL Zsh test sequence:
     - Data pipeline smoke test (`make load && make data` or `python scripts/train.py --epochs 1`).
     - Numerical stability check (zero `NaN`/`Inf` tolerance).
     - Analysis script validation (`scripts/analyze_dynamics.py`).
     - Full test suite regression (`pytest tests/ -v`).

4. **Phase 4: Release & Commit (`nsmor_tester`)**
   - Squashes local iterations via `git rebase -i` if multiple commits exist.
   - Enforces structured commit message with footer `Approved-by: Reviewer #2`.
   - Performs `git commit` and `git push`.

---

## 4. Directory Access & Architectural Boundaries

Agents must strictly respect path-level boundary declarations (`BOUNDARY.md`):

| Path / Module | Status | Boundary File | Constraints |
| :--- | :--- | :--- | :--- |
| `nsmor/model_nsmor_core.py`, `nsmor/loss.py` | 🔒 **FROZEN** | `nsmor/BOUNDARY.md` | Core tensor math & loss function. Modification requires explicit user override. |
| `nsmor/pipeline/` | 🔓 **EDITABLE** | `nsmor/pipeline/BOUNDARY.md` | Data ingestion, kinematics, feature collation. Backward compatibility required. |
| `nsmor/analysis/` | 🔓 **SANDBOX** | `nsmor/analysis/BOUNDARY.md` | Fixed-point finders, Jacobian computations, dynamical manifolds. Free to extend. |
| `scripts/` | 🔓 **EDITABLE** | — | Training, simulation, and evaluation entry points. Must preserve interface contracts. |
| `tests/` | 🔓 **EDITABLE** | — | Regression and integration test suite. Must accompany any new feature or fix. |

---

## 5. Engineering Standards & Quality Invariants

- **Shape Assertions**: Every tensor transformation must include runtime shape assertions.
- **Numerical Protection**: Defensive guards against division-by-zero, log-zero, and exploding gradients (`torch.clamp`, `torch.nan_to_num`).
- **Statistical Audit**: Multi-condition evaluations must report effect sizes (Cohen's $d$ or $\eta^2$) and adjusted p-values (FDR/Bonferroni).
- **Convergence Verification**: Core modifications must demonstrate stable loss convergence within 20 training epochs.

---

## 6. WSL Execution Command Template

All subagents must use the following atomic command chain for any Python/pytest execution:

```bash
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <command>"
```

**Requirements:**
- Must use `-i` flag (interactive) to ensure `.zshrc` aliases load correctly.
- Must NOT split the chain — keep atomic.
- All Windows paths must be converted to WSL mount paths (`/mnt/d/...`).

---

## 7. Orchestrator Skill Reference

The full loop engine, watchdog configuration, heartbeat protocol, and breakpoint reporting format are defined in:

```
.claude/skills/orchestrator/SKILL.md
```

Key parameters:
- `MAX_ITER`: Maximum rejection loop iterations (default: 10)
- **Heartbeat Timeout**: 5 minutes without response → kill and restart agent
- **Long Task Injection**: Forces epoch-level progress logs for training runs
- **Breakpoint Reports**: Structured emoji-tagged status updates to user on reject/fail/success
