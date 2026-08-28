# HARNESS.md — AI Agent Harness Master Engineering Specification

This specification documents the architecture, lifecycle management, quality enforcement loops, double-blind review system, and environment invariants of the NSMoR AI Agent Harness.

---

## 1. System Architecture Overview

The NSMoR AI Agent Harness is a multi-agent orchestration framework designed to ensure top-tier scientific rigor, software reliability, and deterministic reproducibility in computational neuroscience models.

```
                  +-----------------------------------+
                  |      User / Orchestrator          |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |       1. nsmor_developer          |
                  |  (PyTorch / Math Implementation)  |
                  +-----------------------------------+
                                    |
                                    v [Proposal & Diff]
                  +-----------------------------------+
                  |       2. nsmor_reviewer (#2)      |
                  | (Double-Blind Peer Review Audit)  |
                  +-----------------------------------+
                             /             \
                   REJECT   /               \   ACCEPT
                           v                 v
          +-------------------+   +-----------------------+
          | Developer Fix Loop|   |    3. nsmor_tester    |
          +-------------------+   | (E2E Physics & CI/CD) |
                                  +-----------------------+
                                              |
                                              v [Pass Gate]
                                  +-----------------------+
                                  |    4. Git Release     |
                                  | (Rebase & Push w/ ID) |
                                  +-----------------------+
```

---

## 2. Layered Architecture (5-Layer Suite)

The harness consists of 5 tightly integrated governance layers:

1. **Root Protocol Layer (`AGENTS.md`)**: Specifies subagent roles, rights, state transitions, and approval gates.
2. **Session Guidance Layer (`CLAUDE.md`)**: Controls LLM CLI behavior, coding idioms, PEP 8 standards, and prompt directives.
3. **Hierarchical Boundary Matrix (`BOUNDARY.md`)**: Enforces directory-level immutability rules:
   - Root (`/BOUNDARY.md`): Git identity and architecture layout.
   - Core (`nsmor/BOUNDARY.md`): Frozen PyTorch mathematical core (`NSMoRCore`, `BioJointLoss`).
   - Pipeline (`nsmor/pipeline/BOUNDARY.md`): Data ingestion and feature extraction contracts.
   - Sandbox (`nsmor/analysis/BOUNDARY.md`): Free-form dynamical systems analysis tools.
4. **State Machine & Watchdog Layer (`HARNESS.md`)**: Defines lifecycle execution, failure handling, and double-blind verification loops.
5. **Executable Agent Definitions (`.claude/agents/*.md` & `.claude/skills/`)**: Houses subagent system prompts and orchestrator workflows.

---

## 3. Double-Blind Peer Review Protocol

The Orchestrator dispatches the developer's code snapshot to **two independent reviewer instances** (`nsmor_reviewer_A` and `nsmor_reviewer_B`) in isolated sandbox sessions. Neither reviewer sees the other's output — this implements true double-blind review.

**Acceptance Gate**: Both reviewers must return `[is_accepted: TRUE]` for work to proceed to testing. If either returns `[is_accepted: FALSE]`, feedback from both is merged losslessly and sent back to the developer.

Each reviewer audits against three non-negotiable scientific criteria:

### A. Biological Plausibility
- Validate that parameters maintain physiological plausibility (LIF membrane dynamics, leakage rate, absolute/relative refractory periods, ATP metabolic consumption).
- Reject artificial machine-learning shortcuts that lack biological grounding in insect escape circuits or giant fiber pathways.

### B. Mathematical & Dynamical Stability
- Verify numerical stability of matrix operations, continuous-time integration, and Jacobian spectrum extraction around fixed points.
- Ensure proper handling of non-differentiable spiking boundaries and manifold discontinuities.

### C. Statistical Rigor
- Enforce mandatory reporting of effect sizes (Cohen's $d$, $\eta^2$) alongside p-values.
- Enforce multiple testing corrections (Bonferroni / FDR) for high-dimensional feature sweeps.
- Block conclusions based solely on unadjusted $p < 0.05$.

---

## 4. Quality Enforcement & Testing Gates

The tester agent (`nsmor_tester`) acts as the final gatekeeper before Git commits:

1. **Environment Initialization**: Runs under WSL Zsh using `t` conda activate alias.
2. **Data Pipeline Reset**: Standardizes test inputs (`make load && make data` or synthetic fixtures).
3. **Smoke & Stability Gate**: Executes `python scripts/train.py --config config/default.yaml --epochs 1 --output_dir runs/test` and checks for zero `NaN`/`Inf` tensor values.
4. **Regression Gate**: Executes complete pytest suite (`pytest tests/ -v`).
5. **Commit Gate**: Performs interactive squash/rebase if needed and injects `Approved-by: Reviewer #2` in commit footers under identity `wray-lee <i@wray7.top>`.

---

## 5. Error Recovery & Watchdog Interventions

- **Loop Interception**: If Developer and Reviewer reach a 3-turn rejection loop, the Orchestrator interrupts to request human intervention or explicit scope adjustment.
- **Numerical Overflow**: Any `NaN` or `Inf` immediately aborts the pipeline stage, triggering an automatic roll-back and sending log diagnostics back to `nsmor_developer`.
- **Boundary Violation**: Unsanctioned attempts to modify frozen core files (`nsmor/model_nsmor_core.py`, `nsmor/loss.py`) without prior user authorization trigger hard failure.

---

## 6. Cross-Reference: Orchestrator Skill

The full automated loop engine implementation lives in `.claude/skills/orchestrator/SKILL.md` and defines:

- **WSL Atomic Command Chain**: `wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && <cmd>"`
- **Parallel Reviewer Dispatch**: Independent A/B sessions with no cross-contamination.
- **Heartbeat Timeout**: 5 minutes without response triggers kill/restart.
- **Long Task Watchdog Injection**: Forces epoch-level heartbeat logs for training runs.
- **Breakpoint Report Protocol**: Structured emoji-tagged status updates (`💥 REJECT`, `⚠️ TEST FAIL`, `✅ SUCCESS`) to user.
- **Dead-Loop Termination**: `MAX_ITER` (default 10) prevents infinite rejection loops.
