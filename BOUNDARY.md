# Project Boundaries & Constraints — NSMoR

## 1. Git Identity & Commit Boundary
- **Git Author & Committer**:
  - `user.name`: `wray-lee`
  - `user.email`: `i@wray7.top` (Must match verified GitHub primary address to guarantee avatar and contribution graph association)
- **Constraint**:
  - Do NOT commit using unverified emails or secondary aliases.
  - Verify `git config user.email` returns `i@wray7.top` before creating any commits.

## 2. Architectural Boundaries & Permissions
- **Frozen Mathematical Core (`nsmor/BOUNDARY.md`)**:
  - `nsmor/model_nsmor_core.py` and `nsmor/loss.py` are strictly frozen.
  - Modifying the frozen core requires explicit user override.
- **Pipeline Layer (`nsmor/pipeline/BOUNDARY.md`)**:
  - Data ingestion, feature extraction, and batch collation. Safe to extend.
- **Analysis Sandbox (`nsmor/analysis/BOUNDARY.md`)**:
  - Dynamical systems analysis, fixed-point discovery, Jacobian computation, and manifold visualization. Free to create and modify.

## 3. Mandatory Engineering Standards
- **Strict Typing & Shape Assertions**: Every `forward()` method and data transformation must contain explicit shape assertions.
- **Backward Compatibility**: Sub-modules must maintain decoupled interfaces without breaking existing entry points (`scripts/train.py`, `pytest tests/`).
