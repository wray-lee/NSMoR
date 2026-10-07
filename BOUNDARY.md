# Project Boundaries & Constraints — NSMoR

## 1. Git Identity & Commit Boundary
- **Git Author & Committer**:
  - `user.name`: `wray-lee`
  - `user.email`: `i@wray7.top` (Must match verified GitHub primary address to guarantee avatar and contribution graph association)
- **Constraint**:
  - Do NOT commit using unverified emails or secondary aliases.
  - Verify `git config user.email` returns `i@wray7.top` before creating any commits.

## 2. Architectural Boundaries & Permissions
- **Controlled Mathematical Core (`nsmor/model_nsmor_core.py`, `nsmor/loss.py`)**:
  - Core changes are allowed for scoped optimization and correctness fixes.
  - Every core change must use the normative protocol in `nsmor/BOUNDARY.md`: user-authorized scope, two independent `ACCEPT` reviews, focused and full regression tests, numerical checks, and backward-compatibility evidence.
  - This is not a blanket permission to alter protected data, canonical loaders, formal artifacts, or experiment protocols.
- **Pipeline Layer (`nsmor/pipeline/BOUNDARY.md`)**:
  - Data ingestion, feature extraction, and batch collation. Safe to extend.
- **Analysis Sandbox (`nsmor/analysis/BOUNDARY.md`)**:
  - Dynamical systems analysis, fixed-point discovery, Jacobian computation, and manifold visualization. Free to create and modify.

## 3. Mandatory Engineering Standards
- **Strict Typing & Shape Assertions**: Every `forward()` method and data transformation must contain explicit shape assertions.
- **Backward Compatibility**: Sub-modules must maintain decoupled interfaces without breaking existing entry points (`scripts/train.py`, `pytest tests/`).

## 4. Controlled Core-Change Protocol
The normative protocol is [nsmor/BOUNDARY.md](nsmor/BOUNDARY.md#controlled-core-change-protocol).
It requires a user-authorized task scope, two independent `ACCEPT` reviews,
focused tests, complete regression (`pytest tests/`), numerical checks, and
compatibility evidence. Existing user authorization covers necessary core
changes within that scope; do not request it again for each file or fix.
The protected evidence and infrastructure listed there remain protected.
