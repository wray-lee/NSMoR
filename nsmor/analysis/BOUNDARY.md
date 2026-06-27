# BOUNDARY — `nsmor/analysis/` (Dynamical Systems Sandbox)

## Status: 🔓 FREE TO MODIFY

This directory is the **sandbox for dynamical systems analysis tools**. Safe to create, modify, and experiment freely.

---

## Purpose

Provide tools for analyzing the internal dynamics of the NSMoR GRU pathway:

- **Fixed-point analysis** — find equilibrium states of the recurrent dynamics
- **Jacobian computation** — characterize stability of fixed points
- **Trajectory extraction** — collect un-padded hidden states from datasets
- **Manifold analysis** — visualize low-dimensional structure of neural states

---

## Current Tools

### `FixedPointAdapter` (dynamics.py)

**State Extraction:**

```python
adapter = FixedPointAdapter(model, device=device)

# Extract un-padded GRU trajectories
trajectories = adapter.extract_gru_states(dataloader)
# Returns: List[Tensor] — each tensor has shape (T_i, H)
```

**Jacobian Computation:**

```python
# Single state Jacobian
h_t = torch.randn(H, requires_grad=True)
x_t = torch.randn(H)  # sensory encoding output
J = adapter.compute_jacobian_at_state(h_t, x_t)  # (H, H)

# Batch Jacobian
J_batch = adapter.compute_jacobian_batch(h_states, x_inputs)  # (N, H, H)
```

**Eigenvalue Analysis:**

```python
eigenvalues = torch.linalg.eigvals(J)
# |eigenvalue| < 1 → stable fixed point
# |eigenvalue| > 1 → unstable fixed point
```

---

## Input/Output Contract

### `extract_gru_states`

```
Input:  dataloader  — yields (X_batch [B,T,8], Y_batch [B,T], lengths [B])
Output: trajectories  — List[Tensor(T_i, H)]  (un-padded GRU hidden states)
```

### `compute_jacobian_at_state`

```
Input:  h_t  [H]  — hidden state (requires_grad=True)
        x_t  [D]  — sensory encoding input
Output: J    [H, H]  — Jacobian ∂h_{t+1}/∂h_t
```

### `compute_jacobian_batch`

```
Input:  h_states  [N, H]  — batch of hidden states
        x_inputs  [N, D]  — corresponding inputs
Output: jacobians [N, H, H]  — Jacobian for each state
```

---

## Modification Rules

1. **DO** create new analysis tools freely.
2. **DO** experiment with different fixed-point finders.
3. **DO** add visualization utilities.
4. **DO NOT** modify `nsmor/model_nsmor_core.py` from here — import it instead.
5. **ALWAYS** maintain shape assertions in new functions.

---

## Extension Ideas

- Add `FixedPointFinder` using optimization (Sussillo & Barak 2013)
- Add `ManifoldVisualizer` for PCA/t-SNE projections
- Add `StabilityClassifier` for fixed-point characterization
- Add `BifurcationAnalyzer` for parameter sweeps
- Add `LinearResponseAnalyzer` for perturbation analysis

---

## Import Pattern

Always import from frozen core — never copy:

```python
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.loss import BioJointLoss
from nsmor.checkpoint import load_checkpoint
```
