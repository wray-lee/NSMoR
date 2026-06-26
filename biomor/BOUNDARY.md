# BOUNDARY — `biomor/` (Frozen Core)

## Status: 🔒 FROZEN

This directory contains the **mathematical and architectural core** of BioMoR. All modules here are mathematically verified and stable.

**Modifications require explicit user override.** Do not modify without direct instruction.

---

## Input/Output Contract

### `BioMoRCore` (model_biomor_core.py)

**Forward Pass:**
```
Input:  X_batch  [B, T, 8]    — padded feature tensor
        lengths  [B]          — true (unpadded) sequence lengths

Output: Y_pred   [B, T]      — predicted output

Internals (when return_internals=True):
        routing_gates   [B, T, 2]  — [g_lif, g_gru] per timestep
        lif_potentials  [B, T, H]  — membrane potentials
        lif_spikes      [B, T, H]  — spike events
        gru_hidden      [B, T, H]  — GRU hidden states
```

**Feature Layout (dim=8):**
```
[0] v_vis(t)        — visual angle (degrees)
[1] wind(t)         — wind state (0/1)
[2] v_kine(t-1)     — previous velocity (cm/s)
[3] a_kine(t-1)     — previous acceleration (cm/s²)
[4] P_startle       — MCMC prior
[5] P_walk          — MCMC prior
[6] P_pre_active    — MCMC prior
[7] P_no_response   — MCMC prior
```

### `BioJointLoss` (loss.py)

**Forward Pass:**
```
Input:  y_pred     [B, T]      — model predictions
        y_true     [B, T]      — ground truth targets
        lengths    [B]         — true sequence lengths
        g_gru      [B, T, 1]   — GRU routing gate
        lambda_reg float       — regularization weight

Output: loss       scalar      — joint loss value
```

### `save_checkpoint` / `load_checkpoint` (checkpoint.py)

**Checkpoint Dictionary:**
```
{
    "model_state_dict":      OrderedDict,
    "optimizer_state_dict":  OrderedDict,
    "scheduler_state_dict":  OrderedDict (optional),
    "epoch":                 int,
    "loss":                  float,
    "rng_state":             Tensor,
    "cuda_rng_state":        list[Tensor] (optional),
    "config":                dict,
}
```

---

## Sub-modules

| Module | Class | I/O |
|--------|-------|-----|
| `SensoryEncoder` | `nn.Module` | `[B, T, 4]` → `[B, T, H]` |
| `LIFCell` | `nn.Module` | `[B, H]` → `[B, H]` (step-by-step) |
| `GRUUnit` | `nn.Module` | `[B, T, H]` → `[B, T, H]` (packed) |
| `MoRRouter` | `nn.Module` | `[B, H+M]` → `[B, 2]` (softmax) |
| `DirectionHead` | `nn.Module` | `[B, T, H]` → `[B, T]` |

---

## Modification Rules

1. **DO NOT** add new sub-modules without user approval.
2. **DO NOT** change tensor shapes or the feature layout.
3. **DO NOT** remove shape assertions in `forward()` methods.
4. **DO NOT** alter the checkpoint dictionary structure.
5. **ALWAYS** maintain backward compatibility with existing imports.

---

## Override Protocol

To modify frozen core files:
1. State the specific change needed.
2. Explain why it cannot be done in `biomor/analysis/` or `scripts/`.
3. Wait for explicit user approval before proceeding.
