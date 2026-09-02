# BOUNDARY — `nsmor/` (Frozen Core)

## Status: 🔒 FROZEN

This directory contains the **mathematical and architectural core** of NSMoR. All modules here are mathematically verified and stable.

**Modifications require explicit user override.** Do not modify without direct instruction.

---

## Input/Output Contract

### `NSMoRCore` (model_nsmor_core.py)

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

### `NSMoRDataset` (nsmor_dataloader.py)

**Item Contract (unchanged):**

```
__len__()        -> int                     — number of sequences
__getitem__(i)   -> (X_seq, Y_seq)          — Tensors; see feature layout above
.sequences       -> list[(X_seq, Y_seq, label)]
```

**Split Provenance (added 2026-09-02 under user override):**

```
__init__(..., source_indices: Sequence[int] | None = None)
.source_indices  -> list[int]   — row index each sequence occupied in the
                                  UNSPLIT dataset artifact; aligned 1:1 with
                                  .sequences.  Defaults to range(n) when the
                                  caller did no subsetting.
```

`source_indices` is a read-only sidecar. The `.sequences` tuple layout,
`__getitem__`, and `_fill_priors` are deliberately untouched, so no training
behaviour depends on it. Its purpose is auditability: a caller that hands
this dataset a train/val subset records which original rows went where, so
the split can be verified without reverse-engineering it from tensor
contents. A length mismatch raises `ValueError`.

---

### `data_extractor.py` — Snapshot anchor (added 2026-09-02 under user override)

```
resolve_snapshot_anchor(trial_data, stimulus_onset_ms)
    -> (anchor_ms: float, anchor_rule: str)

anchor_rule is one of:
  "stimulus_onset"     — any trial carrying wind.  UNCHANGED behaviour:
                         anchor = stimulus_onset_ms.
  "looming_collision"  — visual-only trials.  anchor = time of the
                         visual-angle peak, which locates the collision.

extract_mcmc_snapshot(...)    — offsets from the resolved anchor, not from
                                stimulus_onset_ms directly.  snapshot_dim
                                stays 5; feature layout unchanged.

build_snapshot_dataset(..., return_anchor_rules=False,
                       on_unanchorable="raise")
    -> (snapshots, labels[, kept_indices][, anchor_rules])
```

Why this required touching a frozen module: the anchor was
`stimulus_onset_ms − 50 ms` for every condition, but visual-only trials begin
looming at the `TrialStart -> Looming` transition — the same instant as
`trial_start` — so their anchor fell before the first frame and
`extract_mcmc_snapshot` raised. `build_snapshot_dataset` swallowed that with a
bare `except ValueError: continue`, so **every visual-only trial in the corpus
was deleted in silence**: 36 of 396 on the reference data, all `NO_RESPONSE`,
none surviving. Because the retention identity carries a snapshot drop
forward, those trials also disappeared from the regression sequence set.

The edit is deliberately confined. On the reference corpus 360 trials resolve
via `"stimulus_onset"` and are bit-identical to before; only the 36
visual-only trials take the new rule. `on_unanchorable` now defaults to
`"raise"`, so a caller that tolerates drops must opt in and report what it
lost. `PIPELINE_SEMANTICS_VERSION` moved to `2.2` because the retained trial
population changed.

---

## Sub-modules

| Module           | Class       | I/O                                |
| ---------------- | ----------- | ---------------------------------- |
| `SensoryEncoder` | `nn.Module` | `[B, T, 4]` → `[B, T, H]`          |
| `LIFCell`        | `nn.Module` | `[B, H]` → `[B, H]` (step-by-step) |
| `GRUUnit`        | `nn.Module` | `[B, T, H]` → `[B, T, H]` (packed) |
| `MoRRouter`      | `nn.Module` | `[B, H+M]` → `[B, 2]` (softmax)    |
| `DirectionHead`  | `nn.Module` | `[B, T, H]` → `[B, T]`             |

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
2. Explain why it cannot be done in `nsmor/analysis/` or `scripts/`.
3. Wait for explicit user approval before proceeding.
