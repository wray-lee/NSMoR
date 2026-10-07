# BOUNDARY — `nsmor/` (Controlled Core)

## Status: 🔐 CONTROLLED

This directory contains the mathematical and architectural core of NSMoR. Core
modules may evolve for scoped optimization or a correctness fix, subject to the
controlled-change protocol below.

Unscoped rewrites, contract-breaking edits, and changes that bypass review or
testing are not permitted.

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

### `data_extractor.py` — Snapshot anchor (reconciled 2026-09-28 under user override)

```
resolve_snapshot_anchor(trial_data, stimulus_onset_ms)
    -> (anchor_ms: float, anchor_rule: str)

anchor_rule is one of:
  "stimulus_onset"     — any trial carrying wind (and no-stimulus fallback).
                         anchor = stimulus_onset_ms, unchanged.
  "looming_collision"  — visual-only trials. Physical collision in trial ms:
                         looming_onset_ms + lv_ratio_ms / tan(radians(init_deg/2)).

extract_mcmc_snapshot(...)    — samples resolved anchor + ttc_offset_ms
                                (default -50 ms). snapshot_dim stays 5:
                                visual_angle, looming_velocity, wind_state,
                                avg_velocity_bg, max_acceleration_bg.

build_snapshot_dataset(..., return_anchor_rules=False,
                       on_unanchorable="raise")
    -> (snapshots, labels[, kept_indices][, anchor_rules])
```

Geometry comes from each trial, with no visual-angle peak selection:

- Canonical `trial_start` event details supply `lv_ratio_ms` and, when declared,
  `init_deg`. Legacy canonical fixtures without declared l/v use their constant
  positive `l_v_ratio` trace. A visual trial must have a positive l/v trace;
  declared l/v must agree with its nonzero values.
- The production raw schema stores `lv_ratio_ms` in `trial_start.details` and
  does not declare the initial angle. The current conversion layer reconstructs
  that angle with `init_deg=2.0`; its converted trace retains the initial geometry.
  Each positive, unclipped angle gives the onset-to-collision duration as
  `max(0, sample_time - looming_onset) + lv_ratio_ms / tan(angle/2)`.
  At least three such samples are required. Without a declaration, use their
  median duration; with `init_deg`, use the declared duration. In both cases a
  strict majority must agree within `max(1 ms, 0.1% of that duration)` or the
  visual trial fails. This tolerates one early 35-degree artifact while rejecting
  a coherent 4-degree trace against a 2-degree declaration. Pre-onset samples
  hold the initial angle; delayed or negative looming onsets use the same clock.
- Looming onset is the earliest `phase_transition` to `Looming` or explicit
  `looming_onset` event, on the shared trial clock. `stimulus_onset_ms` is the
  canonical fallback. Looming onset need not equal `trial_start`.
- l/v must be finite and positive, the initial angle finite and strictly between
  0 and 180 degrees, and collision finite and after onset. Invalid declared
  metadata (including null/boolean values), nonfinite geometry, and an
  unverifiable clipped visual trace raise `ValueError`. The collision itself
  and its visual snapshot must fall inside the recorded time range; nearest-
  frame sampling cannot substitute the last frame for an unrecorded instant.

The original onset-minus-50-ms rule could precede frame one for visual-only
trials and silently remove them. `on_unanchorable="raise"` remains the default;
explicit `"skip"` retains the existing kept-index contract for reporting drops.
The anchor rule remains a sidecar; sequence and feature layouts are unchanged.

**Reconciliation with the approved spec:** the rejected SOURCE4 visual-angle
peak proxy is superseded by the physical geometry above. The approved preserved
wind decision still applies; wind geometry is not parsed and its five snapshot
features retain the onset rule. Earlier reference counts (36 visual-only and
360 wind-bearing trials) are historical audit evidence, not a new corpus run.
`PIPELINE_SEMANTICS_VERSION` remains `2.2`, the already approved 2.1-to-2.2
retention change. This correction belongs to the unapproved v6 candidate and
requires a fresh source seal and independent source review. `CONTEXT.md`'s
"looming begins at trial start" describes the original reference condition;
the clock contract above also covers delayed and negative looming onsets. Core
model and loss changes follow this controlled boundary; this extractor
correction does not grant permission to alter unrelated modules.

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

## Controlled Core-Change Protocol

This section is the normative policy; other governance files summarize it.
On 2026-10-04 the user authorized replacing the permanent core freeze with
this policy for continued core optimization. User task authorization persists
across turns and covers necessary core fixes within its stated scope.

1. Declare the core files, behavior, tensor contracts, compatibility
   requirements, and experiment scope before editing. Routine implementation
   details within a user-authorized task do not require repeated approval.
2. Use authorization from the user or an existing task scope they authorized.
   Agent-to-agent agreement does not grant user authorization. A new scope must
   respect any explicit restrictions on loss, loaders, or experiment design.
3. Obtain two independent source reviews of the actual diff before integration
   testing. Both reviewers must report `ACCEPT` independently. Focused developer
   tests and reviewer reproductions may run while preparing that review.
4. Preserve shape assertions, units, causal inputs, gradients/state behavior,
   imports, and checkpoint compatibility. Run focused tests and the complete
   regression suite (`pytest tests/`) with restricted artifact loading, plus
   applicable numerical/integration checks. Do not silently exclude failures
   or unsafe fixtures; repair them and report any unavailable gate.
5. Preserve protected evidence and infrastructure unless the user separately
   authorizes a change: canonical loaders and data contracts; formal DATA/NEST
   artifacts, dataset/prior/split pins, existing checkpoints and receipts;
   finalized baseline outputs; and paths explicitly protected by the user.
   Never read, traverse, hash, modify, delete, or stage the protected paths
   `nested-acceptance-4843c3c3935a4020a246f21d2d93383d.json`,
   `nested-acceptance-4843c3c3935a4020a246f21d2d93383.json`, or `tmp0f3g8lkx/`.

The active [optimization protocol](../docs/model-optimization-protocol-20261004.md)
continues to govern the declared candidate set, data/split pins, unchanged loss
and loaders, scoring family, and release gates. This policy does not amend that
experiment retrospectively. Core changes within authorized scope are permitted;
changes outside that scope or bypassing review/testing violate this boundary.
