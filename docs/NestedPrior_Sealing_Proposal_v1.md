# Nested-Prior Seam Sealing — Proposal Report (v1)

**Scope:** `nsmor/pipeline/nested_prior.py`, `scripts/evaluate_nested_prior.py`, `scripts/train.py`, `tests/test_nested_train_integration.py`, `tests/test_nested_prior_tool.py`
**Goal:** Make the `--nested_prior_artifact` path airtight against provenance misrepresentation, silent incoherence, and metric/ckpt lineage gaps — then hand to Reviewer #2.

---

## 1. Motivation (动机)

The nested-prior seam (load persisted outer split + OOF priors verbatim) was already fail-closed at the *fingerprint / partition / recording-prefix-disjointness* layer. Historical `train_animals`/`val_animals` and `n_train_animals`/`n_val_animals` denote recording prefixes, not verified animals; animal-level independence and cross-recording leakage control remain unverified. Four gaps remained, each one able to make an audit trail lie or vanish:

| # | Gap | Why it matters |
|---|-----|----------------|
| A | **`source_dataset_path` optional at generation.** `compute_source_fingerprint(...) if path else ""` emitted artifacts with `source_fingerprint=""`, which the loader *rejects by construction* (64-char SHA-256 required). The generator could write an unloadable artifact — a landmine, not a product. | Provenance is the artifact's reason to exist. An artifact that cannot bind to a dataset cannot be audited at all. |
| B | **Split metadata unverified on load.** `split_seed` / `val_split` were read and echoed, never recomputed. A sidecar claiming `split_seed=42, val_split=0.2` next to a partition from another seed would be consumed as truth — every downstream "seed=…" claim would be a lie. | Split reproducibility is the load-bearing assumption of the whole nested protocol. |
| C | **Honest numbers vs overclaiming.** Grouped split *cannot* honor an exact trial fraction (granularity = recording prefixes). Recording `val_split=0.34` and letting consumers report it as the realized hold-out fraction misstates the experiment in the paper's Table 1. | Statistical honesty: report realized trial fraction from persisted indices, keep the requested prefix-grouped split clearly labeled. |
| D | **No lineage on checkpoints / metrics.json / train() return.** Nested runs and legacy runs looked identical downstream. A reviewer could not tell whether a given `best_model.pth` was trained on the nested artifact or the legacy global-OOF path. | Auditability across the whole run chain, not just the artifact. |
| E | **Class-coverage gate leaked val labels (indirect).** Coverage checked `np.unique(labels)` (incl. val) instead of the canonical class vocabulary. Whether the gate passed depended on held-out labels. | Val-label disclosure in the *generator*, which the immunity tests claim to exclude. |
| F | **NaN/Inf targets poison everything silently.** One bad `Y_seqs` entry → non-finite mean/std → every standardized loss/metric is NaN for the whole run, with no warning. | Fail-closed on corrupted kinematics, at the earliest boundary. |
| G | **Dual fingerprint helpers.** Generator had its own `compute_source_fingerprint` (Path-only, silent `""` on missing) alongside the loader's (str/Path, raises). One contract, two behaviors. | Contract drift is how the empty-fingerprint path got in. |

---

## 2. Implementation (实现)

### 2.1 Single canonical fingerprint + mandatory source (`evaluate_nested_prior.py`, `nested_prior.py`)
- Deleted the local `compute_source_fingerprint`; the script now **imports** the loader's implementation (`nsmor.pipeline.nested_prior`) — one contract, one function. `str` and `Path` both accepted, raises on `None`/missing.
- `generate_nested_priors` computes the fingerprint at **step 0** (before any MCMC work) and refuses empty/non-64-char digests. Signature note: `source_dataset_path` is *functionally required* (the dict it returns cannot otherwise be loadable).
- Docstring updated to say so explicitly.

### 2.2 Split-metadata honesty on load (`nested_prior.load_nested_prior_split`)
After the existing fingerprint/pivot/partition/recording-prefix checks, and **using only the artifact's own fields + dataset grouping** (no external labels):
1. `split_seed` must be a real `int` (bools/refused), `val_split` a real float in `(0,1)`.
2. Recompute `grouped_train_val_split(session_ids, n_total, val_split, random_seed=split_seed)` and require the recomputed partition to be **exactly** the persisted one (sorted comparison). Mismatch → `ValueError`, message names the claimed vs recomputed sizes.
3. Recompute realized `val_trial_fraction`, `n_train`, `n_val`, `n_train_animals`, `n_val_animals` **from the persisted indices** (legacy field names counting recording prefixes, not verified animals); if the sidecar *claims* `val_trial_fraction` / `n_train` / `n_val`, require exact agreement.
4. Return a 4th element `info: Dict[str, Any]` carrying: `nested_prior_artifact`, `nested_prior_fingerprint`, `mcmc_prior_provenance`, `is_nested_cv`, `split_seed`, `val_split`, `val_trial_fraction`, `n_train`, `n_val`, `n_train_animals`, `n_val_animals`.
5. Log line now distinguishes *requested* `val_split` (recording-prefix-grouped target) from *realized* trial fraction — never echoes a target as if realized.

### 2.3 Split metadata written honestly at generation (`evaluate_nested_prior.py`)
- `result` now records, alongside the requested fields: `val_trial_fraction`, `n_train`, `n_val`, `n_train_animals`, `n_val_animals`.
- Class-coverage gate switched to the **canonical vocabulary** `range(fc.num_classes)`: validation labels are no longer consulted anywhere in the generator (regression-tested by the existing `test_val_label_leakage_immunity`, which now must still hold).

### 2.4 Checkpoint / metrics / return provenance (`scripts/train.py`)
- `_PROVENANCE_KEYS` extended with `nested_prior_artifact`, `nested_prior_fingerprint`, `nested_split_seed`, `nested_val_split`, `is_nested_cv`, `mcmc_prior_provenance` — atomic save path already patches these into every checkpoint.
- `build_dataloaders` stashes `nested_info` on the train/val datasets; `train()` reads it back (guard: artifact set but no info → `RuntimeError`, fail closed).
- All three `_atomic_save_checkpoint` calls (periodic, best, final) and the `metrics.json` write and the `train()` return dict now carry `nested_provenance`. Historical legacy runs recorded `is_nested_cv=False`, `mcmc_prior_provenance="global_oof_animal_grouped_cv"`; this legacy string denotes prefix grouping only, not verified animal identity or zero cross-recording leakage.
- Real lineage is used everywhere; the previous hardcoded `"nested CV artifact"` log string is gone.

### 2.5 Fail-closed non-finite targets (`scripts/train.py`)
- New shared `assert_finite_targets(Y_seqs)` called from `build_dataloaders` and `compute_target_stats`. Message states the failure mode (silent NaN poisoning of normalization/loss/metrics) and the fix (regenerate dataset after sanitizing kinematics).

### 2.6 Call-site / test coherence (4-tuple unpack, honest fixtures)
- Both `load_nested_prior_split` call sites in `train.py` unpack the new `info` tuple.
- `tests/test_nested_train_integration.py::_make_nested_artifact` default split now comes from the **real** `grouped_train_val_split` with the same `(split_seed, val_split)` recorded in the sidecar — honest artifacts pass the new recomputation check. Tests that hand-craft a partition are deliberately crafting *dishonest* metadata and are expected to be refused (no behavior change to the refusal tests themselves).
- `tests/test_nested_prior_tool.py`: every `generate_nested_priors` call now passes a real `source_dataset_path` via new `_write_ds_file` helper, matching the now-mandatory provenance contract.

---

## 3. Predicted evidence (预判依据)

- **A:** Generator can no longer write an artifact the loader rejects. Fingerprint is computed and length-checked before any training work; the unit test for empty-fingerprint *refusal on load* remains as the inverse guard.
- **B:** A sidecar whose `split_seed`/`val_split` don't reproduce the persisted partition raises before any consumer sees the indices. I can flip `val_split` in a sidecar and the loader refuses — the recomputation is the ground truth, not the echo.
- **C:** Realized `val_trial_fraction` is computed from persisted indices and cross-checked against any claimed value; the loader returns it so downstream code reports what the grouped split actually did. The requested `val_split` is labeled as a recording-prefix-grouped target in the log line.
- **D:** `best_model.pth`, `final_model.pth`, `epoch_*.pth`, `metrics.json`, and the `train()` return all carry the same provenance keys. The historical 20-epoch smoke run recorded `is_nested_cv=False, mcmc_prior_provenance='global_oof_animal_grouped_cv'`; nested runs populate their own lineage keys. Neither path verified animal independence. `metrics.json` exposes the same fields so a reader never has to unpickle a checkpoint to learn the split lineage.
- **E:** Coverage gate uses `range(fc.num_classes)` only. Perturbing val labels cannot change whether the gate passes or fails, which is exactly the property `test_val_label_leakage_immunity` locks in.
- **F:** Any NaN/Inf in any `Y_seqs[i]` raises `ValueError` at loader-construction and stats-computation time, naming the offending index and count. No silent NaN propagation into `target_mean`/`target_std`.
- **G:** Single import of `compute_source_fingerprint`; `tests/test_nested_train_integration.py` already imports it from the same module as the loader, so the two cannot drift again.

**Convergence:** `python scripts/train.py --config config/default.yaml --epochs 20 --output_dir runs/test` → train_loss 8.13 → 6.87 (monotone after warmup unfreeze), val_loss 8.37 → 8.20 (`best_val_loss=8.197`), `final_train_loss=6.868`. No NaN/skip storms; membrane stats stable (`spike_rate≈0.021`). Full test suite: **403 passed, 1 skipped** (372 legacy + 31 nested/integration).

---

## 4. What I did *not* change (and why)

- `split_seed` remains in the required-keys tuple at load (behavior preserved), but is now *verified* rather than merely *present* — stricter, still fail-closed, no fallback path added.
- No new dependencies, no new modules. One function extracted into `nsmor/pipeline/nested_prior.py` (was 2 competing copies).
