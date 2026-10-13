# Publication robustness supplement — 2026-10-13

## Motivation

Add recording-prefix point-estimate sensitivity and seed optimisation variability to
`realdata_phase2_family.json`, without changing the frozen PHASE-2 scorer, protocol,
trial-level mean/d_z, pooled skill, or adjusted p-values. Recording prefixes do not
establish animal identity or biological independence. Validation-guided selection
makes this supplement exploratory; it supplies no new testing or family correction.

## Implementation

Only four new paths, based on `cb88824` in the isolated publication-analysis worktree:

- `nsmor/analysis/publication_robustness.py`: validates the exact 48-slot family,
  recorded dataset/prior/arm input lineage, three seeds 42/43/44 per primary scope,
  globally unique checkpoint paths and SHA256s, arm input transformations, and
  retained-vector shapes/finite nonnegative-or-null observations and stored-mean
  agreement, and frozen availability/p-value relationships (without recalculating Holm). Writes summaries, provenance, and publication figures.
- `scripts/analyze_publication_robustness.py`: the sole caller; CPU1 CLI taking
  `--input` and `--output-dir`. No changes to exports, fixtures, configuration,
  checkpoints, datasets, or the existing scorer.
- `tests/test_publication_robustness.py`: small inline JSON fixture and focused
  numerical, integrity, missing-support, overwrite, and CLI/artifact tests.
- `docs/proposals/publication-robustness-20261013.md`: this proposal.

Outputs: `publication_robustness.json`, `provenance.json`, `family_summary.csv`,
`prefix_detail.csv`, `skill_summary.csv`, `seed_optimisation_variability.csv`, and
`primary_skill`, `prefix_robustness`, `horizon_skill` in PDF, SVG, and PNG.
All MSE/delta axes and JSON/CSV summaries declare `(cm/s)^2`; skill is dimensionless.
The primary figure has a separate seed row using stored `per_seed_trial_mean`
(unweighted per-trial MSE means), not pooled MSE or biological n or twelve replicates.
Unavailable tests retain their cells and reasons; supplied descriptive prefix means
remain usable regardless of p-value availability. Missing secondary metrics stay NA.
The output directory must be new; file creation is exclusive and figure errors propagate.

## Rationale

Equal-prefix weighting is a separate estimand from the frozen trial-level mean and
frame-weighted pooled skill. Each deletion estimate averages the remaining supplied
prefix means with the fitted model held fixed: this is deletion sensitivity, not
cross validation or retraining. At least two prefixes are needed for deletion;
missing/nonfinite support is never silently replaced or filtered. Zero variance is
identified explicitly; no standardized prefix effect or degenerate CI is supplied.
The optional bootstrap is omitted to keep this minimal and avoid conditional
independence claims. Prefix wins mean negative candidate-minus-comparator MSE.
Frozen adjusted p-values and unavailable reasons are copied, never recomputed.

Lineage validation checks what the frozen JSON actually records: dataset/prior paths
from its scorer command and ordered checkpoint path/hash triples. It does not open
checkpoints or data, rehash them, assert corpus identity, or invent dataset/prior hashes absent from that JSON.
The parent owns evidence assessment, condition analysis, and behavior figures.
No training, GPU use, large dataset load, commit, or push is part of this work.

## Validation

```powershell
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 && cd /mnt/d/Projects/NSMoR-publication-analysis && python -m pytest -p no:cacheprovider tests/test_publication_robustness.py tests/test_model_comparison.py tests/test_uq_holm_validation.py -q"
```

Example invocation inside that same CPU1 WSL environment:
`python scripts/analyze_publication_robustness.py --input /path/to/realdata_phase2_family.json --output-dir /path/to/new-supplement`.
