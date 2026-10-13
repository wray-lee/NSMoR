# Evidence support proposal — 2026-10-13

**Motivation.** Publication figures need visible sample support and eligibility denominators. Behavior remains **EXPLORATORY / IN-SAMPLE**; `nested_outer_validation` supports validation-guided model selection, with no untouched test. Condition-pure recording prefixes have unverified animal identity and cannot establish condition causality.

**Implementation.** Add the standalone `scripts/figure_evidence_support.py`, its focused tests, and this proposal. Capture each small source's bytes once and decode the prior using existing `load_artifact_bytes(..., map_location="cpu")`. Validate counts/keys/condition mapping, complete unique bounded disjoint indices, prefix separation and declarations, unverified identity, existing 250 ms / 10 and 5 cm/s gates, and aggregate eligibility/responder denominators and subsets. Refuse existing output paths and fail on missing/conflicting data.

Export `recording_prefix_support.csv` (prefix, condition, split, trial count), `condition_summary.csv`, numerical `evidence_support.json`, and vector PDF/SVG plus PNG `figure_evidence_support`. Panel A shows exact trial/prefix split support, including missing wind-only validation. Panel B shows eligible/excluded and eligible responder/nonresponder counts. Panel C compares existing gates with counts/denominators. Figures identify their source artifacts and scope; stored labels are post-stimulus categories, **not prestimulus states**. No threshold optimization, trial-independent binomial intervals, unsupported condition/TTC results, or dense raw matrix.

**Rationale.** Reuse installed NumPy/Matplotlib/PyTorch and the restricted decoder. Preserve sources, model behavior, and other developers' files. Derive support from indices without assuming equal prefix sizes:

| Condition | Train trials | Validation trials | Train prefixes | Validation prefixes |
| --- | ---: | ---: | ---: | ---: |
| visual_only | 540 | 180 | 15 | 5 |
| wind_only | 396 | **0 (missing)** | 11 | **0 (missing)** |
| multisensory | 936 | 252 | 26 | 7 |

Totals: 2,304 trials, 64 prefixes; train 1,872/52, validation 432/12. The 128-session count is explicitly reported from the summary.

**Provenance limit.** The behavior JSON records no dataset/prior SHA binding. Structural reconciliation does **not** byte-verify a common corpus. Output records `behavior_dataset_binding="not_recorded_in_source_summary"`, SHA256 of both consumed byte payloads and the script, and the prior-declared source fingerprint. Corpus bytes are neither loaded nor rehashed. Aggregate gate subset inequalities cannot verify individual trial membership without masks.

**Validation.** 47 focused tests passed, including unequal prefix sizes, invalid splits/mappings/counts, real restricted Torch loading, vector export, source preservation, and collision refusal. Real-source smoke exported all six files; final PNG visually inspected. Outputs: `/tmp/nsmor-evidence-support-20261013-ch0pyipu/support-readable`. Independent reviews remain with the parent. No GPU, training, full corpus/full regression, or commit.

```powershell
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= && cd /mnt/d/Projects/NSMoR-publication-analysis && pytest -p no:cacheprovider tests/test_figure_evidence_support.py -q"
```

Real-source CLI command (inside the same atomic WSL environment):

```text
python scripts/figure_evidence_support.py --behavior-summary /mnt/d/Projects/NSMoR/results/behavior-20261010/behavior/behavior_summary.json --nested-prior /mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/nested-causal-complete-dku91_m0/nested_split_seed42.pt --output-dir <new-directory>
```
