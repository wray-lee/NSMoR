# Routing-aux uses whole-trial mean and a calibrated hinge

The MoR router should send pure-wind trials toward LIF and visual-present
trials toward GRU. The auxiliary hinge
``max(0, margin − (mean(g_wind) − mean(g_visual)))`` is the instrument.

Calibration on ``nsmor_dataset_full_backup.pt`` (48 wind / 48 other,
``runs/subset_retrain/best_model.pth``, max_seq_len=2400):

- whole-trial masked mean: wind 0.485±0.032, visual 0.439±0.012,
  separation 0.046, Cohen's d=1.87, pooled std 0.024
- peri-stimulus top-k (window 250, k=25): both groups saturate ~0.514,
  separation ≈ 0

So the loss stays on the whole-trial mean. The hinge default is 0.024
(1.0× pooled std), not the previous hardcoded 0.2, which sat above the
observed 5–95 frame-level range (~0.185) and could never saturate.

``nsmor_dataset_3cond_v2.pt`` / ``nsmor_subset_small.pt`` have zero
wind-only trials and cannot calibrate or train this term. Use the backup
corpus (or a subset carved from it) until ETL regenerates v2.2 with a
live wind channel.

## Considered Options

- **Peri-stimulus top-k** (rejected): saturates both conditions; no
  between-group signal.
- **margin=0.2** (rejected): larger than the empirical trial-mean
  separation and larger than the frame-level dynamic range.
- **Whole-trial mean + margin=0.024** (chosen): the only aggregation that
  currently separates, with a hinge the model can actually close.

## Consequences

``lambda_routing_aux`` remains 0 by default. Enabling it on a corpus
without ``is_pure_wind`` True rows is a no-op. Regenerating 3cond without
``pre_load_adapt`` still drops wind-only; the loader now hard-fails on
raw ``sys_time/stim_state`` schema instead of writing a silent
no_stimulus corpus.
