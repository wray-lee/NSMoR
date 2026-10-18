# Stochastic time-consuming depth (CAMoR-L2), default stays off

ADR 0009 made processing depth consume model-grid time, but depth itself was a
deterministic function of the frame (a halt threshold / accumulator). A
behavioral latency distribution is variable, and the phase-3 question is
whether an adaptive processing duration can reproduce condition-relative
latency differences. A deterministic depth can only ever emit one latency per
frame, so it cannot represent that distribution.

Decision: add two opt-in `time_consuming_mode` values under the CAMoR-L
family (ADR 0010):

- `"stochastic"` (**CAMoR-L-S**): depth is SAMPLED from the halt distribution;
  the emitted value is the hard iterate `u_N`; gradients use a hard-forward /
  expectation-gradient (STE) estimator.
- `"hard_depth"` (**CAMoR-L-H**): a deterministic matched control that uses the
  SAME pmf and the SAME hard `u_N` readout, with
  `N = clamp(floor(E[N] + 0.5), 1, K)`.

`"off"` remains the default and constructs no module, so default numerics,
`state_dict` keys and construction RNG are byte-unchanged (verified against the
fixed pre-change ref `5eadb226`). The two new modes are script/phase-3 opt-ins;
no existing ADR 0009 mode changes.

## Why a matched deterministic control

The primary L2 A/B is **stochastic vs hard_depth**: both use the first-success
pmf, hard `u_N` value and expectation-gradient STE (including the halt
head), so the only algorithmic difference is whether depth is sampled.
Comparing stochastic against ADR 0009's `depth_delay` would change
BOTH the depth distribution and the value estimator (ACT full-prefix mixture
vs hard `u_N`), so a difference could not be attributed to stochastic depth.
That secondary contrast is allowed only with the confound declared.

## Depth distribution

For halt probabilities `g_1..g_K = sigmoid(halt(u_k))`, survival
`S_k = prod_{j<=k}(1 - g_j)`, the first-success pmf is

```
p_k = S_{k-1} g_k   (k < K)
p_K = S_{K-1}       (truncation atom)
```

and `N = min{k : S_k < U}` for `U ~ Uniform[0, 1)`, else `K`. Sampling uses a
float64 inverse-CDF helper (`_inverse_cdf_from_uniform`): a float32 grid cannot
resolve atoms narrower than ~6e-8 near 1.0. The pmf is validated (finite,
in `[0, 1]`, correct shapes) before use; `hard_depth` fails closed on a
non-finite `E[N]` because a deterministic control cannot be defined.

## Value readout and gradient

The forward value is hard: `v_hard = u_N`, emitted at
`t + round(delay_scale * (N - 1))` through the ADR 0009 causal map. The
backward pass uses

```
E[u]      = sum_k p_k u_k
E_u_safe  = E[u] on rows whose iterates and expectation are finite
            (recomputed on finite rows if any row failed); zero otherwise
v_ste     = u_N.detach() + (E_u_safe - E_u_safe.detach())
```

The parentheses are load-bearing: `E - E.detach()` is exactly zero, so the
forward value is bitwise `u_N` (the unparenthesised form rounds by up to one
ULP). The gradient is the exact gradient of the expectation target `E[u]`,
including the halt-head path through `p_k`. If `E[u]` is non-finite while
`u_N` is finite, that frame keeps the finite hard value and receives no
expectation gradient; the count is exposed as
`time_consuming_nonfinite_expected` and the training script must alarm on it.
A non-finite `u_N` is left to the standard training non-finite guard (this
guard does not hide it). A `where` after the expectation is insufficient:
backward can still evaluate `0 * Inf`. On the exceptional path the poisoned
graph is abandoned; finite rows are recomputed from the original input and
failed rows receive an empty-sum zero connection to inputs and parameters.
Their pmf/expected-delay diagnostics are detached, so auxiliary losses cannot
re-enter that graph. Both new modes use this same protection and surrogate.
This fallback requires a valid finite pmf:
a non-finite halt probability rejects the whole batch before sampling or the
STE fallback, in both new modes, even if it occurs only
in one future frame. It does not isolate invalid pmfs or promise continuation
with non-finite inputs. ADR 0009 `depth_delay` has a different propagation
policy; its unchanged behavior does not imply the same guarantee here.

Round-two review found that the initial hard-depth arm used hard gather alone:
without latency supervision its halt gradient was `None`. That version is not
a matched control and must not be used for the primary A/B. CAMoR-L-H now uses
the same STE as CAMoR-L-S; task-loss halt gradients are tested for both. The
integer depth and emission location remain detached in both arms.

This estimator is biased for the scored loss because the target frame depends
on the detached `N`. That bias is accepted and recorded; an unbiased
REINFORCE+baseline alternative would widen the loss seam.

## Randomness

Training draws uniforms from a private per-device `torch.Generator` seeded by
`model.time_consuming_seed`; the global RNG is never consumed. Evaluation is
stateless: `eval_seed` (stochastic only) builds a throw-away local generator,
so the same input and seed reproduce the same draw. Validation/selection uses
the MEAN over `S = 5` fixed eval seeds by default and reports the SD; a single
stochastic draw is never presented as a noise-free validation loss.
`hard_depth` is deterministic and must use `S = 1`. Script MC declarations
belong to their specific config/call and are excluded from `to_dict()`; direct
programmatic runs resolve defaults from their current mode. Stochastic
validation always passes `eval_seed_base + s`, even at S=1; provenance records
the actual seed set and a single-draw flag. Final `compute_metrics` remains a
single hard draw at the same seed base and explicitly labels its sampling
noise and distinction from the multi-seed selection mean.

Evaluation uses common random numbers: each forward restarts the generator,
so different batches with the same `(B, T)` reuse the same position-wise
uniform grid. Moving a trial to another row or changing padding/batch shape
can change its draw; seeds are not derived from trial identity. The reported
SD across seeds is not a trial-level standard error and cannot establish
independence between batches. Treating these draws as independent can
overstate effective MC replication/precision; report the fixed seed set and
batch layout, and use prefix-level inference for biological comparisons.

## Latency auxiliary loss (opt-in)

`LossConfig.lambda_latency` (default `0.0`) adds a stimulus-anchored contrast:

```
d_i      = E[delay][i, a_i]                      # anchor frame, source-indexed
Delta D  = mean_{target prefixes}(prefix mean d)
         - mean_{reference prefixes}(prefix mean d)
Delta T  = the same contrast on observed latency
L_lat    = (Delta D - Delta T)^2
```

Only the pre-registered target/reference contrast enters; `visual_only` and
`no_stimulus` trials contribute to the MSE/bio losses but not to this term.
Aggregation is prefix-level and unweighted (a prefix gets ONE group weight).
A batch missing a group returns a differentiable zero and records
`last_latency_stats`; it does not raise. Startup coverage over the whole
training split fails closed when either group lacks the configured minimum
prefixes/trials. Prefixes must be nested in conditions; a prefix spanning two
groups is rejected as non-identifiable. Only an explicit
`latency_pairing="pair_id"` selects the paired variant; it requires pair ids.
The default `"none"` ignores pair ids even when the table contains them.

An enabled loss requires nonempty target and anchor recipes, a one-to-one
condition-name/id mapping covering the table and both contrast groups, and
floating `target_ms` observations plus finite positive `dt_ms` equal to the
active model clock. The table loader converts `target_ms / dt_ms` to frames;
any separately supplied `target_frames` must agree on observed rows. The
original boolean mask and int64 group/prefix/anchor/pair tensors are validated
before any conversion. Frames-only legacy tables remain diagnostic inputs,
but cannot enable supervision. Checkpoint provenance retains the target-ms
vector, its unsplit row indices and dt_ms alongside the frozen table hash.

Training reconciles every table condition, including masked rows, through
`source_indices` against the canonical physical-channel condition derivation.
Input ablation preserves these derived conditions before zeroing sensory
channels; it cannot redefine a trial's physical stimulus condition.
A table cannot manufacture cross-condition coverage by relabeling trials.
Training checks every table anchor against the dataset's actual deterministic
crop using `resolve_anchor_crop`: raw anchor 1800 with a `(600, 3000)` crop
must be supplied as saved anchor 1200. A raw-coordinate table anchor, an
out-of-window anchor or unknown/random crop is rejected before the first
batch. Canonical dataset item/crop behavior is unchanged. `lambda_latency > 0`
also requires `time_consuming_max_steps >= 2`; K=1's constant zero delay has no
trainable latency path and is refused at configuration validation.

The analysis CLI requires the original processed `--dataset` and internals
metadata `dt_ms`, `max_seq_len`, `pre_anchor_frames`. It applies the same
condition, crop-anchor and target-definition checks, including to lambda=0
controls and holdout reports; frames-only tables remain usable by direct
in-memory diagnostics but cannot produce an aligned CLI contrast report.
Missing alignment evidence or mismatched saved lengths fails closed. Internals
`source_indices` must be unique: duplicate trials cannot reweight prefix means.
Legacy processed datasets without `anchor_frames` use `derive_anchor_frames`
from the original physical channels, matching training startup.

Loaders are rebuilt with indexed sidecars and the original collate/worker/RNG
options; validation keeps its original sampler. Validation `total` includes
the latency term for every MC draw; `mse` selection retains its MSE definition.
Batch-level missing contrasts are counted. Training reserves one trial per
required prefix before filling the group trial quota. An epoch consumes every
dataset row at least once, including masked and non-contrast rows. Rare rows
can recur to satisfy the per-batch constraints; this changes sampling weights,
so it is not distribution preserving. Draws, repeats, unique trials and prefix
coverage are logged and stamped in provenance.

Biological decomposition: `L_obs = L_fixed(g) + L_proc(g) + eps`. This term
aligns `Delta L_obs`, so it identifies `Delta L_proc` only under the declared
assumption that `L_fixed` is condition-independent. Processing time is
SUPERVISED, not emergent. No absolute-latency match and no continuous
ex-Gaussian shape are claimed; the support is the `K` atoms.

## Config, resume and provenance

`ModelConfig.time_consuming_seed` and `LossConfig.lambda_latency` are
protected config leaves; `lambda_latency > 0` requires a stochastic/hard_depth
mode and is incompatible with phase-1 frontend training. Both are added to the
architecture-resume leaves. The checkpoint top level additionally stamps
`lambda_latency` plus script-owned provenance (`eval_mc_samples`,
`eval_seed_base`, target-table hash/recipe, condition mapping, contrast groups,
pairing, sampler minima, skipped batches, coverage, grad checkpointing).
Latency declarations, checkpointing and memory budgets are attached to their
specific config as script metadata, excluded from `to_dict()`. A later CLI
parse cannot replace an earlier config's startup targets, loss contrast or
checkpoint provenance. Fresh programmatic configs use independent defaults;
coverage and skipped-batch counters are local to each `train` call.
`_require_latency_provenance_match` refuses a resume that changes the guarded
MC count/seed, target hash, contrast groups, pairing or sampler minima;
a legacy checkpoint missing a key is accepted only at the documented default.
An enabled time-consuming resume is already refused (ADR 0009), so this guard
is defense in depth.

Backends that do not implement the module (raw JAX, Flax, `JAXEvalWrapper`,
full-system Jacobian) refuse an enabled mode through the EXECUTED-module check;
the new modes need no separate refusal path.

## Consequences

- `config.to_dict()` gains two leaves, so every run's `config_sha256` moves,
  including `off`; numerical outputs and `state_dict` keys do not.
- Default (`off`, `lambda_latency=0.0`) loss/forward paths are byte-identical
  to the pre-change ref; the loss validates `lambda_latency` unconditionally
  but reads no latency tensor when it is zero.
- Optional gradient checkpointing (`time_consuming_grad_checkpointing`) trades
  recompute for memory and is refused for the ADR 0009 deterministic modes.
  It checkpoints each block, retaining K inputs/outputs; memory still scales
  with K. It does not promise memory independent of recursion depth.
- New-mode training checks an incremental graph/optimizer budget before its
  first batch (default 8192 MiB, script-owned control). The conservative formula
  is `c*N*K*H*4 + 6*N*H*4 + N*K*8 + 4*parameter_bytes`, with saved-slot
  coefficients calibrated by the CPU benchmark. Imported-process RSS and
  other full-model activations require additional headroom. Unknown dataset
  shapes fail closed. The benchmark uses B=64/T=1500/H=64, K=4/8/16,
  forward and backward, one warm-up plus five runs, allocator events, saved
  tensors and RSS deltas. Estimates are engineering guards, not measurements.
- The stochastic modes are an approximation of a processing-duration
  distribution, not evidence that the cricket samples depth. The claim ceiling
  remains: "an adaptive processing duration exists, and CAMoR is one discrete
  approximation of it", and only if pre-registered tests support it.

## Alternatives rejected

- **Soft/fractional delay emission**: rejected in ADR 0009 (non-causal or
  non-integer frame interpolation); not revisited.
- **Gumbel-softmax relaxation of depth**: changes the value estimator and the
  forward value; the hard `u_N` readout with an expectation gradient keeps the
  forward honest and the A/B clean.
- **REINFORCE with a baseline**: unbiased for the scored loss but higher
  variance and a larger loss seam; deferred until evidence requires it.
- **Using `depth_delay` as the control**: confounds depth sampling with the
  ACT value mixture (see above).
