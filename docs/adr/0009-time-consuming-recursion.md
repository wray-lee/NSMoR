# Time-consuming recursion (opt-in), default stays off

ADR 0008 introduced `AdaptiveLatentRefinement`: an ACT-style adaptive depth
applied to the post-fusion latent. It is **representational refinement only** —
one external timestep still advances the LIF/GRU state exactly once, so
processing depth never advances biological time. That is deliberate for a
per-frame regression target, but it makes depth incomparable with the animal's
response **latency** (the escape latency distribution and its variability are
the behavioral quantities of interest; `docs/behavior-exploratory-20261010.md`).

Decision: add an opt-in **time-consuming** recursion mode in which processing
depth is tied to the output's position in model-grid time. `ModelConfig`
gains `time_consuming_mode: "off" | "depth_delay" | "accumulate"` plus
`time_consuming_max_steps`, `time_consuming_eps`,
`time_consuming_update_scale`, `time_consuming_delay_scale`, and
`time_consuming_threshold`. `"off"` (default) constructs no module, so
historical numerics, `state_dict` keys and construction RNG are byte-unchanged.

## Semantics vs ADR 0008

| | ADR 0008 (refinement) | ADR 0009 (time-consuming) |
| :-- | :-- | :-- |
| what varies | the *value* of the readout latent | the *emission time* of the readout |
| external time | never advances | output frame `t` is emitted at `t + delay` |
| depth meaning | representation depth | **processing duration** |
| default | `off` | `off` |

The two are **orthogonal and separately opt-in**. Enabling both is allowed
(the refinement value is then the frame that is delayed), but neither is
implied by the other.

## Causality

The delayed map is a strictly causal shift: source frame `t` is written only
to `t + delay` (`delay >= 0`), never earlier. No future information leaks
backward; a frame's output depends only on frames up to and including it. The
last `delay` frames of each trial are **truncated** (no source), so the
effective output length shrinks by up to `max_steps - 1` frames. Padding rows
never recurse (depth 0, no delay) and are never emitted.

The shift is **not injective**: when depth varies within a trial a long-delay
(deep) source can overtake a shallow one, so two sources can target the same
frame. This is arbitrated deterministically — the source with the **smallest
delay wins, ties broken by the earliest source frame** — so the surviving
value is never "whichever frame was processed last" and `out_valid` is a pure
function of the surviving targets. Sources whose target falls outside the trial
are **truncated** (`internals["time_consuming_truncated"]`); sources that lose
in-trial arbitration are **collisions**
(`internals["time_consuming_collisions"]`). Both counts are reported. The
scored quantity is the delayed emission itself
(`internals["time_consuming_hidden"]`), decoded by the direction head and
masked by `out_valid`, so causality holds at the **scored** quantity:
perturbing `x[t+k]` cannot change any emitted frame at or before `t`.

## Alignment and scoring

Delayed outputs are aligned with a per-frame mask
`internals["out_valid"]` (frames that received a delayed write), the per-source
`internals["output_delay"]` (`delay_scale * (depth - 1)`, indexed by
**source** frame), and the post-arbitration `internals["realized_delay"]`
(the surviving source's delay, indexed by **target** frame). The training and
validation losses consume `out_valid` (via the `out_valid=` argument of
`BioJointLoss` / `FrontendLoss`), and the reported `compute_metrics` headline
and the checkpoint-selection MSE apply the same mask (finding R2-B2), so no
fabricated value is scored at a frame the model never emitted — the zero-filled
trailing frames, whose target is a real value, cannot bias the reported skill.
Because the shift truncates the trial tail, the pre-registered phase-3 protocol
fixes the scored window to the aligned, `out_valid`-masked region and reports
the truncation and collision counts; no metric may silently compare a truncated
and a full-length window. The emitted set is exactly the frames some source
targets and wins; it is **not** a fixed leading-frame drop: a frame
`j < max_delay` is emitted iff some source `t = j - delay(t) >= 0` targets it
and wins arbitration (a shallow source can reach an early frame), so the only
frames the code proves unemitted are the **trailing** truncated frames and any
interior frame that is the target of no valid source. The scorer asserts the
stimulus anchor frame is **emitted** (or reports the trial unavailable) rather
than asserting a head drop. Latency comparisons use the delay directly:
`latency_frames = delay` (there is **no** additive frame-offset term: the delay
is measured relative to the **source** frame whose value the emission carries —
source `t` emits at `t + delay`, and the delay is read on that emission), and
the step → ms conversion uses `dt_ms` (`config.model.dt_ms`, 4.0 ms).  The
scored R6 statistic uses the post-arbitration `internals["realized_delay"]`
(indexed by **target** frame), not the per-source `output_delay`, so a collided
target is not double-counted; the frozen formula
`latency_ms = delay_scale * (depth - 1) * dt_ms` is identical to the protocol
§4.1.

## Identifiability (the claim boundary)

Behaviorally, "stop after `k` recursions" (`depth_delay`) and "accumulate
continuously to a threshold" (`accumulate`) are **nearly indistinguishable**
from the output alone; both reduce to "the emission time depends on an
adaptive processing duration". The strongest licensable claim is therefore:

> an adaptive processing duration exists, and MoR is one discrete
> approximation of it.

Never "the cricket has MoR". This is the phase-3 pre-registration's stated
boundary (`docs/realdata-phase3-protocol-20261010.md` §12 — claim boundaries).

## Gradient reality (documented, not hidden)

- The discrete integer depth `N` (hence the emission delay) has **no pathwise
  derivative**; it is computed detached and recorded separately
  (`time_consuming_depth`, `output_delay`).
- The frame **value** is the ACT full-prefix mixture
  `sum_{j<N} g_j u_j + R u_N`, `R = 1 - sum_{j<N} g_j` (the same estimator
  ADR 0008 uses), so the shared block and halt head still receive task
  gradient. Because `eps` and `threshold` are bounded (see below), the
  pre-crossing accumulator is `< 1` in every branch, so `R ∈ (0, 1]` and the
  mixture stays inside the convex hull of the iterates — no extrapolation.
  `K == 1` or immediate halting has zero halt gradient; a negative halt bias
  *decreases* the halt probability and typically *increases* depth — it is not
  used to encourage early halting.
- The block update is bounded (`scale * tanh(...)`); LayerNorm alone is not a
  stability proof. Finiteness is checked empirically, not asserted.

## Compute, and why it is not "time"

`time_consuming_updates` (`= valid_rows * K`) is the ACTUAL executed block AND
halt rows (both the shared block and the halt head run on every step for every
valid row, so `block_rows == halt_rows == updates`), so budget matching
(per-token refinement work, as in `scripts/final_control_harness.py`) stays
available. The module is a representational timing device: the delay is in
**model-grid frames**, NOT measured neural latency, NOT work/MAC, and NOT ATP.

## Config, checkpoints, and trust boundaries

The option leaves are added to the protected `ModelConfig` schema, so
`ExperimentConfig.to_dict()` grows the new keys; this is an additive schema
extension, and the default values reproduce the historical **behaviour**
exactly. **Documented exception (finding R2-M2):** `config_sha256` is
`sha256(json.dumps(config.to_dict(), ...))` and `to_dict()` is `asdict(self)`,
so the six new leaves are serialized for EVERY config — including `off` runs —
and the segment `config_sha256` therefore **differs from a pre-0009 run**. The
numerical outputs, `state_dict` keys and construction RNG are unaffected; only
the config-hash lineage string moves. (This is why the phase-2 script controls
were kept out of the schema; the time-consuming options were deliberately
placed IN it so a resume cannot silently switch them — the hash drift is the
accepted cost.)
The executed mode is recorded as an additive top-level checkpoint key
`time_consuming_mode` (riding the same pop-then-patch provenance seam as
`selection_metric`), and a resume that switches it — or any of the six option
leaves — is refused
(`scripts/train.py::_require_architecture_config_match`), because an enabled
mode changes the output map. The top-level stamp makes a mode switch
detectable even against a checkpoint whose stored config predates the option
leaves. All six options — `mode`, `max_steps`, `eps`, `update_scale`,
`delay_scale`, `threshold` — are validated at the config and constructor
boundaries, including under `"off"`; `eps ∈ [MIN_TIME_CONSUMING_EPS, 1)` with
`MIN_TIME_CONSUMING_EPS = 1e-6` (a smaller `eps` would round `1 - eps` to
`1.0`, forcing the never-crossed branch — finding R1-m2), `update_scale > 0`,
`delay_scale > 0` (a non-integer scale manufactures rounding collisions in the
integer emission shift; the pre-registered 10.0 is integer) and
`threshold ∈ (0, 1]`.

## Backends

The raw-JAX (`NSMoRCoreJAX`), Flax (`NSMoRModel`) and `JAXEvalWrapper`
entry points **explicitly refuse** an enabled time-consuming mode (they do not
implement the delayed map) rather than silently dropping it, keyed on the
EXECUTED module (`time_consuming_module_present`), not the mode string. The
full-system Jacobian is likewise refused while the mode is enabled: the output
frame is a delayed emission, so `dh_out/dx` would be a partial derivative at an
unstated coordinate. The GRU-pathway fixed-point/Jacobian coordinate stays the
**raw** GRU state.

## Considered Options

- **Reuse ADR 0008's refinement as "time"** (rejected): it never advances
  external time, so depth cannot be compared with latency.
- **A single mode only** (rejected): `depth_delay` and `accumulate` are the two
  natural behavioral hypotheses, and the protocol must show the claim does not
  depend on which one is chosen.
- **Make delay differentiable through a soft expectation** (rejected):
  emission is a discrete event; a differentiable soft-delay would emit
  fractional-frame outputs that no animal produces and would corrupt alignment.
- **Opt-in `off`/`depth_delay`/`accumulate` sharing one module** (chosen):
  capacity-matched controls with bit-exact defaults.

## Consequences

- Time-consuming recursion need not outperform fixed depth; A/B evidence is
  reported honestly and no superiority is asserted before measurement.
- A model-grid delay is not a measured latency; the module is a computational
  sufficiency device, not proof of a cricket neural mechanism.
