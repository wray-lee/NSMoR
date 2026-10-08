# Adaptive latent refinement (opt-in ACT recursion), default stays off

The model was described as a "Mixture-of-Recursions" but the only learned,
input-dependent gate (the MoR router) blends the LIF and GRU pathway outputs
per timestep. That is representational *pathway* routing, not a
depth/recursion controller: nothing changes how many computation steps the
model spends on a token. A genuine input/state-dependent adaptive recursion
was requested.

Decision: add an opt-in `AdaptiveLatentRefinement` module applied at the
existing post-fusion / pre-direction-head seam, with
`ModelConfig.refinement_mode: "off" | "fixed" | "adaptive"`. `"off"` (default)
constructs no module, so historical numerics and `state_dict` keys are
bitwise unchanged. `"fixed"` and `"adaptive"` share the *same* module
architecture and parameter tree (LayerNorm + two Linear + one shared halt
head), so they are capacity-matched controls.

## What it is (and is not)

- One **shared** bounded residual block and one **shared** halt head, reused
  across internal depth *and* external time `t`.
- **Representational refinement only.** One external timestep still advances
  the LIF membrane/synaptic/refractory state and the GRU temporal state
  exactly once. The module never calls `LIFCell`/`GRUUnit` and introduces no
  new temporal carry. Padding / empty rows never recurse (depth 0).

## ACT equations

For each valid row, with per-step halt probability
`p_k = sigmoid(HaltHead(u_k))`:

```
N = first k with cum_k = sum_{j<=k} p_j >= 1 - eps,   else N = K (max steps)
R = 1 - sum_{j<N} p_j
out = sum_{j<N} p_j * u_j + R * u_N
```

At forced max depth the remainder mass is fully allocated to `u_N`
(`R = 1 - sum_{j<K} p_j`). Weights are nonnegative and sum to one; `N == 1`
gives `out = u_1`. This is the **full prefix** mixture, not a two-state
interpolation of the last two iterates.

## Real compute saving, not all-K weighting

In `adaptive` mode only the *active* valid rows (not yet halted) are gathered
each step and the shared block/halt head run on that subset; halted rows stop
executing. This is genuine active-row computation, not "run all K then weight".

## Gradient reality (documented, not hidden)

- The discrete first-crossing index `N` has **no pathwise derivative**. The
  trainable ponder surrogate is `N.detach() + R` (the ACT estimator): only the
  remainder `R` is differentiable. It is a **compute proxy**, not measured
  work / MAC / ATP. Actual integer depth is recorded separately
  (`refinement_depth` / `refinement_updates`) and is detached.
- `K == 1` or immediate halting can have **zero** halt gradient. Sensitive
  gradient tests use a nondegenerate `depth > 1` case.
- The block update is bounded (`scale * tanh(...)`); LayerNorm alone is not a
  stability proof. Initialization is checked empirically (nondegenerate
  depth), not asserted.
- A negative halt-head bias *decreases* `p` and typically *increases* depth; it
  is not used to encourage early halting.

## Internals (enabled modes only)

`refinement_depth [B,T]` (int, padding 0, no gradient),
`refinement_updates` (int scalar = `depth.sum()`),
`refinement_ponder_cost` (differentiable scalar mean over valid tokens;
all-empty => differentiable zero),
`refinement_weights [B,T,K]` (valid sum one / padding zero),
`refined_hidden [B,T,H]` (readout latent, distinct from raw GRU and routed
fusion). An `off` model exposes none of these.

## Cost, config, and trust boundaries

`LossConfig.lambda_compute` (default 0.0) owns the compute-cost coefficient at
the loss seam; the default path is bitwise unchanged. Adaptive *training*
requires `lambda_compute > 0` (refused otherwise — no bypass flag). Bool/NaN/
illegal modes, max steps, eps and cost are rejected at the config and
constructor trust boundaries.

## Backends

Raw JAX (`NSMoRCoreJAX`), Flax (`NSMoRModel`) and their factories/trainers/
analysis converters **explicitly reject** an enabled refinement mode until it
is implemented there, rather than silently dropping the module. The GRU-pathway
fixed-point/Jacobian coordinate stays the **raw** GRU state
(`gru_hidden_raw`); the refined latent is a readout, not a new autonomous
recurrent state. The full-system Jacobian is refused while adaptive refinement
is enabled (the discrete ACT halt has no pathwise derivative).

## Considered Options

- **Always-on refinement** (rejected): breaks every historical checkpoint and
  the frozen optimization experiment's contract.
- **All-K compute plus weighting** (rejected): not a real compute saving;
  would be a static pretence of adaptivity.
- **Separate architectures for fixed vs adaptive** (rejected): would confound
  a fixed-vs-adaptive comparison with a capacity difference.
- **Opt-in off/fixed/adaptive sharing one module** (chosen).

## Consequences

- Adaptive need not outperform fixed. A/B/C evidence is reported honestly;
  no superiority is asserted before measurement.
- Computational cost is not ATP and is not animal evidence.
