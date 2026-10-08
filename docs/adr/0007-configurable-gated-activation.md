# Configurable gated activation (opt-in SwiGLU), default stays ReLU

The sensory encoder and the direction head have hard-coded ReLU activations.
A deferred request asked for SwiGLU. SwiGLU is a *gated* activation
``SiLU(W_gate x) ⊙ (W_value x)`` (Shazeer 2020, "GLU Variants Improve
Transformer"); a single ``SiLU(W x)`` is not gated and is not what is meant.

Decision: add ``ModelConfig.activation: "relu" | "swiglu"`` and thread it to
both modules. ``"relu"`` (default) keeps the historical module order and every
existing ``state_dict`` key, so old checkpoints load with ``strict=True`` and
default predictions are unchanged. ``"swiglu"`` adds ``gate_proj`` /
``value_proj`` / ``out_proj`` parameters, so the new keys appear only when it
is selected.

## Considered Options

- **Replace ReLU with SwiGLU unconditionally** (rejected): breaks every
  historical checkpoint and the frozen optimization experiment's model
  contract, for no measured benefit.
- **Single-projection ``SiLU(W x)``** (rejected): not a gated activation;
  presenting it as SwiGLU would be a mislabel.
- **Opt-in gated SwiGLU, default ReLU** (chosen): satisfies the request while
  preserving bit-exact defaults and checkpoint compatibility.

## Stochastic (training) equation

In training mode the direction head applies dropout to the *normalized*
input, then feeds the dropped activation into both projections:

    n = Dropout(LayerNorm(h))
    out = Linear(H,1)( SiLU(gate_proj(n)) ⊙ value_proj(n) )

Torch and Flax must place the dropout at the same point. Dropping the gated
*product* instead (an earlier Flax formulation) changes the training
distribution and MC-dropout uncertainty even with identical parameters, so the
Flax ``DirectionHeadJAX`` now drops the normalized input before the
gate/value projections, matching Torch. LayerNorm also matches Torch exactly
(``epsilon=1e-5``, centered variance), which matters on near-constant inputs.
The default ReLU branch is unchanged.

## Consequences

- The JAX consumers that hard-coded the ``Sequential`` layout
  (``model_nsmor_core_jax.py``, ``jax/model.py``, ``analysis/jax_eval.py``)
  branch on the activation; otherwise torch and JAX would silently disagree
  once a gated branch adds parameters.
- The prospective experiment is separate: ``config/default.yaml`` keeps
  ``relu``. Enabling ``swiglu`` is an explicit, documented opt-in and adds no
  new science controls to the frozen optimization experiment.
- Gating is a *coarse* model of multiplicative dendritic/synaptic gating and
  is stated as a hypothesis. It is not evidence about the real cricket escape
  circuit; the GRU/SwiGLU modules are not equated with that circuit.
