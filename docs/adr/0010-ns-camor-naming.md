# Name the opt-in recursion family "NS-CAMoR" (short form CAMoR)

The project name says "Mixture-of-Recursions" (MoR). The user has clarified that
MoR means Bae et al., *Mixture-of-Recursions*, arXiv 2507.10524: a shared
recursive block with input-dependent, token-level recursion-depth routing,
realized by expert-choice or token-choice routers plus recursion-wise KV
caching. The recursion this repository implements (ADR 0008, ADR 0009) keeps
that core idea but differs in algorithm and purpose. Calling it plain "MoR"
invites reading it as a reproduction of Bae et al.

Decision (user, 2026-10-14): in documentation, papers and figures the opt-in
recursion family is called **NS-CAMoR**, with the short form **CAMoR** (Causal
Adaptive Mixture-of-Recursions) used in running text.
- "NS" stands for Neuroscience, as in the project name NSMoR. The name keeps "MoR" to
  credit the original work and adds "Causal Adaptive" for what differs.
- On first use, write "NS-CAMoR (CAMoR), MoR-inspired (Bae et al. 2025)".
  Afterwards CAMoR may stand in for NS-CAMoR.
- The name applies only to models with a recursion mode enabled. A model with
  both modes `off` (phase-2 R0–R3, phase-3 `C0`) is the NSMoR dual-pathway
  baseline, not NS-CAMoR. No code identifier, config key, checkpoint, ADR 0008/0009 text or
frozen protocol changes; the phase-3 protocol keeps its arm ids (`C1`–`C6`).

| Name | Code | Phase-3 arms | What it is |
| :-- | :-- | :-- | :-- |
| **NS-CAMoR** (short: CAMoR) | umbrella | — | One shared bounded residual block reused across internal depth, with a per-frame ACT-style halt head (present but not executed in the fixed-K control). Strictly causal; one external timestep advances the LIF/GRU state exactly once. |
| **CAMoR-R** (refinement) | `refinement_mode: "adaptive"` (ADR 0008) | `C6` | Depth refines the post-fusion representation only; it never consumes model time. |
| CAMoR-R fixed-K control | `refinement_mode: "fixed"` | `C3`/`C4`/`C5` (K2/K3/K4) | Same module and parameters with no early exit: the capacity-matched control. |
| **CAMoR-L** (latency-coupled) | `time_consuming_mode: "depth_delay" \| "accumulate"` (ADR 0009) | `C1`/`C2` | The output for frame `t` is emitted `delay_scale·(depth−1)` frames later, so depth is a model-grid latency comparable with the animal's response latency. |

`refinement_mode: "off"` together with `time_consuming_mode: "off"` (all
phase-2 R0–R3 checkpoints) contains no CAMoR at all.

## Why not reproduce Bae et al. directly

- **Expert-choice routing is non-causal here.** At each recursion step it keeps
  the top-k tokens across the whole sequence, so a frame's depth would depend
  on later frames. An online, per-frame behavioral predictor cannot use that.
- **Token-choice routing is the closest causal relative.** It fixes each
  token's depth once, up front, with a router. CAMoR instead decides
  sequentially with a per-step halt probability, ACT-style (Graves 2016). This
  is a design choice, not an equivalence claim.
- **No attention, so no KV caching.** The recursion-wise KV caching and sharing
  in MoR have nothing to act on in this LIF/GRU model.
- **Different objective.** MoR saves language-model compute. CAMoR asks
  whether processing depth relates to stimulus condition and response latency;
  CAMoR-L's depth-to-delay map exists for that comparison.
- **Biological-time constraint.** Recursion may not advance the LIF/GRU state.
  Only CAMoR-L moves the output in time, and only on the model grid.

## Related work

- Graves 2016, Adaptive Computation Time (arXiv 1603.08983): the halting rule.
- Dehghani et al. 2019, Universal Transformers (ICLR 2019; arXiv 1807.03819):
  a shared block with ACT depth.
- Veerabadran et al. 2023, adaptive recurrent vision (NeurIPS; arXiv
  2311.06964): an ACT-halting ConvRNN whose iteration count scales with
  difficulty. This is the closest precedent for "processing time grows with
  difficulty".
- Bae et al. 2025, Mixture-of-Recursions (arXiv 2507.10524): the inspiration
  and the name.

A name check on 2026-10-14 found no existing use of "NS-CAMoR". It found
unrelated uses of the short letters "CAMoR" in other fields:

- "CAMOR": Congestion Aware Multipath Optimal Routing, an SDN routing method
  (Shah et al., PlatCon 2017; IEEE Xplore document 7883694).
- "CaMor": Raghuraman, Barrash and Rossignac, "CaMor: Screw Interpolation
  between Perspective Projections of Partial Views of Rectangular Images",
  Eurographics WICED 2017, DOI 10.2312/wiced.20171066.
- "CaMoRe": the DFKI research group Causal Models and Representations.
- "CAMoRo": an Interreg project on collaborative autonomous mobile robots.

None of these is an adaptive-recursion method, which is why the full name
NS-CAMoR is used on first mention. The check was not exhaustive; repeat it
before publication.

## Limits on claims

- The name describes the model, not a biological claim.
- The phase-3 protocol (§12) states its claim ceiling in terms of MoR: "an
  adaptive processing duration exists; MoR is one discrete approximation", and
  never "the cricket has MoR". Under this ADR's renaming, the paraphrase is that
  an adaptive processing duration exists and CAMoR is one discrete
  approximation of it.
- That paraphrase is the **upper bound** of what may be concluded, and only if
  the pre-registered phase-3 tests support it. It is not a result. Without that
  support, nothing about adaptive processing duration is claimed, and "the
  cricket has CAMoR/MoR" is never claimed.
- CAMoR-L's delay is a model-grid timing device, not measured neural latency,
  work/MAC or ATP (ADR 0009).
- The `[g_lif, g_gru]` `MoRRouter` gate is pathway readout mixing and is not part
  of CAMoR; its historical name is kept.
