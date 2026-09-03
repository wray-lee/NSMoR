# Gradient Isolation via requires_grad Toggling

The Hybrid Funnel's two-phase training isolates gradients between FrontendEncoder and BioDecisionCore by toggling requires_grad on parameter groups, rather than inserting an explicit .detach() call at the boundary.

The class docstrings document a .detach() boundary, but the actual forward() method does not apply one. An explicit .detach() would sever the only gradient path to the trainable frontend in Phase 1 (where the frontend is the target of training, and the backend is frozen but still part of the computation graph). The requires_grad approach works correctly for all three modes: Phase 1 (frontend trainable, backend frozen), Phase 2 (frontend frozen, backend trainable), and single-phase (all trainable). The trade-off is fragility -- new code paths that forget to toggle requires_grad correctly could violate the isolation invariant.

## Considered Options

- **Explicit .detach()**: robust boundary, but breaks Phase 1 gradient flow to the frontend.
- **requires_grad toggling** (chosen): supports all training phases, but relies on correct parameter-group management at each phase transition.

## Consequences

The docstrings in FrontendEncoder and NSMoRCore are misleading -- they describe a .detach() boundary that does not exist. Phase transition code must correctly toggle requires_grad for the invariant to hold. The docstring-vs-code divergence is a known documentation debt.
