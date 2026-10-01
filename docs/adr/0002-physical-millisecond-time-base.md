# Physical-Millisecond Time Base for All LIF Parameters

All LIF time constants (tau_syn, tau_w, tau_fac, tau_rec, abs_refract_ms, rel_refract_ms, inhib_tau_ms, dendritic_tau) are declared in physical milliseconds and converted internally via exp(-dt_ms/tau_ms). This prevents silent biophysical rescaling when the model-grid interval ``dt_ms`` changes.

Before this decision, time constants were in frame units -- a tau_w of 100 meant 100 frames, not 100 ms. At the legacy 10.0 ms grid this gave tau_w=1000ms, but switching to the 4.0 ms model grid would silently change it to 400ms, fundamentally altering the biophysical dynamics without any code change or error. The millisecond convention requires every checkpoint and config to carry dt_ms explicitly, and the provenance guard rejects artifacts lacking it.

## Considered Options

- **Frame-unit convention**: simpler (no dt_ms tracking needed), but model-grid-interval-coupled. Changing the grid silently alters the biophysics.
- **Physical-millisecond convention** (chosen): requires dt_ms in every checkpoint and config. All pre-v2.0 checkpoints need conversion or regeneration. Chosen for scientific reproducibility.

## Consequences

Every checkpoint must carry dt_ms. The Pipeline Semantics Version guard rejects artifacts without it, breaking all pre-v2.0 workflows. Config/default.yaml documents dt_ms=4.0 as the model-grid interval (a causal previous-source-hold grid containing estimates, not observations), superseding the legacy grid value of 10.0 ms. The nominal 200 Hz (about 5 ms) firmware source cadence is a separate, unverified quantity: the historically flashed firmware is not verified, and no timing or emission calibration is established. (The 4.0 ms model grid corresponds to 250 Hz; the 5 ms source cadence corresponds to about 200 Hz.) The ~4.006 ms figure retained elsewhere is a host-arrival batching diagnostic, not a measured acquisition rate; the nominal source cadence is not the host-arrival residual P95, and the two are not contradictory. The change from 10.0 ms to 4.0 ms does not by itself inflate sequence duration by 25%; the model grid preserves trial duration. Pure-wind trials receive 1425 frames (5.7 s) of prepended padding at 4.0 ms.
