# Physical-Millisecond Time Base for All LIF Parameters

All LIF time constants (tau_syn, tau_w, tau_fac, tau_rec, abs_refract_ms, rel_refract_ms, inhib_tau_ms, dendritic_tau) are declared in physical milliseconds and converted internally via exp(-dt_ms/tau_ms). This prevents silent biophysical rescaling when the acquisition rate changes.

Before this decision, time constants were in frame units -- a tau_w of 100 meant 100 frames, not 100 ms. At 100 Hz (dt=10ms) this gave tau_w=1000ms, but switching to 250 Hz recording (dt=4ms) would silently change it to 400ms, fundamentally altering the biophysical dynamics without any code change or error. The millisecond convention requires every checkpoint and config to carry dt_ms explicitly, and the provenance guard rejects artifacts lacking it.

## Considered Options

- **Frame-unit convention**: simpler (no dt_ms tracking needed), but acquisition-rate-coupled. A hardware change silently alters the biophysics.
- **Physical-millisecond convention** (chosen): requires dt_ms in every checkpoint and config. All pre-v2.0 checkpoints need conversion or regeneration. Chosen for scientific reproducibility.

## Consequences

Every checkpoint must carry dt_ms. The Pipeline Semantics Version guard rejects artifacts without it, breaking all pre-v2.0 workflows. Config/default.yaml documents dt_ms=10.0 as the reference acquisition rate.
