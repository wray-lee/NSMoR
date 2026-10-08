"""JAX-Accelerated Uncertainty Quantification for NSMoR.

Corresponds to :mod:`nsmor.analysis.uq` (PyTorch/NumPy version).

Acceleration strategy:
  - MC dropout forward passes parallelized via ``jax.vmap`` over
    independent PRNG keys (each key produces a different dropout mask).
  - Model inference JIT-compiled via ``jax.jit``.
  - Statistical utilities (bootstrap CI, Cohen's d, Holm-Bonferroni)
    delegate to the NumPy originals in ``nsmor.analysis.uq`` — these
    operate on scalar/1-D data where JAX offers no speedup.

All public functions match the PyTorch API semantics; outputs are
numerically compatible.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    JAX_AVAILABLE = False

# Re-export pure-NumPy UQ utilities unchanged — they are not
# compute-bottlenecked and the JAX module should provide the
# same API surface.
from nsmor.analysis.uq import (
    bootstrap_ci,
    cohens_d,
    holm_bonferroni,
    log_pca_variance,
)

if JAX_AVAILABLE:
    from nsmor.jax.model import NSMoRModel, validate_input_and_lengths

logger = logging.getLogger(__name__)

__all__ = [
    # Re-exported from uq.py (pure NumPy, no JAX needed)
    "bootstrap_ci",
    "cohens_d",
    "holm_bonferroni",
    "log_pca_variance",
    # JAX-accelerated MC inference
    "mc_dropout_predict_jax",
    "mc_dropout_uncertainty_jax",
    "MCDropoutAnalyzerJAX",
]


# ===============================================================
# MC Dropout via vmap over PRNG keys
# ===============================================================

class MCDropoutAnalyzerJAX:
    """Monte Carlo dropout uncertainty quantification with JAX.

    Uses ``jax.vmap`` to run *n_samples* stochastic forward passes
    in parallel (one per independent PRNG key), then summarizes
    the predictive distribution.

    Attributes:
        model: Flax NSMoRModel instance.
        params: Frozen Flax parameter PyTree.
        n_samples: Number of MC dropout samples.
        seed: Base PRNG seed for reproducibility.
    """

    def __init__(
        self,
        model: "NSMoRModel",
        params: Dict[str, Any],
        n_samples: int = 30,
        seed: int = 42,
        include_sensory_noise: bool = False,
    ) -> None:
        """Initialize the MC dropout analyzer.

        Args:
            model: Flax NSMoRModel instance.
            params: Flax parameter PyTree.
            n_samples: Number of MC dropout forward passes.
            seed: Random seed.
            include_sensory_noise: If ``False`` (default) the intrinsic
                Gaussian sensory noise is suppressed for the MC forward
                passes, so the reported ``y_std`` is a *dropout-only*
                epistemic estimate.  Set ``True`` to instead report the
                mixed dropout + input-noise dispersion (which must not be
                labelled purely epistemic).

        Raises:
            RuntimeError: If JAX is not installed.
            ValueError: If n_samples < 2.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for MCDropoutAnalyzerJAX.")
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        if not isinstance(include_sensory_noise, bool):
            raise ValueError(
                "include_sensory_noise must be a bool, got "
                f"{type(include_sensory_noise).__name__}"
            )

        self.model = model
        self.params = params
        self.n_samples = n_samples
        self.seed = seed
        self.include_sensory_noise = include_sensory_noise

        # Pre-split PRNG keys for all MC samples
        self._rng_keys = jax.random.split(
            jax.random.PRNGKey(seed), n_samples
        )  # (n_samples, 2)
        assert self._rng_keys.shape == (n_samples, 2), (
            f"PRNG keys shape {self._rng_keys.shape} != ({n_samples}, 2)"
        )

    def attribution(self) -> Dict[str, Any]:
        """Machine-readable description of what the reported dispersion is.

        Returns a truthful attribution of the ``y_std`` / ``trial_uncertainty``
        dispersion: dropout-only epistemic (default) or a MIXED dropout +
        injected sensory-noise estimate (``include_sensory_noise=True`` with a
        nonzero noise level), which must not be labelled purely epistemic.
        """
        dropout = float(getattr(self.model, "dropout_rate", 0.0) or 0.0)
        noise = (
            float(getattr(self.model, "sensory_noise_std", 0.0) or 0.0)
            if self.include_sensory_noise else 0.0
        )
        mixed = self.include_sensory_noise and noise > 0.0
        return {
            "dispersion": (
                "mixed_dropout_and_sensory_noise" if mixed
                else "epistemic_dropout_only"
            ),
            "dropout_rate": dropout,
            "sensory_noise_std": noise,
            "include_sensory_noise": self.include_sensory_noise,
            "epistemic": not mixed,
            "includes_aleatoric": mixed,
        }

    def predict(
        self,
        x: jnp.ndarray,
        lengths: jnp.ndarray,
        return_internals: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run MC dropout forward passes and summarize predictions.

        Args:
            x: (B, T, D) input batch.
            lengths: (B,) true sequence lengths.
            return_internals: If True, also return per-sample internals.

        Returns:
            Dict containing:
            - ``y_mean``: (B, T) mean prediction across MC samples.
            - ``y_std``: (B, T) std of predictions (epistemic uncertainty).
            - ``y_samples``: (n_samples, B, T) all MC predictions.
            - ``attribution``: machine-readable dispersion attribution (see
              :meth:`attribution`); dropout-only epistemic by default.
            - ``gates_mean``: (B, T, 2) mean routing gates (if return_internals).
            - ``gates_std``: (B, T, 2) std of routing gates (if return_internals).
        """
        B, T, D = x.shape
        H = self.model.hidden_dim
        n = self.n_samples

        assert x.ndim == 3, f"Input must be 3-D (B, T, D), got {x.ndim}-D"
        assert lengths.shape == (B,), f"lengths shape {lengths.shape} != ({B},)"
        # Host preflight BEFORE vmap/JIT: reject fractional/negative/overlong
        # lengths and nonfinite values in VALID frames, so a real NaN/Inf
        # observation fails closed instead of yielding nonfinite samples and
        # confidence summaries.  An invalid padded suffix remains harmless.
        validate_input_and_lengths(
            np.asarray(x), np.asarray(lengths), context="MCDropoutAnalyzerJAX",
        )

        def _single_forward(rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Single stochastic forward pass with dropout enabled.

            Sensory noise is suppressed unless ``include_sensory_noise`` is
            set, so the across-sample dispersion is attributable to dropout
            alone (epistemic) rather than mixed with injected input noise.
            """
            y_pred, internals = self.model.apply(
                self.params,
                x,
                lengths,
                deterministic=False,  # Enable dropout
                return_internals=True,
                rngs={"dropout": rng_key},
                sensory_noise=self.include_sensory_noise,
            )
            return y_pred, internals["routing_gates"]

        # vmap over PRNG keys: (n_samples,) -> (n_samples, B, T) and (n_samples, B, T, 2)
        y_all, gates_all = jax.vmap(_single_forward)(self._rng_keys)

        assert y_all.shape == (n, B, T), (
            f"y_all shape {y_all.shape} != ({n}, {B}, {T})"
        )
        assert gates_all.shape == (n, B, T, 2), (
            f"gates_all shape {gates_all.shape} != ({n}, {B}, {T}, 2)"
        )

        y_mean = jnp.mean(y_all, axis=0)  # (B, T)
        # MINOR-3: Use ddof=1 (Bessel correction) for epistemic uncertainty
        # estimate — more appropriate for small n_samples (4-30).
        y_std = jnp.std(y_all, axis=0, ddof=1)    # (B, T)

        # Fail closed on nonfinite sampled or derived uncertainty.  The valid
        # input frames were already checked finite above, so a nonfinite MC
        # sample/summary is a numerical failure and must not be returned as a
        # confidence estimate.
        if not (
            np.isfinite(np.asarray(y_all)).all()
            and np.isfinite(np.asarray(y_mean)).all()
            and np.isfinite(np.asarray(y_std)).all()
        ):
            raise ValueError(
                "MCDropoutAnalyzerJAX produced nonfinite sampled or derived "
                "predictions; refusing to return a nonfinite uncertainty "
                "estimate."
            )

        result: Dict[str, Any] = {
            "y_mean": np.asarray(y_mean),
            "y_std": np.asarray(y_std),
            "y_samples": np.asarray(y_all),
            "attribution": self.attribution(),
        }

        if return_internals:
            gates_mean = jnp.mean(gates_all, axis=0)  # (B, T, 2)
            gates_std = jnp.std(gates_all, axis=0, ddof=1)    # (B, T, 2)
            # root 6: the returned gate representation must fail closed
            # INDEPENDENTLY.  A corrupt router kernel can yield finite y
            # samples while the routing gates (and their summaries) are
            # nonfinite; a finite y does NOT imply finite gates.  No clipping /
            # nan_to_num — refuse the nonfinite representation.
            if not (
                np.isfinite(np.asarray(gates_all)).all()
                and np.isfinite(np.asarray(gates_mean)).all()
                and np.isfinite(np.asarray(gates_std)).all()
            ):
                raise ValueError(
                    "MCDropoutAnalyzerJAX produced nonfinite routing gates or "
                    "gate summaries; refusing to return a nonfinite gate "
                    "representation."
                )
            result["gates_mean"] = np.asarray(gates_mean)
            result["gates_std"] = np.asarray(gates_std)

        assert result["y_mean"].shape == (B, T)
        assert result["y_std"].shape == (B, T)
        assert result["y_samples"].shape == (n, B, T)
        return result

    def uncertainty_per_trial(
        self,
        x: jnp.ndarray,
        lengths: jnp.ndarray,
    ) -> Dict[str, Any]:
        """Compute per-trial uncertainty summary statistics.

        For each trial, computes the mean dispersion (std across MC samples)
        over its valid timesteps.  When ``include_sensory_noise=True`` and the
        model injects noise, that dispersion is MIXED (dropout + sensory noise)
        and is labelled as such in ``attribution`` — it is not purely
        epistemic.  The default is dropout-only epistemic.

        Args:
            x: (B, T, D) input batch.
            lengths: (B,) true sequence lengths.

        Returns:
            Dict containing:
            - ``trial_uncertainty``: (B,) mean std per trial.
            - ``trial_cv``: (B,) coefficient of variation per trial.
            - ``y_mean``: (B, T) mean prediction.
            - ``y_std``: (B, T) prediction std.
            - ``attribution``: machine-readable dispersion attribution.
        """
        result = self.predict(x, lengths)

        B, T = result["y_mean"].shape
        lengths_np = np.asarray(lengths)

        trial_unc = np.zeros(B, dtype=np.float32)
        trial_cv = np.zeros(B, dtype=np.float32)

        for i in range(B):
            L = int(lengths_np[i])
            if L == 0:
                continue
            std_i = result["y_std"][i, :L]
            mean_i = result["y_mean"][i, :L]

            # r6 R9: accumulate temporal reductions in float64 so a large-but-
            # finite float32 per-frame value cannot overflow the sum to +inf
            # (which would fabricate a zero coefficient of variation and
            # conceal real dispersion).  The representable float32 result is
            # preserved; an unrepresentable result fails closed rather than
            # silently returning Inf.
            std_sum = float(np.sum(std_i.astype(np.float64)))
            mean_abs = float(np.mean(np.abs(mean_i).astype(np.float64)))
            unc = std_sum / float(L)
            if not math.isfinite(unc) or not math.isfinite(mean_abs):
                raise ValueError(
                    "uncertainty_per_trial: derived per-trial reduction is "
                    "nonfinite; refusing to report unrepresentable statistics."
                )
            # r7 R5: validate the ACTUAL exported per-trial uncertainty in its
            # float32 output representation, not only the float64 reduction.
            # A finite float64 reduction (e.g. 1e40) overflows the float32
            # ``trial_unc`` array to +inf; returning that would conceal real
            # dispersion while the separately-checked CV stays finite.
            unc32 = np.float32(unc)
            if not np.isfinite(unc32):
                raise ValueError(
                    "uncertainty_per_trial: per-trial uncertainty is not "
                    "representable in float32; refusing to report a nonfinite "
                    "dispersion statistic."
                )
            trial_unc[i] = unc32
            cv = unc / (mean_abs + 1e-8)
            cv32 = np.float32(cv)
            if not np.isfinite(cv32):
                raise ValueError(
                    "uncertainty_per_trial: per-trial coefficient of variation "
                    "is not representable in float32; refusing to report a "
                    "nonfinite dispersion statistic."
                )
            trial_cv[i] = float(cv32)

        return {
            "trial_uncertainty": trial_unc,
            "trial_cv": trial_cv,
            "y_mean": result["y_mean"],
            "y_std": result["y_std"],
            "attribution": result["attribution"],
        }


# ===============================================================
# Module-level convenience functions
# ===============================================================

def mc_dropout_predict_jax(
    model: "NSMoRModel",
    params: Dict[str, Any],
    x: Any,
    lengths: Any,
    n_samples: int = 30,
    seed: int = 42,
    include_sensory_noise: bool = False,
) -> Dict[str, Any]:
    """Run MC dropout prediction (convenience function).

    Args:
        model: Flax NSMoRModel.
        params: Flax parameter PyTree.
        x: (B, T, D) input.
        lengths: (B,) lengths.
        n_samples: Number of MC samples.
        seed: Random seed.
        include_sensory_noise: If ``False`` (default) the reported ``y_std``
            is dropout-only (epistemic); set ``True`` for mixed dispersion.

    Returns:
        Dict with y_mean, y_std, y_samples, and a machine-readable
        ``attribution`` of the reported dispersion.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for mc_dropout_predict_jax.")

    analyzer = MCDropoutAnalyzerJAX(
        model, params, n_samples=n_samples, seed=seed,
        include_sensory_noise=include_sensory_noise,
    )
    # Host preflight on the ORIGINAL concrete x/lengths BEFORE ``jnp.array``
    # (r4 R5): an int64/uint64 length outside 0..T would otherwise be narrowed
    # to a legal int32 value by ``jnp.array`` and silently accepted.
    validate_input_and_lengths(
        np.asarray(x), np.asarray(lengths), context="mc_dropout_predict_jax",
    )
    x_jax = jnp.array(x) if not isinstance(x, jnp.ndarray) else x
    l_jax = jnp.array(lengths) if not isinstance(lengths, jnp.ndarray) else lengths
    return analyzer.predict(x_jax, l_jax)


def mc_dropout_uncertainty_jax(
    model: "NSMoRModel",
    params: Dict[str, Any],
    x: Any,
    lengths: Any,
    n_samples: int = 30,
    seed: int = 42,
    include_sensory_noise: bool = False,
) -> Dict[str, Any]:
    """Compute per-trial MC dropout uncertainty (convenience function).

    Args:
        model: Flax NSMoRModel.
        params: Flax parameter PyTree.
        x: (B, T, D) input.
        lengths: (B,) lengths.
        n_samples: Number of MC samples.
        seed: Random seed.
        include_sensory_noise: If ``False`` (default) the per-trial
            uncertainty is dropout-only (epistemic); set ``True`` for mixed
            dispersion.

    Returns:
        Dict with trial_uncertainty, trial_cv, y_mean, y_std, and a
        machine-readable ``attribution`` of the reported dispersion.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for mc_dropout_uncertainty_jax.")

    analyzer = MCDropoutAnalyzerJAX(
        model, params, n_samples=n_samples, seed=seed,
        include_sensory_noise=include_sensory_noise,
    )
    # Host preflight on the ORIGINAL concrete x/lengths BEFORE ``jnp.array``
    # (r4 R5): an int64 negative or uint64 overlong length would otherwise be
    # narrowed to a legal int32 value by ``jnp.array`` and silently accepted.
    validate_input_and_lengths(
        np.asarray(x), np.asarray(lengths),
        context="mc_dropout_uncertainty_jax",
    )
    x_jax = jnp.array(x) if not isinstance(x, jnp.ndarray) else x
    l_jax = jnp.array(lengths) if not isinstance(lengths, jnp.ndarray) else lengths
    return analyzer.uncertainty_per_trial(x_jax, l_jax)
