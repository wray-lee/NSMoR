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
    from nsmor.jax.model import NSMoRModel

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
    ) -> None:
        """Initialize the MC dropout analyzer.

        Args:
            model: Flax NSMoRModel instance.
            params: Flax parameter PyTree.
            n_samples: Number of MC dropout forward passes.
            seed: Random seed.

        Raises:
            RuntimeError: If JAX is not installed.
            ValueError: If n_samples < 2.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for MCDropoutAnalyzerJAX.")
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")

        self.model = model
        self.params = params
        self.n_samples = n_samples
        self.seed = seed

        # Pre-split PRNG keys for all MC samples
        self._rng_keys = jax.random.split(
            jax.random.PRNGKey(seed), n_samples
        )  # (n_samples, 2)
        assert self._rng_keys.shape == (n_samples, 2), (
            f"PRNG keys shape {self._rng_keys.shape} != ({n_samples}, 2)"
        )

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
            - ``gates_mean``: (B, T, 2) mean routing gates (if return_internals).
            - ``gates_std``: (B, T, 2) std of routing gates (if return_internals).
        """
        B, T, D = x.shape
        H = self.model.hidden_dim
        n = self.n_samples

        assert x.ndim == 3, f"Input must be 3-D (B, T, D), got {x.ndim}-D"
        assert lengths.shape == (B,), f"lengths shape {lengths.shape} != ({B},)"

        def _single_forward(rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Single stochastic forward pass with dropout enabled."""
            y_pred, internals = self.model.apply(
                self.params,
                x,
                lengths,
                deterministic=False,  # Enable dropout
                return_internals=True,
                rngs={"dropout": rng_key},
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

        result: Dict[str, np.ndarray] = {
            "y_mean": np.asarray(y_mean),
            "y_std": np.asarray(y_std),
            "y_samples": np.asarray(y_all),
        }

        if return_internals:
            gates_mean = jnp.mean(gates_all, axis=0)  # (B, T, 2)
            gates_std = jnp.std(gates_all, axis=0, ddof=1)    # (B, T, 2)
            result["gates_mean"] = np.asarray(gates_mean)
            result["gates_std"] = np.asarray(gates_std)

        return result

    def uncertainty_per_trial(
        self,
        x: jnp.ndarray,
        lengths: jnp.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute per-trial uncertainty summary statistics.

        For each trial, computes the mean epistemic uncertainty
        (std across MC samples) over its valid timesteps.

        Args:
            x: (B, T, D) input batch.
            lengths: (B,) true sequence lengths.

        Returns:
            Dict containing:
            - ``trial_uncertainty``: (B,) mean std per trial.
            - ``trial_cv``: (B,) coefficient of variation per trial.
            - ``y_mean``: (B, T) mean prediction.
            - ``y_std``: (B, T) prediction std.
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

            trial_unc[i] = float(np.mean(std_i))
            mean_abs = float(np.mean(np.abs(mean_i)))
            trial_cv[i] = trial_unc[i] / (mean_abs + 1e-8)

        return {
            "trial_uncertainty": trial_unc,
            "trial_cv": trial_cv,
            "y_mean": result["y_mean"],
            "y_std": result["y_std"],
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
) -> Dict[str, np.ndarray]:
    """Run MC dropout prediction (convenience function).

    Args:
        model: Flax NSMoRModel.
        params: Flax parameter PyTree.
        x: (B, T, D) input.
        lengths: (B,) lengths.
        n_samples: Number of MC samples.
        seed: Random seed.

    Returns:
        Dict with y_mean, y_std, y_samples.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for mc_dropout_predict_jax.")

    analyzer = MCDropoutAnalyzerJAX(model, params, n_samples=n_samples, seed=seed)
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
) -> Dict[str, np.ndarray]:
    """Compute per-trial MC dropout uncertainty (convenience function).

    Args:
        model: Flax NSMoRModel.
        params: Flax parameter PyTree.
        x: (B, T, D) input.
        lengths: (B,) lengths.
        n_samples: Number of MC samples.
        seed: Random seed.

    Returns:
        Dict with trial_uncertainty, trial_cv, y_mean, y_std.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for mc_dropout_uncertainty_jax.")

    analyzer = MCDropoutAnalyzerJAX(model, params, n_samples=n_samples, seed=seed)
    x_jax = jnp.array(x) if not isinstance(x, jnp.ndarray) else x
    l_jax = jnp.array(lengths) if not isinstance(lengths, jnp.ndarray) else lengths
    return analyzer.uncertainty_per_trial(x_jax, l_jax)
