"""
Unit and Integration Tests for NSMoR JAX Optimization Pipeline.

Verifies:
  - Configuration loading and runtime parameters.
  - JAXDataset and JAXDataLoader batching and shape integrity.
  - Forward pass tensor shape assertions (y_pred, routing_gates, potentials).
  - Bidirectional PyTorch <-> JAX state_dict mapping.
  - BioJointLoss numerical parity and penalty formulations.
  - JIT compiled train_step gradient update sanity.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import numpy as np
import torch

try:
    import jax
    import jax.numpy as jnp
    import optax
    from nsmor.jax.config import JAXExperimentConfig, load_config
    from nsmor.jax.dataloader import JAXDataLoader, JAXDataset
    from nsmor.jax.model import (
        NSMoRModel,
        load_from_torch_state_dict,
        to_torch_state_dict,
    )
    from nsmor.jax.train import JAXTrainState, build_optimizer, compute_bio_joint_loss
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestJAXPipeline:

    @pytest.fixture(autouse=True)
    def setup_env(self):
        import os
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    def test_jax_config_loading(self):
        cfg = load_config("config/default.yaml")
        assert isinstance(cfg, JAXExperimentConfig)
        assert cfg.model.sensory_dim == 4
        assert cfg.model.mcmc_dim == 4
        assert cfg.model.hidden_dim == 64
        assert cfg.training.batch_size == 128
        assert cfg.training.learning_rate > 0

    def test_jax_dataloader_and_dataset(self):
        N = 10
        T = 50
        X_seqs = [np.random.randn(T, 4).astype(np.float32) for _ in range(N)]
        Y_seqs = [np.random.randn(T).astype(np.float32) for _ in range(N)]
        priors = np.zeros((N, 4), dtype=np.float32)
        priors[:, 0] = 1.0

        ds = JAXDataset(X_seqs, Y_seqs, priors, max_seq_len=60)
        assert len(ds) == N
        assert ds.X.shape == (N, 60, 8)
        assert ds.Y.shape == (N, 60)
        assert ds.lengths.shape == (N,)
        assert np.allclose(ds.X[:, :50, 4], 1.0)

        loader = JAXDataLoader(ds, batch_size=4, shuffle=False, pad_last_batch=True)
        batches = list(loader)
        assert len(batches) == 3  # 4 + 4 + 2 (padded to 4)
        xb, yb, lb = batches[0]
        assert xb.shape == (4, 60, 8)
        assert yb.shape == (4, 60)
        assert lb.shape == (4,)

    def test_nsmor_model_forward_shapes(self):
        B, T, H = 2, 40, 64
        model = NSMoRModel(hidden_dim=H, dt_ms=4.0)
        rng = jax.random.PRNGKey(0)

        dummy_x = jnp.zeros((B, T, 8), dtype=jnp.float32)
        dummy_l = jnp.array([40, 30], dtype=jnp.int32)
        params = model.init(rng, dummy_x, dummy_l)

        y_pred, internals = model.apply(
            params, dummy_x, dummy_l, deterministic=True, return_internals=True
        )

        assert y_pred.shape == (B, T), f"y_pred shape mismatch: {y_pred.shape}"
        assert internals["routing_gates"].shape == (B, T, 2), f"gates shape: {internals['routing_gates'].shape}"
        assert internals["lif_potentials"].shape == (B, T, H), f"potentials shape: {internals['lif_potentials'].shape}"
        assert internals["lif_spikes"].shape == (B, T, H), f"spikes shape: {internals['lif_spikes'].shape}"
        assert internals["gru_hidden"].shape == (B, T, H), f"gru shape: {internals['gru_hidden'].shape}"

    def test_bidirectional_checkpoint_conversion(self):
        H = 64
        model = NSMoRModel(hidden_dim=H, lif_lateral_inhibition=0.1)
        rng = jax.random.PRNGKey(42)
        x = jnp.zeros((2, 10, 8))
        l = jnp.array([10, 10])
        params = model.init(rng, x, l)

        # Convert to PyTorch state dict
        torch_sd = to_torch_state_dict(params)
        assert "backend.lif_cell.W_in.weight" in torch_sd
        assert "backend.gru_unit.gru.weight_ih_l0" in torch_sd
        assert "backend.router.gate.weight" in torch_sd
        assert "backend.direction_head.net.3.weight" in torch_sd

        # Convert back to JAX params
        flax_params = load_from_torch_state_dict(model, torch_sd)
        p_orig = params["params"]
        p_restored = flax_params["params"]

        assert np.allclose(p_orig["sensory_encoder"]["dense"]["kernel"], p_restored["sensory_encoder"]["dense"]["kernel"])
        assert np.allclose(p_orig["lif_w_in"], p_restored["lif_w_in"])
        assert np.allclose(p_orig["gru_w_ih"], p_restored["gru_w_ih"])
        assert np.allclose(p_orig["router"]["gate"]["kernel"], p_restored["router"]["gate"]["kernel"])

    def test_bio_joint_loss_parity(self):
        B, T, H = 2, 20, 64
        y_pred = jnp.ones((B, T)) * 2.0
        y_true = jnp.ones((B, T)) * 1.0
        lengths = jnp.array([20, 10], dtype=jnp.int32)
        g_gru = jnp.ones((B, T, 1)) * 0.5
        spikes = jnp.zeros((B, T, H))

        loss, metrics = compute_bio_joint_loss(
            y_pred, y_true, lengths, g_gru, spikes,
            lambda_reg=0.01, lambda_energy=0.0, lambda_sparse=0.0, lambda_jerk=0.0,
        )

        # (2 - 1)^2 = 1.0; reg = 0.01 * 0.5 = 0.005; total = 1.005
        assert abs(float(metrics["mse"]) - 1.0) < 1e-5
        assert abs(float(metrics["reg"]) - 0.005) < 1e-5
        assert abs(float(loss) - 1.005) < 1e-5

    def test_jit_train_step_gradient_flow(self):
        B, T, H = 4, 30, 64
        model = NSMoRModel(hidden_dim=H, dt_ms=4.0)
        rng = jax.random.PRNGKey(123)
        rng, init_rng = jax.random.split(rng)

        x = jax.random.normal(rng, (B, T, 8))
        y = jax.random.normal(rng, (B, T))
        lengths = jnp.array([30, 25, 20, 15], dtype=jnp.int32)

        params = model.init(init_rng, x, lengths)
        tx = optax.adamw(1e-3)
        opt_state = tx.init(params)
        state = JAXTrainState(params=params, opt_state=opt_state, rng=rng)

        @jax.jit
        def step(s):
            new_rng, drop_rng = jax.random.split(s.rng)
            def loss_fn(p):
                y_p, internals = model.apply(
                    p, x, lengths, deterministic=False, return_internals=True,
                    rngs={"dropout": drop_rng}
                )
                loss, _ = compute_bio_joint_loss(
                    y_p, y, lengths, internals["routing_gates"][..., 1:2], internals["lif_spikes"]
                )
                return loss
            val, grads = jax.value_and_grad(loss_fn)(s.params)
            updates, new_opt = tx.update(grads, s.opt_state, s.params)
            new_p = optax.apply_updates(s.params, updates)
            return s.replace(params=new_p, opt_state=new_opt, step=s.step + 1, rng=new_rng), val

        # First step (JIT)
        state, loss1 = step(state)
        # Second step
        state, loss2 = step(state)

        assert np.isfinite(float(loss1))
        assert np.isfinite(float(loss2))
        assert state.step == 2
