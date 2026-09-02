"""
JAX Training Pipeline for NSMoR.

Features:
  - JIT-compiled train and evaluation steps with Optax.
  - Multi-group AdamW optimization (separate lower LR for LIF parameters).
  - Gradient clipping by global norm.
  - Gradient accumulation support via optax.MultiSteps.
  - Full BioJointLoss parity (Masked MSE, MoR gate regularization,
    ATP metabolic cost, Population sparsity L1, Temporal coherence jerk).
  - Checkpointing: saves native JAX checkpoints and PyTorch-compatible .pth
    artifacts for seamless evaluation by downstream PyTorch tools.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from nsmor.config_parser import ExperimentConfig
from nsmor.jax.dataloader import (
    JAXDataLoader,
    JAXDataset,
    compute_target_stats,
    load_nsmor_dataset,
    session_grouped_train_val_split,
)
from nsmor.jax.model import (
    NSMoRModel,
    load_from_torch_state_dict,
    to_torch_state_dict,
)

logger = logging.getLogger("nsmor.jax.train")


# ===============================================================
# Loss Function (XLA Compiled)
# ===============================================================

def compute_bio_joint_loss(
    y_pred: jnp.ndarray,
    y_true: jnp.ndarray,
    lengths: jnp.ndarray,
    g_gru: jnp.ndarray,
    lif_spikes: Optional[jnp.ndarray] = None,
    *,
    lambda_reg: float = 0.01,
    lambda_energy: float = 0.001,
    lambda_sparse: float = 0.005,
    lambda_jerk: float = 0.005,
    target_rate: float = 0.05,
    hidden_dim: int = 64,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute masked MSE + biological constraints in pure JAX.

    Args:
        y_pred: (B, T) predicted velocity.
        y_true: (B, T) target velocity.
        lengths: (B,) sequence lengths.
        g_gru: (B, T, 1) GRU gating values.
        lif_spikes: (B, T, H) LIF binary spike events.

    Returns:
        total_loss: scalar loss.
        metrics: dict of sub-loss scalars for logging.
    """
    B, T = y_pred.shape
    t_idx = jnp.arange(T)[None, :]
    mask = (t_idx < lengths[:, None]).astype(jnp.float32)  # (B, T)
    total_valid = jnp.maximum(1.0, jnp.sum(mask))

    # 1. Masked MSE
    sq_err = (y_pred - y_true) ** 2 * mask
    mse_loss = jnp.sum(sq_err) / total_valid

    # 2. MoR Router regularization (prevents GRU collapse)
    reg_raw = jnp.sum(g_gru.squeeze(-1) * mask)
    reg_loss = lambda_reg * (reg_raw / total_valid)

    # 3. Spike-based losses
    energy_loss = jnp.array(0.0, dtype=jnp.float32)
    sparse_loss = jnp.array(0.0, dtype=jnp.float32)
    p_hat = jnp.array(0.0, dtype=jnp.float32)

    if lif_spikes is not None:
        mask_3d = mask[..., None]
        valid_spikes = jnp.sum(lif_spikes * mask_3d)
        total_steps = jnp.maximum(1.0, total_valid * float(hidden_dim))
        p_hat = valid_spikes / total_steps

        if lambda_energy > 0.0:
            energy_loss = lambda_energy * p_hat

        if lambda_sparse > 0.0:
            sparse_scale = math.sqrt(float(hidden_dim))
            sparse_loss = lambda_sparse * sparse_scale * jnp.abs(p_hat - target_rate)

    # 4. Temporal coherence (jerk penalty)
    jerk_loss = jnp.array(0.0, dtype=jnp.float32)
    if lambda_jerk > 0.0 and T >= 4:
        dy1 = y_pred[:, 1:] - y_pred[:, :-1]
        dy2 = dy1[:, 1:] - dy1[:, :-1]
        dy3 = dy2[:, 1:] - dy2[:, :-1]
        jerk_mask = (jnp.arange(T - 3)[None, :] + 3 < lengths[:, None]).astype(jnp.float32)
        jerk_count = jnp.maximum(1.0, jnp.sum(jerk_mask))
        jerk_loss = lambda_jerk * (jnp.sum((dy3 ** 2) * jerk_mask) / jerk_count)

    total_loss = mse_loss + reg_loss + energy_loss + sparse_loss + jerk_loss

    metrics = {
        "loss": total_loss,
        "mse": mse_loss,
        "reg": reg_loss,
        "energy": energy_loss,
        "sparse": sparse_loss,
        "jerk": jerk_loss,
        "spike_rate": p_hat,
    }
    return total_loss, metrics


# ===============================================================
# Optimizer Builder
# ===============================================================

def build_optimizer(
    config: ExperimentConfig,
    grad_accum_steps: int = 1,
) -> optax.GradientTransformation:
    """
    Build multi-group AdamW optimizer with optional gradient accumulation.

    Assigns 0.3x learning rate to LIF parameters to prevent surrogate
    gradient instability, matching the PyTorch configuration.
    """
    base_lr = config.training.learning_rate
    lif_lr = base_lr * 0.3
    weight_decay = config.training.weight_decay
    grad_clip_norm = config.training.grad_clip_norm

    # Parameter partitioning label function
    def label_fn(params: Any) -> Any:
        def _leaf_label(path: Tuple[Any, ...], val: Any) -> str:
            for p in path:
                key_str = p.key if hasattr(p, "key") else str(p)
                if "lif" in key_str:
                    return "lif"
            return "non_lif"
        return jax.tree_util.tree_map_with_path(_leaf_label, params)

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.multi_transform(
            {
                "lif": optax.adamw(lif_lr, weight_decay=weight_decay),
                "non_lif": optax.adamw(base_lr, weight_decay=weight_decay),
            },
            label_fn,
        ),
    )

    if grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=grad_accum_steps)

    return tx


from flax import struct

# ===============================================================
# Training State & Checkpoint
# ===============================================================

@struct.dataclass
class JAXTrainState:
    """Container holding model parameters, optimizer state, and metadata."""
    params: Dict[str, Any]
    opt_state: Any
    rng: jax.Array
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")


# ===============================================================
# Core JAX Training Pipeline
# ===============================================================

def train_jax(
    config: ExperimentConfig,
    dataset_path: str = "data/processed/nsmor_dataset_3cond_v2.pt",
    output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    grad_accum_steps: int = 1,
    val_split: float = 0.2,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute full JAX-accelerated training pipeline.

    Args:
        config: Experiment configuration dataclass.
        dataset_path: Path to preprocessed .pt dataset.
        output_dir: Output directory for checkpoints and logs.
        num_epochs: Override number of training epochs.
        batch_size: Override training batch size.
        learning_rate: Override base learning rate.
        grad_accum_steps: Microbatch accumulation count.
        val_split: Validation split fraction.
        resume_from: Optional path to checkpoint to resume from.

    Returns:
        Summary dict containing training logs and execution benchmarks.
    """
    # Disable preallocation by default to avoid GPU OOM on shared cards
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    epochs = num_epochs if num_epochs is not None else config.training.num_epochs
    bs = batch_size if batch_size is not None else config.training.batch_size
    lr = learning_rate if learning_rate is not None else config.training.learning_rate
    config.training.batch_size = bs
    config.training.learning_rate = lr

    out_path = Path(output_dir or config.checkpoint.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing JAX Training Pipeline")
    logger.info("Output dir: %s | Epochs: %d | Batch size: %d | LR: %.2e", out_path, epochs, bs, lr)

    # 1. Load Dataset
    raw_data = load_nsmor_dataset(dataset_path)
    X_seqs = raw_data["X_seqs"]
    Y_seqs = raw_data["Y_seqs"]
    mcmc_priors = raw_data["mcmc_priors"]
    session_ids = raw_data.get("session_ids")
    lengths = raw_data["lengths"]
    n_total = len(X_seqs)

    train_idx, val_idx = session_grouped_train_val_split(
        session_ids, n_total, val_split=val_split, random_seed=config.training.random_seed
    )
    logger.info("Split: %d train, %d val (session-grouped)", len(train_idx), len(val_idx))

    # Target statistics
    target_mean, target_std = 0.0, 1.0
    if config.training.normalize_targets:
        target_mean, target_std = compute_target_stats(
            Y_seqs, train_idx, lengths, target_clip_cm_s=config.training.target_clip_cm_s
        )
        logger.info("Target normalization: mean=%.4f, std=%.4f", target_mean, target_std)

    max_len = config.training.max_seq_len or 2400

    # Auto-scale microbatch for high sequence length (2400) to keep memory safe
    if bs > 64 and grad_accum_steps == 1:
        micro_bs = 64
        effective_accum = bs // micro_bs
        logger.info(
            "Batch size %d mapped to microbatch %d with %d accumulation steps",
            bs, micro_bs, effective_accum,
        )
    else:
        micro_bs = bs
        effective_accum = grad_accum_steps

    # MAJOR-3 fix: Extract anchor_frames from dataset for anchor-aligned cropping
    anchor_frames = raw_data.get("anchor_frames", None)

    train_dataset = JAXDataset(
        X_seqs, Y_seqs, mcmc_priors, indices=train_idx, max_seq_len=max_len,
        target_mean=target_mean, target_std=target_std,
        normalize_targets=config.training.normalize_targets,
        target_clip_cm_s=config.training.target_clip_cm_s,
        anchor_frames=anchor_frames,
    )
    val_dataset = JAXDataset(
        X_seqs, Y_seqs, mcmc_priors, indices=val_idx, max_seq_len=max_len,
        target_mean=target_mean, target_std=target_std,
        normalize_targets=config.training.normalize_targets,
        target_clip_cm_s=config.training.target_clip_cm_s,
        anchor_frames=anchor_frames,
    )

    train_loader = JAXDataLoader(train_dataset, batch_size=micro_bs, shuffle=True, seed=config.training.random_seed)
    val_loader = JAXDataLoader(val_dataset, batch_size=micro_bs, shuffle=False)

    # 2. Build Model
    model = NSMoRModel(
        sensory_dim=config.model.sensory_dim,
        mcmc_dim=config.model.mcmc_dim,
        hidden_dim=config.model.hidden_dim,
        dt_ms=config.model.dt_ms,
        lif_alpha=config.model.lif_alpha,
        lif_threshold=config.model.lif_threshold,
        lif_beta=config.model.lif_beta,
        lif_tau_syn=config.model.lif_tau_syn,
        lif_tau_w=config.model.lif_tau_w,
        lif_b_adapt=config.model.lif_b_adapt,
        lif_lateral_inhibition=config.model.lif_lateral_inhibition,
        lif_inhib_tau_ms=config.model.lif_inhib_tau_ms,
        lif_rel_refract_ms=config.model.lif_rel_refract_ms,
        lif_abs_refract_ms=config.model.lif_abs_refract_ms,
        lif_v_rest=config.model.lif_v_rest,
        lif_tbptt_steps=config.model.lif_tbptt_steps,
        gru_neuromod_gain=config.model.gru_neuromod_gain,
        dropout_rate=config.model.dropout,
        sensory_noise_std=getattr(config.model, "sensory_noise_std", 0.0),
    )

    rng = jax.random.PRNGKey(config.training.random_seed)
    rng, init_rng = jax.random.split(rng)

    dummy_x = jnp.zeros((micro_bs, max_len, 8), dtype=jnp.float32)
    dummy_l = jnp.full((micro_bs,), max_len, dtype=jnp.int32)
    params = model.init(init_rng, dummy_x, dummy_l)

    # Resume from checkpoint if requested
    resume_path = resume_from or config.checkpoint.resume_from
    if resume_path and Path(resume_path).exists():
        logger.info("Loading weights from checkpoint: %s", resume_path)
        ckpt_th = torch.load(resume_path, weights_only=False)
        sd = ckpt_th.get("model_state_dict", ckpt_th)
        params = load_from_torch_state_dict(model, sd)

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logger.info("Model initialized: %d parameters", param_count)

    # 3. Build Optimizer
    optimizer = build_optimizer(config, grad_accum_steps=effective_accum)
    opt_state = optimizer.init(params)
    rng, train_rng = jax.random.split(rng)
    state = JAXTrainState(params=params, opt_state=opt_state, rng=train_rng)

    # 4. JIT-compiled Train and Eval Steps
    # MAJOR-2 fix: Read lambda_reg from config instead of hardcoding.
    # Previously lambda_reg was set to config.loss.lambda_energy (wrong field)
    # and then overridden by a hardcoded 0.01.
    lambda_reg_val = getattr(config.loss, "lambda_reg", 0.01)
    lambda_energy_val = getattr(config.loss, "lambda_energy", 0.001)
    lambda_sparse_val = getattr(config.loss, "lambda_sparse", 0.005)
    lambda_jerk_val = getattr(config.loss, "lambda_jerk", 0.005)
    target_rate_val = getattr(config.loss, "target_rate", 0.05)

    @jax.jit
    def train_step(
        train_state: JAXTrainState,
        x: jnp.ndarray,
        y: jnp.ndarray,
        lengths: jnp.ndarray,
    ) -> Tuple[JAXTrainState, Dict[str, jnp.ndarray]]:
        new_rng, dropout_rng = jax.random.split(train_state.rng)

        def loss_fn(p):
            y_pred, internals = model.apply(
                p, x, lengths, deterministic=False, return_internals=True,
                rngs={"dropout": dropout_rng},
            )
            g_gru = internals["routing_gates"][:, :, 1:2]
            spikes = internals["lif_spikes"]
            loss, metrics = compute_bio_joint_loss(
                y_pred, y, lengths, g_gru, spikes,
                lambda_reg=lambda_reg_val,
                lambda_energy=lambda_energy_val,
                lambda_sparse=lambda_sparse_val,
                lambda_jerk=lambda_jerk_val,
                target_rate=target_rate_val,
                hidden_dim=config.model.hidden_dim,
            )
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, step_metrics), grads = grad_fn(train_state.params)
        updates, new_opt_state = optimizer.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)

        new_train_state = train_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=train_state.step + 1,
            rng=new_rng,
        )
        return new_train_state, step_metrics

    @jax.jit
    def eval_step(
        p: Dict[str, Any],
        x: jnp.ndarray,
        y: jnp.ndarray,
        lengths: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        y_pred, internals = model.apply(
            p, x, lengths, deterministic=True, return_internals=True
        )
        g_gru = internals["routing_gates"][:, :, 1:2]
        spikes = internals["lif_spikes"]
        _, metrics = compute_bio_joint_loss(
            y_pred, y, lengths, g_gru, spikes,
            lambda_reg=lambda_reg_val,
            lambda_energy=lambda_energy_val,
            lambda_sparse=lambda_sparse_val,
            lambda_jerk=lambda_jerk_val,
            target_rate=target_rate_val,
            hidden_dim=config.model.hidden_dim,
        )
        return metrics

    # 5. Training Loop
    logger.info("=" * 60)
    logger.info("Starting JAX training loop for %d epochs", epochs)
    logger.info("=" * 60)

    epoch_times = []
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_losses = []
        train_mses = []

        # Training epoch
        for x_b, y_b, l_b in train_loader:
            state, step_metrics = train_step(state, x_b, y_b, l_b)
            train_losses.append(step_metrics["loss"])
            train_mses.append(step_metrics["mse"])

        # Asynchronously wait and compute means at epoch end
        train_loss = float(jnp.mean(jnp.stack(train_losses)))
        train_mse = float(jnp.mean(jnp.stack(train_mses)))

        # Validation epoch
        val_losses = []
        val_mses = []
        val_spks = []

        for x_v, y_v, l_v in val_loader:
            v_metrics = eval_step(state.params, x_v, y_v, l_v)
            val_losses.append(v_metrics["loss"])
            val_mses.append(v_metrics["mse"])
            val_spks.append(v_metrics["spike_rate"])

        val_loss = float(jnp.mean(jnp.stack(val_losses)))
        val_mse = float(jnp.mean(jnp.stack(val_mses)))
        val_spk = float(jnp.mean(jnp.stack(val_spks)))

        ep_duration = time.time() - t0
        epoch_times.append(ep_duration)
        state = state.replace(epoch=ep)

        logger.info(
            "Epoch %3d/%3d | train_loss: %.4f (mse: %.4f) | val_loss: %.4f (mse: %.4f) | spk: %.4f | time: %.2fs",
            ep, epochs, train_loss, train_mse, val_loss, val_mse, val_spk, ep_duration,
        )

        ep_stats = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "spike_rate": val_spk,
            "time_sec": ep_duration,
        }
        history.append(ep_stats)

        # Checkpointing
        is_best = val_loss < state.best_val_loss
        if is_best:
            state = state.replace(best_val_loss=val_loss)
            # MINOR-3 fix: Atomic write via .tmp + os.replace
            best_pth = out_path / "best_model.pth"
            best_tmp = out_path / "best_model.pth.tmp"
            torch_sd = to_torch_state_dict(state.params)
            torch.save({
                "model_state_dict": torch_sd,
                "epoch": ep,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "target_mean": target_mean,
                "target_std": target_std,
            }, best_tmp)
            os.replace(best_tmp, best_pth)
            logger.info("Saved new best model checkpoint to %s (val_loss=%.4f)", best_pth, val_loss)

        if ep % config.training.checkpoint_interval == 0 or ep == epochs:
            ep_pth = out_path / f"epoch_{ep}.pth"
            ep_tmp = out_path / f"epoch_{ep}.pth.tmp"
            torch_sd = to_torch_state_dict(state.params)
            torch.save({
                "model_state_dict": torch_sd,
                "epoch": ep,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "target_mean": target_mean,
                "target_std": target_std,
            }, ep_tmp)
            os.replace(ep_tmp, ep_pth)

    avg_ep_time = np.mean(epoch_times[1:]) if len(epoch_times) > 1 else epoch_times[0]
    logger.info("=" * 60)
    logger.info("Training complete. Average epoch time (post-JIT): %.2fs", avg_ep_time)
    logger.info("=" * 60)

    return {
        "best_val_loss": state.best_val_loss,
        "avg_epoch_time_s": avg_ep_time,
        "first_epoch_time_s": epoch_times[0],
        "history": history,
    }
