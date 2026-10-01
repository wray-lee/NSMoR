"""Physical prediction units for analysis only; canonical model math is unchanged.

Restore y_norm * target_std + target_mean at the backend output, including
per-step and direct backend calls. Checkpoint clip policy describes target
preprocessing/scoring, not an inference bound: raw predictions stay unclipped.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import torch

from nsmor.model_utils import load_model_from_checkpoint as _canonical_load_model
from nsmor.pipeline.nested_prior import load_artifact_bytes
from nsmor.pipeline.resampling import validate_checkpoint_clock, validate_dt_ms

logger = logging.getLogger(__name__)


def _target_transform(checkpoint: Mapping) -> tuple[float, float, float]:
    config = checkpoint.get('config', {})
    if not isinstance(config, Mapping) or not isinstance(config.get('training', {}), Mapping):
        raise ValueError('Invalid checkpoint normalization config')
    training = config.get('training', {})
    enabled = training.get('normalize_targets', False)
    if type(enabled) is not bool:
        raise ValueError('normalize_targets must be a boolean')
    present = {'target_mean', 'target_std'} & checkpoint.keys()
    if (present and len(present) != 2) or (enabled and len(present) != 2):
        raise ValueError('Incomplete checkpoint target normalization metadata: target_mean and target_std required')
    values = []
    for key, default in (('target_mean', 0.0), ('target_std', 1.0), ('target_clip_cm_s', 0.0)):
        value = checkpoint.get(key, default)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f'Invalid checkpoint {key}: finite numeric scalar required')
        values.append(float(value))
    mean, std, clip = values
    if std <= 0.0 or clip < 0.0:
        raise ValueError('Invalid checkpoint target normalization metadata: std must be positive, clip nonnegative')
    normalized = enabled or mean != 0.0 or std != 1.0
    if normalized and 'target_clip_cm_s' not in checkpoint:
        raise ValueError('Incomplete normalized checkpoint target_clip_cm_s policy')
    if 'normalize_targets' in training and not enabled and (mean != 0.0 or std != 1.0):
        raise ValueError('Checkpoint target normalization metadata contradicts normalize_targets=False')
    return mean, std, clip


def resolve_dt_ms(model: torch.nn.Module, requested: float | None = None) -> float:
    """Use the saved clock; an explicit cadence must match the model biophysics."""
    saved = validate_dt_ms(getattr(model, 'dt_ms', None), name='model.dt_ms')
    if requested is not None:
        requested = validate_dt_ms(requested, name='requested dt_ms')
    if requested is not None and not math.isclose(requested, saved, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(f'Explicit dt_ms={requested} conflicts with saved model.dt_ms={saved}')
    return saved


def prediction_to_physical(prediction: torch.Tensor, model) -> torch.Tensor:
    """Convert only the velocity prediction; preserve shape, dtype and gradients."""
    if not isinstance(prediction, torch.Tensor) or not prediction.is_floating_point():
        raise ValueError('Expected a floating prediction tensor')
    mean, std = getattr(model, 'target_mean', 0.0), getattr(model, 'target_std', 1.0)
    physical = prediction if (mean == 0.0 and std == 1.0) else prediction * std + mean
    assert physical.shape == prediction.shape, 'Prediction unit restoration changed shape'
    if not torch.isfinite(physical).all():
        raise ValueError('Nonfinite physical prediction after target unit restoration')
    return physical


def _physical_backend_output(backend, _inputs, output):
    if isinstance(output, tuple):
        return (prediction_to_physical(output[0], backend), *output[1:])
    return prediction_to_physical(output, backend)


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device,
) -> torch.nn.Module:
    """Load one checkpoint payload through the canonical loader, then restore units.

    Modern checkpoints require a dataset SHA-256; nested source digests must agree.
    Dataset-byte verification is performed by load_analysis_priors.
    Unmarked legacy checkpoints with no transform metadata use identity units.
    Normalized checkpoints must carry complete, finite scalar metadata. Hooks
    do not change state_dict, internals or recurrent state carry; all NSMoRCore
    forward variants delegate to backend, so restoration happens exactly once.
    """
    checkpoint_bytes = Path(checkpoint_path).read_bytes()
    checkpoint = load_artifact_bytes(checkpoint_bytes, map_location='cpu')
    validate_checkpoint_clock(checkpoint, require_dt_ms=True)
    mean, std, clip = _target_transform(checkpoint)
    lineage_keys = ('is_nested_cv', 'nested_prior_artifact', 'nested_prior_artifact_sha256',
                    'nested_prior_fingerprint',
                    'nested_split_seed', 'nested_val_split', 'dataset_path',
                    'dataset_source_sha256', 'dataset_source_binding', 'mcmc_prior_provenance',
                    'animal_identity_status', 'validation_scope')
    lineage = {key: checkpoint[key] for key in lineage_keys if key in checkpoint}
    from nsmor.pipeline.grouping import prior_identity_status
    if lineage.get("is_nested_cv") is True:
        identity_status = prior_identity_status(
            lineage.get("mcmc_prior_provenance"), lineage.get("animal_identity_status"), nested=True
        )
    elif lineage.get("mcmc_prior_provenance") in (None, "global_oof_animal_grouped_cv"):
        if lineage.get("animal_identity_status") not in (None, "historical_unknown"):
            raise ValueError("Historical checkpoint animal_identity_status cannot certify animals")
        identity_status = "historical_unknown"
    else:
        identity_status = prior_identity_status(
            lineage["mcmc_prior_provenance"], lineage.get("animal_identity_status")
        )
    if identity_status != "historical_unknown":
        digest = lineage.get("dataset_source_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Modern checkpoint requires valid dataset_source_sha256")
        if lineage.get("dataset_source_binding", "sha256_bound") != "sha256_bound":
            raise ValueError("Modern checkpoint requires sha256_bound dataset_source_binding")
        if lineage.get("is_nested_cv") is True and lineage.get("nested_prior_fingerprint") != digest:
            raise ValueError("Modern nested checkpoint dataset_source_sha256 conflicts with nested_prior_fingerprint")
    # Config, weights and analysis metadata all come from this single deserialization.
    model = _canonical_load_model(checkpoint_path, device, checkpoint)
    del checkpoint  # Canonical reconstruction copied the CPU state into model parameters.
    model.analysis_checkpoint_lineage = lineage
    model.analysis_checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    model.analysis_animal_identity_status = identity_status
    for consumer in (model, model.backend):
        consumer.target_mean = mean
        consumer.target_std = std
        consumer.target_clip_cm_s = clip
    model.backend.register_forward_hook(_physical_backend_output)
    logger.info('Analysis prediction units: cm/s = y * %.8g + %.8g; saved target clip %.8g cm/s (raw output unclipped)',
                std, mean, clip)
    return model
