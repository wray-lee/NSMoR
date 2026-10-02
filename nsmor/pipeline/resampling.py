"""Explicit causal model-grid estimates; source observations remain unchanged."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

_CHANNELS = (
    "x_pos", "y_pos", "heading", "velocity", "acceleration",
    "visual_angle", "wind_state", "l_v_ratio",
)

# Lazy (tensor-free metadata) model-clock contract.  Source time is evidence;
# the model grid is a declared causal estimate.  This lives in a namespace
# distinct from the eager grid keys (``model_dt_ms``/``model_grid_provenance``)
# so tensor-free metadata never trips the eager completeness gate.
LAZY_MODEL_CLOCK_SCHEMA = "lazy-model-clock-v1"
LAZY_MODEL_CLOCK_RESAMPLER = "nsmor.pipeline.resampling.resample_trial_for_model"


def build_lazy_model_clock_contract(dt_ms: float) -> dict[str, Any]:
    """Declare the model grid a lazy producer resamples its source onto."""
    return {
        "schema_version": LAZY_MODEL_CLOCK_SCHEMA,
        "dt_ms": validate_dt_ms(dt_ms, name="lazy model clock dt_ms"),
        "resampler": LAZY_MODEL_CLOCK_RESAMPLER,
    }


def validate_lazy_model_clock_contract(
    contract: Any, expected_dt_ms: float | None = None,
) -> float:
    """Validate a lazy model-clock contract and bind it to the consumer clock.

    Returns the declared model dt.  A malformed/partial contract or a declared
    dt that contradicts ``expected_dt_ms`` fails closed rather than letting a
    resampled artifact be consumed at the wrong clock.
    """
    if not isinstance(contract, Mapping):
        raise ValueError("lazy model clock contract must be a mapping")
    if contract.get("schema_version") != LAZY_MODEL_CLOCK_SCHEMA:
        raise ValueError(
            f"Unsupported lazy model clock schema: {contract.get('schema_version')!r}"
        )
    if contract.get("resampler") != LAZY_MODEL_CLOCK_RESAMPLER:
        raise ValueError(
            f"Unknown lazy model clock resampler: {contract.get('resampler')!r}"
        )
    dt_ms = validate_dt_ms(contract.get("dt_ms"), name="lazy model clock dt_ms")
    if expected_dt_ms is not None:
        expected = validate_dt_ms(expected_dt_ms, name="expected_dt_ms")
        if not math.isclose(dt_ms, expected, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError(
                f"Lazy model clock dt_ms={dt_ms} conflicts with "
                f"expected_dt_ms={expected}"
            )
    return dt_ms


_LAZY_CLOCK_CONTRACT_KEY = "lazy_model_clock_contract"
_LAZY_CLOCK_FLAG_KEY = "lazy_model_clock"
_TRIAL_SPECS_KEY = "trial_specs"
# The keys the frozen lazy reader resolves a spec's raw source CSVs from
# (``lazy_dataloader.source_paths`` reads all three); a source carrying the
# complete set is a genuine lazy source pointer rather than an inert stub.
_LAZY_SOURCE_KEYS = ("session_dir", "kinematics_file", "events_file")

# Eager model-grid namespace keys: the stored tensor payload and the grid clock
# that describes it.  ``anchor_frames`` is deliberately excluded: it is a shared
# key that genuine tensor-free lazy metadata also carries, so its presence cannot
# discriminate a mixed declaration.  Any of these keys beside a lazy contract --
# or beside a raw ``trial_specs`` container with no contract -- means the
# artifact carries two contradictory descriptions of the same fingerprint
# (tensor payload vs. lazy resampler).
_EAGER_NAMESPACE_KEYS = (
    "X_seqs", "Y_seqs", "lengths", "model_dt_ms", "model_grid_provenance",
)

# The complete eager model-grid clock/anchor contract.  Mirrors the frozen
# restricted loader's own completeness set; a ``trial_specs`` artifact that
# carries *any* of these beside the lazy description is an incomplete hybrid.
# ``trial_specs`` is written by every lazy producer (enhanced or markerless
# legacy) and by no eager producer, so its presence is itself proof that the
# artifact carries a lazy trial description -- and therefore must resolve a
# representation verdict, not silently fall to the source-cadence path the
# frozen inner reader would take.
#
# ``labeling_eligibility`` / ``source_clock_provenance`` are deliberately *not*
# eager markers: they are source-side, non-grid ledgers that a genuine
# tensor-free legacy artifact may carry (``prepare_data`` writes both).  Alone
# they never denote an eager model-grid payload -- only the tensor keys and the
# grid clock do -- so they cannot turn a true legacy control into a hybrid.
_EAGER_COMPLETE_KEYS = (
    "model_dt_ms", "model_grid_provenance", "anchor_frames",
    "X_seqs", "Y_seqs", "lengths",
)


def _trial_specs_of(metadata: Any) -> list[Any] | None:
    """Return the trial-spec elements of *metadata*, or ``None`` when absent.

    A genuine lazy artifact stores ``trial_specs`` as a list, tuple, or an
    object ``ndarray`` (the public root loader accepts any of them).  Any other
    container is refused at the common seam: an unrecognised container must not
    let a missing marker bypass the representation verdict.
    """
    if not isinstance(metadata, Mapping) or _TRIAL_SPECS_KEY not in metadata:
        return None
    container = metadata[_TRIAL_SPECS_KEY]
    if isinstance(container, (list, tuple)):
        return list(container)
    if isinstance(container, np.ndarray) and container.dtype == object:
        return list(container)
    raise ValueError(
        "trial_specs must be a list, tuple, or object ndarray of specs"
    )


def _is_raw_source_pointer(source: Any) -> bool:
    """Whether *source* carries a key the frozen reader resolves CSVs from.

    A mapping bearing any of ``session_dir`` / ``kinematics_file`` /
    ``events_file`` is a raw-source pointer -- the frozen reader would try to
    resolve it (and fail closed on a partial pointer) rather than treat it as
    inert.  A non-mapping (e.g. the subset tool's opaque string stubs) or a
    mapping bearing none of them is inert lineage.
    """
    return isinstance(source, Mapping) and any(
        key in source for key in _LAZY_SOURCE_KEYS
    )


def _has_lazy_source_description(specs: list[Any]) -> bool:
    """Whether any spec is a genuine lazy source description.

    The frozen readers resolve a spec's raw CSVs through ``spec['source_pairs']``
    when present, iterating each entry, and fall back to the spec itself
    otherwise (``lazy_dataloader.source_paths``).  So a raw-source pointer is a
    lazy source description whether it sits at the spec's top level or inside
    ``source_pairs``; a spec with neither -- e.g. the subset tool's inert
    per-trial lineage stubs (strings, or dicts bearing no source key) -- is not,
    and cannot make an eager artifact a hybrid.
    """
    for spec in specs:
        if not isinstance(spec, Mapping):
            continue
        sources = spec["source_pairs"] if "source_pairs" in spec else [spec]
        if isinstance(sources, Mapping):
            sources = [sources]
        if any(_is_raw_source_pointer(source) for source in sources):
            return True
    return False


def validate_lazy_artifact_clock(
    metadata: Any, expected_dt_ms: float | None = None,
) -> float | None:
    """Fail-closed lazy-clock validation shared by every ingestion seam.

    Returns the declared model dt, or ``None`` for genuine legacy metadata that
    carries no lazy-clock declaration at all.  A *present-but-null*, partial,
    malformed, or flag-contradicted declaration raises: a lazy artifact must
    never silently degrade to source cadence.  A declared lazy artifact must
    also mark every trial spec and must not carry eager model-grid tensors or
    grid provenance; a complete or partial eager namespace beside the lazy
    contract is refused even at the same dt, so the two consumers can never
    serve divergent values for one fingerprint.  Both the restricted loader and
    the enhanced dataset constructor call this so their verdicts cannot drift.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("lazy-clock metadata must be a mapping")
    dt_ms = None
    if _LAZY_CLOCK_CONTRACT_KEY in metadata:
        dt_ms = validate_lazy_model_clock_contract(
            metadata[_LAZY_CLOCK_CONTRACT_KEY], expected_dt_ms,
        )
    declared = dt_ms is not None
    if declared:
        eager = [key for key in _EAGER_NAMESPACE_KEYS if key in metadata]
        if eager:
            raise ValueError(
                "lazy model clock contract cannot coexist with the eager "
                f"model-grid namespace keys: {sorted(eager)}"
            )
    if _LAZY_CLOCK_FLAG_KEY in metadata:
        flag = metadata[_LAZY_CLOCK_FLAG_KEY]
        if not isinstance(flag, bool):
            raise ValueError("lazy_model_clock must be a boolean")
        if flag != declared:
            raise ValueError(
                "lazy_model_clock flag contradicts the presence of a "
                "lazy_model_clock_contract declaration"
            )
    # A genuine lazy artifact stores a ``trial_specs`` container; an
    # unrecognised container (or a non-mapping element) is refused here rather
    # than silently skipped, so a missing marker cannot hide inside an array.
    specs = _trial_specs_of(metadata)
    if specs is not None:
        for spec in specs:
            if not isinstance(spec, Mapping):
                raise ValueError("every trial spec must be a mapping")
        if declared:
            # On a declared lazy artifact every trial spec must carry an
            # explicit True marker: the top contract resamples every trial, so a
            # spec without one is a declaration gap, not unverified legacy.
            for spec in specs:
                if spec.get(_LAZY_CLOCK_FLAG_KEY) is not True:
                    raise ValueError(
                        "declared lazy artifact requires an explicit "
                        "lazy_model_clock=True marker on every trial spec"
                    )
        else:
            # No declaration.  Genuine markerless legacy carries no eager
            # model-grid key, so it stays unverified.  But a ``trial_specs``
            # container beside *any* eager model-grid key -- the stored tensors,
            # the grid clock, or the grid provenance -- is a hybrid: a lazy trial
            # description next to eager tensors/provenance whose lazy contract
            # was deleted.  It cannot silently fall to the source-cadence path
            # (which would serve a divergent Y under the eager fingerprint).
            # A ``trial_specs`` container is a *lazy description* only when its
            # specs carry a resolvable raw source (the frozen reader resolves
            # ``session_dir``/``kinematics_file``/``events_file`` either at the
            # spec's top level or inside its ``source_pairs`` entries).
            # A container of inert per-trial lineage stubs -- e.g. the subset
            # tool slices the source's specs beside its eager tensors -- is not a
            # competing representation, so it does not make an eager artifact a
            # hybrid.  Beside a genuine lazy description, any eager model-grid
            # key (the stored tensors, the grid clock, or the grid provenance) is
            # refused.  Supported eager-only artifacts carry no ``trial_specs``
            # and are untouched.  The verdict reuses the restricted loader's own
            # completeness rule (the shared seam is the only verdict source)
            # rather than a parallel constructor gate.  Non-grid source ledgers
            # (``labeling_eligibility``/``source_clock_provenance``) are not
            # eager keys and never make a true legacy artifact a hybrid.
            eager_present = [
                key for key in _EAGER_NAMESPACE_KEYS if key in metadata
            ]
            if eager_present and _has_lazy_source_description(specs):
                incomplete = [
                    key for key in _EAGER_COMPLETE_KEYS if key not in metadata
                ]
                if incomplete:
                    raise ValueError(
                        "trial_specs beside an incomplete eager model-grid "
                        f"clock/anchor contract: {sorted(eager_present)}"
                    )
                raise ValueError(
                    "trial_specs beside the eager model-grid namespace is a "
                    "hybrid with no lazy declaration; declare "
                    "lazy_model_clock_contract or mark the trials"
                )
            # A leftover per-spec flag beside an absent declaration likewise
            # contradicts the artifact and is never silently ignored.
            for spec in specs:
                if _LAZY_CLOCK_FLAG_KEY in spec:
                    raise ValueError(
                        "trial spec lazy_model_clock flag contradicts the "
                        "artifact lazy model-clock declaration"
                    )
    return dt_ms


def validate_source_frame_count(spec: Mapping[str, Any], observed_frames: int) -> int:
    """Fail closed unless contributing source pairs reconstruct ``spec['n_frames']``.

    ``n_frames`` is a *source-only* frame count; the synthetic pure-wind prepend
    is a separate, model-grid quantity.  The check runs on raw source frames
    *before* any resampling, so a missing/partial source pair, or an altered
    declaration, is refused instead of being silently resampled onto the model
    grid and published.  The declared count is required and validated for every
    spec the caller has resolved onto the model grid; callers gate this on the
    artifact's resolved model-grid state, never on the redundant per-spec flag,
    so a dropped flag cannot hide an omitted source pair or a missing count.
    """
    if "n_frames" not in spec:
        raise ValueError("lazy spec is missing the declared source n_frames")
    declared = spec["n_frames"]
    if (isinstance(declared, (bool, np.bool_))
            or not isinstance(declared, (int, np.integer)) or int(declared) < 1):
        raise ValueError("spec n_frames must be a positive nonboolean integer")
    declared = int(declared)
    if int(observed_frames) != declared:
        raise ValueError(
            f"trial {spec.get('session_id')!r}/{spec.get('trial_id')!r}: "
            f"contributing source pairs reconstruct {int(observed_frames)} frames, "
            f"expected declared source n_frames={declared}; missing or altered "
            "source pair (no truncation accepted)"
        )
    return declared


def validate_dt_ms(value: Any, *, name: str = "dt_ms") -> float:
    """Require a finite, positive, nonboolean real scalar clock."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive interval")
    try:
        interval = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive interval") from exc
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError(f"{name} must be a finite positive interval")
    return interval


def validate_checkpoint_clock(
    checkpoint: Mapping[str, Any],
    active_dt_ms: float | None = None,
    *,
    require_dt_ms: bool = True,
) -> float | None:
    """Validate serialized cadence before construction or any state restoration."""
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("Invalid checkpoint config: dt_ms required")
    model_config = config.get("model", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Invalid checkpoint model config: dt_ms required")
    active = (
        validate_dt_ms(active_dt_ms, name="active dt_ms")
        if active_dt_ms is not None else None
    )
    if "dt_ms" not in model_config:
        if require_dt_ms:
            raise ValueError("Checkpoint config lacks explicit dt_ms")
        return None
    saved = validate_dt_ms(model_config["dt_ms"], name="checkpoint dt_ms")
    if active is not None and not math.isclose(
        saved, active, rel_tol=1e-9, abs_tol=0.0,
    ):
        raise ValueError(
            f"Checkpoint dt_ms={saved} conflicts with active dt_ms={active}"
        )
    return saved


def resample_trial_for_model(
    trial_data: dict[str, Any], dt_ms: float,
) -> dict[str, Any]:
    """Hold the last source sample on a uniform grid inside source support.

    No future source sample contributes to a model frame. This matters for
    both stimulus onset and the velocity/acceleration inputs shifted by one
    model step downstream. Held values are estimates, not extra observations.
    Labels and MCMC snapshots must be computed from the original trial.
    Events and source-row clock provenance retain their original coordinates.
    """
    if isinstance(dt_ms, (bool, np.bool_)) or not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be positive and finite")
    times = np.asarray(trial_data["time_ms"], dtype=np.float64)
    if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Source time_ms must be finite and strictly increasing")
    arrays = {}
    for name in _CHANNELS:
        values = np.asarray(trial_data[name], dtype=np.float64)
        if values.shape != times.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid source channel {name}: shape/finite check")
        arrays[name] = values
    # Subtraction can undercount an aligned endpoint; filter the extra candidate.
    count = int(np.floor((times[-1] - times[0]) / dt_ms)) + 2
    grid = times[0] + np.arange(count, dtype=np.float64) * dt_ms
    grid = grid[grid <= times[-1]]
    if not grid.size or np.any(np.diff(grid) <= 0):
        raise ValueError("Model grid is empty or not representable at this origin")
    indices = np.searchsorted(times, grid, side="right") - 1
    assert indices.shape == grid.shape
    assert np.all((indices >= 0) & (indices < times.size))
    assert np.all(times[indices] <= grid)
    # Every active stimulus interval must reach a distinct held-grid run.
    # Otherwise discretization silently changes the condition or pulse count.
    # Abort rather than fabricate a sample beyond source support or drop a trial.
    for name in ("visual_angle", "wind_state"):
        active = np.abs(arrays[name]) > 0.0
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
        represented = (
            np.searchsorted(indices, ends) > np.searchsorted(indices, starts)
        )
        if not represented.all():
            raise ValueError(
                f"Model grid loses {int((~represented).sum())} active {name} "
                "interval(s); trial unavailable on this grid (no trials dropped)"
            )
        held_active = active[indices]
        held_starts = np.flatnonzero(
            held_active & ~np.r_[False, held_active[:-1]]
        )
        if held_starts.size != starts.size:
            raise ValueError(
                f"Model grid merges active {name} intervals; trial unavailable "
                "on this grid (no trials dropped)"
            )
    result = dict(trial_data)
    result["time_ms"] = grid
    for name, values in arrays.items():
        result[name] = values[indices].copy()
        assert result[name].shape == grid.shape
    result["model_grid_provenance"] = {
        "method": "causal_previous_source_sample_hold",
        "interpretation": "model_grid_estimates_not_observations",
        "source_n": int(times.size), "model_n": int(grid.size),
        "dt_ms": float(dt_ms), "origin_ms": float(grid[0]),
        "source_end_ms": float(times[-1]), "model_end_ms": float(grid[-1]),
        "omitted_tail_ms": float(times[-1] - grid[-1]),
        "max_source_sample_age_ms": float(np.max(grid - times[indices])),
        "singleton": bool(times.size == 1),
        "source_coordinate_scope": "source_CSV_rows_not_model_grid_frames",
        "synthetic_prepend_frames": 0,
    }
    return result


def resolve_model_anchor_frame(provenance: dict[str, Any]) -> int:
    """Map the authoritative source event to its first supported model frame.

    Source timestamps, not held-channel peaks, define the anchor. Include the
    actual sequence prepend without extrapolating past recorded support.
    """
    numeric = (
        "source_anchor_ms", "origin_ms", "source_end_ms", "model_end_ms", "dt_ms",
    )
    integers = ("source_n", "model_n", "synthetic_prepend_frames")
    required = (*numeric, *integers, "source_anchor_rule")
    if not isinstance(provenance, dict) or any(
        name not in provenance for name in required
    ):
        raise ValueError("Incomplete source anchor/model grid provenance")
    for name in numeric:
        value = provenance[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError(f"source anchor/model grid {name} must be finite numeric")
    for name in integers:
        value = provenance[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise ValueError(f"source anchor/model grid {name} must be integer")
    if (not isinstance(provenance["source_anchor_rule"], str)
            or provenance["source_anchor_rule"] not in (
                "looming_collision", "stimulus_onset",
            )):
        raise ValueError("Invalid source anchor rule")
    anchor_ms = float(provenance["source_anchor_ms"])
    origin = float(provenance["origin_ms"])
    end = float(provenance["source_end_ms"])
    dt_ms = float(provenance["dt_ms"])
    count = int(provenance["model_n"])
    prepend = int(provenance["synthetic_prepend_frames"])
    if (not np.isfinite([anchor_ms, origin, end, dt_ms]).all()
            or dt_ms <= 0 or count < 1 or prepend < 0 or provenance["source_n"] < 1
            or not origin <= anchor_ms <= end):
        raise ValueError(f"Invalid source anchor/model grid: {anchor_ms!r}")
    grid = origin + np.arange(count, dtype=np.float64) * dt_ms
    assert grid.shape == (count,)
    if (grid[-1] > end or grid[-1] != provenance["model_end_ms"]
            or np.any(np.diff(grid) <= 0)):
        raise ValueError("Model grid exceeds or disagrees with recorded support")
    index = int(np.searchsorted(grid, anchor_ms, side="left"))
    if index == count:
        raise ValueError(
            f"source anchor {anchor_ms:.17g} ms has no model frame at or after "
            f"it within support (model ends {grid[-1]:.17g} ms)"
        )
    return index + prepend
