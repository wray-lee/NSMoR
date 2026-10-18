"""Latency target sidecar, contrast sampler and collate helpers (ADR 0012).

The latency auxiliary loss needs four per-trial fields that are NOT part of
the protected model schema: a stimulus anchor frame (saved coordinates), an
observed latency in frames, a condition id, and a recording-prefix id.  They
are script-owned provenance (recorded in the checkpoint top level), never
``ExperimentConfig`` leaves, so they do not move ``config_sha256``.

Contract:

* :class:`LatencyTargetTable` is the frozen, hash-stamped target table.
* :func:`build_latency_sidecar` aligns it to a dataset's
  ``source_indices`` (row order of the unsplit artifact).
* :class:`IndexedDataset` + :func:`collate_with_latency_metadata` attach the
  per-trial metadata to a batch.
* :class:`LatencyContrastBatchSampler` guarantees each training batch has
  enough target/reference prefixes and trials; :func:`validate_latency_coverage`
  is the startup fail-closed check.

Only synthetic fixtures are exercised in this implementation batch; the real
Cercus-cli target table integration requires a fresh pre-registration.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class LatencyTargetTable:
    """Frozen per-trial latency targets and their provenance.

    All tensors are aligned 1:1 with the rows of the UNSplit dataset artifact
    (``dataset.source_indices`` maps a split back to these rows).

    Attributes:
        target_frames: ``(N,)`` observed latency in FRAMES (stimulus anchor
            to response onset).
        target_mask: ``(N,)`` bool; True where a latency was observed.
        group_ids: ``(N,)`` int64 condition ids (pre-registered mapping).
        prefix_ids: ``(N,)`` int64 recording-prefix ids.
        anchor_frames: ``(N,)`` int64 stimulus anchor frames in SAVED
            coordinates (post ``resolve_anchor_crop``).
        pair_ids: Optional ``(N,)`` int64 pair ids for a future paired corpus.
        sha256: SHA-256 of the source table file.
        recipe: Latency definition recipe (e.g. the D1 trigger / 250 ms
            stationarity rule).
        anchor_recipe: Anchor derivation recipe.
        condition_mapping: Condition name -> group id mapping.
        target_ms: Optional ``(N,)`` floating observed latency in milliseconds;
            required together with dt_ms when supervision is enabled.
        dt_ms: Positive model-grid interval in milliseconds. The loader converts
            target_ms / dt_ms to frames, checking any supplied target_frames.
    """

    target_frames: torch.Tensor
    target_mask: torch.Tensor
    group_ids: torch.Tensor
    prefix_ids: torch.Tensor
    anchor_frames: torch.Tensor
    pair_ids: Optional[torch.Tensor] = None
    sha256: str = ""
    recipe: str = ""
    anchor_recipe: str = ""
    condition_mapping: Dict[str, int] = field(default_factory=dict)
    target_ms: Optional[torch.Tensor] = None
    dt_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_frames, torch.Tensor)
            or self.target_frames.ndim != 1
        ):
            raise ValueError("target_frames must be a 1-D torch.Tensor")
        n = self.target_frames.shape[0]
        for name, value, dtype in (
            ("target_frames", self.target_frames, None),
            ("target_mask", self.target_mask, torch.bool),
            ("group_ids", self.group_ids, torch.long),
            ("prefix_ids", self.prefix_ids, torch.long),
            ("anchor_frames", self.anchor_frames, torch.long),
        ):
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if value.shape != (n,):
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} != ({n},)"
                )
            if dtype is not None and value.dtype != dtype:
                raise ValueError(f"{name} dtype {value.dtype} != {dtype}")
        if not self.target_frames.is_floating_point():
            raise ValueError("target_frames must be floating (frames)")
        if bool((self.group_ids < 0).any()):
            raise ValueError("group_ids must be >= 0")
        if bool((self.prefix_ids < 0).any()):
            raise ValueError("prefix_ids must be >= 0")
        if bool((self.anchor_frames < 0).any()):
            raise ValueError("anchor_frames must be >= 0")
        masked = self.target_mask
        if self.target_ms is not None or self.dt_ms is not None:
            if (
                isinstance(self.dt_ms, bool)
                or not isinstance(self.dt_ms, (int, float))
                or not math.isfinite(self.dt_ms)
                or self.dt_ms <= 0
            ):
                raise ValueError("dt_ms must be finite and positive")
            if (
                not isinstance(self.target_ms, torch.Tensor)
                or self.target_ms.shape != (n,)
                or not self.target_ms.is_floating_point()
            ):
                raise ValueError("target_ms must be a floating (N,) tensor")
            observed_ms = self.target_ms[masked].double()
            if (
                not bool(torch.isfinite(observed_ms).all())
                or bool((observed_ms < 0).any())
            ):
                raise ValueError("target_ms must be finite and >= 0 on masked rows")
            if not torch.allclose(
                self.target_frames[masked].double(), observed_ms / self.dt_ms,
                rtol=1e-6, atol=1e-6,
            ):
                raise ValueError("target_ms / dt_ms disagrees with target_frames")
        if bool(~torch.isfinite(self.target_frames[masked]).all()):
            raise ValueError("target_frames must be finite where mask is True")
        if bool((self.target_frames[masked] < 0).any()):
            raise ValueError("target_frames must be >= 0 where mask is True")
        if self.pair_ids is not None:
            if (
                not isinstance(self.pair_ids, torch.Tensor)
                or self.pair_ids.shape != (n,)
                or self.pair_ids.dtype != torch.long
            ):
                raise ValueError("pair_ids must be (N,) int64")
            if bool((self.pair_ids < 0).any()):
                raise ValueError("pair_ids must be >= 0")


def load_latency_target_table(path: str | Path) -> LatencyTargetTable:
    """Load a ``.pt`` latency target table and stamp its file SHA-256.

    The payload is a dict with ``mask``, ``group_ids``, ``prefix_ids`` and
    saved-coordinate ``anchor_frames``. Supply floating ``target_ms`` with
    positive ``dt_ms`` to convert observations to frames. A supplied floating
    ``target_frames`` must agree on observed rows. Frames-only legacy tables
    remain readable for diagnostics but cannot enable latency supervision.
    Optional ``pair_ids`` and supervision recipes/mapping retain strict types;
    no raw mask or identity tensor is coerced before validation.

    Args:
        path: Path to the frozen table.

    Returns:
        The validated table.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: On a malformed payload.
    """
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Latency target table not found: {path}")
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    try:
        payload = torch.load(
            table_path, map_location="cpu", weights_only=True,
        )
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(table_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("latency target table must be a dict of tensors")
    required = (
        "mask", "group_ids", "prefix_ids",
        "anchor_frames",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"latency target table is missing {missing}"
        )
    target_frames = payload.get("target_frames")
    target_ms = payload.get("target_ms")
    dt_ms = payload.get("dt_ms")
    if target_frames is None:
        if (
            not isinstance(target_ms, torch.Tensor)
            or target_ms.ndim != 1 or not target_ms.is_floating_point()
        ):
            raise ValueError("table requires target_frames or floating target_ms")
        if (
            isinstance(dt_ms, bool) or not isinstance(dt_ms, (int, float))
            or not math.isfinite(dt_ms) or dt_ms <= 0
        ):
            raise ValueError("dt_ms must be finite and positive for target_ms")
        target_frames = target_ms.double() / dt_ms
        assert target_frames.shape == target_ms.shape
    for name in ("recipe", "anchor_recipe"):
        value = payload.get(name, "")
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
    mapping = payload.get("condition_mapping", {})
    if not isinstance(mapping, dict) or any(
        not isinstance(name, str) or not name.strip()
        or isinstance(group, bool) or not isinstance(group, int) or group < 0
        for name, group in mapping.items()
    ):
        raise ValueError("condition_mapping must map condition names to integer ids")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("condition_mapping ids must be distinct")
    return LatencyTargetTable(
        # Validate the original tensors: bool()/long() here would silently
        # alter masks, condition membership, prefix identity and anchor time.
        target_frames=target_frames,
        target_ms=target_ms,
        dt_ms=dt_ms,
        target_mask=payload["mask"],
        group_ids=payload["group_ids"],
        prefix_ids=payload["prefix_ids"],
        anchor_frames=payload["anchor_frames"],
        pair_ids=payload.get("pair_ids"),
        sha256=digest,
        recipe=str(payload.get("recipe", "")),
        anchor_recipe=str(payload.get("anchor_recipe", "")),
        condition_mapping=dict(payload.get("condition_mapping", {})),
    )


def validate_latency_target_definition(
    table: LatencyTargetTable,
    *,
    target_group: int,
    reference_group: int,
    dt_ms: float,
) -> None:
    """Require an auditable supervision definition on the active model clock.

    Args:
        table: Validated frozen target table.
        target_group: Declared target condition id.
        reference_group: Declared reference condition id.
        dt_ms: Active model-grid interval in milliseconds.

    Raises:
        ValueError: If recipes, condition mapping or physical units are missing
            or contradict the active clock/contrast.
    """
    table.__post_init__()
    for name in ("recipe", "anchor_recipe"):
        value = getattr(table, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"latency target definition requires a nonempty {name}"
            )
    mapping = table.condition_mapping
    if (
        not isinstance(mapping, dict) or not mapping
        or any(
            not isinstance(name, str) or not name.strip()
            or isinstance(group, bool) or not isinstance(group, int) or group < 0
            for name, group in mapping.items()
        )
        or len(set(mapping.values())) != len(mapping)
        or not {target_group, reference_group}.issubset(set(mapping.values()))
        or not set(table.group_ids.tolist()).issubset(set(mapping.values()))
    ):
        raise ValueError(
            "condition_mapping must uniquely declare table and contrast groups"
        )
    if table.target_ms is None or table.dt_ms is None:
        raise ValueError(
            "latency target definition requires target_ms and dt_ms evidence"
        )
    if table.dt_ms != dt_ms:
        raise ValueError(
            f"latency dt_ms={table.dt_ms} disagrees with model dt_ms={dt_ms}"
        )


def build_latency_sidecar(
    table: LatencyTargetTable,
    source_indices: Sequence[int],
    *,
    dataset: Optional[torch.utils.data.Dataset] = None,
) -> Dict[str, torch.Tensor]:
    """Align a target table to a dataset split's source rows.

    Args:
        table: Frozen target table (UNsplit row order).
        source_indices: The dataset's ``source_indices`` (1:1 with its rows).
        dataset: Optional dataset for physical-condition and saved-anchor
            validation against its actual deterministic crop. Training and
            analysis entry points always supply it. Input ablation uses the
            canonical physical conditions captured before channel zeroing.
            Unknown or random crops fail closed rather than guessing an anchor.
            Direct in-memory sidecar consumers may omit it when no alignment
            is claimed.

    Returns:
        A dict of per-dataset-row tensors with the exact loss keyword names
        (``latency_anchor_frames``, ``latency_target_frames``,
        ``latency_target_mask``, ``latency_group_id``,
        ``latency_prefix_id``, optional ``latency_pair_id``).

    Raises:
        ValueError: If an index is out of range, if dataset metadata is missing
            or length-mismatched, or if dataset conditions or anchor frames
            disagree with the table.
    """
    indices = torch.as_tensor(list(source_indices), dtype=torch.long)
    n_rows = table.target_frames.shape[0]
    if indices.numel() > 0 and (
        bool((indices < 0).any()) or bool((indices >= n_rows).any())
    ):
        raise ValueError(
            f"source_indices must lie in [0, {n_rows}); got "
            f"{indices.tolist()}"
        )
    sidecar: Dict[str, torch.Tensor] = {
        "latency_anchor_frames": table.anchor_frames[indices],
        "latency_target_frames": table.target_frames[indices],
        "latency_target_mask": table.target_mask[indices],
        "latency_group_id": table.group_ids[indices],
        "latency_prefix_id": table.prefix_ids[indices],
    }
    if dataset is not None:
        from nsmor.pipeline.conditions import (
            derive_stimulus_metadata, resolve_anchor_crop,
        )

        sequences = getattr(dataset, "sequences", None)
        raw_anchors = getattr(dataset, "anchor_frames", None)
        if sequences is None or raw_anchors is None:
            raise ValueError(
                "Latency anchor alignment requires deterministic dataset "
                "sequences and anchor_frames; random/unknown crops are refused"
            )
        if len(sequences) != len(indices) or len(raw_anchors) != len(indices):
            raise ValueError("Latency anchor metadata length disagrees with dataset")
        conditions = getattr(dataset, "latency_physical_conditions", None)
        if conditions is None:
            physical = [np.asarray(sequence[0]) for sequence in sequences]
            for values in physical:
                assert values.ndim == 2 and values.shape[1] == 8, values.shape
            conditions, _ = derive_stimulus_metadata(
                physical, [len(values) for values in physical],
            )
        if len(conditions) != len(indices):
            raise ValueError(
                "Latency physical conditions length disagrees with dataset"
            )
        assert np.asarray(conditions).shape == (len(indices),)
        names_by_group = {
            group: name for name, group in table.condition_mapping.items()
        }
        for row, condition in enumerate(conditions):
            supplied = names_by_group.get(int(sidecar["latency_group_id"][row]))
            if supplied != condition:
                raise ValueError(
                    f"Latency condition mismatch at dataset row {row}, source "
                    f"index {int(indices[row])}: table={supplied!r}, "
                    f"physical condition={condition!r}"
                )
        for row, (sequence, raw_anchor) in enumerate(zip(sequences, raw_anchors)):
            n_frames = int(sequence[0].shape[0])
            if (
                isinstance(raw_anchor, (bool, np.bool_))
                or not isinstance(raw_anchor, (int, np.integer))
                or not 0 <= raw_anchor < n_frames
            ):
                raise ValueError(f"Invalid dataset anchor at row {row}")
            start, end = resolve_anchor_crop(
                n_frames=n_frames, anchor_frame=int(raw_anchor),
                max_seq_len=dataset.max_seq_len,
                pre_anchor_frames=dataset.pre_anchor_frames,
            )
            saved_anchor = int(raw_anchor) - start
            supplied = int(sidecar["latency_anchor_frames"][row])
            if not 0 <= saved_anchor < end - start or supplied != saved_anchor:
                raise ValueError(
                    f"Latency anchor mismatch at dataset row {row}: table saved "
                    f"anchor={supplied}, expected={saved_anchor}, raw anchor="
                    f"{raw_anchor}, crop=({start}, {end})"
                )
        assert sidecar["latency_anchor_frames"].shape == (len(indices),)
    if table.pair_ids is not None:
        sidecar["latency_pair_id"] = table.pair_ids[indices]
    return sidecar


def _as_arrays(
    group_ids: Sequence[int],
    prefix_ids: Sequence[int],
    mask: Optional[Sequence[bool]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(group_ids, dtype=np.int64)
    prefixes = np.asarray(prefix_ids, dtype=np.int64)
    if groups.ndim != 1 or prefixes.shape != groups.shape:
        raise ValueError(
            f"group_ids/prefix_ids shapes {groups.shape}/{prefixes.shape} "
            "must be equal 1-D arrays"
        )
    if mask is None:
        available = np.ones(groups.shape[0], dtype=bool)
    else:
        available = np.asarray(mask, dtype=bool)
        if available.shape != groups.shape:
            raise ValueError(
                f"mask shape {available.shape} != {groups.shape}"
            )
    if bool((groups < 0).any()) or bool((prefixes < 0).any()):
        raise ValueError("group_ids/prefix_ids must be >= 0")
    return groups, prefixes, available


def validate_latency_coverage(
    group_ids: Sequence[int],
    prefix_ids: Sequence[int],
    *,
    target_group: int,
    reference_group: int,
    min_prefixes_per_group: int = 2,
    min_trials_per_group: int = 4,
    mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Fail closed unless both contrast groups have enough prefixes/trials.

    This is the STARTUP (whole training set) check.  Batch-level missing
    groups are handled by the loss (differentiable zero + skip-and-log), not
    here.

    Args:
        group_ids: Per-row condition ids (full dataset order).
        prefix_ids: Per-row recording-prefix ids.
        target_group: Pre-registered target condition id.
        reference_group: Pre-registered reference condition id.
        min_prefixes_per_group: Minimum available prefixes per group.
        min_trials_per_group: Minimum available masked trials per group.
        mask: Optional per-row observed-latency mask.

    Returns:
        A plain dict of counts and the configured minima.

    Raises:
        ValueError: If a group is missing or below either minimum.
    """
    if isinstance(target_group, bool) or isinstance(reference_group, bool):
        raise ValueError("target/reference group must be non-bool ints")
    if target_group == reference_group:
        raise ValueError("target_group and reference_group must differ")
    if min_prefixes_per_group < 1 or min_trials_per_group < 1:
        raise ValueError("minimum prefix/trial counts must be >= 1")
    groups, prefixes, available = _as_arrays(group_ids, prefix_ids, mask)
    counts: Dict[str, Any] = {}
    problems: List[str] = []
    for name, group in (
        ("target", target_group), ("reference", reference_group),
    ):
        selected = available & (groups == group)
        n_trials = int(selected.sum())
        n_prefixes = int(np.unique(prefixes[selected]).size)
        counts[f"{name}_group"] = int(group)
        counts[f"{name}_trials"] = n_trials
        counts[f"{name}_prefixes"] = n_prefixes
        if n_prefixes < min_prefixes_per_group:
            problems.append(
                f"{name} group {group}: {n_prefixes} prefix(es) < "
                f"min_prefixes_per_group={min_prefixes_per_group}"
            )
        if n_trials < min_trials_per_group:
            problems.append(
                f"{name} group {group}: {n_trials} trial(s) < "
                f"min_trials_per_group={min_trials_per_group}"
            )
    counts["min_prefixes_per_group"] = int(min_prefixes_per_group)
    counts["min_trials_per_group"] = int(min_trials_per_group)
    if problems:
        raise ValueError(
            "latency coverage insufficient (fail closed): "
            + "; ".join(problems)
            + f" — counts={counts}"
        )
    return counts


class LatencyContrastBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Deterministic batch sampler guaranteeing both contrast groups.

    Every batch contains at least ``min_prefixes_per_group`` prefixes and
    ``min_trials_per_group`` trials from each of the target and reference
    groups. Every dataset row appears at least once per epoch, including other
    conditions and masked targets. Rare rows can recur across batches to meet
    the contrast quotas; the resulting oversampling is reported in
    ``last_epoch_stats`` and changes the empirical training weights.

    Args:
        group_ids: Per-row condition ids (full dataset order).
        prefix_ids: Per-row recording-prefix ids.
        batch_size: Batch size; must be
            ``>= 2 * (min_prefixes_per_group + min_trials_per_group)``.
        target_group: Pre-registered target condition id.
        reference_group: Pre-registered reference condition id.
        min_prefixes_per_group: Minimum prefixes per group per batch.
        min_trials_per_group: Minimum trials per group per batch.
        seed: Private sampler seed (does not touch the global RNG).
        epoch: Epoch index; use :meth:`set_epoch` between epochs.
        mask: Optional per-row observed-latency mask.  Masked-out rows never
            satisfy the contrast guarantee (they carry no latency target),
            but they ARE still drawn for the fill slots so they keep
            contributing to the MSE / bio losses (spec §8.5: the sampler must
            not exclude visual_only / no_stimulus trials).
    """

    def __init__(
        self,
        group_ids: Sequence[int],
        prefix_ids: Sequence[int],
        batch_size: int,
        *,
        target_group: int,
        reference_group: int,
        min_prefixes_per_group: int = 2,
        min_trials_per_group: int = 4,
        seed: int = 0,
        epoch: int = 0,
        mask: Optional[Sequence[bool]] = None,
    ) -> None:
        self.groups, self.prefixes, self.available = _as_arrays(
            group_ids, prefix_ids, mask,
        )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError(f"batch_size must be an int, got {batch_size!r}")
        minimum = 2 * (min_prefixes_per_group + min_trials_per_group)
        if batch_size < minimum:
            raise ValueError(
                f"batch_size={batch_size} < 2 * (min_prefixes_per_group + "
                f"min_trials_per_group)={minimum}; fail closed."
            )
        self.batch_size = int(batch_size)
        self.target_group = int(target_group)
        self.reference_group = int(reference_group)
        self.min_prefixes = int(min_prefixes_per_group)
        self.min_trials = int(min_trials_per_group)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.coverage = validate_latency_coverage(
            self.groups, self.prefixes,
            target_group=self.target_group,
            reference_group=self.reference_group,
            min_prefixes_per_group=self.min_prefixes,
            min_trials_per_group=self.min_trials,
            mask=self.available,
        )
        self._available_indices = np.flatnonzero(self.available)
        # Fill pool = ALL rows: masked-out rows (no latency target) still
        # train the primary/bio losses, only the contrast guarantee uses the
        # masked subset (spec §8.5).
        self._by_group: Dict[int, Dict[int, np.ndarray]] = {}
        for group in (self.target_group, self.reference_group):
            mapping: Dict[int, np.ndarray] = {}
            selected = self._available_indices[
                self.groups[self._available_indices] == group
            ]
            for prefix in np.unique(self.prefixes[selected]).tolist():
                mapping[int(prefix)] = selected[
                    self.prefixes[selected] == prefix
                ]
            self._by_group[group] = mapping
        self._target_prefixes = np.asarray(
            sorted(self._by_group[self.target_group]), dtype=np.int64,
        )
        self._reference_prefixes = np.asarray(
            sorted(self._by_group[self.reference_group]), dtype=np.int64,
        )
        self._epoch_batches: Optional[List[List[int]]] = None
        self.last_epoch_stats: Dict[str, Any] = {}

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch (changes the deterministic batch sequence)."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative int, got {epoch!r}")
        self.epoch = int(epoch)
        self._epoch_batches = None

    def __len__(self) -> int:
        return len(self._batches())

    def _batches(self) -> List[List[int]]:
        """Build one exhaustive epoch, repeating only to satisfy constraints.

        Every row is consumed from a shuffled pending pool at least once. Rare
        contrast rows may recur in different batches to maintain the minima;
        this necessary oversampling is reported, never called distribution
        preserving. Within-batch duplication occurs only for tiny datasets.
        """
        if self._epoch_batches is not None:
            return self._epoch_batches
        generator = torch.Generator().manual_seed(
            (self.seed + self.epoch * 1_000_003) % (2**63),
        )
        order = torch.randperm(len(self.groups), generator=generator).tolist()
        pending = dict.fromkeys(order)
        rank = {row: i for i, row in enumerate(order)}
        batches: List[List[int]] = []
        counts = np.zeros(len(self.groups), dtype=np.int64)
        while pending:
            chosen: List[int] = []
            taken: set[int] = set()

            def add(row: int) -> None:
                chosen.append(row)
                taken.add(row)
                pending.pop(row, None)

            for group, mapping in self._by_group.items():
                # First reserve one row from EACH selected prefix. Prefer
                # unseen rows, then rotate previously seen rare prefixes.
                selected = sorted(
                    mapping,
                    key=lambda p: (
                        not any(int(i) in pending for i in mapping[p]),
                        sum(int(counts[i]) for i in mapping[p]), p,
                    ),
                )[:self.min_prefixes]
                for prefix in selected:
                    row = min(
                        (int(i) for i in mapping[prefix]),
                        key=lambda i: (i not in pending, counts[i], rank[i]),
                    )
                    add(row)
                candidates = [
                    int(i) for rows in mapping.values() for i in rows
                    if int(i) not in taken
                ]
                candidates.sort(
                    key=lambda i: (i not in pending, counts[i], rank[i]),
                )
                need = max(0, self.min_trials - len(selected))
                for row in candidates[:need]:
                    add(row)
            # Exhaustive fill, including masked and non-contrast trials.
            for row in list(pending):
                if len(chosen) >= self.batch_size:
                    break
                add(row)
            if len(chosen) < self.batch_size:
                for row in sorted(order, key=lambda i: (counts[i], rank[i])):
                    if len(chosen) >= self.batch_size:
                        break
                    if row not in taken:
                        add(row)
            while len(chosen) < self.batch_size:
                chosen.append(order[len(chosen) % len(order)])
            perm = torch.randperm(len(chosen), generator=generator).tolist()
            batch = [chosen[i] for i in perm]
            for group in (self.target_group, self.reference_group):
                rows = [i for i in batch if self.available[i]
                        and self.groups[i] == group]
                assert len(set(rows)) >= self.min_trials
                assert len(set(self.prefixes[rows])) >= self.min_prefixes
            np.add.at(counts, batch, 1)
            batches.append(batch)
        assert bool((counts > 0).all()), "latency sampler omitted dataset rows"
        self.last_epoch_stats = {
            "unique_trials": int((counts > 0).sum()),
            "draws": int(counts.sum()),
            "repeated_draws": int(counts.sum() - len(counts)),
            "max_trial_draws": int(counts.max()),
            "batches": len(batches),
            "target_prefixes": int(self._target_prefixes.size),
            "reference_prefixes": int(self._reference_prefixes.size),
        }
        self._epoch_batches = batches
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from (list(batch) for batch in self._batches())


def collate_with_latency_metadata(
    batch: Sequence[Tuple[Any, ...]],
    *,
    base_collate: Optional[Callable[[Any], Any]] = None,
) -> Tuple[Any, ...]:
    """Pad a batch that carries per-trial latency metadata.

    Each item is ``(X, Y, length[, wind_only], latency_meta)`` where
    ``latency_meta`` is a dict of Python scalars/tensors.  Returns
    ``(X, Y, lengths, wind_only, meta)`` with ``meta`` a dict of stacked
    tensors (the exact loss keyword names).
    """
    if not batch:
        raise ValueError("collate_with_latency_metadata received an empty batch")
    has_meta = isinstance(batch[0][-1], dict)
    if has_meta:
        metas = [item[-1] for item in batch]
        items = [item[:-1] for item in batch]
    else:
        metas = None
        items = list(batch)
    if base_collate is not None:
        # Preserve the canonical dataset's (idx, X, Y) item contract and its
        # existing padding/condition logic. Sidecars follow the same row order.
        padded = tuple(base_collate(items))
        if len(padded) == 3:
            padded = (*padded, None)
        assert len(padded) == 4, f"Expected 3/4 base fields, got {len(padded)}"
        x_pad, y_pad, lengths, _wind = padded
        b, t, features = x_pad.shape
        assert x_pad.shape == (b, t, features)
        assert y_pad.shape == (b, t)
        assert lengths.shape == (b,)
        if metas is None:
            return padded
        meta = {
            key: torch.stack([torch.as_tensor(m[key]) for m in metas])
            for key in sorted(metas[0])
        }
        for value in meta.values():
            assert value.shape == (b,)
        return (*padded, meta)
    has_wind = len(items[0]) == 4
    x_seqs = [item[0] for item in items]
    y_seqs = [item[1] for item in items]
    lengths = torch.tensor([int(item[2]) for item in items], dtype=torch.long)
    x_pad = torch.nn.utils.rnn.pad_sequence(x_seqs, batch_first=True)
    y_pad = torch.nn.utils.rnn.pad_sequence(y_seqs, batch_first=True)
    wind = (
        torch.tensor([bool(item[3]) for item in items], dtype=torch.bool)
        if has_wind else None
    )
    if metas is None:
        return x_pad, y_pad, lengths, wind
    keys = sorted(metas[0])
    meta = {
        key: torch.as_tensor([m[key] for m in metas])
        for key in keys
    }
    return x_pad, y_pad, lengths, wind, meta


class IndexedDataset(torch.utils.data.Dataset):
    """Attach a per-row latency sidecar to an existing dataset.

    ``__getitem__`` returns the original item with a metadata dict appended.
    The sidecar values are per-dataset-row (already aligned by
    :func:`build_latency_sidecar`).
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        sidecar: Dict[str, Sequence[Any]],
    ) -> None:
        self.dataset = dataset
        self.sidecar = sidecar
        n = len(dataset)
        for key, values in sidecar.items():
            if len(values) != n:
                raise ValueError(
                    f"sidecar[{key!r}] has {len(values)} rows, dataset has {n}"
                )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        """Delegate dataset attributes so wrapping stays transparent.

        Downstream code reads ``is_pure_wind``, ``source_indices``,
        ``sequences`` and provenance attributes off ``loader.dataset``;
        forwarding them keeps the wrapper from changing that contract.
        """
        if name in ("dataset", "sidecar", "__getitems__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        item = self.dataset[index]
        meta = {key: values[index] for key, values in self.sidecar.items()}
        return (*item, meta)
