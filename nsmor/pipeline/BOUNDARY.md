# BOUNDARY — `nsmor/pipeline/` (Data Pipeline)

## Status: 🟢 EXTENDABLE

This directory handles the transformation from **raw CSV data** to **PyTorch DataLoaders**. Safe to extend with new extraction functions or data sources.

---

## Purpose

Convert raw experimental data (kinematics CSVs, events CSVs) into structured tensors for the NSMoR model.

**Data Flow:**

```
Raw CSVs (kinematics + events)
    ↓
load_and_concat_sessions()        →  pd.DataFrame
    ↓
extract_trial_data()              →  Dict per trial
    ↓
assign_ground_truth_labels()      →  Label enum
    ↓
extract_mcmc_snapshot()           →  [5] vector at TTC-50ms
extract_trial_sequence()          →  (X_seq [T,8], Y_seq [T])
    ↓
build_sequence_dataset()          →  List[(X_seq, Y_seq, label)]
    ↓
create_dataloader()               →  DataLoader yielding (X_batch, Y_batch, lengths)
```

---

## Input/Output Contract

### `io.py` — CSV Loading

```python
load_and_concat_sessions(
    kinematics_paths: List[Path],
    events_paths: List[Path],
) -> Dict[str, pd.DataFrame]
```

### `kinematics.py` — Smoothing

```python
smooth_kinematics(
    trial_data: Dict,
    window_length: int = 11,
    polyorder: int = 3,
) -> Dict
```

### `labeling.py` — Ground Truth

```python
assign_ground_truth_labels(
    trials: List[Dict],
    config: ThresholdConfig = DEFAULT_THRESHOLD,
) -> List[Dict]  # each dict has "label": Label enum
```

### `data_extractor.py` — Feature Extraction

```python
extract_mcmc_snapshot(
    trial: Dict,
    stimulus_onset_ms: float,
) -> np.ndarray  # shape (5,)

extract_trial_sequence(
    trial: Dict,
) -> Tuple[np.ndarray, np.ndarray]  # (X_seq [T,8], Y_seq [T])
```

---

## Modification Rules

1. **DO** add new extraction functions for new data sources.
2. **DO** add new smoothing or filtering methods.
3. **DO NOT** change the output shape of existing functions without updating downstream code.
4. **DO NOT** modify the feature layout (indices 0-7) without updating `config.py` and all consumers.
5. **ALWAYS** maintain shape assertions in extraction functions.

---

## Extension Points

- Add new event types in `io.py`
- Add new labeling strategies in `labeling.py`
- Add new feature extraction methods in `data_extractor.py`
- Add new data augmentation in `kinematics.py`
