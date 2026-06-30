import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

KIN_TARGET = ["session_id", "trial_id", "time_ms", "x_pos", "y_pos", "heading", "velocity", "acceleration", "visual_angle", "wind_state", "l_v_ratio"]
EVT_TARGET = ["session_id", "trial_id", "time_ms", "event_type", "event_value"]


def reconstruct_visual_angle(time_ms, lv_ratio_ms, init_deg=2.0):
    """
    Reconstruct visual looming angle θ(t) = 2 × arctan(lv / (TTC - t))

    From the experiment code (paradigm.py):
        lv_s = lv_ratio_ms / 1000.0
        init_rad = math.radians(init_deg / 2)
        t_col = lv_s / math.tan(init_rad)  # TTC in seconds
        theta = math.degrees(2 * math.atan(lv_s / delta))  # delta = t_col - elapsed_s

    Args:
        time_ms: array of timestamps in ms (relative to looming onset)
        lv_ratio_ms: l/v ratio in ms (object size / approach speed)
        init_deg: initial visual angle in degrees (default 2.0)

    Returns:
        visual_angle: array of visual angles in degrees
    """
    visual_angle = np.zeros_like(time_ms, dtype=np.float64)

    if lv_ratio_ms is None or lv_ratio_ms <= 0:
        return visual_angle

    # Compute TTC from experiment parameters (matching paradigm.py)
    lv_s = lv_ratio_ms / 1000.0
    init_rad = math.radians(init_deg / 2)
    tan_val = math.tan(init_rad)
    if tan_val <= 0:
        return visual_angle
    t_col_s = lv_s / tan_val  # TTC in seconds from looming onset
    ttc_ms = t_col_s * 1000.0  # Convert to ms

    # Compute visual angle for all time points where t < TTC
    # time_ms is relative to looming onset (0 = start of looming)
    mask = (time_ms >= 0) & (time_ms < ttc_ms)
    t_s = time_ms[mask] / 1000.0  # Convert to seconds

    # θ(t) = 2 × arctan(lv_s / (t_col - t))
    delta = np.maximum(t_col_s - t_s, 0.001)
    visual_angle[mask] = np.degrees(2.0 * np.arctan(lv_s / delta))

    # Clamp to max_deg
    visual_angle = np.clip(visual_angle, 0.0, 179.0)

    return visual_angle


def parse_trial_events(evt_path):
    """
    Parse events CSV to extract trial types and stimulus parameters.

    Returns:
        dict mapping trial_id -> {
            'type': str,  # 'baseline_wind', 'baseline_visual', 'looming_wind'
            'lv_ratio_ms': float or None,
            'target_ttc_ms': float or None,
            'wind_dir': str or None,
            'stimulus_onset_ms': float or None,
            'looming_onset_ms': float or None,
            'ttc_ms': float or None,  # Time of Collision_TTC0 phase
        }
    """
    df_e = pd.read_csv(evt_path)
    trial_info = {}

    # Normalize column names
    evt_col = 'event_type' if 'event_type' in df_e.columns else 'event_name'
    tid_col = 'trial_id' if 'trial_id' in df_e.columns else 'global_trial_id'
    val_col = 'event_value' if 'event_value' in df_e.columns else 'details'
    ts_col = 'time_ms' if 'time_ms' in df_e.columns else 'timestamp'

    # ── trial_start events: extract trial metadata ──
    ts_mask = df_e[evt_col] == 'trial_start'
    for _, row in df_e[ts_mask].iterrows():
        trial_id = row[tid_col]
        try:
            details = json.loads(str(row[val_col]))
        except Exception:
            details = {}
        trial_info[trial_id] = {
            'type': details.get('type', 'unknown'),
            'lv_ratio_ms': details.get('lv_ratio_ms'),
            'target_ttc_ms': details.get('target_ttc_ms'),
            'wind_dir': details.get('wind_dir', 'none'),
            'stimulus_onset_ms': None,
            'looming_onset_ms': None,
            'ttc_ms': None,
        }

    # ── phase_transition events: vectorized processing ──
    pt_mask = df_e[evt_col] == 'phase_transition'
    if trial_info and pt_mask.any():
        pt_df = df_e.loc[pt_mask, [tid_col, val_col, ts_col]].copy()
        # Only process transitions for known trials
        pt_df = pt_df[pt_df[tid_col].isin(trial_info)]

        # Parse details JSON in batch
        parsed = pt_df[val_col].apply(lambda s: json.loads(str(s)) if pd.notna(s) else {})
        pt_df['from_phase'] = parsed.apply(lambda d: d.get('from_phase', ''))
        pt_df['to_phase'] = parsed.apply(lambda d: d.get('to_phase', ''))

        # Looming onset: to_phase == 'Looming'
        looming = pt_df[pt_df['to_phase'] == 'Looming']
        for _, r in looming.iterrows():
            trial_info[r[tid_col]]['looming_onset_ms'] = float(r[ts_col])

        # TTC: to_phase == 'Collision_TTC0'
        ttc = pt_df[pt_df['to_phase'] == 'Collision_TTC0']
        for _, r in ttc.iterrows():
            trial_info[r[tid_col]]['ttc_ms'] = float(r[ts_col])

        # Wind onset: from_phase == 'Baseline' -> to_phase == 'PostStimulus'
        wind = pt_df[(pt_df['from_phase'] == 'Baseline') & (pt_df['to_phase'] == 'PostStimulus')]
        for _, r in wind.iterrows():
            trial_info[r[tid_col]]['stimulus_onset_ms'] = float(r[ts_col])

    return trial_info


def adapt_cercus_to_nsmor(raw_dir="data/raw"):
    base_path = Path(raw_dir)
    kin_files = list(base_path.rglob("*kinematics*.csv"))

    count_k, count_e = 0, 0
    for kin_path in kin_files:
        session_id = kin_path.parent.name
        evt_path = kin_path.parent / kin_path.name.replace("kinematics", "events")

        # ── 1. Parse trial events for visual stimulus info ──
        trial_info = {}
        if evt_path.exists():
            trial_info = parse_trial_events(evt_path)

        # ── 2. Always re-process kinematics (force overwrite) ──
        df_k = pd.read_csv(kin_path)
        stim_starts = []

        # Check if this is raw sensor data (has sys_time but not x_pos)
        is_raw = "sys_time" in df_k.columns and "x_pos" not in df_k.columns

        if is_raw:
            print(f"正在重构轨迹与相对时间轴: {kin_path.name}")
            df_k["session_id"] = session_id
            df_k["trial_id"] = df_k["global_trial_id"]

            df_k["abs_time"] = df_k["sys_time"] * 1000.0
            df_k["time_ms"] = df_k.groupby("trial_id")["abs_time"].transform(lambda x: x - x.iloc[0])

            df_k["heading"] = np.cumsum(np.degrees(df_k["dz"] / 30.0))
            rad = np.radians(df_k["heading"])
            dx_glob = df_k["dx"] * np.cos(rad) - df_k["dy"] * np.sin(rad)
            dy_glob = df_k["dx"] * np.sin(rad) + df_k["dy"] * np.cos(rad)

            df_k["x_pos"] = np.cumsum(dx_glob) / 10.0
            df_k["y_pos"] = np.cumsum(dy_glob) / 10.0

            # Compute velocity from raw sensor data (matching kinematics.py)
            # velocity = sqrt(dx^2 + dy^2) / dt, converted to cm/s
            # dx, dy are in mm; dt is in seconds from sys_time
            step_dist_mm = np.sqrt(df_k["dx"]**2 + df_k["dy"]**2)
            dt_s = df_k.groupby("trial_id")["abs_time"].transform(
                lambda x: x.diff().fillna(0) / 1000.0
            )
            # Avoid division by zero
            dt_s = dt_s.clip(lower=0.001)
            # velocity in mm/s, then convert to cm/s
            df_k["velocity"] = (step_dist_mm / dt_s) / 10.0

            # Compute acceleration as diff of velocity
            df_k["acceleration"] = df_k.groupby("trial_id")["velocity"].transform(
                lambda x: x.diff().fillna(0)
            )

            df_k["visual_angle"] = 0.0
            df_k["l_v_ratio"] = 0.0
            df_k["wind_state"] = df_k["stim_state"] if "stim_state" in df_k.columns else 0

        # ── 3. Reconstruct visual angle per trial (always run) ──
        print(f"正在重构视觉角度: {kin_path.name}")

        # Pre-build trial parameter mapping for vectorized assignment
        tid_series = df_k["trial_id"]
        unique_tids = tid_series.unique()

        for tid in unique_tids:
            trial_id = int(tid)
            info = trial_info.get(trial_id, {})
            trial_type = info.get('type', 'unknown')
            lv_ratio_ms = info.get('lv_ratio_ms')

            has_visual = trial_type in ('baseline_visual', 'looming_wind')
            has_wind = trial_type in ('baseline_wind', 'looming_wind')

            # Boolean mask for this trial (faster than groupby + loc)
            trial_mask = tid_series == tid
            time_vals = df_k.loc[trial_mask, "time_ms"].values

            if has_visual and lv_ratio_ms and lv_ratio_ms > 0:
                visual_angle = reconstruct_visual_angle(
                    time_vals, lv_ratio_ms, init_deg=2.0,
                )
                df_k.loc[trial_mask, "visual_angle"] = visual_angle
                df_k.loc[trial_mask, "l_v_ratio"] = lv_ratio_ms
            else:
                df_k.loc[trial_mask, "visual_angle"] = 0.0
                df_k.loc[trial_mask, "l_v_ratio"] = 0.0

            if not has_wind:
                df_k.loc[trial_mask, "wind_state"] = 0

            # Detect stimulus onset for events
            if has_wind:
                wind_mask = trial_mask & (df_k["wind_state"] > 0)
                if wind_mask.any():
                    stim_time = df_k.loc[wind_mask, "time_ms"].iloc[0]
                    stim_starts.append({
                        "session_id": session_id,
                        "trial_id": tid,
                        "time_ms": stim_time,
                        "event_type": "stimulus_onset",
                        "event_value": '{"source": "kinematics_injected"}'
                    })
            elif has_visual:
                stim_starts.append({
                    "session_id": session_id,
                    "trial_id": tid,
                    "time_ms": 0.0,
                    "event_type": "stimulus_onset",
                    "event_value": '{"source": "kinematics_injected"}'
                })

        df_k[KIN_TARGET].to_csv(kin_path, index=False)
        count_k += 1

        # ── 2. 事件流净化与合并 ──
        if evt_path.exists():
            df_e = pd.read_csv(evt_path)
            needs_update = False

            # 若还未映射结构
            if "event_type" not in df_e.columns:
                print(f"正在格式化兜底事件流: {evt_path.name}")
                df_e["session_id"] = session_id
                df_e["trial_id"] = df_e["global_trial_id"]
                df_e["abs_time"] = df_e["timestamp"] * 1000.0

                # 修复核心：必须在这里生成兜底的 time_ms
                df_e["time_ms"] = df_e.groupby("trial_id")["abs_time"].transform(lambda x: x - x.min())

                df_e["event_type"] = df_e["event_name"]
                df_e["event_value"] = df_e["details"].fillna("")
                needs_update = True

            # 合并 stimulus_onset 事件（保留原有事件）
            if stim_starts:
                df_stim = pd.DataFrame(stim_starts)
                # 只添加不存在的 stimulus_onset 事件
                existing_stim = df_e[df_e["event_type"] == "stimulus_onset"]
                if len(existing_stim) == 0:
                    df_e = pd.concat([df_e, df_stim], ignore_index=True)
                else:
                    # 替换已有的 stimulus_onset 事件
                    df_e = df_e[df_e["event_type"] != "stimulus_onset"]
                    df_e = pd.concat([df_e, df_stim], ignore_index=True)
                needs_update = True

            if needs_update:
                df_e = df_e[EVT_TARGET]
                df_e.to_csv(evt_path, index=False)
                count_e += 1

    print(f"绝对对齐完毕: 重构 {count_k} 份轨迹, 覆盖 {count_e} 份基准事件。")

if __name__ == "__main__":
    adapt_cercus_to_nsmor()