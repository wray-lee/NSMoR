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

    for _, row in df_e.iterrows():
        # Handle both column name formats
        event_name = str(row.get('event_type', row.get('event_name', '')))
        trial_id = row.get('trial_id', row.get('global_trial_id', None))
        details_str = str(row.get('event_value', row.get('details', '{}')))
        timestamp_ms = float(row.get('time_ms', row.get('timestamp', 0)))

        if event_name == 'trial_start' and trial_id is not None:
            try:
                details = json.loads(details_str)
            except:
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

        # Track phase transitions for stimulus timing
        if event_name == 'phase_transition' and trial_id in trial_info:
            try:
                details = json.loads(details_str)
            except:
                details = {}

            from_phase = details.get('from_phase', '')
            to_phase = details.get('to_phase', '')

            # Looming onset: transition TO Looming phase
            if to_phase == 'Looming':
                trial_info[trial_id]['looming_onset_ms'] = timestamp_ms

            # TTC: transition TO Collision_TTC0 phase
            if to_phase == 'Collision_TTC0':
                trial_info[trial_id]['ttc_ms'] = timestamp_ms

            # Wind onset: transition from Baseline to PostStimulus (for wind trials)
            if from_phase == 'Baseline' and to_phase == 'PostStimulus':
                trial_info[trial_id]['stimulus_onset_ms'] = timestamp_ms

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
        for tid, group in df_k.groupby("trial_id"):
            trial_id = int(tid)
            info = trial_info.get(trial_id, {})
            trial_type = info.get('type', 'unknown')
            lv_ratio_ms = info.get('lv_ratio_ms')

            # Determine visual stimulus parameters
            has_visual = trial_type in ('baseline_visual', 'looming_wind')
            has_wind = trial_type in ('baseline_wind', 'looming_wind')

            if has_visual and lv_ratio_ms and lv_ratio_ms > 0:
                # Reconstruct visual angle using experiment parameters
                # time_ms is relative to trial start; looming starts at trial start
                visual_angle = reconstruct_visual_angle(
                    group["time_ms"].values,
                    lv_ratio_ms,
                    init_deg=2.0,  # Default from paradigm.py
                )
                df_k.loc[group.index, "visual_angle"] = visual_angle
                df_k.loc[group.index, "l_v_ratio"] = lv_ratio_ms
            else:
                # No visual stimulus
                df_k.loc[group.index, "visual_angle"] = 0.0
                df_k.loc[group.index, "l_v_ratio"] = 0.0

            # Set wind state from trial type (overwrite raw sensor data)
            if has_wind:
                # Keep wind_state from raw sensor data (already set)
                pass
            else:
                # No wind stimulus for this trial type
                df_k.loc[group.index, "wind_state"] = 0

            # Detect stimulus onset for events
            # For wind trials: use wind onset time
            # For visual-only trials: use looming onset time (trial start)
            if has_wind:
                mask = group["wind_state"] > 0
                if mask.any():
                    stim_time = group.loc[mask, "time_ms"].iloc[0]
                    stim_starts.append({
                        "session_id": session_id,
                        "trial_id": tid,
                        "time_ms": stim_time,
                        "event_type": "stimulus_onset",
                        "event_value": '{"source": "kinematics_injected"}'
                    })
            elif has_visual:
                # Baseline visual: stimulus onset at trial start (looming begins immediately)
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