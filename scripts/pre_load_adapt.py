import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

KIN_TARGET = ["session_id", "trial_id", "time_ms", "x_pos", "y_pos", "heading", "velocity", "acceleration", "visual_angle", "wind_state", "l_v_ratio"]
EVT_TARGET = ["session_id", "trial_id", "time_ms", "event_type", "event_value"]

def adapt_cercus_to_nsmor(raw_dir="data/raw"):
    base_path = Path(raw_dir)
    kin_files = list(base_path.rglob("*kinematics*.csv"))

    count_k, count_e = 0, 0
    for kin_path in kin_files:
        session_id = kin_path.parent.name
        evt_path = kin_path.parent / kin_path.name.replace("kinematics", "events")

        # ── 1. 轨迹映射与绝对相对化 ──
        df_k = pd.read_csv(kin_path)
        stim_starts = []

        if "x_pos" not in df_k.columns:
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

            df_k["velocity"] = 0.0
            df_k["acceleration"] = 0.0
            df_k["visual_angle"] = 0.0
            df_k["l_v_ratio"] = 0.0
            df_k["wind_state"] = df_k["stim_state"] if "stim_state" in df_k.columns else 0

            for tid, group in df_k.groupby("trial_id"):
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

            df_k[KIN_TARGET].to_csv(kin_path, index=False)
            count_k += 1

        # ── 2. 事件流净化与强制替换 ──
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

            # 若存在物理注入点，直接整体覆盖
            if stim_starts:
                df_e = pd.DataFrame(stim_starts)
                needs_update = True

            if needs_update:
                df_e = df_e[EVT_TARGET]
                df_e.to_csv(evt_path, index=False)
                count_e += 1

    print(f"绝对对齐完毕: 重构 {count_k} 份轨迹, 覆盖 {count_e} 份基准事件。")

if __name__ == "__main__":
    adapt_cercus_to_nsmor()