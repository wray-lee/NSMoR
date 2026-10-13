"""Phase-2 model-internal exploratory secondary battery; CPU, one trial at a time."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from nsmor.analysis.analysis_priors import describe_analysis_population
from nsmor.analysis.model_comparison import paired_mse_comparison
from nsmor.analysis.prediction_units import load_model_from_checkpoint, resolve_dt_ms
from nsmor.config_parser import ExperimentConfig
from nsmor.pipeline.nested_prior import compute_source_fingerprint as _sha, load_artifact_bytes
from scripts.evaluate_realdata_phase2 import _validate_arm_config
from scripts.simulate_lesion import LESION_OVERRIDES, load_dataset

SCORER_SHA256 = "786f0bc80bed8e45e686e61018f5d43825971663e09ff9796c9ed3e8f1a21a9d"
REPLACEMENTS = {"lif_replaced": "lif_lesioned", "gru_replaced": "gru_lesioned"}
WINDOWS = ("onset", "sustained", "overall")


def _finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item)
    elif isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite tensor in input, prediction, or internals")
    elif isinstance(value, (np.ndarray, float)) and not np.isfinite(value).all():
        raise ValueError("Nonfinite analysis value")


def scoring_windows(observed, reference, dt_ms):
    """Full-sequence replacement scoring masks, with no intervention timing."""
    length = len(observed)
    frames = np.arange(length)
    masks = {name: np.zeros(length, dtype=bool) for name in WINDOWS}
    response = None
    if reference is not None:
        if not 0 <= reference < length:
            raise ValueError("Reference outside cropped trial")
        masks["overall"] = frames >= reference
        hits = np.flatnonzero(masks["overall"] & (observed >= 10))
        if hits.size:
            response = int(hits[0])
            boundary = response + int(round(200 / dt_ms))
            masks["onset"] = (frames >= reference) & (frames < boundary)
            masks["sustained"] = (frames >= boundary) & (observed >= 10)
    assert all(mask.shape == (length,) for mask in masks.values())
    return masks, response


def departure_timing(rms, reference, dt_ms):
    """Scale-dependent RMS departure; invalid baselines are never censoring."""
    n = int(round(250 / dt_ms))
    result = {"status": "invalid_baseline", "event_observed": None,
              "right_censored": None, "event_frame": None, "latency_ms": None,
              "baseline_requested_ms": 250, "baseline_n_frames": n,
              "baseline_effective_ms": n * dt_ms, "baseline_available_frames": 0,
              "baseline_mean": None, "baseline_sd": None, "threshold": None,
              "opportunity_end_frame_exclusive": None, "opportunity_end_ms": None}
    _finite(rms)
    if reference is None:
        result["status"] = "invalid_reference"
        return result
    if not 0 <= reference < len(rms):
        raise ValueError("Reference outside timing trajectory")
    result.update(baseline_available_frames=min(reference, n),
                  opportunity_end_frame_exclusive=len(rms),
                  opportunity_end_ms=(len(rms) - reference) * dt_ms)
    if reference < n or n < 2:
        return result
    baseline = rms[reference - n:reference]
    assert baseline.shape == (n,)
    mean, sd = float(np.mean(baseline)), float(np.std(baseline, ddof=0))
    threshold = mean + 3 * max(sd, 1e-6)
    hits = np.flatnonzero(rms[reference:] > threshold)
    event = int(reference + hits[0]) if hits.size else None
    result.update(status="observed" if event is not None else "right_censored",
                  event_observed=event is not None, right_censored=event is None,
                  event_frame=event, latency_ms=(event - reference) * dt_ms if event is not None else None,
                  baseline_mean=mean, baseline_sd=sd, threshold=threshold)
    _finite(result)
    return result


def residual_moments(predicted, observed):
    residual = predicted - observed
    assert residual.shape == predicted.shape == observed.shape and residual.size > 0
    mse, bias, variance = float(np.mean(residual ** 2)), float(np.mean(residual)), float(np.var(residual, ddof=0))
    _finite((mse, bias, variance))
    if not np.isclose(mse, bias ** 2 + variance, rtol=1e-10, atol=1e-12):
        raise ValueError("Residual MSE != bias squared + population variance")
    return mse, bias, variance


def _bout(mask):
    """Two adjacent frames; holes stay in the mask and break runs."""
    return bool(np.any(mask[:-1] & mask[1:]))


def prestim_counts(predicted, observed, reference):
    end = reference if reference is not None else 0
    pred, obs = predicted[:end] >= 10, observed[:end] >= 10
    below = observed[:end] < 10
    observed_bout, predicted_bout = _bout(obs), _bout(pred)
    return {"prestim_frames": end, "prestim_trials": int(end > 0),
            "prestim_observed_escape_frames": int(obs.sum()),
            "prestim_predicted_escape_frames": int(pred.sum()),
            "prestim_observed_bout_trials": int(observed_bout),
            "prestim_predicted_bout_trials": int(predicted_bout),
            "prestim_false_alarm_frames": int((pred & below).sum()),
            "prestim_below_observed_frames": int(below.sum()),
            "prestim_below_observed_predicted_bout_trials": int(_bout(pred & below)),
            "prestim_observed_no_bout_trials": int(end > 0 and not observed_bout),
            "prestim_strict_false_alarm_bout_trials": int(end > 0 and not observed_bout and predicted_bout),
            "prestim_proxy_exactly_zero_frames": int((observed[:end] == 0).sum()),
            "prestim_proxy_0_to_10_frames": int(((observed[:end] > 0) & below).sum()),
            "prestim_proxy_ge10_frames": int(obs.sum())}


def stream_trials(model, loader, references, arm, seed, dt_ms):
    """Yield scalar trial rows; no predictions or trajectories accumulate."""
    model.eval()
    seen = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            xb, yb, lengths = batch[:3]
            ref = references[index]
            length = int(lengths[0])
            assert xb.shape == (1, length, 8) and yb.shape == (1, length)
            assert lengths.shape == (1,)
            if xb.device.type != "cpu" or yb.device.type != "cpu":
                raise ValueError("Secondary battery requires CPU inputs")
            _finite((xb, yb))
            if torch.any(yb < 0):
                raise ValueError("Scalar speed target must be nonnegative")
            if length != ref["cropped_length"]:
                raise ValueError("Loader length disagrees with reference crop")
            if arm == "R2":
                xb = xb.clone()
                xb[:, :, [2, 3]] = 0
                assert xb.shape == (1, length, 8)
            predicted, internals = model(xb, lengths, return_internals=True)
            assert predicted.shape == (1, length)
            _finite((predicted, internals))
            natural = predicted[0].double().numpy()
            observed = yb[0].double().numpy()
            reference = ref["cropped_reference_frame"]
            masks, response = scoring_windows(observed, reference, dt_ms)
            row = dict(ref, arm=arm, seed=seed, response_onset_frame=response)
            for pathway, key in (("lif", "lif_spikes"), ("gru", "gru_hidden")):
                trajectory = internals[key]
                assert trajectory.ndim == 3 and trajectory.shape[:2] == (1, length)
                rms = trajectory[0].double().square().mean(dim=-1).sqrt().numpy()
                assert rms.shape == (length,)
                row.update({f"timing_{pathway}_{key}": value for key, value in
                            departure_timing(rms, reference, dt_ms).items()})
            lif, gru = row["timing_lif_latency_ms"], row["timing_gru_latency_ms"]
            delta = lif - gru if lif is not None and gru is not None else None
            row["timing_lif_minus_gru_ms"] = delta
            row["timing_order"] = ("unavailable" if delta is None else "exact_zero_tie" if delta == 0
                                   else "lif_first" if delta < 0 else "gru_first")
            row.update(prestim_counts(natural, observed, reference))
            for window, mask in masks.items():
                count = int(mask.sum())
                available = count >= (5 if window == "sustained" else 1)
                row[f"{window}_frame_count"] = count
                row[f"{window}_unavailable_reason"] = (None if available else "invalid_reference" if reference is None
                    else "no_observed_responder" if response is None and window != "overall" else "insufficient_eligible_frames")
                mse, bias, variance = residual_moments(natural[mask], observed[mask]) if available else (None, None, None)
                row.update({f"{window}_natural_mse": mse, f"{window}_natural_bias_cm_s": bias,
                            f"{window}_natural_residual_variance_cm_s2": variance})
            for replacement, condition in REPLACEMENTS.items():
                changed, changed_internals = model(xb, lengths, return_internals=True,
                    override_gates=LESION_OVERRIDES[condition])
                assert changed.shape == (1, length)
                _finite((changed, changed_internals))
                changed = changed[0].double().numpy()
                assert changed.shape == (length,)
                for window, mask in masks.items():
                    base = row[f"{window}_natural_mse"]
                    mse = float(np.mean((changed[mask] - observed[mask]) ** 2)) if base is not None else None
                    row[f"{window}_{replacement}_mse"] = mse
                    row[f"{window}_{replacement}_delta_mse"] = mse - base if mse is not None else None
            _finite(row)
            seen += 1
            yield row
    if seen != len(references):
        raise ValueError("Loader and source references differ in trial count")


def _distribution(values):
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {"n": len(values), "median": float(median), "q1": float(q1), "q3": float(q3)}


def summarize(rows):
    timing = {}
    for pathway in ("lif", "gru"):
        eligible = [r for r in rows if r[f"timing_{pathway}_event_observed"] is not None]
        observed = [r[f"timing_{pathway}_latency_ms"] for r in eligible if r[f"timing_{pathway}_event_observed"]]
        censored = sum(r[f"timing_{pathway}_right_censored"] for r in eligible)
        timing[pathway] = dict(_distribution(observed), eligible_trials=len(eligible),
            invalid_trials=len(rows) - len(eligible),
            invalid_baseline_trials=sum(r[f"timing_{pathway}_status"] == "invalid_baseline" for r in rows),
            invalid_reference_trials=sum(r[f"timing_{pathway}_status"] == "invalid_reference" for r in rows),
            censored_trials=censored,
            censor_rate=censored / len(eligible) if eligible else None)
    paired = [r["timing_lif_minus_gru_ms"] for r in rows if r["timing_lif_minus_gru_ms"] is not None]
    timing["common_uncensored_lif_minus_gru_ms"] = dict(_distribution(paired),
        exact_zero_ties=sum(d == 0 for d in paired), lif_first=sum(d < 0 for d in paired),
        gru_first=sum(d > 0 for d in paired), interpretation="descriptive common-uncensored pairs only; RMS-scale dependent")
    effects = {}
    for window in WINDOWS:
        valid = [r for r in rows if r[f"{window}_natural_mse"] is not None]
        effects[window] = {}
        for replacement in REPLACEMENTS:
            effects[window][replacement] = paired_mse_comparison(
                candidate_mse=[r[f"{window}_{replacement}_mse"] for r in valid],
                comparator_mse=[r[f"{window}_natural_mse"] for r in valid],
                trial_ids=[f"{r['session_id']}:{r['source_trial_id']}" for r in valid],
                prefix_ids=[r["recording_prefix"] for r in valid],
                frame_counts=[r[f"{window}_frame_count"] for r in valid],
                exchangeability_asserted=False) if valid else {"available": False, "reason": "no_eligible_trials"}
    counts = {key: sum(r[key] for r in rows) for key in rows[0] if key.startswith("prestim_")}
    rates = {}
    for name, numerator, denominator in (
        ("observed_escape_frame_rate", "observed_escape_frames", "frames"),
        ("predicted_escape_frame_rate", "predicted_escape_frames", "frames"),
        ("observed_bout_trial_rate", "observed_bout_trials", "trials"),
        ("predicted_bout_trial_rate_all_prestim", "predicted_bout_trials", "trials"),
        ("false_alarm_frame_rate", "false_alarm_frames", "below_observed_frames"),
        ("strict_observed_no_bout_false_alarm_trial_rate", "strict_false_alarm_bout_trials", "observed_no_bout_trials")):
        num, den = counts[f"prestim_{numerator}"], counts[f"prestim_{denominator}"]
        rates[name] = {"numerator": num, "denominator": den, "value": num / den if den else None}
    return {"timing": timing, "readout_window_effects": effects, "prestim_counts": counts,
            "prestim_rates": rates, "by_condition": {"available": False,
                "reason": "validated reference helper does not expose independently trustworthy condition metadata"}}


def _figure(rows, summary, output_dir, arm, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    colors = ("#0072B2", "#D55E00")
    for i, window in enumerate(WINDOWS):
        for j, replacement in enumerate(REPLACEMENTS):
            result = summary["readout_window_effects"][window][replacement]
            x = i + (j - 0.5) * 0.22
            values = list(result.get("prefix_means", {}).values())
            if values:
                axes[0].scatter([x] * len(values), values, s=14, alpha=0.4, color=colors[j])
                axes[0].scatter(x, result["cluster_statistic"], marker="D", color=colors[j],
                                label=replacement.replace("_", " ") if i == 0 else None)
            else:
                axes[0].text(x, 0.5, "NA", transform=axes[0].get_xaxis_transform(), ha="center")
    axes[0].axhline(0, color="grey", linewidth=0.6)
    axes[0].set(xticks=range(3), xticklabels=WINDOWS, ylabel="Replacement − natural MSE (cm/s)²",
                title="Full-sequence readout replacements\nWindowed scoring; prefix means")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)
    for j, pathway in enumerate(("lif", "gru")):
        eligible = [r for r in rows if r[f"timing_{pathway}_event_observed"] is not None]
        times = sorted(r[f"timing_{pathway}_latency_ms"] for r in eligible if r[f"timing_{pathway}_event_observed"])
        stats = summary["timing"][pathway]
        if eligible:
            end = max(r[f"timing_{pathway}_opportunity_end_ms"] for r in eligible)
            axes[1].step([0, *times, end], [0, *[k / len(eligible) for k in range(1, len(times) + 1)],
                            len(times) / len(eligible)], where="post", color=colors[j],
                            label=f"{pathway.upper()}: censored {stats['censor_rate']:.1%}; invalid {stats['invalid_trials']}")
        else:
            axes[1].text(0.03, 0.9 - j * 0.25, f"{pathway.upper()}: NA (invalid baseline/reference)", transform=axes[1].transAxes)
        text = f"{pathway.upper()} observed median/IQR: "
        text += f"{stats['median']:g} / [{stats['q1']:g}, {stats['q3']:g}] ms" if times else "NA"
        axes[1].text(0.03, 0.65 - j * 0.1, text, transform=axes[1].transAxes, fontsize=8)
    axes[1].set(xlabel="RMS departure latency (ms)", ylabel="Events / all eligible timing trials", ylim=(0, 1.05),
                title="Defective CDF, varying available followup\nRMS-scale dependent; not Kaplan–Meier")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=8, loc="upper right")
    keys = list(summary["prestim_rates"])
    labels = ("Obs. frame", "Model frame", "Obs. bout", "Model bout", "Frame FA", "Strict trial FA")
    for i, key in enumerate(keys):
        value = summary["prestim_rates"][key]["value"]
        if value is None:
            axes[2].text(i, 0.5, "NA", ha="center")
        else:
            axes[2].bar(i, value, color=colors[i % 2])
    axes[2].set(xticks=range(6), xticklabels=labels, ylabel="Rate (denominators in JSON)", ylim=(0, 1.05),
                title="Natural prestimulus observed vs model\nSpeed proxies only; annotated states unavailable")
    axes[2].tick_params(axis="x", rotation=35)
    fig.suptitle(f"{arm}, seed {seed} — model-internal exploratory; recording prefixes are not animals")
    try:
        for extension in ("pdf", "svg", "png"):
            with (output_dir / f"secondary_battery.{extension}").open("xb") as stream:
                fig.savefig(stream, format=extension, dpi=300)
    finally:
        plt.close(fig)


def run_battery(checkpoint, arm, dataset, nested_prior, output_dir):
    checkpoint, dataset, nested_prior, output_dir = map(Path, (checkpoint, dataset, nested_prior, output_dir))
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Refusing nonempty output directory or overwrite")
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[2]
    if _sha(repo / "scripts/evaluate_realdata_phase2.py") != SCORER_SHA256:
        raise ValueError("Frozen phase-2 scorer SHA mismatch")
    payload = checkpoint.read_bytes()
    ckpt = load_artifact_bytes(payload, map_location="cpu")
    channels = ckpt.get("zero_input_channels")
    if not isinstance(channels, list) or any(type(c) is not int for c in channels):
        raise ValueError("Exact integer zero_input_channels required")
    _validate_arm_config(ckpt, arm)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    seed = ckpt["config"].get("training", {}).get("random_seed")
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError("Exact training seed 42/43/44 required")
    if (ckpt.get("is_nested_cv") is not True or ckpt.get("animal_identity_status") != "unverified"
            or ckpt.get("validation_scope") != "nested_outer_validation"):
        raise ValueError("Require unverified prefixes and nested_outer_validation")
    if cfg.model.refinement_mode != "off" or cfg.model.time_consuming_mode != "off":
        raise ValueError("Adaptive/refinement/time-delay modes are incompatible")
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    if model.analysis_checkpoint_sha256 != hashlib.sha256(payload).hexdigest():
        raise ValueError("Checkpoint changed while loading")
    dt_ms = resolve_dt_ms(model, requested=4.0)
    _finite(model.state_dict())
    prior_sha = _sha(nested_prior)
    loader, _, _ = load_dataset(dataset, batch_size=1, max_seq_len=cfg.training.max_seq_len,
        pre_anchor_frames=1200, nested_prior_artifact=nested_prior, checkpoint_model=model)
    source = loader.dataset
    val_indices = source.analysis_population["outer_val_indices"]
    if not val_indices:
        raise ValueError("Empty or missing nested validation indices")
    if prior_sha != ckpt.get("nested_prior_artifact_sha256"):
        raise ValueError("Nested prior SHA-256 differs from checkpoint")
    references = [source.analysis_reference_trials[i] for i in val_indices]
    if any(r["source_trial_id"] is None or not r["recording_prefix"] or not r["session_id"] for r in references):
        raise ValueError("Require source trial IDs and recording prefixes")
    selected = DataLoader(Subset(source, val_indices), batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=loader.collate_fn)
    population = describe_analysis_population(len(source), val_indices, val_indices, "nested_outer_validation")
    rows = list(stream_trials(model, selected, references, arm, seed, dt_ms))
    summary = summarize(rows)
    code_paths = ("nsmor/analysis/secondary_battery.py", "scripts/analyze_phase2_secondary.py",
                  "scripts/evaluate_realdata_phase2.py", "scripts/simulate_lesion.py",
                  "nsmor/analysis/prediction_units.py", "nsmor/analysis/analysis_priors.py",
                  "nsmor/pipeline/nested_prior.py", "nsmor/pipeline/conditions.py",
                  "nsmor/nsmor_dataloader.py", "nsmor/model_nsmor_core.py", "nsmor/analysis/model_comparison.py")
    summary.update(arm=arm, seed=seed, analysis_claim="model-internal exploratory", population=population,
        config=ckpt["config"], zero_input_channels=ckpt["zero_input_channels"],
        provenance={"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": model.analysis_checkpoint_sha256,
            "dataset": str(dataset.resolve()), "dataset_sha256": ckpt["dataset_source_sha256"],
            "nested_prior": str(nested_prior.resolve()), "nested_prior_sha256": prior_sha,
            "checkpoint_lineage": model.analysis_checkpoint_lineage,
            "code_sha256": {p: _sha(repo / p) for p in code_paths},
            "protocol_sha256": _sha(repo / "docs/realdata-phase2-protocol-20261010.md"),
            "a1_reference_sha256": _sha(repo / "config/realdata-preliminary-20261009/A1-swiglu-off.yaml"),
            "device": "cpu", "batch_size": 1, "torch_threads": 1, "num_workers": 0},
        definitions={"dt_ms": dt_ms, "crop": {"max_seq_len": cfg.training.max_seq_len, "pre_anchor_frames": 1200},
            "readout": "full-sequence gate replacements; score same trials/frames; no state lesions or intervention timing",
            "replacement_gates": {name: LESION_OVERRIDES[key] for name, key in REPLACEMENTS.items()},
            "windows": "r=first observed>=10 at/after cropped reference; onset=[ref,r+200ms); sustained=[r+200ms,L) & observed>=10, >=5 frames; overall=[ref,L); no responder: onset/sustained unavailable",
            "natural_residual_moments": "predicted-observed bias (cm/s), population variance and MSE ((cm/s)^2), MSE=bias^2+variance checked per window; no across-seed bias/variance or calibration claim",
            "prediction_units": "scalar physical cm/s, signed predicted>=10; targets nonnegative, raw/unclipped",
            "timing": "natural lif_spikes/gru_hidden RMS; first > baseline mean+3*max(population SD,1e-6); full available tail; scale dependent",
            "baseline_requested_ms": 250, "baseline_n_frames": int(round(250 / dt_ms)),
            "baseline_effective_ms": int(round(250 / dt_ms)) * dt_ms,
            "timing_censor": "valid complete baseline, no departure through last frame; opportunity end exclusive; invalid baseline/reference excluded from timing denominator",
            "timing_curve": "defective CDF with all eligible trials denominator and varying followup; not KM or common-followup survival",
            "effect_size": "trial d_z and prefix trial-mean deltas; unweighted prefix mean cluster statistic; prefixes are not animals; no new p-values",
            "paired_timing": "common-uncensored lif minus gru only; exact zero tie; no full-population ordering",
            "prestim": "[0,ref); >=10 frame and >=2 adjacent-frame bout; holes break runs; strict trial FA conditional on no observed bout; frame FA conditional on observed<10",
            "speed_proxy": "exactly0, 0<speed<10, >=10 frame counts; NOT resting/grooming/walking states",
            "independently_annotated_state_control": "unavailable", "exchangeability_asserted": False})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trial_level.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _figure(rows, summary, output_dir, arm, seed)
    with (output_dir / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary
