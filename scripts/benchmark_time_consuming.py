"""Synthetic CPU timing, tensor-memory profiling and RSS for ADR 0012.

Representative defaults: B=64, T=1500, H=64, one warm-up + five timed runs.
Run each configuration in a fresh process when comparing peak RSS. This is
engineering cost, not measured ATP or biological work. No dataset is loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsmor.analysis.recursion_resources import require_memory_budget
from nsmor.model_nsmor_core import TimeConsumingRecursion

_MODES = ("depth_delay", "accumulate", "stochastic", "hard_depth")


def _cpu_model() -> str:
    """Resolve the CPU model under WSL, where platform.processor is empty."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def _current_rss_bytes() -> int:
    """Return current resident bytes from Linux procfs without dependencies."""
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * int(os.sysconf("SC_PAGE_SIZE"))


def _peak_rss_mb() -> float:
    """Return process lifetime peak RSS in MiB (WSL/Linux)."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _memory_events(trace_path: Path) -> List[tuple[float, int]]:
    """Read allocator instants without loading a large operator trace in RAM."""
    decoder = json.JSONDecoder()
    events: List[tuple[float, int]] = []
    with trace_path.open() as stream:
        buffer = ""
        while '"traceEvents":' not in buffer:
            chunk = stream.read(65536)
            if not chunk:
                raise ValueError("Profiler trace has no traceEvents array")
            buffer += chunk
        buffer = buffer.split('"traceEvents":', 1)[1].lstrip()
        if not buffer.startswith("["):
            raise ValueError("Profiler traceEvents must be an array")
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip(" \r\n\t,")
            if buffer.startswith("]"):
                break
            try:
                event, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(65536)
                if not chunk:
                    raise ValueError("Incomplete profiler trace") from None
                buffer += chunk
                continue
            if event.get("name") == "[memory]":
                events.append((float(event["ts"]), int(event["args"]["Bytes"])))
            buffer = buffer[end:]
    return events


def _profile_step(step: Callable[[], float]) -> Dict[str, Any]:
    """Measure CPU allocation events and sampled RSS on a backward step."""
    baseline = _current_rss_bytes()
    rss_samples = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.005):
            rss_samples.append(_current_rss_bytes())

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        experimental = None
        try:
            experimental = torch._C._profiler._ExperimentalConfig(trace_only=True)
        except TypeError:
            pass  # Older torch: parsed events consume more profiler overhead.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True, experimental_config=experimental,
        ) as profiler:
            step()
        rss_samples.append(_current_rss_bytes())
    finally:
        stop.set()
        worker.join()
    # The parsed FunctionEvent list folds allocations into operator events;
    # only the raw trace retains ALL positive/negative allocator instants.
    # Nested op totals double count memory and must not be summed.
    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "memory.json"
        profiler.export_chrome_trace(str(trace_path))
        events = _memory_events(trace_path)
    running = peak = allocated = freed = 0
    for _timestamp, delta in sorted(events):
        allocated += max(delta, 0)
        freed += max(-delta, 0)
        running += delta
        peak = max(peak, running)
    return {
        "rss_baseline_mb": baseline / 1024**2,
        "profiled_rss_peak_mb": max(rss_samples) / 1024**2,
        "profiled_rss_delta_mb": (max(rss_samples) - baseline) / 1024**2,
        "profiler_tensor_peak_delta_mb": peak / 1024**2,
        "profiler_allocated_mb": allocated / 1024**2,
        "profiler_freed_mb": freed / 1024**2,
        "profiler_memory_events": len(events),
    }


def benchmark(
    mode: str,
    *,
    K: int = 8,
    hidden_dim: int = 64,
    batch: int = 64,
    timesteps: int = 1500,
    repeats: int = 5,
    grad_checkpointing: bool = False,
    profile_memory: bool = True,
    budget_mb: float = 8192.0,
) -> Dict[str, Any]:
    """Measure synthetic forward/backward time, saved tensors and peak RSS.

    Args:
        mode: Enabled recursion mode.
        K: Maximum number of block evaluations per valid frame.
        hidden_dim: Block width.
        batch: Trials in a batch.
        timesteps: Valid frames per trial.
        repeats: Timed runs, excluding one warm-up.
        grad_checkpointing: Recompute block activations during backward.
        profile_memory: Capture allocator events and sampled RSS separately.
        budget_mb: Declared incremental memory budget in MiB.

    Returns:
        Timing, memory, hardware and numerical diagnostics.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if repeats < 2:
        raise ValueError("repeats must be >= 2, excluding warm-up")
    estimate = require_memory_budget(
        batch=batch, timesteps=timesteps, hidden_dim=hidden_dim, K=K,
        grad_checkpointing=grad_checkpointing, budget_mb=budget_mb,
    )
    torch.manual_seed(0)
    baseline_rss = _current_rss_bytes()
    module = TimeConsumingRecursion(
        hidden_dim, mode=mode, max_steps=K, delay_scale=10.0,
        grad_checkpointing=grad_checkpointing,
    ).train()
    generator = torch.Generator().manual_seed(0)
    h = torch.randn(batch, timesteps, hidden_dim, generator=generator)
    valid = torch.ones(batch, timesteps, dtype=torch.bool)
    assert h.shape == (batch, timesteps, hidden_dim)
    assert valid.shape == (batch, timesteps)

    def forward() -> float:
        with torch.no_grad():
            out = module(h, valid)[0]
            return float(out.abs().mean())

    def step() -> float:
        module.zero_grad(set_to_none=True)
        out = module(h, valid)[0]
        loss = out.square().mean()
        loss.backward()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite benchmark loss")
        for parameter in module.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError("Nonfinite benchmark gradient")
        return float(loss.detach())

    def timed(fn: Callable[[], float]) -> List[float]:
        fn()
        times: List[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            value = fn()
            times.append(time.perf_counter() - start)
            assert math.isfinite(value)
        return times

    forward_times = timed(forward)
    fb_times = timed(step)
    # Save-hook bytes measure aggregate saved tensor slots, not peak live
    # allocator bytes. Parameters/aliases are included; the metric is labeled.
    saved_bytes = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss_value = step()
    timed_peak = _peak_rss_mb()
    measured_delta = timed_peak - baseline_rss / 1024**2
    memory = _profile_step(step) if profile_memory else {}

    def summary(times: List[float]) -> Dict[str, float]:
        values = torch.tensor(times, dtype=torch.float64)
        return {
            "mean_s": float(values.mean()),
            "sd_s": float(values.std(unbiased=True)),
            "median_s": float(values.median()),
            "iqr_s": float(torch.quantile(values, 0.75)
                           - torch.quantile(values, 0.25)),
            "ms_per_valid_frame": float(values.median()) * 1000 / (batch * timesteps),
        }

    return {
        "mode": mode, "K": K, "hidden_dim": hidden_dim,
        "batch": batch, "timesteps": timesteps,
        "grad_checkpointing": grad_checkpointing, "repeats": repeats,
        "warmup": 1, "forward_only": summary(forward_times),
        "forward_backward": summary(fb_times),
        "peak_rss_mb": _peak_rss_mb(),
        "timed_peak_rss_mb": timed_peak,
        "timed_rss_delta_mb": measured_delta,
        "saved_tensor_slots_mb": saved_bytes / 1024**2,
        "measured_saved_slots_c": (
            saved_bytes / (batch * timesteps * K * hidden_dim * 4)
        ),
        "memory_estimate": estimate, "memory_profile": memory,
        "within_rss_budget": measured_delta <= budget_mb,
        "loss": loss_value, "synthetic_data_only": True,
        "hardware": {
            "cpu": _cpu_model(), "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a configuration; fresh processes isolate RSS comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="stochastic")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--no-profile-memory", action="store_true")
    parser.add_argument("--budget-mb", type=float, default=8192.0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    reports = [benchmark(
        mode.strip(), K=args.K, hidden_dim=args.hidden_dim, batch=args.batch,
        timesteps=args.timesteps, repeats=args.repeats,
        grad_checkpointing=args.grad_checkpointing,
        profile_memory=not args.no_profile_memory, budget_mb=args.budget_mb,
    ) for mode in args.modes.split(",") if mode.strip()]
    text = json.dumps(reports, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0 if all(r["within_rss_budget"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
