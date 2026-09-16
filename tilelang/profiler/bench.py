"""Common entry point for GPU and wall-clock benchmarking."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from math import isfinite
from typing import Literal

import torch
import tilelang
import tilelang.language as T

from tilelang.utils.device import IS_CUDA, IS_NPU, Event, device_synchronize

from .torch_bench import (
    _CACHE_FLUSH_ID as _CACHE_FLUSH_ID,
    _cuda_synchronize as _cuda_synchronize,
    bench_with_cuda_events as _bench_with_cuda_events,
    bench_with_cudagraph as _bench_with_cudagraph,
    bench_with_cupti as _bench_with_cupti,
    suppress_stdout_stderr as suppress_stdout_stderr,
)
from .wall import bench_with_wall


@tilelang.jit(out_idx=[], target="ascend")
def get_msprof_cache_flush_kernel(numel: int, dtype: T.dtype):
    # Ascend-only helper: T.SimtVF and l2_cache_ctrl="WTS_FV" are Ascend dialect
    # extensions, so they come from tilelang.ascend.language rather than from the
    # default facade (which is the CUDA dialect).
    from tilelang.ascend import language as TA

    tile_elems = (32 * 1024) // (4 if dtype == "int32" else 1)
    tiles_per_core = numel // tile_elems // 64

    @TA.prim_func
    def _tilelang_profiler_cache_flush(Cache: TA.Tensor((numel,), dtype)):
        with TA.Kernel(64) as bx:
            ub = TA.alloc_shared((tile_elems,), dtype)

            with TA.SimtVF(threads=2048):
                for i in TA.Parallel(tile_elems):
                    ub[i] = TA.cast(0, dtype)

            for it in TA.serial(tiles_per_core):
                begin = (it * 64 + bx) * tile_elems
                if begin < numel:
                    TA.copy(ub, Cache[begin : begin + tile_elems], l2_cache_ctrl="WTS_FV")

    return _tilelang_profiler_cache_flush


logger = logging.getLogger(__name__)

device = "cuda:0" if IS_CUDA else "npu" if IS_NPU else "mps:0"


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    quantiles: list[float] | None = None,
    fast_flush: bool = True,
    backend: Literal["event", "cupti", "cudagraph", "wall", "msprof", "msprof_detail"] = "event",
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
    device: int | torch.device | None = None,
    cache_size: int = 256,
    early_stop_baseline: float | None = None,
) -> float | list[float]:
    """Benchmark a callable with GPU timing or host wall-clock timing.

    The existing GPU timing methods provide accurate kernel timing by:
    - Clearing L2 cache between runs for consistent measurements
    - Auto-calculating warmup and repeat counts based on kernel runtime
    - Supporting multiple profiling backends (CUDA/MPS events, CUPTI, or CUDA graph replay)
    - Offering flexible result aggregation (mean/median/min/max/quantiles)

    Wall timing measures host elapsed time without cache flushing. With no
    device or a CPU device, the callable is assumed to be synchronous. An
    explicit CUDA/HIP, MPS, or NPU device enables synchronization before and after
    each wall-clock sample, including launch and synchronization overhead.

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds (default: 25)
        rep: Target total benchmark time in milliseconds (default: 100)
        _n_warmup: Manual override for warmup iterations (default: 0 = auto)
        _n_repeat: Manual override for benchmark iterations (default: 0 = auto)
        quantiles: Performance percentiles to compute (e.g., [0.5, 0.95])
        fast_flush: Use faster GPU L2 cache flush with int32 vs int8 (default: True); ignored by "wall"
        backend: Timing method - "event", "cupti", "cudagraph", "wall", "msprof", or "msprof_detail" (default: "event")
        return_mode: Result aggregation method - "mean", "median", "min", or "max"
        device: Optional device to benchmark on. CUDA/HIP events, streams,
            cache buffers, and synchronization are scoped to that device.
            Event timing also accepts MPS and NPU; wall timing accepts CPU, MPS, and NPU.
            Integer device indices select CUDA/HIP; use an NPU device object for NPU selection.
        cache_size: GPU L2 cache flush buffer size in MB (default: 256); ignored by "wall"

    Returns:
        Runtime in milliseconds (float) or list of quantile values if quantiles specified
    """
    if backend == "wall":
        if return_mode not in ("min", "max", "mean", "median"):
            raise ValueError(f"Invalid return_mode: {return_mode}")
        if any(not isfinite(duration) or duration < 0 for duration in (warmup, rep)):
            raise ValueError("warmup and rep must be finite, non-negative millisecond budgets")
        if _n_warmup < 0 or _n_repeat < 0:
            raise ValueError("Iteration overrides must be non-negative")
        if quantiles is not None and any(not 0 <= quantile <= 1 for quantile in quantiles):
            raise ValueError("quantiles must be between 0 and 1")

        resolved_device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device) if device is not None else None
        device_context = nullcontext()
        synchronize = None
        if resolved_device is not None:
            if resolved_device.type == "cuda":
                device_context = torch.cuda.device(resolved_device)
            elif resolved_device.type == "npu":
                device_context = torch.npu.device(resolved_device)
            elif resolved_device.type not in ("cpu", "mps"):
                raise ValueError(f"Wall timing supports CPU, CUDA/HIP, MPS, or NPU devices, got {resolved_device}")
            if resolved_device.type in ("cuda", "mps", "npu"):
                synchronize = partial(device_synchronize, resolved_device)

        with device_context:
            fn()
            if synchronize is not None:
                synchronize()

            estimate_ms = bench_with_wall(fn, n_repeat=5, synchronize=synchronize)
            if early_stop_baseline is not None and estimate_ms > early_stop_baseline:
                if quantiles is not None:
                    return estimate_ms if len(quantiles) == 1 else [estimate_ms] * len(quantiles)
                return estimate_ms

            estimate_ms = max(estimate_ms, 1e-6)
            n_warmup = _n_warmup if _n_warmup > 0 else int(warmup / estimate_ms)
            n_repeat = _n_repeat if _n_repeat > 0 else max(1, int(rep / estimate_ms))

            for _ in range(n_warmup):
                fn()
            if synchronize is not None:
                synchronize()

            return bench_with_wall(fn, n_repeat=n_repeat, quantiles=quantiles, return_mode=return_mode, synchronize=synchronize)

    assert return_mode in ["min", "max", "mean", "median"], f"Invalid return_mode: {return_mode}"

    if device is not None and not isinstance(device, int) and torch.device(device).type in ("mps", "npu"):
        device_idx = torch.device(device)
    else:
        device_idx = _normalize_cuda_device(device)
    if isinstance(device_idx, int) or (isinstance(device_idx, torch.device) and device_idx.type == "npu"):
        device_context = torch.cuda.device(device_idx) if isinstance(device_idx, int) else torch.npu.device(device_idx)
        with device_context:
            return _do_bench_impl(
                fn,
                warmup=warmup,
                rep=rep,
                _n_warmup=_n_warmup,
                _n_repeat=_n_repeat,
                quantiles=quantiles,
                fast_flush=fast_flush,
                backend=backend,
                return_mode=return_mode,
                device_idx=device_idx,
                cache_size=cache_size,
                early_stop_baseline=early_stop_baseline,
            )

    return _do_bench_impl(
        fn,
        warmup=warmup,
        rep=rep,
        _n_warmup=_n_warmup,
        _n_repeat=_n_repeat,
        quantiles=quantiles,
        fast_flush=fast_flush,
        backend=backend,
        return_mode=return_mode,
        device_idx=device_idx,
        cache_size=cache_size,
        early_stop_baseline=early_stop_baseline,
    )


def _normalize_cuda_device(benchmark_device: int | torch.device | None) -> int | None:
    """Return a concrete CUDA device index, preserving implicit mode for None."""
    if benchmark_device is None:
        return None
    if isinstance(benchmark_device, int):
        return benchmark_device

    torch_device = torch.device(benchmark_device)
    if torch_device.type != "cuda":
        raise ValueError(f"do_bench device must be a CUDA device, got {torch_device}")
    if torch_device.index is None:
        return torch.cuda.current_device()
    return torch_device.index


def _cache_device(device_idx: int | torch.device | None) -> str | torch.device:
    if device_idx is None:
        return device
    if isinstance(device_idx, torch.device):
        return device_idx
    return torch.device("cuda", device_idx)


def _do_bench_impl(
    fn: Callable,
    warmup: float,
    rep: float,
    _n_warmup: int,
    _n_repeat: int,
    quantiles: list[float] | None,
    fast_flush: bool,
    backend: Literal["event", "cupti", "cudagraph", "msprof", "msprof_detail"],
    return_mode: Literal["min", "max", "mean", "median"],
    device_idx: int | torch.device | None,
    cache_size: int,
    early_stop_baseline: float | None = None,
) -> float | list[float]:
    if backend in ("cupti", "cudagraph"):
        cache_device_type = torch.device(_cache_device(device_idx)).type
        if cache_device_type == "mps":
            raise ValueError(f'{backend} timing requires CUDA/HIP; use "event" or "wall" for MPS')
        if cache_device_type == "npu":
            raise ValueError(f'{backend} timing requires CUDA/HIP; use "event", "msprof", or "wall" for NPU')

    # Initial function call and synchronization
    fn()
    _cuda_synchronize(device_idx)

    # Create L2 cache flush buffer (`cache_size` MB)
    # Fast flush uses int32 (4 bytes), regular uses int8 (1 byte)
    cache_bytes = cache_size * 1024 * 1024
    cache_numel = cache_bytes // 4 if fast_flush else cache_bytes
    cache_dtype = torch.int if fast_flush else torch.int8
    cache = torch.empty(cache_numel, dtype=cache_dtype, device=_cache_device(device_idx))

    # Warm the flush buffer once outside the timed estimate: the first
    # kernel launch may pay one-time backend init (e.g. ~250 ms on torch_npu).
    cache.zero_()
    _cuda_synchronize(device_idx)

    # Estimate kernel runtime with 5 iterations
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    start_event.synchronize()
    end_event.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # Early stop: skip full benchmark if estimate exceeds baseline
    if early_stop_baseline is not None and estimate_ms > early_stop_baseline:
        logger.debug(
            "Early stop: estimate_ms=%.3fms exceeds baseline=%.3fms, skipping full benchmark.",
            estimate_ms,
            early_stop_baseline,
        )
        if quantiles is not None:
            return [estimate_ms] * len(quantiles)
        return estimate_ms

    # Calculate warmup and repeat counts (minimum 1 iteration each)
    n_warmup = _n_warmup if _n_warmup > 0 else max(1, int(warmup / estimate_ms))
    n_repeat = _n_repeat if _n_repeat > 0 else max(1, int(rep / estimate_ms))

    # Warmup phase
    for _ in range(n_warmup):
        fn()

    # Benchmarking phase
    if backend == "event":
        return _bench_with_cuda_events(fn, cache, n_repeat, quantiles, return_mode, device_idx)
    elif backend == "cupti":
        return _bench_with_cupti(fn, cache, n_repeat)
    elif backend == "cudagraph":
        return _bench_with_cudagraph(fn, cache, n_repeat, quantiles, return_mode, device_idx)
    elif backend.startswith("msprof"):
        return _bench_with_msprof(fn, cache, n_repeat, detailed=backend == "msprof_detail")
    else:
        raise ValueError(f"Unknown profiler backend: {backend}")


def _bench_with_msprof(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    detailed: bool = False,
) -> float:
    """Benchmark using torch_npu msprof timing."""
    import shutil
    import signal
    import struct
    import tempfile
    from collections import defaultdict
    from contextlib import contextmanager, redirect_stderr, redirect_stdout
    from dataclasses import dataclass
    from pathlib import Path

    import torch_npu.profiler

    @dataclass
    class KernelProfile:
        """Per-kernel profiling result with AIC and AIV pipe utilization.

        All pipe fields are ratios in [0, 1] relative to their own total_cycles.
        """

        dur_ns: float
        # AIC pipes
        aic_total_cycles: int = 0
        aic_mad: float = 0.0
        aic_scalar: float = 0.0
        aic_mte1: float = 0.0
        aic_mte2: float = 0.0
        aic_mte3: float = 0.0
        aic_fixpipe: float = 0.0
        # AIV pipes
        aiv_total_cycles: int = 0
        aiv_vec: float = 0.0
        aiv_scalar: float = 0.0
        aiv_mte2: float = 0.0
        aiv_mte3: float = 0.0

        def __add__(self, other: KernelProfile) -> KernelProfile:
            """Sum two profiles: dur_ns adds, pipe ratios are cycle-weighted averages."""
            aic_c = self.aic_total_cycles + other.aic_total_cycles
            aiv_c = self.aiv_total_cycles + other.aiv_total_cycles
            wa = self.aic_total_cycles / aic_c if aic_c else 0
            wb = other.aic_total_cycles / aic_c if aic_c else 0
            va = self.aiv_total_cycles / aiv_c if aiv_c else 0
            vb = other.aiv_total_cycles / aiv_c if aiv_c else 0
            return KernelProfile(
                dur_ns=self.dur_ns + other.dur_ns,
                aic_total_cycles=aic_c,
                aic_mad=self.aic_mad * wa + other.aic_mad * wb,
                aic_scalar=self.aic_scalar * wa + other.aic_scalar * wb,
                aic_mte1=self.aic_mte1 * wa + other.aic_mte1 * wb,
                aic_mte2=self.aic_mte2 * wa + other.aic_mte2 * wb,
                aic_mte3=self.aic_mte3 * wa + other.aic_mte3 * wb,
                aic_fixpipe=self.aic_fixpipe * wa + other.aic_fixpipe * wb,
                aiv_total_cycles=aiv_c,
                aiv_vec=self.aiv_vec * va + other.aiv_vec * vb,
                aiv_scalar=self.aiv_scalar * va + other.aiv_scalar * vb,
                aiv_mte2=self.aiv_mte2 * va + other.aiv_mte2 * vb,
                aiv_mte3=self.aiv_mte3 * va + other.aiv_mte3 * vb,
            )

        @property
        def dur_us(self) -> float:
            return self.dur_ns / 1000

        def tflops(self, flops: float) -> float:
            return flops / self.dur_ns / 1000

        def gbps(self, bytes: float) -> float:
            return bytes / self.dur_ns

        def __repr__(self):
            def _fmt(label, val):
                return f"{label}={val * 100:.1f}%" if val > 0.001 else None

            parts = [f"{self.dur_ns / 1000:.1f}us"]
            for s in [
                _fmt("mad", self.aic_mad),
                _fmt("aic_mte2", self.aic_mte2),
                _fmt("aic_mte1", self.aic_mte1),
                _fmt("aic_fix", self.aic_fixpipe),
                _fmt("aiv_vec", self.aiv_vec),
                _fmt("aiv_mte2", self.aiv_mte2),
                _fmt("aiv_mte3", self.aiv_mte3),
            ]:
                if s:
                    parts.append(s)
            if self.aic_total_cycles:
                parts.append(f"aic_cycles={self.aic_total_cycles}")
            if self.aiv_total_cycles:
                parts.append(f"aiv_cycles={self.aiv_total_cycles}")
            return f"KernelProfile({', '.join(parts)})"

    @contextmanager
    def _suppress_output(suppress: bool):
        if not suppress:
            yield
            return

        with open(os.devnull, "w") as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
            yield

    @contextmanager
    def _defer_signals():
        pending_signal = None

        def _handler(signum, frame):
            nonlocal pending_signal
            pending_signal = signum

        old_sigint = signal.signal(signal.SIGINT, _handler)
        old_sigterm = signal.signal(signal.SIGTERM, _handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            if pending_signal == signal.SIGINT:
                raise KeyboardInterrupt()
            if pending_signal == signal.SIGTERM:
                raise SystemExit(128 + signal.SIGTERM)

    @contextmanager
    def _prof_work_path():
        key = "ASCEND_WORK_PATH"
        old = os.environ.get(key)
        tmp_dir = tempfile.mkdtemp(prefix="tilelang_prof_")
        os.environ[key] = tmp_dir
        try:
            yield tmp_dir
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _nop_trace_ready(_prof):
        pass

    def _parse_ffts_profile(prof_path: Path, n_repeat: int):
        hash_to_name: dict[int, str] = {}
        for f in prof_path.rglob("*hash_dic.slice_*"):
            if f.name.endswith(".done"):
                continue
            text = f.read_bytes().decode("utf-8", errors="replace")
            for line in text.strip().split("\n"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        h = int(parts[0])
                        if h >= 2**63:
                            h -= 2**64
                        hash_to_name[h] = parts[1]
                    except ValueError:
                        pass

        seq_to_name: dict[int, str] = {}
        for f in prof_path.rglob("*task_track.slice_*"):
            if f.name.endswith(".done"):
                continue
            data = f.read_bytes()
            for i in range(len(data) // 64):
                vals = struct.unpack_from("<8q", data, i * 64)
                seq = (vals[3] >> 32) & 0xFFFF
                name_hash = vals[5]
                if name_hash != 0:
                    seq_to_name[seq] = hash_to_name.get(name_hash, f"<unknown {name_hash}>")

        aic_records: dict[str, list[dict[str, float]]] = defaultdict(list)
        aiv_records: dict[str, list[dict[str, float]]] = defaultdict(list)
        for f in prof_path.rglob("ffts_profile*"):
            if f.name.endswith(".done") or f.stat().st_size == 0:
                continue
            data = f.read_bytes()
            for i in range(len(data) // 128):
                vals = struct.unpack_from("<16q", data, i * 128)
                total_cycles = vals[1]
                if total_cycles == 0:
                    continue
                seq = (vals[0] >> 32) & 0xFFFF
                is_aiv = vals[2] >= (1 << 31)
                dur_ns = float(vals[15] - vals[14])
                name = seq_to_name.get(seq, "<unknown>")
                metrics = {
                    "dur_ns": dur_ns,
                    "total_cycles": float(total_cycles),
                    "vec": vals[4] / total_cycles,
                    "mad": vals[5] / total_cycles,
                    "scalar": vals[6] / total_cycles,
                    "mte1": vals[7] / total_cycles,
                    "mte2": vals[8] / total_cycles,
                    "mte3": vals[9] / total_cycles,
                    "fixpipe": vals[12] / total_cycles,
                }
                if is_aiv:
                    aiv_records[name].append(metrics)
                else:
                    aic_records[name].append(metrics)

        profiles: dict[str, KernelProfile] = {}
        for name in set(aic_records.keys()) | set(aiv_records.keys()):
            aic_list = aic_records.get(name, [])
            aiv_list = aiv_records.get(name, [])
            dur_list = aic_list or aiv_list
            n_aic = len(aic_list) or 1
            n_aiv = len(aiv_list) or 1
            profiles[name] = KernelProfile(
                dur_ns=sum(m["dur_ns"] for m in dur_list) / n_repeat,
                aic_total_cycles=int(sum(m["total_cycles"] for m in aic_list) / n_repeat),
                aic_mad=sum(m["mad"] for m in aic_list) / n_aic,
                aic_scalar=sum(m["scalar"] for m in aic_list) / n_aic,
                aic_mte1=sum(m["mte1"] for m in aic_list) / n_aic,
                aic_mte2=sum(m["mte2"] for m in aic_list) / n_aic,
                aic_mte3=sum(m["mte3"] for m in aic_list) / n_aic,
                aic_fixpipe=sum(m["fixpipe"] for m in aic_list) / n_aic,
                aiv_total_cycles=int(sum(m["total_cycles"] for m in aiv_list) / n_repeat),
                aiv_vec=sum(m["vec"] for m in aiv_list) / n_aiv,
                aiv_scalar=sum(m["scalar"] for m in aiv_list) / n_aiv,
                aiv_mte2=sum(m["mte2"] for m in aiv_list) / n_aiv,
                aiv_mte3=sum(m["mte3"] for m in aiv_list) / n_aiv,
            )
        return profiles

    def _sum_profiles(profiles):
        total = profiles[0]
        for profile in profiles[1:]:
            total += profile
        return total

    cache_dtype = "int32" if cache.dtype == torch.int32 else "int8"
    cache_flush_kernel = get_msprof_cache_flush_kernel(int(cache.numel()), cache_dtype)
    cache_flush_kernel(cache)
    torch.npu.synchronize()

    with _defer_signals(), _prof_work_path(), _suppress_output(True):
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=_nop_trace_ready,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                data_simplification=False,
            ),
        ) as prof:
            for _ in range(n_repeat):
                cache_flush_kernel(cache)
                fn()
            torch.npu.synchronize()
            prof.step()

        kernel_profiles = _parse_ffts_profile(Path(prof.prof_if.prof_path), n_repeat)

    kernel_profiles = {name: profile for name, profile in kernel_profiles.items() if "_tilelang_profiler_cache_flush_kernel" not in name}
    if not kernel_profiles:
        raise RuntimeError("No profiled kernels remain after excluding the msprof flush kernel.")

    profile = _sum_profiles(list(kernel_profiles.values()))
    if detailed:
        return profile
    return profile.dur_ns / 1e6
