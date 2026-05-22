from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.distributed as dist


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _resolve_bench_group(group):
    if not _dist_ready():
        return None
    return group if group is not None else dist.group.WORLD


def _reduce_benchmark_times(
    times: "np.ndarray",
    *,
    group=None,
    aggregate: Literal["max", "mean", "min", "none"] = "max",
    device: str | torch.device = "cuda",
) -> "np.ndarray":
    if aggregate == "none" or not _dist_ready():
        return times

    group = _resolve_bench_group(group)
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return times

    time_tensor = torch.as_tensor(times, dtype=torch.float64, device=device)
    if aggregate == "max":
        op = dist.ReduceOp.MAX
    elif aggregate == "min":
        op = dist.ReduceOp.MIN
    elif aggregate == "mean":
        op = dist.ReduceOp.SUM
    else:
        raise ValueError(f"unsupported benchmark aggregate mode: {aggregate}")

    dist.all_reduce(time_tensor, op=op, group=group)
    if aggregate == "mean":
        time_tensor /= world_size
    return time_tensor.cpu().numpy()


def do_bench(
    fn,
    warmup: int = 50,
    rep: int = 50,
    post_fn=None,
    *,
    group=None,
    aggregate: Literal["max", "mean", "min", "none"] = "max",
    flush_l2: bool = True,
    l2_flush_bytes: int = int(256e6),
    barrier_comm_profiling: bool = True,
    barrier: Callable | None = None,
    discard_first: bool = True,
    return_mode: Literal["mean", "median", "min", "max", "all"] = "mean",
):
    """Benchmark a CUDA function and normalize timing across distributed ranks."""
    import numpy as np

    if rep <= 0:
        raise ValueError("rep must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if l2_flush_bytes <= 0:
        raise ValueError("l2_flush_bytes must be positive")

    group = _resolve_bench_group(group)

    torch.cuda.synchronize()
    if group is not None:
        dist.barrier(group)
    world_size = dist.get_world_size(group) if group is not None else 1
    use_barrier_comm_profiling = barrier_comm_profiling and group is not None and world_size > 1

    cache = None
    if flush_l2:
        cache = torch.empty(int(l2_flush_bytes // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    if group is not None:
        dist.barrier(group)

    dummy = None
    if use_barrier_comm_profiling and barrier is None:
        dummy = torch.ones(1, dtype=torch.float32, device="cuda")

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        if cache is not None:
            cache.zero_()
        if use_barrier_comm_profiling:
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(int(2e7))
            if barrier is None:
                dist.all_reduce(dummy, group=group)
            else:
                barrier()
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    local_times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=np.float64)
    if discard_first and len(local_times) > 1:
        local_times = local_times[1:]
    times = _reduce_benchmark_times(local_times, group=group, aggregate=aggregate)
    if group is not None:
        dist.barrier(group)

    if return_mode == "all":
        return times
    if return_mode == "mean":
        return np.average(times).item()
    if return_mode == "median":
        return np.median(times).item()
    if return_mode == "min":
        return np.min(times).item()
    if return_mode == "max":
        return np.max(times).item()
    raise ValueError(f"unsupported benchmark return_mode: {return_mode}")


perf_fn = do_bench
