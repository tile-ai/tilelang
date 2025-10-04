"""The profiler and convert to torch utils"""

import torch
from typing import Callable, List, Literal, Optional, Union
import os
import sys


def _do_bench_event(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    grad_to_none: Optional[List[torch.Tensor]] = None,
    quantiles: Optional[List[float]] = None,
    fast_flush: bool = True,
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
) -> Union[float, List[float]]:
    """Benchmarks the runtime of a PyTorch function.

    This function handles:
    - L2 cache flushing between runs for consistent timing
    - Automatic warmup and repeat count calculation
    - Optional gradient clearing for backward passes
    - Multiple measurement modes (mean, median, min, max)

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds
        rep: Target number of repetitions
        _n_warmup: Override for number of warmup iterations
        _n_repeat: Override for number of timing iterations
        grad_to_none: Tensors whose gradients should be cleared between runs
        quantiles: Optional performance percentiles to compute
        fast_flush: Whether to use faster L2 cache flushing
        return_mode: How to aggregate timing results ("mean", "median", "min", "max")

    Returns:
        float: Aggregated runtime in milliseconds
    """
    assert return_mode in ["min", "max", "mean", "median"]
    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    if _n_warmup > 0:
        n_warmup = _n_warmup
    if _n_repeat > 0:
        n_repeat = _n_repeat
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


class empty_suppress:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:

    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# from https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/bench.py
def _do_bench_kineto(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    fast_flush: bool = True,
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
    trace_path: str = None,
    suppress_kineto_output: bool = False,
) -> Union[float, List[float]]:
    """Benchmarks the runtime of a PyTorch function using Kineto."""
    assert return_mode in ["min", "max", "mean", "median"]
    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    if _n_warmup > 0:
        n_warmup = _n_warmup
    if _n_repeat > 0:
        n_repeat = _n_repeat

    # warm-up
    for _ in range(n_warmup):
        fn()

    # benchmark
    using_nsys = int(os.environ.get('DG_NSYS_PROFILING', 0))
    suppress = suppress_stdout_stderr if suppress_kineto_output and not using_nsys else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(
            wait=1, warmup=0, active=1, repeat=1) if not using_nsys else None
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule) if not using_nsys else empty_suppress()
        with profiler:
            for _i in range(2):
                for _ in range(n_repeat):
                    cache.zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    if using_nsys:
        return 1

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # # Return average kernel times
    total_cuda_time = 0.0
    excluded_time = 0.0
    excluded_kernels = "at::native::vectorized_elementwise"
    for e in profiler.key_averages():
        total_cuda_time += e.self_device_time_total
        if excluded_kernels in e.key:
            excluded_time += e.self_device_time_total

    kernel_time = (total_cuda_time - excluded_time) / n_repeat

    return kernel_time * 1e-3


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    grad_to_none: Optional[List[torch.Tensor]] = None,
    quantiles: Optional[List[float]] = None,
    fast_flush: bool = True,
    backend: Literal["event", "cupti"] = "event",
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
    trace_path: str = None,
    suppress_kineto_output: bool = False,
) -> Union[float, List[float]]:
    return _do_bench_event(
        fn,
        warmup,
        rep,
        _n_warmup,
        _n_repeat,
        grad_to_none,
        quantiles,
        fast_flush,
        return_mode,
    ) if backend == "event" else _do_bench_kineto(
        fn,
        warmup,
        rep,
        _n_warmup,
        _n_repeat,
        fast_flush,
        return_mode,
        trace_path,
        suppress_kineto_output,
    )