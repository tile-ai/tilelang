# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from typing import List, Literal, Optional, Callable
from functools import partial
import torch
from contextlib import suppress

import tvm
from tvm.relay import TensorType

from tilelang.engine import lower
from tilelang.jit.adapter import TorchDLPackKernelAdapter
from tilelang.utils.tensor import (
    get_tensor_supply,
    TensorSupplyType,
    torch_assert_close,
    adapt_torch2tvm,
)


class Profiler(TorchDLPackKernelAdapter):

    def __init__(
        self,
        mod,
        params: List[TensorType],
        result_idx: List[int],
        supply_type: TensorSupplyType = TensorSupplyType.Normal,
    ):
        super().__init__(mod, params, result_idx)
        self.supply = get_tensor_supply(supply_type)

    def _get_inputs(self, with_output=False):
        ins = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                ins.append(self.supply(self.params[i]))
        return ins

    def assert_allclose(
        self,
        reference_program: Callable,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        max_mismatched_ratio=0.01,
    ):
        ins = self._get_inputs()
        ref_outs = reference_program(*ins)
        torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        assert len(lib_outs) == len(ref_outs)
        # torch.set_printoptions(edgeitems=torch.inf)
        for lhs, rhs in zip(lib_outs, ref_outs):
            # close_mask = torch.isclose(lhs, rhs, rtol=rtol, atol=atol)
            # total_elements = lhs.numel()
            # num_not_close = (~close_mask).sum().item()
            # percentage_not_close = (num_not_close / total_elements) * 100
            # print(f"{percentage_not_close:.2f}% of the elements are not close.")
            # print(f"Total elements: {total_elements}, Not close elements: {num_not_close}")
            torch_assert_close(
                lhs,
                rhs,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
            )

    def assert_consistent(self, repeat=10):
        # Used to check no race condition inside the kernel
        ins = self._get_inputs()
        ref_outs = self.func(*ins)

        for _ in range(repeat):
            lib_outs = self.func(*ins)
            for lhs, rhs in zip(lib_outs, ref_outs):
                assert torch.allclose(lhs, rhs), [
                    "result is not consistent",
                    lhs,
                    rhs,
                ]

    def run_once(self, func: Optional[Callable] = None):
        ins = self._get_inputs()
        if not func:
            func = self.__call__
        return func(*ins)

    def determine_profiler(self,
                           func: Optional[Callable] = None,
                           profiler: Literal["torch", "tvm", "auto"] = "auto"):
        if profiler == "auto":
            if func is None or isinstance(func, tvm.runtime.Module):
                return "tvm"
            else:
                return "torch"
        return profiler

    def do_bench(
        self,
        func: Optional[Callable] = None,
        warmup: int = 25,
        rep: int = 100,
        n_warmup: int = 1,
        n_repeat: int = 1,
        profiler: Literal["torch", "tvm", "auto"] = "auto",
        input_tensors: List[torch.Tensor] = None,
    ) -> float:
        profiler = self.determine_profiler(func, profiler)
        if profiler == "torch":
            ins = self._get_inputs() if input_tensors is None else input_tensors
            bench_func = partial(func, *ins)
            return do_bench(
                bench_func,
                warmup=warmup,
                rep=rep,
                _n_warmup=n_warmup,
                _n_repeat=n_repeat,
            )
        elif profiler == "tvm":
            if func is None:
                func = self.mod
            assert isinstance(
                func, tvm.runtime.Module), f"func should be a TVM module, but got {type(func)}"
            ins = (self._get_inputs(with_output=True) if input_tensors is None else input_tensors)
            target = "cuda"

            with suppress(Exception):
                target = self.mod.imported_modules[0].type_key

            assert target in ["cuda", "hip"], f"Unknown target: {target}"

            device = tvm.cuda(0) if target == "cuda" else tvm.rocm(0)
            time_evaluator = self.mod.time_evaluator(
                self.mod.entry_name, device, number=rep, repeat=n_repeat)
            tvm_inputs = [adapt_torch2tvm(inp) for inp in ins]
            # Transform Latency to ms
            return time_evaluator(*tvm_inputs).mean * 1e3
        else:
            raise ValueError(f"Unknown profiler: {profiler}")


def do_bench(
    fn,
    warmup=25,
    rep=100,
    _n_warmup=0,
    _n_repeat=0,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
) -> float:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    
    Returns:
        float: The median runtime of :code:`fn` along with
        the 20-th and 80-th performance percentile.
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


_cached = {}


def cached(func, result_idx: List[int], *args):
    global _cached
    key = (func, tuple(result_idx), *args)
    if key not in _cached:
        program = func(*args)
        mod, params = lower(program)
        mod = TorchDLPackKernelAdapter(mod, params, result_idx)
        _cached[key] = mod
    return _cached[key]
