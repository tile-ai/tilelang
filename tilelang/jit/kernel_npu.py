from typing import Callable, Any, List, Dict, Literal, Optional, Tuple, Union
from dataclasses import dataclass
from functools import partial
import torch
import torch_npu
import importlib.util
from tilelang.engine.param import KernelParam
from tilelang.utils.tensor import (
    TensorSupplyType,
    torch_assert_close,
)
from tvm import tir
from tilelang.jit.adapter import BaseKernelAdapter

def do_bench(
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
    assert return_mode in ["min", "max", "mean", "median"]
    fn()
    torch.npu.synchronize()
    device: torch.device = torch.npu.current_device()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device=device)

    # Estimate the runtime of the function
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.npu.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    if _n_warmup > 0:
        n_warmup = _n_warmup
    if _n_repeat > 0:
        n_repeat = _n_repeat
    start_event = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
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
    torch.npu.synchronize()
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


def get_tensor_supply(supply_type: TensorSupplyType = TensorSupplyType.Integer):

    from tilelang.engine.param import KernelParam

    def get_tensor(param: KernelParam) -> torch.Tensor:
        dtype: torch.dtype = param.torch_dtype()
        device =  torch.npu.current_device()

        if hasattr(param, "shape") and not param.shape:
            raise ValueError(
                f"TensorType must have a shape, but got {type(param)}, "
                "likely you are trying to generate a random tensor with a dynamic symbolic shape."
            )

        # Check if with dynamic symbolic shape
        for shape in param.shape:
            if isinstance(shape, tir.Var):
                raise ValueError(
                    f"TensorType must have a static shape, but got {shape}, "
                    "likely you are trying to generate a random tensor with a dynamic symbolic shape."
                )

        shape = list(map(int, param.shape))
        if supply_type == TensorSupplyType.Auto:
            is_unsigned = param.is_unsigned()
            is_float8 = param.is_float8()
            is_float4 = param.is_float4()
            is_boolean = param.is_boolean()
            if is_unsigned:
                return torch.randint(low=0, high=3, size=shape, device=device, dtype=dtype)
            elif is_float8:
                return torch.randint(low=-128, high=128, size=shape, device=device, dtype=torch.int8).to(dtype)
            elif is_float4:
                return torch.randint(low=0, high=16, size=shape, device=device, dtype=dtype)
            elif is_boolean:
                return torch.randint(low=0, high=2, size=shape, device=device, dtype=dtype)
            elif dtype in {torch.float16, torch.float32, torch.bfloat16}:
                return torch.empty(*shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
            else:
                return torch.randint(low=-2, high=3, size=shape, device=device, dtype=dtype)

        if dtype == torch.int8 and supply_type in [
            TensorSupplyType.Uniform,
            TensorSupplyType.Normal,
        ]:
            return torch.ones(*shape, device=device, dtype=dtype)

        if supply_type == TensorSupplyType.Integer:
            is_unsigned = param.is_unsigned()
            is_float8 = param.is_float8()
            is_float4 = param.is_float4()
            is_boolean = param.is_boolean()
            if is_unsigned:
                return torch.randint(low=0, high=3, size=shape, device=device, dtype=dtype)
            elif is_float8:
                return torch.randint(low=-128, high=128, size=shape, device=device, dtype=torch.int8).to(dtype)
            elif is_float4:
                return torch.randint(low=0, high=16, size=shape, device=device, dtype=dtype)
            elif is_boolean:
                return torch.randint(low=0, high=2, size=shape, device=device, dtype=dtype)
            else:
                return torch.randint(low=-2, high=3, size=shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.Uniform:
            return torch.empty(*shape, device=device, dtype=torch.float32).uniform_(-1.0, 1.0).to(dtype)
        elif supply_type == TensorSupplyType.Normal:
            return torch.empty(*shape, device=device, dtype=torch.float32).normal_(-1.0, 1.0).to(dtype)
        elif supply_type == TensorSupplyType.Randn:
            return torch.randn(*shape, device=device).to(dtype)
        elif supply_type == TensorSupplyType.Zero:
            return torch.zeros(*shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.One:
            return torch.ones(*shape, device=device, dtype=dtype)
        else:
            raise NotImplementedError(supply_type)

    return get_tensor

@dataclass
class Profiler:
    params: List[KernelParam]
    result_idx: List[int]
    supply_type: TensorSupplyType
    adapter: Optional[BaseKernelAdapter] = None
    def __post_init__(self):
        """Initialize tensor supply after dataclass initialization"""
        self.result_idx = self._legalize_result_idx(self.result_idx)
        self.supply = get_tensor_supply(self.supply_type)

    def _legalize_result_idx(self, result_idx: Optional[List[int]] = None) -> List[int]:
        params = self.params
        # result_idx is a list of indices of the output tensors
        if result_idx is None:
            result_idx = []
        elif isinstance(result_idx, int):
            if result_idx > len(params) or result_idx < -len(params):
                raise ValueError(
                    f"result_idx should be an integer between {-len(params)} and {len(params) - 1}")
            if result_idx < 0:
                result_idx = len(params) + result_idx
            result_idx = [result_idx]
        elif not isinstance(result_idx, list):
            raise ValueError("result_idx should be a list of integers")
        return result_idx

    def with_default_adapter(self, adapter: BaseKernelAdapter) -> "Profiler":
        self.adapter = adapter
        return self

    def _get_full_args(self, input_tensors: Optional[List[torch.Tensor]] = None):
        if input_tensors is not None:
            ins = input_tensors
            full_args = []
            input_iter = iter(ins)
            for i in range(len(self.params)):
                if i in self.result_idx:
                    full_args.append(self.supply(self.params[i]))
                else:
                    full_args.append(next(input_iter))
            return full_args
        else:
            return [self.supply(param) for param in self.params]
    def _get_inputs(self, with_output=False):
        ins = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                ins.append(self.supply(self.params[i]))
        return ins
    
    def _get_params(self, with_output=False):
        params = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                params.append(self.params[i])
        return params

    def assert_allclose(
        self,
        reference_program: Callable,
        input_tensors: Optional[List[torch.Tensor]] = None,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        max_mismatched_ratio=0.01,
    ):
        """Validates kernel output against a reference implementation.
        
        Args:
            reference_program: Reference implementation to compare against
            input_tensors: Optional pre-generated input tensors
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            max_mismatched_ratio: Maximum allowed ratio of mismatched elements
        """
        full_args = self._get_full_args(input_tensors)
        input_only = [full_args[i] for i in range(len(full_args)) if i not in self.result_idx]
        ref_outs = reference_program(*input_only)
        torch.npu.synchronize()
        self.func(*full_args)
        torch.npu.synchronize()
        actual_outs = [full_args[i] for i in self.result_idx]
        
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        elif ref_outs is None:
            ref_outs = []

        assert len(actual_outs) == len(ref_outs), "len(actual_outs) not equals to len(ref_outs) !"
        for lhs, rhs in zip(actual_outs, ref_outs):
            torch_assert_close(
                lhs,
                rhs,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
                base_name="tilelang",
                ref_name="ref",
            )
            
    def determine_profiler(self, func: Optional[Callable] = None):
        # we only support torch profiler
        return "torch"    

    def do_bench(
        self,
        func: Optional[Callable] = None,
        warmup: int = 25,
        rep: int = 100,
        n_warmup: int = 1,
        n_repeat: int = 1,
        input_tensors: List[torch.Tensor] = None,
    ) -> float:
        """Benchmarks the execution time of a given function.
        
        Args:
            func: Function to benchmark (uses adapter if None)
            warmup: Warmup time in milliseconds
            rep: Number of repetitions for timing
            n_warmup: Number of warmup iterations
            n_repeat: Number of timing iterations
            profiler: Which profiling backend to use
            input_tensors: Optional pre-generated input tensors
            
        Returns:
            float: Average execution time in milliseconds
        """
        profiler = self.determine_profiler(func)
        if profiler == "torch":
            full_args = self._get_full_args(input_tensors)
            input_only = [full_args[i] for i in range(len(full_args)) if i not in self.result_idx]
            if func is None:
                assert self.adapter is not None, "benchmarking function should be provided"
                func = self.adapter
                bench_func = partial(func, *full_args)
            else:
                bench_func = partial(func, *input_only)
            return do_bench(
                bench_func,
                warmup=warmup,
                rep=rep,
                _n_warmup=n_warmup,
                _n_repeat=n_repeat,
            )
        else:
            raise ValueError(f"Unknown profiler: {profiler}")
   
    @property
    def func(self):
        assert self.adapter is not None, "adapter should be provided"
        return self.adapter
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)   


class JitKernel_NPU:
    
    adapter: BaseKernelAdapter = None
    def __init__(self, compiled_kernel: Any, metadata : dict) -> None:
        self._compiled_kernel = compiled_kernel
        self.params = metadata['tl_params']
        self.out_idx = metadata['tl_out_idx']
        self.adapter = NPUKernelAdapter(self, self.params, self.out_idx)
    def __call__(self, *args, **kwargs):
        return self._compiled_kernel(*args, **kwargs)
    
    def get_profiler(self,
                     tensor_supply_type: TensorSupplyType = TensorSupplyType.Auto) -> Profiler:
        return Profiler(self.params, self.out_idx, tensor_supply_type).with_default_adapter(self.adapter)
    
class NPUKernelAdapter(BaseKernelAdapter):
    def __init__(self, jit_kernel: JitKernel_NPU, params, result_idx):
        super().__init__(mod=jit_kernel, params=params, result_idx=result_idx)

    def _convert_torch_func(self) -> callable:
        return self.mod