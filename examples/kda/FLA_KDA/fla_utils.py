import contextlib
import functools
import inspect
import os
import warnings
from collections.abc import Callable
from typing import Any
from packaging import version
from enum import Enum

import torch
import triton
import triton.language.extra.libdevice as tldevice


device = "cuda"
device_torch_lib = getattr(torch, device)

exp = tldevice.fast_expf
exp2 = tldevice.exp2
log = tldevice.fast_logf
log2 = tldevice.fast_log2f

IS_NVIDIA_HOPPER = True and ("NVIDIA H" in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9)
USE_CUDA_GRAPH = True and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"


FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


# error check，copy from
def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (error_rate < 0.01 or abs_atol <= 0.3):
        if error_rate > ratio:
            warnings.warn(msg, stacklevel=2)
    else:
        assert error_rate < ratio, msg


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: tuple | None = None
    last_kwargs: dict | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if (
            last_args is not None
            and last_kwargs is not None
            and len(args) == len(last_args)
            and len(kwargs) == len(last_kwargs)
            and all(a is b for a, b in zip(args, last_args))
            and all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items())
        ):
            return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


# @functools.cache
# def get_multiprocessor_count(tensor_idx: int = 0) -> int:
#     try:
#         return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']
#     except BaseException:
#         # Maybe we use a NPU device.
#         if triton.runtime.driver.active.get_current_target().backend == 'npu':
#             return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['num_vectorcore']
#         else:
#             return 1
@functools.cache
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    """
    Compatible across Triton versions:
    - 2.0.x
    - 2.1.0
    - 2.2.x and above
    Supports CUDA and NPU.
    """

    # ---- Try the newer Triton 2.2+ API ----
    try:
        drv = triton.runtime.driver.active
        props = drv.utils.get_device_properties(tensor_idx)
        return props.get("multiprocessor_count") or props.get("num_vectorcore") or 1
    except Exception:
        pass

    # ---- Fallback: Triton 2.0 / 2.1 API ----
    try:
        cuda = triton.runtime.driver.CudaDriver
        dev = cuda.get_current_device()
        props = cuda.get_device_properties(dev)
        return props.get("multiprocessor_count", 1)
    except Exception:
        pass

    return 1


def input_guard(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@functools.cache
def check_pytorch_version(version_s: str = "2.4") -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


if check_pytorch_version("2.4"):
    device = "cuda"
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    assert device == "cuda", "Only cuda device is supported for PyTorch version < 2.4.0."
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"] for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        return [-1]


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False
