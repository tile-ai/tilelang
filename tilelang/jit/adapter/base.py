"""The profiler and convert to torch utils"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
from tilelang.engine.param import KernelParam
import torch


class BaseKernelAdapter(ABC):
    func: Callable | None = None

    def __init__(self, mod, params: list[KernelParam], result_idx: list[int]) -> None:
        self.mod = mod
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self._post_init()

    def _legalize_result_idx(self, result_idx: list[int] | None) -> list[int]:
        params = self.params
        # result_idx is a list of indices of the output tensors
        if result_idx is None:
            result_idx = []
        elif isinstance(result_idx, int):
            if result_idx > len(params) or result_idx < -len(params):
                raise ValueError(f"result_idx should be an integer between {-len(params) - 1} and {len(params) - 1}")
            if result_idx < 0:
                result_idx = len(params) + result_idx
            result_idx = [result_idx]
        elif isinstance(result_idx, list):
            for i, idx in enumerate(result_idx):
                if idx >= len(params) or idx < -len(params):
                    raise ValueError(f"result_idx should be an integer between {-len(params) - 1} and {len(params) - 1}")
                if idx < 0:
                    result_idx[i] = len(params) + idx
        else:
            raise ValueError("result_idx should be a list of integers")

        return result_idx

    @abstractmethod
    def _convert_torch_func(self) -> callable:
        pass

    # --- Common helpers to align with PyTorch stream/device semantics ---
    @staticmethod
    def get_current_stream_functor() -> Callable[[], int]:
        """Return a callable that reads Torch's current CUDA stream pointer.

        The returned lambda yields the raw CUDA stream handle of the current
        PyTorch stream on the active device. It's a thunk (evaluated at call
        time) so that any upstream stream guards are respected. If CUDA is
        unavailable, it returns a lambda that yields 0.
        """
        if torch.cuda.is_available():
            try:
                torch.cuda._lazy_init()
                current_device = torch._C._cuda_getDevice
                get_stream = torch._C._cuda_getCurrentRawStream
                return lambda: get_stream(current_device())
            except Exception:
                # Fallback to Python API if internal handles are unavailable
                return lambda: int(torch.cuda.current_stream().cuda_stream)
        # CPU or CUDA unavailable: no stream semantics
        return lambda: 0

    @staticmethod
    def get_current_device_functor() -> Callable[[], torch.device]:
        """Return a callable that yields Torch's current device.

        Similar to the stream functor, we capture a callable that, when called,
        fetches the current device according to PyTorch. On CPU or when CUDA is
        unavailable, returns ``torch.device('cpu')``.
        """
        if torch.cuda.is_available():
            try:
                torch.cuda._lazy_init()
                current_device = torch._C._cuda_getDevice
                return lambda: torch.device("cuda", current_device())
            except Exception:
                return lambda: torch.device("cuda", torch.cuda.current_device())
        # CPU fallback
        return lambda: torch.device("cpu")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if kernel_only:
            return self.mod.imports[0].inspect_source()
        else:
            return self.mod.inspect_source() + "\n\n" + self.mod.imports[0].inspect_source()

    def _post_init(self):
        self.func = self._convert_torch_func()
