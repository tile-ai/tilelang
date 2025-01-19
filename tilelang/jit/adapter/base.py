# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from typing import Any, List
from tvm.relay import TensorType


class BaseKernelAdapter(object):

    def __init__(self, mod, params: List[TensorType], result_idx: List[int]) -> None:
        self.mod = mod
        self.params = params

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

        self.result_idx = result_idx

        self.func = self._convert_torch_func()

    def _convert_torch_func(self) -> callable:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)

    def get_kernel_source(self) -> str:
        return self.mod.imported_modules[0].get_source()
