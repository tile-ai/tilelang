# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from typing import List
from .base import BaseKernelAdapter
from tvm.relay import TensorType


class CtypesKernelAdapter(BaseKernelAdapter):

    target = "cuda"
    prim_func = None

    def __init__(self,
                 mod,
                 params: List[TensorType],
                 result_idx: List[int],
                 target,
                 prim_func,
                 verbose: bool = False):
        self.target = target
        self.prim_func = prim_func
        self.verbose = verbose
        super().__init__(mod, params, result_idx)

        raise NotImplementedError("CtypesKernelAdapter is not implemented yet.")
