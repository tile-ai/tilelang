# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from dataclasses import dataclass
from typing import List, Union
import torch
from tilelang import tvm as tvm
from tvm.tir import Buffer, IntImm, Var
from tilelang.utils.tensor import map_torch_type


@dataclass
class KernelParam:
    dtype: torch.dtype
    shape: List[Union[int, Var]]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        dtype = map_torch_type(buffer.dtype)
        shape = []
        for s in buffer.shape:
            if isinstance(s, IntImm):
                shape.append(s.value)
            elif isinstance(s, Var):
                shape.append(s)
            else:
                raise ValueError(f"Unsupported dimension type: {type(s)}")
        return cls(dtype, shape)

    @classmethod
    def from_var(cls, var: Var):
        return cls(var.dtype, [])

    def is_scalar(self) -> bool:
        return len(self.shape) == 0
