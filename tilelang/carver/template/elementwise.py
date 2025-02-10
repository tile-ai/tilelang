# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from .base import BaseTemplate
from tvm import te
from ..arch import TileDevice
from ..roller import Hint
from typing import List
from ..utils import get_roller_hints_from_func

@dataclass
class ElementwiseTemplate(BaseTemplate):
    
    # OP Related Config
    shape: List[int] = None
    dtype: str = "float16"

    def get_hardware_aware_configs(self,
                                   arch: TileDevice = None,
                                   topk: int = 10) -> List[Hint]:
        roller_hints = get_roller_hints_from_func(
            self._func, arch=arch, topk=topk, allow_gemv=True)
        return roller_hints

    def initialize_function(self) -> None:
        shape, dtype = self.shape, self.dtype

        A = te.placeholder(shape, name="A", dtype=dtype)

        def _compute_elementwise(*indices):
            return A[indices] + 1

        B = te.compute(
            shape,
            fcompute = _compute_elementwise,
            name="B",
        )
        
        args = [A, B]
        self.set_function(te.create_prim_func(args))

    def params_as_dict(self):
        return {
            "shape": self.shape,
            "dtype": self.dtype
        }

    @property
    def class_attributes(self):
        return self.params_as_dict()

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"
