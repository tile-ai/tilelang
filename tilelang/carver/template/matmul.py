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
class MatmulTemplate(BaseTemplate):
    
    # OP Related Config
    M: int = None
    N: int = None
    K: int = None
    trans_A: bool = False
    trans_B: bool = True
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    with_bias: bool = False

    def get_hardware_aware_configs(self,
                                   arch: TileDevice = None,
                                   topk: int = 10) -> List[Hint]:
        roller_hints = get_roller_hints_from_func(
            self._func, arch=arch, topk=topk, allow_gemv=True)
        return roller_hints

    def initialize_function(self) -> None:
        M, N, K = self.M, self.N, self.K
        assert (isinstance(M, int)
                and isinstance(N, int) 
                and isinstance(K, int)), "Only Support Integer M, N, K"
        assert (M > 0 and N > 0 and K > 0), "M, N, K should be positive"
        
        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias
        
        input_shape = (M, K) if not trans_A else (K, M)
        weight_shape = (K, N) if not trans_B else (N, K)
        output_shape = (M, N)
        Bias_shape = (N, )
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)
        B = te.placeholder(weight_shape, name="B", dtype=in_dtype)
        Bias = te.placeholder(
            Bias_shape, name="Bias", dtype=accum_dtype)
        

        k = te.reduce_axis((0, K), name="k")

        def _compute_matmul(i, j):
            A_indices = [i, k] if not trans_A else [k, i]
            B_indices = [k, j] if not trans_B else [j, k]
            return te.sum(
                A[tuple(A_indices)].astype(accum_dtype) * B[tuple(B_indices)].astype(accum_dtype), axis=k)

        C = te.compute(
            output_shape,
            fcompute = _compute_matmul,
            name="C",
        )
        
        if with_bias:
            C = te.compute(
                output_shape,
                lambda i, j: C[i, j] + Bias[j],
                name="Bias",
            )
        
        if out_dtype != accum_dtype:
            C = te.compute(
                output_shape,
                lambda i, j: C[i, j].astype(out_dtype),
                name="D",
            )

        args = [A, B, Bias, C] if self.with_bias else [A, B, C]
        self.set_function(te.create_prim_func(args))

    def params_as_dict(self):
        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "trans_A": self.trans_A,
            "trans_B": self.trans_B,
            "in_dtype": self.in_dtype,
            "out_dtype": self.out_dtype,
            "accum_dtype": self.accum_dtype,
            "with_bias": self.with_bias,
        }

    @property
    def class_attributes(self):
        return self.params_as_dict()

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"
