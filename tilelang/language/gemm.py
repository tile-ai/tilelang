# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tilelang.primitives.gemm.base import GemmWarpPolicy
import tilelang.language as T
from tvm import tir
from typing import Union


def gemm(
    A: Union[tir.Buffer, tir.Var],
    B: Union[tir.Buffer, tir.Var],
    C: Union[tir.Buffer, tir.Var],
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
):
    """
    k_pack: int
        The number of k dimension that is packed into a single warp.
        please ref to mfma macro generator for the detail information.
    """

    def legalize_arguments(arg: Union[tir.Buffer, tir.Var]):
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        else:
            return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)
    M = C.shape[0]
    N = C.shape[1]
    K = A.shape[0] if transpose_A else A.shape[1]
    K_B = B.shape[1] if transpose_B else B.shape[0]
    assert K == K_B, "gemm K shape check failed"
    Aptr = A.access_ptr("r")
    Bptr = B.access_ptr("r")
    Cptr = C.access_ptr("rw")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.gemm"),
        Aptr,
        Bptr,
        Cptr,
        transpose_A,
        transpose_B,
        M,
        N,
        K,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
    )
