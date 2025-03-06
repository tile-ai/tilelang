# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tilelang.primitives.gemm.base import GemmWarpPolicy
from tvm import tir


def gemm(
    A: tir.Buffer,
    B: tir.Buffer,
    C: tir.Buffer,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    k_pack: int = 1,
):
    """
    k_pack: int
        The number of k dimension that is packed into a single warp.
        please ref to mfma macro generator for the detail information.
    """
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
        k_pack,
    )
