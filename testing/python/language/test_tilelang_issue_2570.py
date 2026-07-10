"""Regression test for issue #2570.

T.reduce_bitand/bitor/bitxor on a float buffer should raise a clear
ValueError instead of leaking an internal ICHECK from the TIR lowering.
"""

import pytest
import tilelang
import tilelang.language as T


@pytest.mark.parametrize("op_name", ["bitand", "bitor", "bitxor"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_reduce_bitwise_rejects_float_dtype(op_name, dtype):
    M, N = 32, 32

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M,), dtype)):
        with T.Kernel(1, threads=128):
            A_fr = T.alloc_fragment((M, N), dtype)
            B_fr = T.alloc_fragment((M,), dtype)
            T.copy(A, A_fr)
            getattr(T, f"reduce_{op_name}")(A_fr, B_fr, dim=1)
            T.copy(B_fr, Out)

    with pytest.raises(ValueError, match="requires an integer or bool buffer"):
        tilelang.compile(main, out_idx=[-1], target="cuda")


@pytest.mark.parametrize("op_name", ["bitand", "bitor", "bitxor"])
def test_reduce_bitwise_accepts_int32(op_name):
    M, N = 32, 32

    @T.prim_func
    def main(A: T.Tensor((M, N), "int32"), Out: T.Tensor((M,), "int32")):
        with T.Kernel(1, threads=128):
            A_fr = T.alloc_fragment((M, N), "int32")
            B_fr = T.alloc_fragment((M,), "int32")
            T.copy(A, A_fr)
            getattr(T, f"reduce_{op_name}")(A_fr, B_fr, dim=1)
            T.copy(B_fr, Out)

    # Should NOT raise — int32 is valid for bitwise reduce
    tilelang.compile(main, out_idx=[-1], target="cuda")
