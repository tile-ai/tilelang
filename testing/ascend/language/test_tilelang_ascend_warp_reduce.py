import pytest

import tilelang.ascend.language as T
import tilelang.testing
import tvm
from tilelang.engine.lower import lower


REDUCE_CASES = [
    ("sum", "asc_reduce_add"),
    ("max", "asc_reduce_max"),
    ("min", "asc_reduce_min"),
]
SUPPORTED_DTYPES = ["float32", "int32", "uint32", "float16"]
UNSUPPORTED_DTYPES = [
    "bfloat16",
    "float64",
    "int16",
    "uint16",
    "int64",
    "uint64",
]


def make_warp_reduce_kernel(op, dtype):
    warp_reduce = getattr(T, f"warp_reduce_{op}")

    @T.prim_func
    def kernel(A: T.Tensor((32,), dtype), B: T.Tensor((32,), dtype)):
        with T.Kernel(1) as _, T.SimtVF(threads=32):
            tx = T.get_thread_binding()
            B[tx] = warp_reduce(A[tx])

    return kernel


@pytest.mark.parametrize("op,ascend_intrinsic", REDUCE_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_warp_reduce_codegen(op, ascend_intrinsic, dtype):
    source = lower(make_warp_reduce_kernel(op, dtype), target="ascend").kernel_source

    assert f"{ascend_intrinsic}(" in source


@pytest.mark.parametrize("op", [case[0] for case in REDUCE_CASES])
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_warp_reduce_rejects_unsupported_dtype(op, dtype):
    with pytest.raises(
        tvm.error.InternalError,
        match="supports only scalar float16, float32, int32, and uint32",
    ):
        lower(make_warp_reduce_kernel(op, dtype), target="ascend")


if __name__ == "__main__":
    tilelang.testing.main()
