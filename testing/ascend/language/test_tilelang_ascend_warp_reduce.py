import pytest
import torch

import tilelang
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


PTO_REDUCE_CALL = {"sum": "tl.redux_add", "max": "tl.redux_max", "min": "tl.redux_min"}


@pytest.mark.pto
@pytest.mark.parametrize("op", [case[0] for case in REDUCE_CASES])
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_warp_reduce_pto_codegen(op, dtype):
    source = lower(make_warp_reduce_kernel(op, dtype), target="pto").kernel_source
    assert f"{PTO_REDUCE_CALL[op]}(" in source


# One case per op and dtype; the remaining combinations run the same wrappers.
RUN_CASES = [
    ("min", "int32"),
    ("max", "uint32"),
    ("sum", "float32"),
]


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "target",
    ["ascend", pytest.param("pto", marks=pytest.mark.pto)],
)
@pytest.mark.parametrize("op,dtype", RUN_CASES)
def test_warp_reduce(op, dtype, target):
    compiled = tilelang.compile(make_warp_reduce_kernel(op, dtype), target=target, out_idx=-1)
    if dtype == "float32":
        values = torch.randn(32, dtype=torch.float32)
    elif dtype == "int32":
        values = torch.randint(-100, 100, (32,), dtype=torch.int32)
    else:
        # Values at or above 2**31 are negative when read as signed.
        # This case checks whether a wrong signedness is passed to pto.redux_max/min.
        values = torch.tensor(list(range(16)) + [2**31 + i for i in range(16)], dtype=torch.uint32)
    if dtype == "uint32":
        # torch has no uint32 max/min; compute the reference from Python ints.
        reduced = {"max": max(values.tolist()), "min": min(values.tolist()), "sum": sum(values.tolist()) % 2**32}[op]
    else:
        reduced = {"sum": values.sum(), "max": values.max(), "min": values.min()}[op].item()
    expected = torch.full((32,), reduced, dtype=values.dtype)
    actual = compiled(values.to("npu"))
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
