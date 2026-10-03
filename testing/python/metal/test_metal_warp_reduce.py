"""Metal lowering of scalar warp reductions to SIMD-group functions."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

REDUCERS = {
    "sum": T.warp_reduce_sum,
    "max": T.warp_reduce_max,
    "min": T.warp_reduce_min,
    "bitand": T.warp_reduce_bitand,
    "bitor": T.warp_reduce_bitor,
}
METAL_FUNCTIONS = {
    "sum": "simd_sum",
    "max": "simd_max",
    "min": "simd_min",
    "bitand": "simd_and",
    "bitor": "simd_or",
}


@tilelang.jit(target="metal", execution_backend="torch")
def warp_reduce_kernel(kind, dtype, threads=32):
    reduce = REDUCERS[kind]

    @T.prim_func
    def main(A: T.Tensor((threads,), dtype)):
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding(0)
            value = T.alloc_local([1], dtype)
            value[0] = A[tid]
            A[tid] = reduce(value[0])

    return main


def reference(kind, values):
    if kind == "sum":
        return values.sum()
    if kind == "max":
        return values.max()
    if kind == "min":
        return values.min()
    result = values[0]
    op = torch.bitwise_and if kind == "bitand" else torch.bitwise_or
    for value in values[1:]:
        result = op(result, value)
    return result


@pytest.mark.parametrize("kind", sorted(REDUCERS))
def test_warp_reduce_codegen(kind):
    dtype = "int32" if kind.startswith("bit") else "float32"
    program = warp_reduce_kernel.get_tir(kind, dtype)
    with tvm.target.Target("metal"):
        source = tilelang.lower(program, target="metal").kernel_source
    assert f"{METAL_FUNCTIONS[kind]}(" in source
    assert "simd_shuffle" not in source
    assert "threadgroup_barrier" not in source


@tilelang.testing.requires_metal
@pytest.mark.parametrize("kind", ["sum", "max", "min"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int32"])
def test_warp_reduce_execution(kind, dtype):
    torch_dtype = getattr(torch, dtype)
    if dtype == "int32":
        a = torch.randint(-1000, 1000, (32,), dtype=torch_dtype)
    else:
        a = torch.arange(32, dtype=torch.float32).sub(15.5).to(torch_dtype)
    expected = torch.full_like(a, reference(kind, a.to(torch.float64) if dtype != "int32" else a).to(torch_dtype))
    device = a.to("mps")
    warp_reduce_kernel(kind, dtype)(device)
    torch.testing.assert_close(device.cpu(), expected, atol=0, rtol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("kind", ["bitand", "bitor"])
def test_warp_reduce_bitwise(kind):
    a = torch.randint(-128, 128, (32,), dtype=torch.int32)
    expected = torch.full_like(a, reference(kind, a))
    device = a.to("mps")
    warp_reduce_kernel(kind, "int32")(device)
    torch.testing.assert_close(device.cpu(), expected, atol=0, rtol=0)


@tilelang.testing.requires_metal
def test_warp_reduce_is_per_simdgroup():
    a = torch.arange(64, dtype=torch.float32)
    expected = torch.cat([torch.full((32,), a[:32].sum()), torch.full((32,), a[32:].sum())])
    device = a.to("mps")
    warp_reduce_kernel("sum", "float32", threads=64)(device)
    torch.testing.assert_close(device.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("kind,dtype", [("sum", "int64"), ("max", "bfloat16"), ("bitand", "float32")])
def test_warp_reduce_rejects_unsupported_scalars(kind, dtype):
    program = warp_reduce_kernel.get_tir(kind, dtype)
    with tvm.target.Target("metal"), pytest.raises(tvm.error.InternalError, match="SIMD-group reduction"):
        tilelang.lower(program, target="metal")
