# Regression test for https://github.com/tile-ai/tilelang/issues/3814
# On sm_100+ TileLang enables 256-bit vectorization for global-only access,
# which turns a 16-bit elementwise loop over 16 elements per thread into
# int16x16 / uint16x16. CUDA codegen only knew how to print 16-bit integer
# vectors up to x8, so lowering failed with:
#   Fatal: Cannot convert type int16x16 to CUDA type
# The source checks below need the CUDA backend but no Blackwell GPU (or even a
# CUDA device). The runtime check exercises the same kernel on whatever CUDA GPU
# runs the test suite.
import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import language as T, tvm
from tilelang.contrib import nvcc
from tilelang.engine.lower import lower as tilelang_lower
from tvm import tirx


def _elementwise_kernel(dtype):
    @T.prim_func
    def main(X: T.Tensor((512,), dtype), O: T.Tensor((512,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(512):
                O[i] = X[i] + X[i]

    return main


def _vector_carrier(dtype):
    return "ulonglong4" if dtype.startswith("uint") else "longlong4"


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize("arch", ["sm_100a", "sm_120a"])
def test_16bit_integer_256bit_vector_codegen(dtype, arch):
    target = {"kind": "cuda", "arch": arch}
    with tvm.target.Target(target):
        artifact = tilelang_lower(_elementwise_kernel(dtype), target=target)

    source = artifact.kernel_source
    assert _vector_carrier(dtype) in source
    assert "tl::load_global_256" in source
    assert "tl::store_global_256" in source


def _12_lane_constructor_module(dtype):
    # PrintType maps int16x12/uint16x12 to (u)longlong3, so both the Broadcast
    # and the Ramp constructor must pack four lanes into each of the three
    # fields instead of passing one argument per lane to a three-argument
    # CUDA builtin.
    name = f"vector_12_lane_{dtype}"
    broadcast = tirx.Broadcast(tirx.const(1, dtype), 12)
    ramp = tirx.Ramp(tirx.const(0, dtype), tirx.const(1, dtype), 12)
    func = tirx.PrimFunc([], tirx.SeqStmt([tirx.Evaluate(broadcast), tirx.Evaluate(ramp)]))
    func = func.with_attr("global_symbol", name)
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return tvm.IRModule({name: func})


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize(
    "dtype, constructor, packer",
    [("int16", "make_longlong3", "tl::pack_int16x4"), ("uint16", "make_ulonglong3", "tl::pack_uint16x4")],
)
def test_12_lane_integer_vector_constructor_compiles(dtype, constructor, packer):
    module = _12_lane_constructor_module(dtype)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    build_source = tvm.get_global_func("target.build.tilelang_cuda_without_compile")
    source = build_source(module, target).inspect_source()
    assert constructor in source
    assert packer in source
    # nvcc rejects the unpacked form with "too many arguments in function call".
    tvm.get_global_func("target.build.tilelang_cuda")(module, target)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype, torch_dtype, low, high",
    [
        ("int16", torch.int16, -1000, 1000),
        ("uint16", torch.uint16, 0, 30000),
    ],
)
def test_16bit_integer_vector_add_numeric(dtype, torch_dtype, low, high):
    kernel = tilelang.compile(_elementwise_kernel(dtype), out_idx=[1])
    if nvcc.parse_compute_version(nvcc.get_target_compute_version(kernel.target)) >= (10, 0):
        # Below sm_100 the loop is only vectorized to 128 bits, which would
        # not exercise the x16 accessors at all.
        source = kernel.get_kernel_source()
        assert _vector_carrier(dtype) in source
        assert "tl::load_global_256" in source
        assert "tl::store_global_256" in source
    x = torch.randint(low, high, (512,), dtype=torch_dtype, device="cuda")
    out = kernel(x)
    expected = (x.to(torch.int64) + x.to(torch.int64)).to(torch_dtype)
    assert torch.equal(out, expected)


if __name__ == "__main__":
    tilelang.testing.main()
