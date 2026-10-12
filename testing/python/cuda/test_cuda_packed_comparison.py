"""Packed FP16/BF16 comparisons must preserve scalar boolean semantics."""

import operator
import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


OPS = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


def _source(dtype, lanes, op, broadcast=False):
    lhs = tirx.Var("a", f"{dtype}x{lanes}")
    rhs = tirx.Var("b", dtype if broadcast else f"{dtype}x{lanes}")
    value = tirx.Broadcast(rhs, lanes) if broadcast else rhs
    # Var == Var is an identity operation in Python; use explicit TIR constructors.
    comparison = getattr(tirx, op.upper())(lhs, value)
    func = tirx.PrimFunc([lhs, rhs], tirx.Evaluate(comparison))
    func = func.with_attr("global_symbol", "compare")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    build = tvm.get_global_func("target.build.tilelang_cuda_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("CUDA codegen is not enabled")
    return build(tvm.IRModule({"compare": func}), tvm.target.Target({"kind": "cuda", "arch": "sm_90"})).inspect_source()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("broadcast", [False, True])
def test_packed_comparison_codegen(dtype, lanes, op, broadcast):
    source = _source(dtype, lanes, op, broadcast)
    intrinsic = f"__h{'neu' if op == 'ne' else op}2_mask"
    assert source.count(intrinsic + "(") == lanes // 2
    native = "__half2" if dtype == "float16" else "__nv_bfloat162"
    assert f"tl::from_uint1<{native}>" in source
    assert "CUDART_VERSION >= 12000" in source
    assert "#else" in source  # Older CUDA / pre-SM90 BF16 keeps scalar lowering.
    assert source.count(" & 1u)") == lanes


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype,lanes", [("float32", 2), ("float32", 3), ("int32", 4)])
def test_other_comparison_types_keep_scalar_fallback(dtype, lanes):
    assert "2_mask(" not in _source(dtype, lanes, "lt")


def _kernel(dtype, lanes, op, broadcast):
    compare = OPS[op]

    @T.prim_func
    def kernel(A: T.Tensor((128 * lanes,), dtype), B: T.Tensor((128 * lanes,), dtype), C: T.Tensor((3, 128 * lanes), "int32")):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            for j in T.vectorized(lanes):
                pos = tx * lanes + j
                pred = compare(A[pos], B[tx * lanes] if broadcast else B[pos])
                C[0, pos] = T.cast(pred, "int32")
                C[1, pos] = T.Select(pred, 17, -9)
                C[2, pos] = T.cast(T.Not(pred), "int32")

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("broadcast", [False, True])
def test_packed_comparison_special_values(dtype, lanes, op, broadcast):
    torch_dtype = getattr(torch, dtype)
    tiny = torch.finfo(torch_dtype).tiny
    specials = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, tiny / 4, -tiny / 4, 1, -1, 2, -2], dtype=torch_dtype)
    # All pairings exercise NaN on either operand, signed zeros and subnormals.
    a = specials.repeat_interleave(len(specials)).repeat(5)[: 128 * lanes].cuda()
    b = specials.repeat(len(specials)).repeat(5)[: 128 * lanes].cuda()
    kernel = tilelang.compile(_kernel(dtype, lanes, op, broadcast), out_idx=[2])
    result = kernel(a, b)
    if broadcast:
        b = b.reshape(-1, lanes)[:, :1].expand(-1, lanes).reshape(-1)
    expected = OPS[op](a, b)
    torch.testing.assert_close(result[0], expected.int(), rtol=0, atol=0)
    torch.testing.assert_close(result[1], torch.where(expected, 17, -9).int(), rtol=0, atol=0)
    torch.testing.assert_close(result[2], (~expected).int(), rtol=0, atol=0)
    # Simplification may canonicalize a > b into b < a (and >= into <=).
    assert re.search(r"__h(?:eq|neu|lt|le|gt|ge)2_mask", kernel.get_kernel_source())


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_packed_comparison_ptx_and_older_arch_fallback(dtype):
    from tilelang.contrib.nvcc import compile_cuda, default_compile_options, get_nvcc_compiler
    import subprocess

    version = subprocess.check_output([get_nvcc_compiler(), "--version"], text=True)
    if not re.search(r"release (?:1[2-9]|[2-9]\d)\.", version):
        pytest.skip("Mask intrinsics require CUDA 12+")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        source = tilelang.lower(_kernel(dtype, 2, "ne", False), target=target).kernel_source
    ptx = bytes(compile_cuda(source, "ptx", arch="sm_90", options=default_compile_options())).decode()
    suffix = "bf16x2" if dtype == "bfloat16" else "f16x2"
    assert re.search(rf"set\.neu\.(?:u32|s32)\.{suffix}\b", ptx)
    # The old architecture must still compile without BF16 packed comparisons.
    ptx80 = bytes(compile_cuda(source, "ptx", arch="sm_80", options=default_compile_options())).decode()
    if dtype == "bfloat16":
        assert ".bf16x2" not in ptx80


if __name__ == "__main__":
    tilelang.testing.main()
