import importlib.util
import os

import pytest

if os.environ.get("TILELANG_LIGHT_IMPORT") == "1":
    pytest.skip("CuTeDSL execution requires the full frontend", allow_module_level=True)

import tilelang
import tilelang.language as T


pytestmark = pytest.mark.skipif(
    os.environ.get("TILELANG_LIGHT_IMPORT") == "1" or importlib.util.find_spec("cutlass") is None,
    reason="CuTeDSL runtime requires normal TileLang import and CUTLASS DSL",
)


def copy_kernel(length):
    @T.prim_func
    def increment(source: T.Tensor((length,), "float32"), output: T.Tensor((length,), "float32")):
        with T.Kernel(T.ceildiv(length, 64), threads=32) as block:
            shared = T.alloc_shared((64,), "float32")
            T.copy(source[block * 64 : (block + 1) * 64], shared, disable_tma=True)
            for index in T.Parallel(64):
                if block * 64 + index < length:
                    output[block * 64 + index] = shared[index] + 1.0

    return increment


@pytest.mark.parametrize("dynamic", [False, True])
def test_cutedsl_shared_copy_executes(dynamic):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")
    length = T.dynamic("length") if dynamic else 128
    kernel = tilelang.compile(copy_kernel(length), target="cutedsl", execution_backend="cutedsl", out_idx=[1])
    for size in [65, 128, 193] if dynamic else [128]:
        source = torch.arange(size, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(kernel(source), source + 1.0)


def test_cutedsl_host_control_flow_and_stream():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")

    @T.prim_func
    def staged(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32"), repeats: T.int32):
        with T.Kernel(1, threads=32):
            for index in T.Parallel(64):
                output[index] = source[index]
        for phase in T.serial(repeats):
            if phase % 2 == 0:
                with T.Kernel(1, threads=32):
                    for index in T.Parallel(64):
                        output[index] += 1.0
            else:
                with T.Kernel(1, threads=32):
                    for index in T.Parallel(64):
                        output[index] += 2.0

    kernel = tilelang.compile(staged, out_idx=[1], target="cutedsl", execution_backend="cutedsl")
    with torch.cuda.stream(torch.cuda.Stream()):
        for repeats in (0, 1, 4):
            source = torch.randn(64, device="cuda")
            torch.testing.assert_close(kernel(source, repeats), source + repeats // 2 * 3 + repeats % 2)


def test_cutedsl_fragment_reduction():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")

    @T.prim_func
    def row_sum(source: T.Tensor((4, 64), "float32"), output: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((4, 64), "float32")
            reduced = T.alloc_fragment((4,), "float32")
            T.copy(source, fragment)
            T.reduce_sum(fragment, reduced, dim=1)
            T.copy(reduced, output)

    kernel = tilelang.compile(row_sum, out_idx=[1], target="cutedsl", execution_backend="cutedsl")
    source = torch.randn(4, 64, device="cuda")
    torch.testing.assert_close(kernel(source), source.sum(dim=1))


@pytest.mark.parametrize("input_dtype", ["float16", "bfloat16"])
def test_cutedsl_gemm(input_dtype):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")

    @T.prim_func
    def matmul(lhs: T.Tensor((64, 32), input_dtype), rhs: T.Tensor((32, 64), input_dtype), output: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, threads=128):
            lhs_shared = T.alloc_shared((64, 32), input_dtype)
            rhs_shared = T.alloc_shared((32, 64), input_dtype)
            accumulator = T.alloc_fragment((64, 64), "float32")
            T.copy(lhs, lhs_shared)
            T.copy(rhs, rhs_shared)
            T.clear(accumulator)
            T.gemm(lhs_shared, rhs_shared, accumulator)
            T.copy(accumulator, output)

    kernel = tilelang.compile(matmul, out_idx=[2], target="cutedsl", execution_backend="cutedsl")
    lhs = torch.randn(64, 32, device="cuda", dtype=getattr(torch, input_dtype))
    rhs = torch.randn(32, 64, device="cuda", dtype=getattr(torch, input_dtype))
    torch.testing.assert_close(kernel(lhs, rhs), lhs.float() @ rhs.float(), atol=0.02, rtol=0.002)
