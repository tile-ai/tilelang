import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_wgmma_kernel(gemm_op):
    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float16)

            T.copy(A[0:64, 0:16], A_shared)
            T.copy(B[0:16, 0:64], B_shared)
            gemm_op(A_shared, B_shared, C_local)
            T.wait_wgmma(0)
            T.copy(C_local, D[0:64, 0:64])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize(
    "gemm_api",
    [T.wgmma_gemm],
)
def test_wgmma_gemm_has_no_implicit_wait(gemm_api):
    kernel = tilelang.compile(_make_wgmma_kernel(lambda A, B, C: gemm_api(A, B, C, clear_accum=True)), target="cuda")
    src = kernel.get_kernel_source()

    assert src.count("tl::wait_wgmma<0>();") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_dispatch_has_no_implicit_wait():
    kernel = tilelang.compile(
        _make_wgmma_kernel(lambda A, B, C: T.wgmma_gemm(A, B, C, clear_accum=True)),
        target="cuda",
    )

    assert kernel.get_kernel_source().count("tl::wait_wgmma<0>();") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_rejects_mma_fallback():
    @T.prim_func
    def main(
        A: T.Tensor((32, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((32, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float16)

            T.copy(A[0:32, 0:16], A_shared)
            T.copy(B[0:16, 0:64], B_shared)
            T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.wait_wgmma(0)
            T.copy(C_local, D[0:32, 0:64])

    with pytest.raises(
        tvm.error.InternalError,
        match=r"T\.wgmma_gemm\(\) requires Hopper WGMMA lowering",
    ):
        tilelang.compile(main, target="cuda")
