import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_sync_tcgen05_kernel(gemm_op):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.bfloat16),
        B: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A[0:128, 0:128], A_shared)
            T.copy(B[0:128, 0:128], B_shared)
            gemm_op(A_shared, B_shared, C_tmem, mbar)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[0:128, 0:128])

    return main


def _make_async_tcgen05_kernel(gemm_op):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.bfloat16),
        B: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)

            T.copy(A[0:128, 0:128], A_shared)
            T.copy(B[0:128, 0:128], B_shared)
            gemm_op(A_shared, B_shared, C_tmem, mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[0:128, 0:128])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    ("sync_api", "async_api"),
    [
        (
            lambda A, B, C, mbar: T.gemm(A, B, C, transpose_B=True, mbar=mbar, clear_accum=True),
            lambda A, B, C, mbar: T.tcgen05_gemm(A, B, C, transpose_B=True, mbar=mbar, clear_accum=True),
        ),
    ],
)
def test_tcgen05_gemm_matches_sync_gemm_codegen(sync_api, async_api):
    sync_kernel = tilelang.compile(_make_sync_tcgen05_kernel(sync_api), target="cuda")
    async_kernel = tilelang.compile(_make_async_tcgen05_kernel(async_api), target="cuda")

    assert sync_kernel.get_kernel_source() == async_kernel.get_kernel_source()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_dispatch_matches_sync_gemm_codegen():
    sync_kernel = tilelang.compile(
        _make_sync_tcgen05_kernel(lambda A, B, C, mbar: T.gemm(A, B, C, transpose_B=True, mbar=mbar, clear_accum=True)),
        target="cuda",
    )
    async_kernel = tilelang.compile(
        _make_async_tcgen05_kernel(lambda A, B, C, mbar: T.tcgen05_gemm(A, B, C, transpose_B=True, mbar=mbar, clear_accum=True)),
        target="cuda",
    )

    assert sync_kernel.get_kernel_source() == async_kernel.get_kernel_source()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_rejects_non_tcgen05_lowering():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.bfloat16),
        B: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_local = T.alloc_fragment((128, 128), T.float32)
            mbar = T.alloc_barrier(1)

            T.copy(A[0:128, 0:128], A_shared)
            T.copy(B[0:128, 0:128], B_shared)
            T.tcgen05_gemm(A_shared, B_shared, C_local, transpose_B=True, mbar=mbar, clear_accum=True)
            T.copy(C_local, D[0:128, 0:128])

    with pytest.raises(
        tvm.error.InternalError,
        match=r"T\.tcgen05_gemm\(\) requires Blackwell TCGEN5MMA lowering",
    ):
        tilelang.compile(main, target="cuda")


def _make_sliced_tcgen05_kernel(M, N, K, num_k_tiles):
    k_tile = K // num_k_tiles

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.bfloat16),
        D: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), T.bfloat16)
            B_shared = T.alloc_shared((N, K), T.bfloat16)
            C_tmem = T.alloc_tmem((M, N), T.float32)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((M, N), T.float32)
            C_shared = T.alloc_shared((M, N), T.bfloat16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            for j in T.serial(num_k_tiles):
                T.tcgen05_gemm(
                    A_shared[:, j * k_tile : (j + 1) * k_tile],
                    B_shared[:, j * k_tile : (j + 1) * k_tile],
                    C_tmem,
                    transpose_B=True,
                    mbar=mbar,
                    clear_accum=(j == 0),
                )
                T.mbarrier_wait_parity(mbar, j % 2)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tcgen05_gemm_sliced_operand_emits_offset():
    # A sliced (non-zero-origin) UMMA operand must build the descriptor from the
    # buffer base and advance it with increase_descriptor_offset (warp-uniform),
    # mirroring the WGMMA path.
    kernel = tilelang.compile(_make_sliced_tcgen05_kernel(128, 128, 128, 2), target="cuda")
    src = kernel.get_kernel_source()
    assert "increase_descriptor_offset" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    "M,N,K,num_k_tiles",
    [
        (128, 128, 128, 2),
        (128, 128, 256, 4),
        (128, 256, 128, 2),
    ],
)
def test_tcgen05_gemm_sliced_operand_correctness(M, N, K, num_k_tiles):
    import torch

    kernel = tilelang.compile(_make_sliced_tcgen05_kernel(M, N, K, num_k_tiles), target="cuda")
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    kernel(a, b, d)
    ref = (a.float() @ b.float().t()).to(torch.bfloat16)
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)
