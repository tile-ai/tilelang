"""CuTeDSL TMA tile::gather4 / tile::scatter4, mirroring the CUDA coverage in
test_tilelang_language_tma_gather_scatter.py."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def gather_scatter_program(N: int, K: int, K_box: int, in_dtype: str = "float16", eviction_policy=None):

    @T.prim_func
    def main(
        Src: T.Tensor((N, K), in_dtype),
        Idx: T.Tensor((4,), "int32"),
        Dst: T.Tensor((N, K), in_dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            smem = T.alloc_shared((4, K_box), in_dtype)
            mbar = T.alloc_barrier(1)

            r0 = Idx[0]
            r1 = Idx[1]
            r2 = Idx[2]
            r3 = Idx[3]

            if T.shuffle_elect(128):
                T.mbarrier_expect_tx(mbar, T.tma_gather4_bytes(K_box, in_dtype))
                T.tma_gather4(Src, smem, 0, [r0, r1, r2, r3], barrier=mbar, eviction_policy=eviction_policy)
                T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)

            if T.shuffle_elect(128):
                T.tma_scatter4(smem, Dst, 0, [r0, r1, r2, r3], eviction_policy=eviction_policy)
                T.tma_store_arrive()
            T.tma_store_wait(0, read=False)

    return main


def run_gather_scatter(N=64, K=64, K_box=64, eviction_policy=None):
    program = gather_scatter_program(N=N, K=K, K_box=K_box, eviction_policy=eviction_policy)
    kernel = tilelang.compile(
        program,
        target="cutedsl",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    Src = torch.randn(N, K, dtype=torch.float16, device="cuda")
    Idx = torch.tensor([5, 17, 42, 9], dtype=torch.int32, device="cuda")
    Dst = torch.zeros_like(Src)

    kernel(Src, Idx, Dst)
    torch.cuda.synchronize()

    expected = torch.zeros_like(Src)
    for r in Idx.tolist():
        expected[r] = Src[r]

    torch.testing.assert_close(Dst, expected)


def gather_readback_program(N: int, K: int, K_box: int):
    """Gather four rows, then copy shared memory out in order.

    Reading the rows back through an ordinary copy pins the row order, which a
    gather-then-scatter round trip with the same indices cannot.
    """

    @T.prim_func
    def main(
        Src: T.Tensor((N, K), "float16"),
        Idx: T.Tensor((4,), "int32"),
        Out: T.Tensor((4, K_box), "float16"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            smem = T.alloc_shared((4, K_box), "float16")
            mbar = T.alloc_barrier(1)
            if T.shuffle_elect(128):
                T.mbarrier_expect_tx(mbar, T.tma_gather4_bytes(K_box, "float16"))
                T.tma_gather4(Src, smem, 0, [Idx[0], Idx[1], Idx[2], Idx[3]], barrier=mbar)
                T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(smem, Out)

    return main


def scatter_only_program(N: int, K: int, K_box: int):
    """Fill shared memory in order, then scatter it to four chosen rows."""

    @T.prim_func
    def main(
        Src: T.Tensor((4, K_box), "float16"),
        Idx: T.Tensor((4,), "int32"),
        Dst: T.Tensor((N, K), "float16"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            smem = T.alloc_shared((4, K_box), "float16")
            T.copy(Src, smem)
            if T.shuffle_elect(128):
                T.tma_scatter4(smem, Dst, 0, [Idx[0], Idx[1], Idx[2], Idx[3]])
                T.tma_store_arrive()
            T.tma_store_wait(0, read=False)

    return main


@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_gather_scatter_cutedsl():
    _require_cutedsl()
    run_gather_scatter(N=64, K=64, K_box=64)


@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_gather_rows_cutedsl():
    _require_cutedsl()
    N, K = 64, 64
    kernel = tilelang.compile(gather_readback_program(N, K, K), target="cutedsl")
    Src = torch.randn(N, K, dtype=torch.float16, device="cuda")
    rows = torch.tensor([3, 40, 17, 61], dtype=torch.int32, device="cuda")
    Out = torch.zeros(4, K, dtype=torch.float16, device="cuda")
    kernel(Src, rows, Out)
    torch.cuda.synchronize()
    torch.testing.assert_close(Out, Src[rows.tolist()])


@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_scatter_rows_cutedsl():
    _require_cutedsl()
    N, K = 64, 32
    kernel = tilelang.compile(scatter_only_program(N, K, K), target="cutedsl")
    Src = torch.randn(4, K, dtype=torch.float16, device="cuda")
    rows = torch.tensor([9, 2, 33, 58], dtype=torch.int32, device="cuda")
    Dst = torch.zeros(N, K, dtype=torch.float16, device="cuda")
    kernel(Src, rows, Dst)
    torch.cuda.synchronize()
    expected = torch.zeros_like(Dst)
    expected[rows.tolist()] = Src
    torch.testing.assert_close(Dst, expected)


@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_gather_scatter_eviction_cutedsl():
    _require_cutedsl()
    run_gather_scatter(N=64, K=64, K_box=64, eviction_policy="evict_first")


if __name__ == "__main__":
    tilelang.testing.main()
