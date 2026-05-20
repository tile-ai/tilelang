"""Round-trip test for TMA tile::gather4 / tile::scatter4 (sm_100a, Blackwell)."""

import pytest

from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import tilelang


def _has_sm100():
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 10


requires_sm100 = pytest.mark.skipif(not _has_sm100(), reason="tile::gather4/scatter4 require sm_100a (Blackwell)")


def gather_scatter_program(N: int, K: int, K_box: int, in_dtype: str = "float16"):

    @T.prim_func
    def main(
        Src: T.Tensor((N, K), in_dtype),
        Idx: T.Tensor((4,), "int32"),
        Dst: T.Tensor((N, K), in_dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            T.reads(Src[0:N, 0:K], Idx[0:4])
            T.writes(Dst[0:N, 0:K])

            smem = T.alloc_shared((4, K_box), in_dtype)
            mbar = T.alloc_barrier(1)

            r0 = Idx[0]
            r1 = Idx[1]
            r2 = Idx[2]
            r3 = Idx[3]

            if T.shuffle_elect(128):
                T.mbarrier_expect_tx(mbar, T.tma_gather4_bytes(K_box, in_dtype))
                T.tma_gather4(Src, smem, 0, [r0, r1, r2, r3], barrier=mbar)
                T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)

            if T.shuffle_elect(128):
                T.tma_scatter4(smem, Dst, 0, [r0, r1, r2, r3])
                T.tma_store_arrive()
            T.tma_store_wait(0)

    return main


def run_gather_scatter(N=64, K=64, K_box=64):
    program = gather_scatter_program(N=N, K=K, K_box=K_box)
    kernel = tilelang.compile(
        program,
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    src = kernel.get_kernel_source()
    assert "tma_load_gather4" in src, "tma_load_gather4 missing from emitted CUDA"
    assert "tma_store_scatter4" in src, "tma_store_scatter4 missing from emitted CUDA"
    assert "CUtensorMap" in src, "CUtensorMap descriptor missing from kernel signature"

    import torch

    Src = torch.randn(N, K, dtype=torch.float16, device="cuda")
    Idx = torch.tensor([5, 17, 42, 9], dtype=torch.int32, device="cuda")
    Dst = torch.zeros_like(Src)

    kernel(Src, Idx, Dst)
    torch.cuda.synchronize()

    expected = torch.zeros_like(Src)
    rows = Idx.tolist()
    for r in rows:
        expected[r] = Src[r]

    torch.testing.assert_close(Dst, expected)


@requires_sm100
def test_gather_scatter_basic():
    run_gather_scatter(N=64, K=64, K_box=64)


# Swizzled round-trip: LowerBulkGather4 infers desc.swizzle from the annotated
# shared layout via DetectSwizzleMode. K_box * 2 bytes must match the swizzle
# period: 64→128B, 32→64B, 16→32B fp16.


def gather_scatter_swizzled_program(N: int, K: int, K_box: int, swizzle_kind: str, in_dtype: str = "float16"):
    from tilelang.layout import (
        make_full_bank_swizzled_layout,
        make_half_bank_swizzled_layout,
        make_quarter_bank_swizzled_layout,
    )

    swizzle_factories = {
        "128B": make_full_bank_swizzled_layout,
        "64B": make_half_bank_swizzled_layout,
        "32B": make_quarter_bank_swizzled_layout,
    }
    make_layout = swizzle_factories[swizzle_kind]

    @T.prim_func
    def main(
        Src: T.Tensor((N, K), in_dtype),
        Idx: T.Tensor((4,), "int32"),
        Dst: T.Tensor((N, K), in_dtype),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            T.reads(Src[0:N, 0:K], Idx[0:4])
            T.writes(Dst[0:N, 0:K])

            smem = T.alloc_shared((4, K_box), in_dtype)
            T.annotate_layout({smem: make_layout(smem)})

            mbar = T.alloc_barrier(1)

            r0 = Idx[0]
            r1 = Idx[1]
            r2 = Idx[2]
            r3 = Idx[3]

            if T.shuffle_elect(128):
                T.mbarrier_expect_tx(mbar, T.tma_gather4_bytes(K_box, in_dtype))
                T.tma_gather4(Src, smem, 0, [r0, r1, r2, r3], barrier=mbar)
                T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)

            if T.shuffle_elect(128):
                T.tma_scatter4(smem, Dst, 0, [r0, r1, r2, r3])
                T.tma_store_arrive()
            T.tma_store_wait(0)

    return main


def run_gather_scatter_swizzled(N, K, K_box, swizzle_kind):
    program = gather_scatter_swizzled_program(N=N, K=K, K_box=K_box, swizzle_kind=swizzle_kind)
    kernel = tilelang.compile(
        program,
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    src = kernel.get_kernel_source()
    assert "tma_load_gather4" in src
    assert "tma_store_scatter4" in src

    import torch

    Src = torch.randn(N, K, dtype=torch.float16, device="cuda")
    Idx = torch.tensor([5, 17, 42, 9], dtype=torch.int32, device="cuda")
    Dst = torch.zeros_like(Src)

    kernel(Src, Idx, Dst)
    torch.cuda.synchronize()

    expected = torch.zeros_like(Src)
    rows = Idx.tolist()
    for r in rows:
        expected[r] = Src[r]

    torch.testing.assert_close(Dst, expected)


@requires_sm100
@pytest.mark.parametrize(
    "K_box, swizzle_kind",
    [
        (64, "128B"),  # row = 128 bytes fp16 -> full-bank swizzle
        (32, "64B"),  # row =  64 bytes fp16 -> half-bank swizzle
        (16, "32B"),  # row =  32 bytes fp16 -> quarter-bank swizzle
    ],
)
def test_gather_scatter_swizzled(K_box, swizzle_kind):
    run_gather_scatter_swizzled(N=64, K=K_box, K_box=K_box, swizzle_kind=swizzle_kind)


if __name__ == "__main__":
    tilelang.testing.main()
