import pytest
import tilelang
from tilelang import language as T
import torch


requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="TMA tests require CUDA compute capability >= 9.0",
)


def per_warp_tma_kernel():
    M = 32
    K = 1024
    num_threads = 256
    warp_size = 32
    num_warps = num_threads // warp_size

    @T.prim_func
    def main(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(T.ceildiv(M, num_warps), threads=num_threads) as pid:
            tid = T.get_thread_binding()
            warp_idx = tid // warp_size
            row = pid * num_warps + warp_idx
            a_shared = T.alloc_shared((num_warps, K), dtype=T.float32)
            mbars = T.alloc_barrier([warp_size] * num_warps)
            T.tma_copy(
                A[row, 0:K],
                a_shared[warp_idx, 0:K],
                barrier=mbars[warp_idx],
                leader_scope_threads=warp_size,
            )
            T.mbarrier_arrive(mbarrier=mbars[warp_idx])
            T.mbarrier_wait_parity(mbarrier=mbars[warp_idx], parity=0)
            if tid % warp_size == 0:
                B[row] = a_shared[warp_idx, 0]

    return main


@requires_sm90
def test_per_warp_tma_basic_codegen():
    kernel = tilelang.compile(per_warp_tma_kernel(), out_idx=[1])
    source = kernel.get_kernel_source()
    assert "tl_shuffle_elect<32>()" in source, "Expected per-warp elect<32>"


def block_tma_kernel():
    M = 256
    K = 1024
    num_threads = 256

    @T.prim_func
    def main(A: T.Tensor((M, K), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(T.ceildiv(M, num_threads), threads=num_threads) as pid:
            tid = T.get_thread_binding()
            a_shared = T.alloc_shared((1, K), dtype=T.float32)
            mbar = T.alloc_barrier([num_threads])
            T.tma_copy(A[pid, 0:K], a_shared[0, 0:K], barrier=mbar)
            T.mbarrier_arrive(mbarrier=mbar)
            T.mbarrier_wait_parity(mbarrier=mbar, parity=0)
            if tid == 0:
                B[pid] = a_shared[0, 0]

    return main


@requires_sm90
def test_per_warp_tma_default_block_codegen():
    kernel = tilelang.compile(block_tma_kernel(), out_idx=[1])
    source = kernel.get_kernel_source()
    assert "tl_shuffle_elect<256>()" in source, "Expected block-wide elect<256>"
