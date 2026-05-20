import tilelang
from tilelang import language as T
import torch


@tilelang.jit
def gemm(A):
    M = T.dynamic("M")
    K = T.const("K")
    num_threads = 256
    warp_size = 32
    num_warps = num_threads // warp_size
    A: T.Tensor[[M, K], T.float32]
    with T.Kernel(T.ceildiv(M, num_warps), threads=num_threads) as pid:
        tid = T.get_thread_binding()
        warp_idx = tid // warp_size
        a_shared = T.alloc_shared((num_warps, K), dtype=T.float32)
        mbars = T.alloc_barrier([warp_size] * num_warps)
        T.tma_copy(A[pid * num_warps + warp_idx, 0:K], a_shared[warp_idx, 0:K], barrier=mbars[warp_idx], thread_extent=warp_size)
        T.mbarrier_arrive(mbarrier=mbars[warp_idx])
        T.mbarrier_wait_parity(mbarrier=mbars[warp_idx], parity=0)
    return None


def test_per_warp_tma_basic():
    A = torch.randn((32, 1024), dtype=torch.float32, device="cuda")
    kernel = gemm.compile(A)
    source = kernel.get_kernel_source()
    assert "tl_shuffle_elect<32>()" in source, "Expected per-warp elect<32>"
    kernel(A)


@tilelang.jit
def kernel_block_tma(A):
    M = T.dynamic("M")
    K = T.const("K")
    num_threads = 256
    A: T.Tensor[[M, K], T.float32]
    with T.Kernel(T.ceildiv(M, num_threads), threads=num_threads) as pid:
        a_shared = T.alloc_shared((1, K), dtype=T.float32)
        mbar = T.alloc_barrier([num_threads])
        T.tma_copy(A[pid, 0:K], a_shared[0, 0:K], barrier=mbar)
        T.mbarrier_arrive(mbarrier=mbar)
        T.mbarrier_wait_parity(mbarrier=mbar, parity=0)
    return None


def test_per_warp_tma_default_block():
    A = torch.randn((256, 1024), dtype=torch.float32, device="cuda")
    kernel = kernel_block_tma.compile(A)
    source = kernel.get_kernel_source()
    assert "tl_shuffle_elect<256>()" in source, "Expected block-wide elect<256>"
