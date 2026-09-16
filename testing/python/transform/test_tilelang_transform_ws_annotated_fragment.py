"""Regression test: auto producer/consumer WS must not drop statements whose loop layout comes from a
T.annotate_layout fragment (Fragment without thread_range) in the consumer branch.

Fails on the unpatched pass (exp2/AllReduce vanish from the source, outputs stay NaN); passes with
the thread-range binding patch to src/cuda/transform/producer_consumer_ws.cc.
Requires a TMA-capable GPU (sm_90+, verified on sm_120a)."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Fragment

M, N, K, BK, THREADS = 64, 64, 256, 64, 128


def build():
    warp_rows = M // (THREADS // 32)  # FullRow: 16 rows per warp; mma m16n8 C-fragment thread map

    def fwd_thread(i, j):
        return (i // warp_rows) * 32 + (i % 8) * 4 + (j % 8) // 2

    def fwd_index(i, j):
        return (j // 8) * 4 + ((i % warp_rows) // 8) * 2 + j % 2

    g_layout = Fragment((M, N), forward_thread_fn=fwd_thread, forward_index_fn=fwd_index)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
        R: T.Tensor((M,), T.float32),
    ):
        with T.Kernel(1, threads=THREADS) as _:
            A_sh = T.alloc_shared((M, BK), T.float16)
            B_sh = T.alloc_shared((N, BK), T.float16)
            acc = T.alloc_fragment((M, N), T.float32)
            G = T.alloc_fragment((M, N), T.float32)
            rs = T.alloc_fragment((M,), T.float32)
            T.annotate_layout({G: g_layout})  # user fragment: no thread_range
            T.fill(acc, 0)
            T.fill(rs, 0)
            for k in T.Pipelined(K // BK, num_stages=2):
                T.copy(A[0, k * BK], A_sh)  # TMA producers -> pass fires, consumer moves to tx >= 128
                T.copy(B[0, k * BK], B_sh)
                T.gemm(A_sh, B_sh, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(M, N):  # loop layout := G's annotated layout
                    G[i, j] = T.exp2(acc[i, j] * 0.01)
                T.reduce_sum(G, rs, dim=1, clear=False)  # reducer layout inherits G's thread range
            T.copy(G, C)  # copy loop layout := G's annotated layout
            T.copy(rs, R)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_keeps_annotated_fragment_statements():
    cfg = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = tilelang.jit(pass_configs=cfg)(build)()
    src = kernel.get_kernel_source()
    assert src.count("exp2f") >= 1, "exp2 loop over the annotated fragment was dropped"
    assert src.count("AllReduce") >= 1, "reduce_sum over the annotated fragment was dropped"
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="cuda")  # NaN sentinel: dropped stores leave NaN
    r = torch.full((M,), float("nan"), device="cuda")
    kernel(a, b, c, r)
    torch.cuda.synchronize()
    c_ref = torch.exp2((a.float() @ b.float().T) * 0.01)
    r_ref = torch.zeros(M, device="cuda")
    for kt in range(K // BK):  # rs accumulates rowsum(exp2(partial acc)) every k-iteration
        r_ref += torch.exp2((a[:, : (kt + 1) * BK].float() @ b[:, : (kt + 1) * BK].float().T) * 0.01).sum(1)
    torch.testing.assert_close(c, c_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(r, r_ref, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
