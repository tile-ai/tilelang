import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def gemm(M_DIM=256, N_DIM=7168, K_DIM=2048, TILE_M=256, TILE_N=224, TILE_K=128, NUM_STAGES=2):
    NUM_BLOCKS = 32
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES

    l2_ctrl_b = "NOTALLOC_KEEP" if M_TILES <= 2 else "NORMAL_FV"
    l2_ctrl_d = "NOTALLOC_PW"

    @T.macro
    def aswt_swizzle(tile_idx):
        return tile_idx // N_TILES, tile_idx % N_TILES

    @T.prim_func
    def main(
        X: T.Buffer((M_DIM, K_DIM), "bfloat16"),
        W: T.Buffer((N_DIM, K_DIM), "bfloat16"),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), "bfloat16")
            w_l1 = T.alloc_l1((TILE_N, TILE_K), "bfloat16")
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1, l2_cache_ctrl=l2_ctrl_b)
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
                T.dual_copy(res, temp)
                T.dual_copy(
                    temp, C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N], l2_cache_ctrl=l2_ctrl_d
                )

    return main


def ref_program(x, w):
    return x.float() @ w.float().T


def run_regression_perf(M=256, N=7168, K=2048):
    import torch

    device = torch.device("npu")
    program = gemm(M, N, K)
    kernel = tilelang.compile(program, out_idx=-1)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    kernel(x, w)
    torch.npu.synchronize()
    latency_ms = do_bench(lambda: kernel(x, w), backend="msprof", _n_warmup=30, _n_repeat=100)
    us = latency_ms * 1e3
    flops = 2.0 * M * N * K
    tf = flops / (us / 1e6) / 1e12
    print(f"  {us:.1f} us  |  {tf:.1f} TFLOPS")
    return latency_ms


if __name__ == "__main__":
    import torch

    device = torch.device("npu")
    M, N, K = 256, 7168, 2048

    program = gemm(M, N, K)
    kernel = tilelang.compile(program, out_idx=-1)
    print("Compilation succeeded!")

    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    c = kernel(x, w)
    torch.npu.synchronize()

    expected = ref_program(x, w)
    max_diff = torch.max(torch.abs(c - expected)).item()
    print(f"  {'PASS' if max_diff < 1e-2 else 'FAIL'}  max_diff={max_diff:.2e}")

    latency_ms = do_bench(lambda: kernel(x, w), backend="msprof", _n_warmup=30, _n_repeat=100)
    us = latency_ms * 1e3
    flops = 2.0 * M * N * K
    tf = flops / (us / 1e6) / 1e12
    print(f"  {us:.1f} us  |  {tf:.1f} TFLOPS")
