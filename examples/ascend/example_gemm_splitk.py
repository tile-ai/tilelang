"""Split-K GEMM with atomic or deterministic ordered reduction.

All physical cores participate in the inter-core barriers, so each worker group
must process the same number of output tiles. AutoSchedule manages intra-core sync.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def gemm_splitk(M_DIM=8192, K_DIM=8192, N_DIM=8192, split_k=4, deterministic=False):
    """BF16 GEMM with explicit cross-core reduction and FP32 output."""
    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    NUM_STAGES = 2
    INTER_CORE_FLAG = 0
    if split_k <= 1:
        raise ValueError(f"split_k must be greater than 1, got {split_k}")
    if NUM_BLOCKS % split_k != 0:
        raise ValueError(f"split_k must divide {NUM_BLOCKS}, got {split_k}")
    if M_DIM % TILE_M != 0 or N_DIM % TILE_N != 0:
        raise ValueError(f"M and N must be multiples of ({TILE_M}, {TILE_N}), got ({M_DIM}, {N_DIM})")
    if K_DIM % TILE_K != 0:
        raise ValueError(f"K must be a multiple of {TILE_K}, got {K_DIM}")
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    NUM_GROUPS = NUM_BLOCKS // split_k
    OUT_TILES = M_TILES * N_TILES
    if K_TILES % split_k != 0:
        raise ValueError(f"K tile count must be divisible by split_k, got K_TILES={K_TILES}, split_k={split_k}")
    if OUT_TILES % NUM_GROUPS != 0:
        raise ValueError(
            f"output tile count must be divisible by the number of worker groups, got OUT_TILES={OUT_TILES}, NUM_GROUPS={NUM_GROUPS}"
        )
    K_TILES_PER_SPLIT = K_TILES // split_k
    TILES_PER_GROUP = OUT_TILES // NUM_GROUPS
    WINDOW = min(4, M_TILES)
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = tile_idx // WINDOW % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = tail_idx // TAIL_WIN % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile
        return (m_tile, n_tile)

    @T.prim_func
    def main(X: T.Tensor((M_DIM, K_DIM), "bfloat16"), W: T.Tensor((N_DIM, K_DIM), "bfloat16"), C: T.Tensor((M_DIM, N_DIM), "float32")):
        with T.Kernel(NUM_BLOCKS) as bx:
            split_id = bx % split_k
            group_id = bx // split_k
            x_l1 = T.alloc_l1((TILE_M, TILE_K), "bfloat16")
            w_l1 = T.alloc_l1((TILE_N, TILE_K), "bfloat16")
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            if split_id != 0:
                T.set_atomic("add", "float32")
            for out_tile in T.Serial(TILES_PER_GROUP):
                tile_idx = out_tile * NUM_GROUPS + group_id
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for local_kt in T.Pipelined(K_TILES_PER_SPLIT, num_stages=NUM_STAGES):
                    kt = split_id * K_TILES_PER_SPLIT + local_kt
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=local_kt == 0)
                if deterministic:
                    # Every core participates in every phase of the ordered reduction.
                    with T.PerCoreTask():
                        for store_split in T.Serial(split_k):
                            if split_id == store_split:
                                T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])
                            T.ascend_sync_inter_arrive("PIPE_FIX", INTER_CORE_FLAG)
                            T.ascend_sync_inter_wait("PIPE_FIX", INTER_CORE_FLAG)
                else:
                    # Publish split 0 before other cores atomically add their partials.
                    with T.PerCoreTask():
                        if split_id == 0:
                            T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])
                        T.ascend_sync_inter_arrive("PIPE_FIX", INTER_CORE_FLAG)
                        T.ascend_sync_inter_wait("PIPE_FIX", INTER_CORE_FLAG)
                        if split_id != 0:
                            T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])
            if split_id != 0:
                T.set_atomic_none()

    return main


def ref_program(x, w):
    return x.float() @ w.float().T


def run_regression_perf(M_DIM=512, K_DIM=8192, N_DIM=512, split_k=8, deterministic=False):
    kernel = tilelang.compile(gemm_splitk(M_DIM, K_DIM, N_DIM, split_k, deterministic), target="ascend", out_idx=-1)
    x = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16, device="npu")
    w = torch.randn(N_DIM, K_DIM, dtype=torch.bfloat16, device="npu")
    latency = do_bench(lambda: kernel(x, w), backend="msprof", _n_warmup=30, _n_repeat=50)
    tflops = 2.0 * M_DIM * N_DIM * K_DIM / (latency * 1e9)
    print(
        f"gemm_splitk (M={M_DIM}, N={N_DIM}, K={K_DIM}, split_k={split_k}, deterministic={deterministic}): "
        f"{latency * 1000:.2f} us | {tflops:.1f} TFLOPS"
    )
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    args = parser.parse_args()
    x = torch.randn(512, 4096, dtype=torch.bfloat16, device="npu")
    w = torch.randn_like(x)
    kernel = tilelang.compile(gemm_splitk(512, 4096, 512, 8, args.deterministic), target="ascend", out_idx=-1)
    result = kernel(x, w)
    torch.testing.assert_close(result, ref_program(x, w), rtol=1e-2, atol=1e-2)
    if args.deterministic:
        torch.testing.assert_close(kernel(x, w), result, rtol=0, atol=0)
    print(f"gemm_splitk (deterministic={args.deterministic}): correctness passed")
    if args.bench:
        run_regression_perf(deterministic=args.deterministic)
