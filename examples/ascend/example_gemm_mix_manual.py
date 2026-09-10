"""Manual Cube+Vector mix kernel GEMM — teaching reference for manual flag/cross-core sync.

This example demonstrates the legacy manual synchronization pattern using:
- T.Cube() and T.Vector() scopes inside T.Kernel()
- T.Serial loops (manual iteration, no auto-scheduling)
- Double-buffered L1 with T.ascend_set_flag / T.ascend_wait_flag
- Cross-core sync with T.ascend_cross_core_set_flag / T.ascend_cross_core_wait_flag
- T.dual_copy for L0C -> UB split across AIV sub-cores
- ASWT swizzle for tile distribution

For production use, prefer the auto-scheduled equivalent in example_gemm_mixedkernel.py.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192):
    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    TILES_PER_CORE = OUT_TILES // NUM_BLOCKS
    WINDOW = min(4, M_TILES)
    NUM_STAGES = 2
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = (tile_idx // WINDOW) % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = (tail_idx // TAIL_WIN) % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile
        return m_tile, n_tile

    @T.prim_func
    def main(
        X: T.Buffer((M_DIM, K_DIM), "bfloat16"),
        W: T.Buffer((N_DIM, K_DIM), "bfloat16"),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((NUM_STAGES, TILE_M, TILE_K), "bfloat16")
            w_l1 = T.alloc_l1((NUM_STAGES, TILE_N, TILE_K), "bfloat16")
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")

            # --- Cube (AIC): GEMM compute ---
            with T.Cube():
                for f in T.Serial(NUM_STAGES):
                    T.ascend_set_flag("MTE1_MTE2", f)
                T.ascend_set_flag("FIX_M", 0)

                for out_tile in T.Serial(TILES_PER_CORE):
                    tile_idx = out_tile * NUM_BLOCKS + bx
                    m_tile, n_tile = aswt_swizzle(tile_idx)

                    for kt in T.Serial(K_TILES):
                        f = kt % NUM_STAGES
                        T.ascend_wait_flag("MTE1_MTE2", f)
                        T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1[f, :, :])
                        T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1[f, :, :])
                        T.ascend_set_flag("MTE2_MTE1", f)

                        T.ascend_wait_flag("MTE2_MTE1", f)
                        if kt == 0:
                            T.ascend_wait_flag("FIX_M", 0)
                        T.gemm(x_l1[f, :, :], w_l1[f, :, :], res, transpose_B=True, clear_accum=(kt == 0))
                        T.ascend_set_flag("MTE1_MTE2", f)

                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 4)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 20)
                    T.dual_copy(res, temp)
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 6)
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 22)
                    T.ascend_set_flag("FIX_M", 0)

                for f in T.Serial(NUM_STAGES):
                    T.ascend_wait_flag("MTE1_MTE2", f)
                T.ascend_wait_flag("FIX_M", 0)
                T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 4)
                T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 20)

            # --- Vector (AIV): GM writeback ---
            with T.Vector() as sid:
                my_aic_flag = 6
                my_aiv_flag = 4
                T.ascend_cross_core_set_flag(4, "PIPE_MTE3", my_aiv_flag)

                for out_tile in T.Serial(TILES_PER_CORE):
                    tile_idx = out_tile * NUM_BLOCKS + bx
                    m_tile, n_tile = aswt_swizzle(tile_idx)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", my_aic_flag)
                    T.copy(temp, C[m_tile * TILE_M + sid * (TILE_M // 2), n_tile * TILE_N])
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE3", my_aiv_flag)

    return main


def ref_program(x, w):
    return x.float() @ w.float().T


NUM_REPEATS = 500
M_DIM = 8192
K_DIM = 8192
N_DIM = 8192


def run_regression_perf(M_DIM=8192, K_DIM=8192, N_DIM=8192, target="ascend"):
    import torch

    device = torch.device("npu")
    program = gemm(M_DIM, K_DIM, N_DIM)
    kernel = tilelang.compile(
        program,
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    x = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    w = torch.randn(N_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    kernel(x, w)
    torch.npu.synchronize()
    latency_ms = do_bench(lambda: kernel(x, w), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    flops = 2.0 * M_DIM * N_DIM * K_DIM
    print(f"{latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")
    return latency_ms


if __name__ == "__main__":
    import torch
    import argparse

    parser = argparse.ArgumentParser(description="Run the manual Cube+Vector GEMM example.")
    parser.add_argument("--target", choices=["ascend"], default="ascend")
    cli_args = parser.parse_args()

    dtype = torch.float32
    device = torch.device("npu")

    print("Compiling gemm kernel...")
    program = gemm(M_DIM, K_DIM, N_DIM)
    kernel = tilelang.compile(
        program,
        target=cli_args.target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    print("Compilation succeeded!")

    print(f"\n--- Generated {cli_args.target} source ---")
    print(kernel.get_kernel_source())

    x = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    w = torch.randn(N_DIM, K_DIM, dtype=torch.bfloat16, device=device)

    print("Running kernel on NPU...")
    c = kernel(x, w)
    torch.npu.synchronize()
    print("Done")

    expected = ref_program(x, w)
    max_diff = torch.max(torch.abs(c - expected)).item()
    print(f"  {'PASS' if max_diff < 1e-2 else 'FAIL'}  rel={max_diff:.2e}")
    if not max_diff < 1e-2:
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")

    latency_ms = do_bench(lambda: kernel(x, w), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    flops = 2.0 * M_DIM * N_DIM * K_DIM
    print(f"{latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")
