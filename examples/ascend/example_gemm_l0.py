import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192, trans_a=False, trans_b=True):
    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    TILE_K_SUB = 64
    SUB_K = TILE_K // TILE_K_SUB
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    WINDOW = min(4, M_TILES)
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW
    NUM_STAGES = 2

    a_shape = (K_DIM, M_DIM) if trans_a else (M_DIM, K_DIM)
    b_shape = (N_DIM, K_DIM) if trans_b else (K_DIM, N_DIM)

    a_l1_rows = TILE_M if not trans_a else TILE_K
    a_l1_cols = TILE_K if not trans_a else TILE_M
    b_l1_rows = TILE_N if trans_b else TILE_K
    b_l1_cols = TILE_K if trans_b else TILE_N

    a_l0_rows = TILE_M if not trans_a else TILE_K_SUB
    a_l0_cols = TILE_K_SUB if not trans_a else TILE_M
    b_l0_rows = TILE_N if trans_b else TILE_K_SUB
    b_l0_cols = TILE_K_SUB if trans_b else TILE_N

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
        X: T.Buffer(a_shape, "bfloat16"),
        W: T.Buffer(b_shape, "bfloat16"),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            x_l0 = T.alloc_l0a((a_l0_rows, a_l0_cols), "bfloat16")
            w_l0 = T.alloc_l0b((b_l0_rows, b_l0_cols), "bfloat16")
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((a_l1_rows, a_l1_cols), "bfloat16")
            w_l1 = T.alloc_l1((b_l1_rows, b_l1_cols), "bfloat16")
            # Keep the next K tile's GM->L1 loads overlapped with the current
            # tile's four MTE1/Cube sub-K steps.
            T.annotate_buffer_versions({x_l1: NUM_STAGES, w_l1: NUM_STAGES})
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    # GM -> L1
                    if trans_a:
                        T.copy(X[kt * TILE_K : (kt + 1) * TILE_K, m_tile * TILE_M : (m_tile + 1) * TILE_M], x_l1)
                    else:
                        T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    if trans_b:
                        T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    else:
                        T.copy(W[kt * TILE_K : (kt + 1) * TILE_K, n_tile * TILE_N : (n_tile + 1) * TILE_N], w_l1)
                    # L1 -> L0
                    for sk in T.Pipelined(SUB_K, num_stages=NUM_STAGES):
                        if trans_a:
                            T.copy(x_l1[sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB, :], x_l0)
                        else:
                            T.copy(x_l1[:, sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB], x_l0)
                        if trans_b:
                            T.copy(w_l1[:, sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB], w_l0)
                        else:
                            T.copy(w_l1[sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB, :], w_l0)
                        T.gemm(x_l0, w_l0, res, transpose_A=trans_a, transpose_B=trans_b, clear_accum=(kt == 0 and sk == 0))
                T.dual_copy(res, temp)
                T.dual_copy(temp, C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N])

    return main


def ref_program(x, w, trans_a, trans_b):
    am = x.T if trans_a else x
    bm = w.T if trans_b else w
    return am.float() @ bm.float()


NUM_REPEATS = 500
M_DIM = 8192
K_DIM = 8192
N_DIM = 8192


def run_regression_perf(M_DIM=8192, K_DIM=8192, N_DIM=8192, target="ascend"):
    import torch

    device = torch.device("npu")
    program = gemm(M_DIM, K_DIM, N_DIM, trans_a=False, trans_b=True)
    kernel = tilelang.compile(program, target=target, out_idx=-1)
    x = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    w = torch.randn(N_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    kernel(x, w)
    torch.npu.synchronize()
    latency_ms = do_bench(lambda: kernel(x, w), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    flops = 2.0 * M_DIM * N_DIM * K_DIM
    print(f"{latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")
    return latency_ms


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Ascend GEMM L0 example.")
    parser.add_argument("--target", choices=["ascend"], default="ascend")
    return parser.parse_args()


if __name__ == "__main__":
    import torch

    args = _parse_args()
    target = args.target
    device = torch.device("npu")

    # Correctness: test all 4 trans_A/trans_B combos at moderate size
    print(f"=== Correctness checks (2048 x 2048 x 2048, target={target}) ===")
    M, K, N = 2048, 2048, 2048
    all_pass = True
    for ta in (False, True):
        for tb in (False, True):
            tag = f"trans_a={ta}, trans_b={tb}"
            try:
                program = gemm(M, K, N, trans_a=ta, trans_b=tb)
                kernel = tilelang.compile(program, target=target, out_idx=-1)
            except Exception as e:
                print(f"[{tag}] COMPILE FAIL: {type(e).__name__}: {e}")
                all_pass = False
                continue

            a_shape = (K, M) if ta else (M, K)
            b_shape = (N, K) if tb else (K, N)
            x = torch.randn(*a_shape, dtype=torch.bfloat16, device=device)
            w = torch.randn(*b_shape, dtype=torch.bfloat16, device=device)
            try:
                c = kernel(x, w)
                torch.npu.synchronize()
            except Exception as e:
                print(f"[{tag}] RUN FAIL: {type(e).__name__}: {e}")
                all_pass = False
                continue

            expected = ref_program(x, w, ta, tb)
            max_diff = torch.max(torch.abs(c.float() - expected)).item()
            rel = max_diff / (torch.max(torch.abs(expected)).item() + 1e-6)
            passed = rel < 1e-2
            print(f"[{tag}] {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.3e} rel={rel:.6e}")
            if not passed:
                all_pass = False
    if not all_pass:
        raise AssertionError("Correctness check failed!")

    # Performance benchmark: original 8192 x 8192 x 8192 NT path
    print("\n=== Performance benchmark (8192 x 8192 x 8192, NT) ===")
    run_regression_perf(M_DIM, K_DIM, N_DIM, target=target)
