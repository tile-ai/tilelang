"""L0C fp32 -> UB f16/bf16 ``dual_copy`` on-path cast.

Dual + ``quant_pre`` is illegal, so lowering emits two non-dual FixPipes.
Each AIV gets a half-tile already in the destination dtype.

Run on an Ascend NPU; checks torch accuracy and benches M- and N-split.
"""

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192, dtype="bfloat16", dst_dtype="float16", split="M"):
    """GEMM with L0C fp32 -> UB f16/bf16 FixPipe quant.

    Args:
        dtype: input A/B dtype ("bfloat16" or "float16").
        dst_dtype: output/UB dtype of the on-path cast ("float16" or "bfloat16").
        split: "M" (rows split across AIVs) or "N" (columns split across AIVs).
    """
    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    WINDOW = min(4, M_TILES)
    NUM_STAGES = 2
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW

    half_m = TILE_M // 2
    half_n = TILE_N // 2
    dst_shape = (half_m, TILE_N) if split == "M" else (TILE_M, half_n)

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
        X: T.Buffer((M_DIM, K_DIM), dtype),
        W: T.Buffer((N_DIM, K_DIM), dtype),
        C: T.Buffer((M_DIM, N_DIM), dst_dtype),
    ):
        with T.MixedKernel(NUM_BLOCKS) as (bx, _):
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            temp = T.alloc_shared(dst_shape, dst_dtype)

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))

                # fp32 L0C -> f16/bf16 UB, fanned out across the two AIVs, then
                # each AIV stores its half back into the matching half of C.
                T.dual_copy(res, temp)
                T.dual_copy(
                    temp,
                    C[
                        m_tile * TILE_M : (m_tile + 1) * TILE_M,
                        n_tile * TILE_N : (n_tile + 1) * TILE_N,
                    ],
                )

    return main


def ref_program(x, w, dst_dtype):
    out = x.float() @ w.float().T
    return out.to(getattr(torch, dst_dtype))


def run_regression_perf(M=8192, K=8192, N=8192, dtype="bfloat16", dst_dtype="float16", split="M"):
    """Compile + benchmark a single on-path-cast GEMM; returns latency in ms."""
    device = torch.device("npu")
    program = gemm(M, K, N, dtype=dtype, dst_dtype=dst_dtype, split=split)
    kernel = tilelang.compile(program, out_idx=-1)

    x = torch.randn(M, K, dtype=getattr(torch, dtype), device=device)
    w = torch.randn(N, K, dtype=getattr(torch, dtype), device=device)

    kernel(x, w)
    torch.npu.synchronize()

    NUM_REPEATS = 100 if dtype == "bfloat16" else 50

    def run_kernel(kernel=kernel, x=x, w=w):
        return kernel(x, w)

    latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    flops = 2.0 * M * N * K
    print(f"  {latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")
    return latency_ms


if __name__ == "__main__":
    device = torch.device("npu")
    M, K, N = 8192, 8192, 8192
    dtype = "bfloat16"

    for dst_dtype, thresh in [("float16", 2e-2), ("bfloat16", 5e-2)]:
        for split in ["M", "N"]:
            print(f"\n=== fp32 -> {dst_dtype} on-path cast, {split}-split ===")
            program = gemm(M, K, N, dtype=dtype, dst_dtype=dst_dtype, split=split)
            kernel = tilelang.compile(program, out_idx=-1)
            print("Compilation succeeded!")

            x = torch.randn(M, K, dtype=getattr(torch, dtype), device=device)
            w = torch.randn(N, K, dtype=getattr(torch, dtype), device=device)

            c = kernel(x, w)
            torch.npu.synchronize()

            expected = ref_program(x, w, dst_dtype)
            # Relative L-inf error, robust against the ~sqrt(K) output magnitude.
            rel = (c.float() - expected.float()).abs().max() / expected.float().abs().max()
            print(f"  {'PASS' if rel < thresh else 'FAIL'}  rel_max={rel:.2e}")
            if not rel < thresh:
                raise AssertionError(f"Results mismatch! rel_max={rel:.2e}")

            run_regression_perf(M, K, N, dtype=dtype, dst_dtype=dst_dtype, split=split)
