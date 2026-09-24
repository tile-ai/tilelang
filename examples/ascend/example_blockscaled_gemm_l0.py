"""Ascend explicit-L0 block-scaled GEMM example for MXFP8 and MXFP4."""

import argparse

import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

try:
    from .example_blockscaled_gemm import FP4_DTYPE, FP8_DTYPE, make_inputs, ref_program
except ImportError:
    from example_blockscaled_gemm import FP4_DTYPE, FP8_DTYPE, make_inputs, ref_program


def gemm(
    M_DIM=8192,
    K_DIM=8192,
    N_DIM=8192,
    dtype=FP8_DTYPE,
    scale_dtype="uint16",
    scale_packed=True,
):
    """Build a block-scaled GEMM with explicit L1-to-L0 data/SF loads."""

    NUM_BLOCKS = 32
    is_fp4 = dtype == FP4_DTYPE
    if is_fp4 and (scale_dtype != "uint16" or not scale_packed):
        raise ValueError("MXFP4 requires pair-packed uint16 scale factors")

    TILE_M = 256
    TILE_N = 256
    TILE_K = 512 if is_fp4 else 256
    TILE_K_SUB = 256 if is_fp4 else 128
    SUB_K = TILE_K // TILE_K_SUB

    SF_DIV = 64 if scale_dtype == "uint16" else 32
    SF_K = K_DIM // SF_DIV
    TILE_SF_K = TILE_K // SF_DIV
    SF_SUB_K = TILE_K_SUB // SF_DIV
    SF_LOAD_CHUNK_SIZE = 1

    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
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
        X: T.Buffer((M_DIM, K_DIM), dtype),
        W: T.Buffer((N_DIM, K_DIM), dtype),
        SFX: T.Buffer((SF_K, M_DIM) if scale_packed else (M_DIM, SF_K), scale_dtype),
        SFW: T.Buffer((SF_K, N_DIM) if scale_packed else (N_DIM, SF_K), scale_dtype),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            x_l0 = T.alloc_l0a((TILE_M, TILE_K_SUB), dtype)
            w_l0 = T.alloc_l0b((TILE_N, TILE_K_SUB), dtype)
            x_l0_sf = T.alloc_l0a_sf(x_l0, sf_dtype=scale_dtype)
            w_l0_sf = T.alloc_l0b_sf(w_l0, sf_dtype=scale_dtype)
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            xsf_l1 = T.alloc_l1((TILE_M, TILE_SF_K * SF_LOAD_CHUNK_SIZE), scale_dtype)
            wsf_l1 = T.alloc_l1((TILE_N, TILE_SF_K * SF_LOAD_CHUNK_SIZE), scale_dtype)
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")
            sf_int = SF_LOAD_CHUNK_SIZE

            T.annotate_buffer_versions({x_l1: NUM_STAGES, w_l1: NUM_STAGES, xsf_l1: NUM_STAGES, wsf_l1: NUM_STAGES})

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(
                        X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K],
                        x_l1,
                    )
                    T.copy(
                        W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K],
                        w_l1,
                    )

                    if kt % sf_int == 0:
                        if scale_packed:
                            T.copy(
                                SFX[
                                    kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K,
                                    m_tile * TILE_M : (m_tile + 1) * TILE_M,
                                ],
                                xsf_l1,
                                transpose=True,
                            )
                            T.copy(
                                SFW[
                                    kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K,
                                    n_tile * TILE_N : (n_tile + 1) * TILE_N,
                                ],
                                wsf_l1,
                                transpose=True,
                            )
                        else:
                            T.copy(
                                SFX[
                                    m_tile * TILE_M : (m_tile + 1) * TILE_M,
                                    kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K,
                                ],
                                xsf_l1,
                            )
                            T.copy(
                                SFW[
                                    n_tile * TILE_N : (n_tile + 1) * TILE_N,
                                    kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K,
                                ],
                                wsf_l1,
                            )

                    sf_offset = (kt % sf_int) * TILE_SF_K
                    for sk in T.Pipelined(SUB_K, num_stages=NUM_STAGES):
                        sf_start = sf_offset + sk * SF_SUB_K
                        T.copy(
                            x_l1[:, sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB],
                            x_l0,
                        )
                        T.copy(xsf_l1[:, sf_start : sf_start + SF_SUB_K], x_l0_sf)
                        T.copy(
                            w_l1[:, sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB],
                            w_l0,
                        )
                        T.copy(wsf_l1[:, sf_start : sf_start + SF_SUB_K], w_l0_sf)
                        T.gemm_blockscaled(
                            x_l0,
                            w_l0,
                            res,
                            x_l0_sf,
                            w_l0_sf,
                            transpose_B=True,
                            clear_accum=(kt == 0 and sk == 0),
                            unit_flag_ctrl=T.Select(
                                kt == K_TILES - 1 and sk == SUB_K - 1,
                                3,
                                2,
                            ),
                        )

                T.dual_copy(res, temp, unit_flag_ctrl=3)
                T.dual_copy(
                    temp,
                    C[
                        m_tile * TILE_M : (m_tile + 1) * TILE_M,
                        n_tile * TILE_N : (n_tile + 1) * TILE_N,
                    ],
                )

    return main


def run_compile(
    M=8192,
    K=8192,
    N=8192,
    dtype=FP8_DTYPE,
    scale_dtype="uint16",
    scale_packed=True,
    print_source=False,
):
    program = gemm(M, K, N, dtype=dtype, scale_dtype=scale_dtype, scale_packed=scale_packed)
    kernel = tilelang.compile(program, out_idx=-1)
    if print_source:
        print(kernel.get_kernel_source())
    return kernel


def run_regression(
    M=8192,
    K=8192,
    N=8192,
    dtype=FP8_DTYPE,
    scale_dtype="uint16",
    scale_packed=True,
    print_source=False,
    bench=False,
):
    import torch

    device = torch.device("npu")
    kernel = run_compile(M, K, N, dtype, scale_dtype, scale_packed, print_source)
    x, w, sfx, sfw, sfx_e8m0, sfw_e8m0 = make_inputs(M, K, N, dtype, device, scale_dtype, scale_packed)
    result = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()

    expected = ref_program(x, w, sfx_e8m0, sfw_e8m0, K)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel_mean_diff = (result - expected).abs().mean().div(denom).item()
    print(f"L0 {dtype}: rel_mean_diff={rel_mean_diff:.2e}")

    if bench:

        def run_kernel():
            return kernel(x, w, sfx, sfw)

        latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=50)
        flops = 2.0 * M * N * K
        print(f"{latency_ms:.3f} ms/iter | {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")

    return rel_mean_diff


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--dtype", choices=[FP8_DTYPE, FP4_DTYPE], default=FP8_DTYPE)
    parser.add_argument("--scale-dtype", choices=["uint8", "uint16"], default="uint16")
    parser.add_argument("--scale-unpacked", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    scale_packed = not args.scale_unpacked
    if args.compile_only:
        run_compile(
            args.m,
            args.k,
            args.n,
            args.dtype,
            args.scale_dtype,
            scale_packed,
            args.print_source,
        )
        print("Compilation succeeded")
    else:
        run_regression(
            args.m,
            args.k,
            args.n,
            args.dtype,
            args.scale_dtype,
            scale_packed,
            args.print_source,
            args.bench,
        )
