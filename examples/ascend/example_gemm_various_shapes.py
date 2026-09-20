import argparse
import math
import sys
from typing import Generator

import torch

import tilelang
import tilelang.ascend.language as T

from tilelang.profiler import do_bench


def enumerate_shapes() -> Generator:
    """Enumerate (M, N, K) test shapes covering compute-bound and memory-bound regimes."""
    for d in [256, 512, 1024, 2048, 4096, 8192]:
        yield d, d, d
    m_list = [256, 1024, 4096]
    nk_list = [
        (4096, 4096),
        (4096, 14336),
        (14336, 4096),
        (2048, 7168),
        (7168, 2048),
    ]
    for m in m_list:
        for n, k in nk_list:
            if m != n or n != k:
                yield m, n, k
    for m in [64, 128]:
        for n, k in [(4096, 4096), (14336, 4096), (7168, 2048)]:
            yield m, n, k
    for n in [64, 128]:
        for m, k in [(256, 4096), (1024, 4096)]:
            yield m, n, k


_NUM_CORES = 32  # Cores
_L0C_SIZE = 256 * 1024  # L0C bytes (fp32 accumulator)
_L0A_SIZE = 64 * 1024  # L0A bytes
_L0B_SIZE = 64 * 1024  # L0B bytes
_L1_SIZE = 512 * 1024  # L1 bytes per core
_NUM_L0_STAGES = 2  # L0A/L0B double-buffered
_MAD_K = 64  # BF16 inner-K per MMAD
_ELEM_BYTES = 2  # BF16 element size
_BYTES_TO_MADS = 42  # BF16 wall-clock weight: per-core MAD TFLOPS/2 / MTE2_GBps


def derive_mad(block_m: int, block_n: int, mad_align_m: int = 16, mad_align_n: int = 16):
    """DeepGEMM-style MAD tile derivation from block size.

    L0C is double-buffered, so MAD must fit in L0C/2 bytes.
    When it doesn't fit, prefer halving N (keeping M whole for narrow-M throughput);
    fall back to halving M; last resort halve N anyway.
    Returns (mad_m, mad_n) or None if post-shrink dims misalign.
    """
    mad_m, mad_n = block_m, block_n
    cap = _L0C_SIZE // 2  # double-buffered L0C
    if mad_m * mad_n * 4 > cap:
        if (mad_n // 2) % mad_align_n == 0 and mad_m * (mad_n // 2) * 4 <= cap:
            mad_n //= 2  # keep M whole
        elif (mad_m // 2) % mad_align_m == 0:
            mad_m //= 2
        else:
            mad_n //= 2
    if mad_m % mad_align_m != 0 or mad_n % mad_align_n != 0:
        return None
    return mad_m, mad_n


def select_block_mad(M: int, N: int, _K: int):
    """DeepGEMM-style (BLOCK_M, BLOCK_N, MAD_M, MAD_N) selection.

    Sweeps every (bm, bn) in [16, 256] with step 16, filtering by L0
    capacity, MAD derivation, and divisibility.  Picks the candidate
    that minimizes estimated per-core wall time.
    """
    best_cost = (float("inf"), float("inf"), float("inf"))
    best_bm, best_bn = 64, 64
    best_mad_m, best_mad_n = 64, 64

    for bm in range(16, 257, 16):
        for bn in range(16, 257, 16):
            # L0C: full block tile as fp32 (single-buffer check)
            if bm * bn * 4 > _L0C_SIZE:
                continue
            # MAD derivation (L0C double-buffered constraint)
            mad = derive_mad(bm, bn)
            if mad is None:
                continue
            mad_m, mad_n = mad
            # L0A/L0B per-stage capacity (2 = double-buffered)
            if bm * _MAD_K * _ELEM_BYTES > _L0A_SIZE // _NUM_L0_STAGES:
                continue
            if bn * _MAD_K * _ELEM_BYTES > _L0B_SIZE // _NUM_L0_STAGES:
                continue
            # Divisible shapes (our kernel doesn't handle tail tiles)
            if M % bm != 0 or N % bn != 0:
                continue

            waves = math.ceil(math.ceil(M / bm) * math.ceil(N / bn) / _NUM_CORES)
            compute_load = waves * bm * bn
            mte2_load = waves * (bm + bn)
            wall = max(compute_load, mte2_load * _BYTES_TO_MADS)
            cost = (wall, mte2_load, -bm)

            if cost < best_cost:
                best_cost = cost
                best_bm, best_bn = bm, bn
                best_mad_m, best_mad_n = mad_m, mad_n
    return best_bm, best_bn, best_mad_m, best_mad_n


def select_block_k(BM: int, BN: int, K: int):
    """DeepGEMM-style BLOCK_K selection for BF16."""
    # Try larger block (256): must divide AND leave ≥3 L1 stages
    if K % 256 == 0:
        l1_per_stage = (BM + BN) * 256 * _ELEM_BYTES
        if _L1_SIZE // l1_per_stage >= 3:
            return 256
    # Fall back to default (128)
    if K % 128 == 0:
        return 128
    # Walk down powers of 2 from 128 (down to 64 for BF16 minimum)
    bk = 128
    while bk > 64 and K % bk != 0:
        bk //= 2
    return bk


def select_num_l1_stages(BM: int, BN: int, BK: int):
    """DeepGEMM-style L1 pipeline depth."""
    l1_per_stage = (BM + BN) * BK * _ELEM_BYTES
    stages = _L1_SIZE // l1_per_stage
    stages = max(stages, 1)
    stages = min(stages, 8)
    return stages


def bf16_select_config(M: int, N: int, K: int):
    """
    Returns (num_cores, BM, BN, BK, num_l1_stages, MAD_M, MAD_N).
    When MAD_N < BN the kernel uses a half-N split (two W tiles per K-step).
    """
    BM, BN, MAD_M, MAD_N = select_block_mad(M, N, K)
    BK = select_block_k(BM, BN, K)
    num_l1_stages = select_num_l1_stages(BM, BN, BK)
    num_l1_stages = max(num_l1_stages, 2)  # enforced minimum

    total_tiles = (M // BM) * (N // BN)
    num_cores = min(_NUM_CORES, total_tiles)

    return num_cores, BM, BN, BK, num_l1_stages, MAD_M, MAD_N


def gemm(M_DIM, N_DIM, K_DIM):
    NUM_BLOCKS, TILE_M, TILE_N, TILE_K, NUM_STAGES, MAD_M, MAD_N = bf16_select_config(M_DIM, N_DIM, K_DIM)
    assert TILE_M == MAD_M
    assert TILE_N == MAD_N or TILE_N == 2 * MAD_N
    HALF_N = MAD_N < TILE_N
    print(
        f"Selected config for M={M_DIM}, N={N_DIM}, K={K_DIM}: "
        f"TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}, NUM_STAGES={NUM_STAGES}, HALF_N={HALF_N}",
        file=sys.stderr,
    )
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    TILE_N_HALF = TILE_N // 2

    # DeepGEMM l2_ctrl strategy
    l2_ctrl_a = "NOTALLOC_KEEP" if N_TILES <= 2 and M_TILES > N_TILES else "NORMAL_FV"
    l2_ctrl_b = "NOTALLOC_KEEP" if M_TILES <= 2 and N_TILES > M_TILES else "NORMAL_FV"
    l2_ctrl_d = "NOTALLOC_CLEAN"

    @T.prim_func
    def main(
        A: T.Buffer((M_DIM, K_DIM), "bfloat16"),
        B: T.Buffer((N_DIM, K_DIM), "bfloat16"),
        D: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            a = T.alloc_l1((TILE_M, TILE_K), "bfloat16")
            if HALF_N:
                # Half-N kernel: split W tile into two halves to fit L0C double-buffer
                b0 = T.alloc_l1((TILE_N_HALF, TILE_K), "bfloat16")
                b1 = T.alloc_l1((TILE_N_HALF, TILE_K), "bfloat16")
                res0 = T.alloc_l0c((TILE_M, TILE_N_HALF), "float32")
                res1 = T.alloc_l0c((TILE_M, TILE_N_HALF), "float32")
                temp0 = T.alloc_shared((TILE_M // 2, TILE_N_HALF), "float32")
                temp1 = T.alloc_shared((TILE_M // 2, TILE_N_HALF), "float32")
                T.annotate_buffer_versions({a: NUM_STAGES, b0: NUM_STAGES, b1: NUM_STAGES})

                sched = T.AscendTileScheduler(block_m=TILE_M, block_n=TILE_N, num_cores=NUM_BLOCKS, shape_m=M_DIM, shape_n=N_DIM)
                sched.init(bx)
                while sched.valid():
                    m_tile, n_tile = sched.m_idx, sched.n_idx
                    for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                        T.copy(A[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], a, l2_cache_ctrl=l2_ctrl_a)
                        T.copy(
                            B[n_tile * TILE_N : n_tile * TILE_N + TILE_N_HALF, kt * TILE_K : (kt + 1) * TILE_K], b0, l2_cache_ctrl=l2_ctrl_b
                        )
                        T.copy(
                            B[n_tile * TILE_N + TILE_N_HALF : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K],
                            b1,
                            l2_cache_ctrl=l2_ctrl_b,
                        )
                        T.gemm(a, b0, res0, transpose_B=True, clear_accum=(kt == 0))
                        T.gemm(a, b1, res1, transpose_B=True, clear_accum=(kt == 0))
                    T.dual_copy(res0, temp0)
                    T.dual_copy(res1, temp1)
                    T.dual_copy(
                        temp0,
                        D[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : n_tile * TILE_N + TILE_N_HALF],
                        l2_cache_ctrl=l2_ctrl_d,
                    )
                    T.dual_copy(
                        temp1,
                        D[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N + TILE_N_HALF : (n_tile + 1) * TILE_N],
                        l2_cache_ctrl=l2_ctrl_d,
                    )
                    sched.next_block()
            else:
                # Full-N kernel: single W tile, single L0C accumulator
                b = T.alloc_l1((TILE_N, TILE_K), "bfloat16")
                res = T.alloc_l0c((TILE_M, TILE_N), "float32")
                temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")
                T.annotate_buffer_versions({a: NUM_STAGES, b: NUM_STAGES})

                sched = T.AscendTileScheduler(block_m=TILE_M, block_n=TILE_N, num_cores=NUM_BLOCKS, shape_m=M_DIM, shape_n=N_DIM)
                sched.init(bx)
                while sched.valid():
                    m_tile, n_tile = sched.m_idx, sched.n_idx
                    for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                        T.copy(A[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], a, l2_cache_ctrl=l2_ctrl_a)
                        T.copy(B[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], b, l2_cache_ctrl=l2_ctrl_b)
                        T.gemm(a, b, res, transpose_B=True, clear_accum=(kt == 0))
                    T.dual_copy(res, temp)
                    T.dual_copy(
                        temp, D[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N], l2_cache_ctrl=l2_ctrl_d
                    )
                    sched.next_block()

    return main


def ref_program(a, b):
    """Reference implementation: D = A @ B^T"""
    return a.float() @ b.float().T


def run(m: int, n: int, k: int, target: str) -> float:
    a = torch.randn(m, k, dtype=torch.bfloat16, device="npu")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="npu")

    program = gemm(m, n, k)
    compile_kwargs = {"out_idx": -1}
    if target == "pto":
        compile_kwargs["target"] = "pto"
    kernel = tilelang.compile(program, **compile_kwargs)
    d = kernel(a, b)
    torch.npu.synchronize()

    d_ref = ref_program(a, b)

    # Correctness
    max_diff = torch.max(torch.abs(d - d_ref)).item()
    assert max_diff < 1e-2, f"Max difference {max_diff:.2e} exceeds tolerance"

    # Bench
    flops = 2.0 * m * n * k
    prof = do_bench(lambda: kernel(a, b), backend="msprof_detail")

    print(f"m={m:5}, n={n:5}, k={k:5}:  PASS  max_diff={max_diff:.2e}  |  {prof.dur_us:6.1f} us  |  {prof.tflops(flops):5.1f} TFLOPS")
    return prof.dur_ns / 1e6


def run_regression_perf(m=8192, n=8192, k=8192, target: str = "ascend") -> float:
    """Benchmark one representative (largest square) shape from enumerate_shapes()."""
    return run(m, n, k, target)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["ascend", "pto"], default="ascend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for m, n, k in enumerate_shapes():
        run(m, n, k, args.target)
    print("All tests passed!")


if __name__ == "__main__":
    main()
