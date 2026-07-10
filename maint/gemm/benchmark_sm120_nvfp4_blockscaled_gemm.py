"""SM120 NVFP4 block-scaled GEMM maintenance benchmark.

This script keeps the retained high-performance persistent benchmark used to
compare TileLang's SM120 NVFP4 block-scaled GEMM path against CUTLASS.  The
public-facing example lives in ``examples/gemm_sm120`` and intentionally avoids
the persistent scheduler and CUTLASS build harness.

It can optionally compare against the official CUTLASS GeForce NVFP4 example
79a. Run from the repository root:

    python maint/gemm/benchmark_sm120_nvfp4_blockscaled_gemm.py --m 8192 --n 8192 --k 8192 --run-cutlass
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench
from tilelang.quantize import (
    swizzle_blockscaled_chunk_kmajor_scale_words,
    unswizzle_blockscaled_chunk_kmajor_scale_words,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tilelang_nvfp4_quantizer import tilelang_quantize_bf16_to_nvfp4_blockscaled  # noqa: E402
from tilelang.quantize.nvfp4 import (
    blockscaled_chunk_kmajor_word_offset,
    decode_ue4m3_scale_bytes,
)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return 2.0 * m * n * k / (latency_ms * 1.0e-3) / 1.0e12


def _early_bench_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--maxrregcount", type=int)
    parser.add_argument("--ptxas-verbose", action="store_true")
    args, _ = parser.parse_known_args()
    return args


_EARLY_BENCH_OPTIONS = _early_bench_options()


_SM120_SCHEDULE = "pp_stream_tma_copy_store_pipe2"
_SM120_SCALE_LAYOUT = "blockscaled_chunk_kmajor"
_SM120_SCALE_LOAD = "tma"
_SM120_AB_SHARED_STORAGE = "packed"
_SM120_THREADS = 384


# Scale source contract for the SM120 optimized path:
# - semantic SFA/SFB shape is [M or N, K / 64] uint32; each word packs four
#   scale bytes for four consecutive K/16 groups.
# - host storage uses CUTLASS BlockScaledBasicChunk K-major order:
#     atom_shape=((32, 4), (16, 4)), atom_stride=((16, 4), (0, 1)).
#   For one K=64 atom, the scale-byte offset is
#     (row % 32) * 16 + (row // 32) * 4 + ((k // 16) % 4).
#   The uint32 source layout below is the compressed form of that byte layout.
# - the benchmark always passes this swizzled storage to the kernel; reference
#   checking keeps the semantic row-major copy separate.
def _device_compile_flags() -> list[str]:
    flags = []
    if _EARLY_BENCH_OPTIONS.maxrregcount is not None:
        if _EARLY_BENCH_OPTIONS.maxrregcount <= 0:
            raise ValueError("--maxrregcount must be positive")
        flags.append(f"--maxrregcount={_EARLY_BENCH_OPTIONS.maxrregcount}")
    if _EARLY_BENCH_OPTIONS.ptxas_verbose:
        flags.append("--ptxas-options=--verbose")
    return flags


def _sm120_nvfp4_jit_pass_configs() -> dict:
    pass_configs = {}
    pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = _device_compile_flags()
    return pass_configs


@tilelang.jit(
    out_idx=None,
    pass_configs=_sm120_nvfp4_jit_pass_configs(),
)
def tilelang_nvfp4_blockscaled_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    warp_policy,
    out_dtype,
    producer_regs: int,
    consumer_regs: int,
):
    """SM120 NVFP4 block-scaled GEMM implementation.

    This function has a compact external contract. The selected SM120
    scheduling, scale-source layout, and MMA helper are fixed internally by
    the SM120 implementation contract.
    """

    assert M % block_M == 0
    assert N % block_N == 0
    assert K % block_K == 0
    assert block_M % 2 == 0
    assert block_K % 64 == 0
    assert num_stages >= 2
    in_dtype = T.float4_e2m1fn
    ab_shared_dtype = in_dtype
    sf_layout = _SM120_SCALE_LAYOUT
    accum_dtype = T.float32
    half_block_N = block_N // 2
    sf_granularity_k = 16
    sf_words_per_block_k = block_K // 64

    if _SM120_SCHEDULE == "pp_stream_tma_copy_store_pipe2":
        if sf_words_per_block_k < 4:
            raise ValueError("The SM120 NVFP4 optimized path requires block_K >= 256 for scale TMA")
        panel32_tma_store = False
        panel64_tma_store = False
        store_block_N = 32
        epilogue_store_block_M = block_M
        epilogue_store_block_N = store_block_N
        epilogue_store_slots = 2
        if block_N != 128:
            raise ValueError("The SM120 NVFP4 optimized path requires block_N=128")

        sm_num = driver.get_num_sms()
        n_blocks = T.ceildiv(N, block_N)
        m_blocks = T.ceildiv(M, block_M)
        total_tiles = n_blocks * m_blocks
        k_tiles = K // block_K
        stream_iters = T.ceildiv(total_tiles, sm_num)
        owner_iters = T.ceildiv(stream_iters, 2)

        @T.macro
        def copy_blockscaled_chunk_kmajor_scale_tile(SF, SF_shared, tile_row, block_rows, ko, stage, tx):
            # Producer-warp-group staging: 128 producer lanes stride the tile,
            # with the layout addressing shared via ue4m3_scale_tile_source_coords.
            scale_tile_words = block_rows * sf_words_per_block_k
            scale_lane = tx - 256
            for scale_iter in T.serial((scale_tile_words + 127) // 128):
                scale_flat = scale_iter * 128 + scale_lane
                if scale_flat < scale_tile_words:
                    row, col = T.ue4m3_scale_tile_source_coords(SF, block_rows, sf_words_per_block_k, tile_row, ko, scale_flat)
                    SF_shared[
                        stage,
                        scale_flat // sf_words_per_block_k,
                        scale_flat % sf_words_per_block_k,
                    ] = SF[row, col]

        @T.prim_func
        def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((N, K), in_dtype),
            SFA: T.Tensor((M, K // 64), T.uint32),
            SFB: T.Tensor((N, K // 64), T.uint32),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(sm_num, threads=384) as block_id:
                A_shared = T.alloc_shared((num_stages, block_M, block_K), ab_shared_dtype)
                B_shared = T.alloc_shared((num_stages, block_N, block_K), ab_shared_dtype)
                SFA_shared = T.alloc_shared((num_stages, block_M, sf_words_per_block_k), T.uint32)
                SFB_shared = T.alloc_shared((num_stages, block_N, sf_words_per_block_k), T.uint32)
                if panel32_tma_store or panel64_tma_store:
                    C_shared = T.alloc_shared((epilogue_store_slots, epilogue_store_block_M, epilogue_store_block_N), out_dtype)

                loaded = T.alloc_barrier([128] * (2 * num_stages))
                consumed = T.alloc_barrier([128] * (2 * num_stages))
                wg_order = T.alloc_barrier([128] * 2)
                store_order = T.alloc_barrier([128] * 2)

                tx = T.get_thread_binding()

                if tx >= 256:
                    if producer_regs > 0:
                        T.set_max_nreg(producer_regs, 0)
                    for stream in T.serial(stream_iters):
                        tile_id = block_id + stream * sm_num
                        if tile_id < total_tiles:
                            tile_n = tile_id % n_blocks
                            tile_m = tile_id // n_blocks
                            owner = stream & 1
                            owner_iter = stream // 2
                            for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                                stage = ko % num_stages
                                phase = ((owner_iter * k_tiles + ko) // num_stages) & 1
                                barrier_slot = owner * num_stages + stage
                                if stream > 0 and ko < num_stages:
                                    prev_owner = (stream - 1) & 1
                                    prev_owner_iter = (stream - 1) // 2
                                    prev_barrier_slot = prev_owner * num_stages + stage
                                    prev_last_ko = ((k_tiles - 1 - stage) // num_stages) * num_stages + stage
                                    prev_phase = ((prev_owner_iter * k_tiles + prev_last_ko) // num_stages) & 1
                                    T.barrier_wait(consumed[prev_barrier_slot], prev_phase)
                                else:
                                    T.barrier_wait(consumed[barrier_slot], phase ^ 1)
                                T.tma_copy(
                                    A[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        ko * block_K : (ko + 1) * block_K,
                                    ],
                                    A_shared[stage, :, :],
                                    barrier=loaded[barrier_slot],
                                    leader_scope_threads=128,
                                )
                                T.tma_copy(
                                    B[
                                        tile_n * block_N : (tile_n + 1) * block_N,
                                        ko * block_K : (ko + 1) * block_K,
                                    ],
                                    B_shared[stage, :, :],
                                    barrier=loaded[barrier_slot],
                                    leader_scope_threads=128,
                                )
                                copy_blockscaled_chunk_kmajor_scale_tile(SFA, SFA_shared, tile_m, block_M, ko, stage, tx)
                                copy_blockscaled_chunk_kmajor_scale_tile(SFB, SFB_shared, tile_n, block_N, ko, stage, tx)
                                T.barrier_arrive(loaded[barrier_slot])

                elif tx < 128:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    if panel32_tma_store:
                        C0_local_0 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C0_local_1 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C0_local_2 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C0_local_3 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                    elif panel64_tma_store:
                        C0_local_0 = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                        C0_local_1 = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                    else:
                        C0_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for owner_iter in T.serial(owner_iters):
                        stream = owner_iter * 2
                        tile_id = block_id + stream * sm_num
                        if tile_id < total_tiles:
                            tile_n = tile_id % n_blocks
                            tile_m = tile_id // n_blocks

                            if owner_iter > 0:
                                T.barrier_wait(wg_order[0], (owner_iter - 1) & 1)

                            if panel32_tma_store:
                                T.clear(C0_local_0)
                                T.clear(C0_local_1)
                                T.clear(C0_local_2)
                                T.clear(C0_local_3)
                            elif panel64_tma_store:
                                T.clear(C0_local_0)
                                T.clear(C0_local_1)
                            else:
                                T.clear(C0_local)
                            for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                                stage = ko % num_stages
                                phase = ((owner_iter * k_tiles + ko) // num_stages) & 1
                                barrier_slot = stage
                                T.barrier_wait(loaded[barrier_slot], phase)
                                if panel32_tma_store:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 0:store_block_N, :],
                                        C0_local_0,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 0:store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, store_block_N : 2 * store_block_N, :],
                                        C0_local_1,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, store_block_N : 2 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 2 * store_block_N : 3 * store_block_N, :],
                                        C0_local_2,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 2 * store_block_N : 3 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 3 * store_block_N : 4 * store_block_N, :],
                                        C0_local_3,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 3 * store_block_N : 4 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                elif panel64_tma_store:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 0:half_block_N, :],
                                        C0_local_0,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 0:half_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, half_block_N:block_N, :],
                                        C0_local_1,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, half_block_N:block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                else:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, :, :],
                                        C0_local,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, :, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                T.barrier_arrive(consumed[barrier_slot])

                            T.barrier_arrive(wg_order[1])

                            if panel32_tma_store:
                                base_n0 = tile_n * block_N
                                T.copy(C0_local_0, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n0 : base_n0 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n1 = tile_n * block_N + store_block_N
                                T.copy(C0_local_1, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n1 : base_n1 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n2 = tile_n * block_N + 2 * store_block_N
                                T.copy(C0_local_2, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n2 : base_n2 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n3 = tile_n * block_N + 3 * store_block_N
                                T.copy(C0_local_3, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n3 : base_n3 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)
                            elif panel64_tma_store:
                                if owner_iter > 0:
                                    T.barrier_wait(store_order[0], (owner_iter - 1) & 1)
                                base_n0 = tile_n * block_N
                                T.copy(C0_local_0, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n0 : base_n0 + epilogue_store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n2 = tile_n * block_N + half_block_N
                                T.copy(C0_local_1, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(7, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n2 : base_n2 + epilogue_store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)
                                T.barrier_arrive(store_order[1])
                            else:
                                T.copy(C0_local, C[tile_m * block_M, tile_n * block_N])

                else:
                    if consumer_regs > 0:
                        T.set_max_nreg(consumer_regs, 1)
                    if panel32_tma_store:
                        C1_local_0 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C1_local_1 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C1_local_2 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                        C1_local_3 = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                    elif panel64_tma_store:
                        C1_local_0 = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                        C1_local_1 = T.alloc_fragment((block_M, half_block_N), accum_dtype)
                    else:
                        C1_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for owner_iter in T.serial(owner_iters):
                        stream = owner_iter * 2 + 1
                        tile_id = block_id + stream * sm_num
                        if tile_id < total_tiles:
                            tile_n = tile_id % n_blocks
                            tile_m = tile_id // n_blocks

                            T.barrier_wait(wg_order[1], owner_iter & 1)

                            if panel32_tma_store:
                                T.clear(C1_local_0)
                                T.clear(C1_local_1)
                                T.clear(C1_local_2)
                                T.clear(C1_local_3)
                            elif panel64_tma_store:
                                T.clear(C1_local_0)
                                T.clear(C1_local_1)
                            else:
                                T.clear(C1_local)
                            for ko in T.unroll(k_tiles, explicit=False, unroll_factor=1):
                                stage = ko % num_stages
                                phase = ((owner_iter * k_tiles + ko) // num_stages) & 1
                                barrier_slot = num_stages + stage
                                T.barrier_wait(loaded[barrier_slot], phase)
                                if panel32_tma_store:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 0:store_block_N, :],
                                        C1_local_0,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 0:store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, store_block_N : 2 * store_block_N, :],
                                        C1_local_1,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, store_block_N : 2 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 2 * store_block_N : 3 * store_block_N, :],
                                        C1_local_2,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 2 * store_block_N : 3 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 3 * store_block_N : 4 * store_block_N, :],
                                        C1_local_3,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 3 * store_block_N : 4 * store_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                elif panel64_tma_store:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, 0:half_block_N, :],
                                        C1_local_0,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, 0:half_block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, half_block_N:block_N, :],
                                        C1_local_1,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, half_block_N:block_N, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                else:
                                    T.mma_gemm_blockscaled(
                                        A_shared[stage, :, :],
                                        B_shared[stage, :, :],
                                        C1_local,
                                        SFA_shared[stage, :, :],
                                        SFB_shared[stage, :, :],
                                        transpose_B=True,
                                        policy=warp_policy,
                                        clear_accum=False,
                                        k_start=0,
                                        sf_a_granularity_k=sf_granularity_k,
                                        sf_b_granularity_k=sf_granularity_k,
                                        sf_layout=sf_layout,
                                    )
                                T.barrier_arrive(consumed[barrier_slot])

                            T.barrier_arrive(wg_order[0])

                            if panel32_tma_store:
                                base_n0 = tile_n * block_N
                                T.copy(C1_local_0, C_shared[1, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[1, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n0 : base_n0 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n1 = tile_n * block_N + store_block_N
                                T.copy(C1_local_1, C_shared[1, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[1, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n1 : base_n1 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n2 = tile_n * block_N + 2 * store_block_N
                                T.copy(C1_local_2, C_shared[1, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[1, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n2 : base_n2 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n3 = tile_n * block_N + 3 * store_block_N
                                T.copy(C1_local_3, C_shared[1, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[1, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n3 : base_n3 + store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)
                            elif panel64_tma_store:
                                T.barrier_wait(store_order[1], owner_iter & 1)
                                base_n0 = tile_n * block_N
                                T.copy(C1_local_0, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n0 : base_n0 + epilogue_store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)

                                base_n2 = tile_n * block_N + half_block_N
                                T.copy(C1_local_1, C_shared[0, :, :])
                                T.fence_proxy_async()
                                T.sync_threads(8, 128)
                                T.tma_copy(
                                    C_shared[0, :, :],
                                    C[
                                        tile_m * block_M : (tile_m + 1) * block_M,
                                        base_n2 : base_n2 + epilogue_store_block_N,
                                    ],
                                    leader_scope_threads=128,
                                )
                                T.tma_store_wait(0, True)
                                T.barrier_arrive(store_order[0])
                            else:
                                T.copy(C1_local, C[tile_m * block_M, tile_n * block_N])

        return main


_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _make_packed_fp4(rows: int, cols: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randint(-128, 128, (rows, cols // 2), device="cuda", dtype=torch.int8, generator=generator)


def _make_bf16_activation(rows: int, cols: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return (torch.randn((rows, cols), device="cuda", dtype=torch.float32, generator=generator) * 2.0).to(torch.bfloat16)


def _make_ones_packed_fp4(rows: int, cols: int) -> torch.Tensor:
    # Two FP4 e2m1 values with raw code 0x2, packed into one byte.
    return torch.full((rows, cols // 2), 0x22, device="cuda", dtype=torch.int8)


def _make_constant_scale_words(rows: int, k: int, byte: int = 0x38) -> torch.Tensor:
    word = byte | (byte << 8) | (byte << 16) | (byte << 24)
    return torch.full((rows, k // 64), word, device="cuda", dtype=torch.uint32)


def _pack_scale_words(scale_bytes: torch.Tensor) -> torch.Tensor:
    scale_i64 = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], -1, 4)
    words = scale_i64[:, :, 0]
    words = words | (scale_i64[:, :, 1] << 8)
    words = words | (scale_i64[:, :, 2] << 16)
    words = words | (scale_i64[:, :, 3] << 24)
    return words.to(torch.uint32).contiguous()


def _make_binary_scale_words(rows: int, k: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    scale_bytes = (
        torch.randint(
            0,
            2,
            (rows, k // 16),
            device="cuda",
            dtype=torch.int64,
            generator=generator,
        )
        * 0x38
    )
    return _pack_scale_words(scale_bytes)


def _blockscaled_chunk_kmajor_word_offset(row: int, k64_word: int, block_rows: int = 128) -> tuple[int, int]:
    """Return compressed word coordinates for CUTLASS BlockScaledBasicChunk.

    CUTLASS' element-level byte offset for one K=64 atom is:
        (row % 32) * 16 + (row // 32) * 4 + ((k // 16) % 4)

    This benchmark stores the four adjacent ((k // 16) % 4) bytes in one
    uint32, so the physical word coordinate is the byte offset divided by four.
    """
    return blockscaled_chunk_kmajor_word_offset(row, k64_word, block_rows=block_rows)


def _swizzle_scale_words_blockscaled_chunk_kmajor(words: torch.Tensor, block_rows: int = 128, block_words: int = 4) -> torch.Tensor:
    """Convert semantic row-major scale words to BlockScaledBasicChunk K-major.

    Input words[row, k_word] is the natural reference matrix: one uint32 packs
    scale bytes for K groups [4*k_word + 0, ..., +3].  Output storage is the
    uint32-compressed form of CUTLASS BlockScaledBasicChunk K-major order.  TMA
    copies this physical tile directly into shared memory.
    """
    return swizzle_blockscaled_chunk_kmajor_scale_words(words, block_rows=block_rows, block_words=block_words)


def _check_blockscaled_chunk_kmajor_offsets() -> None:
    expected = {
        (0, 0): (0, 0),
        (0, 1): (32, 0),
        (0, 2): (64, 0),
        (0, 3): (96, 0),
        (1, 0): (1, 0),
        (15, 0): (15, 0),
        (16, 0): (16, 0),
        (31, 0): (31, 0),
        (32, 0): (0, 1),
        (33, 0): (1, 1),
        (64, 0): (0, 2),
        (96, 0): (0, 3),
        (127, 0): (31, 3),
    }
    for (row, k64_word), physical in expected.items():
        actual = _blockscaled_chunk_kmajor_word_offset(row, k64_word)
        if actual != physical:
            raise AssertionError(
                f"BlockScaledBasicChunk K-major offset mismatch: row={row} k64_word={k64_word} expected={physical} actual={actual}"
            )


def _decode_binary_scale_words(words: torch.Tensor, k: int) -> torch.Tensor:
    w = words.to(torch.int64)
    scale_bytes = torch.empty((words.shape[0], k // 16), device=words.device, dtype=torch.int64)
    scale_bytes[:, 0::4] = w & 0xFF
    scale_bytes[:, 1::4] = (w >> 8) & 0xFF
    scale_bytes[:, 2::4] = (w >> 16) & 0xFF
    scale_bytes[:, 3::4] = (w >> 24) & 0xFF
    return (scale_bytes != 0).to(torch.float32)


def _decode_ue4m3_scale_words(words: torch.Tensor, k: int) -> torch.Tensor:
    w = words.to(torch.int64)
    scale_bytes = torch.empty((words.shape[0], k // 16), device=words.device, dtype=torch.uint8)
    scale_bytes[:, 0::4] = (w & 0xFF).to(torch.uint8)
    scale_bytes[:, 1::4] = ((w >> 8) & 0xFF).to(torch.uint8)
    scale_bytes[:, 2::4] = ((w >> 16) & 0xFF).to(torch.uint8)
    scale_bytes[:, 3::4] = ((w >> 24) & 0xFF).to(torch.uint8)
    return decode_ue4m3_scale_bytes(scale_bytes)


def _decode_rowmajor_fp4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(_FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def _verify_tilelang_output(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    C: torch.Tensor,
    out_dtype: torch.dtype,
    scale_mode: str,
    block_m: int = 128,
    block_n: int = 128,
    sm_num: int | None = None,
) -> None:
    A_full = _decode_rowmajor_fp4(A, A.shape[0], A.shape[1] * 2)
    B_full = _decode_rowmajor_fp4(B, B.shape[0], B.shape[1] * 2)
    if scale_mode == "constant":
        ref = (A_full @ B_full.T).to(out_dtype)
    elif scale_mode in ("random_binary", "random_sfa", "random_sfb", "ue4m3"):
        if scale_mode == "ue4m3":
            sfa = _decode_ue4m3_scale_words(SFA, A_full.shape[1])
            sfb = _decode_ue4m3_scale_words(SFB, B_full.shape[1])
        else:
            sfa = _decode_binary_scale_words(SFA, A_full.shape[1])
            sfb = _decode_binary_scale_words(SFB, B_full.shape[1])
        ref_f32 = torch.zeros((A_full.shape[0], B_full.shape[0]), device=C.device, dtype=torch.float32)
        for k_sf in range(A_full.shape[1] // 16):
            k0 = k_sf * 16
            k1 = k0 + 16
            a_chunk = A_full[:, k0:k1] * sfa[:, k_sf].unsqueeze(1)
            b_chunk = B_full[:, k0:k1] * sfb[:, k_sf].unsqueeze(1)
            ref_f32 += a_chunk @ b_chunk.T
        ref = ref_f32.to(out_dtype)
    else:
        raise ValueError(f"Unsupported scale_mode={scale_mode!r}")
    try:
        torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)
    except AssertionError:
        diff = ref != C
        mismatch = int(diff.sum().item())
        total = diff.numel()
        print(f"TileLang mismatch summary: {mismatch}/{total} ({mismatch / total:.2%})")
        if C.shape[0] % block_m == 0 and C.shape[1] % block_n == 0:
            m_blocks = C.shape[0] // block_m
            n_blocks = C.shape[1] // block_n
            tile_counts = diff.reshape(m_blocks, block_m, n_blocks, block_n).sum(dim=(1, 3))
            flat = tile_counts.flatten()
            top = torch.topk(flat, k=min(12, flat.numel()))
            for value, index in zip(top.values.cpu().tolist(), top.indices.cpu().tolist()):
                if int(value) == 0:
                    continue
                tile_m = index // n_blocks
                tile_n = index % n_blocks
                tile_id = tile_m * n_blocks + tile_n
                owner = "unknown"
                if sm_num is not None and sm_num > 0:
                    owner = f"WG{(tile_id // sm_num) & 1}"
                print(f"  tile m={tile_m} n={tile_n} tile_id={tile_id} owner={owner} mismatch={int(value)}/{block_m * block_n}")
            if block_n % 64 == 0:
                panel_counts = diff.reshape(m_blocks, block_m, n_blocks, block_n // 64, 64).sum(dim=(1, 4))
                panel_flat = panel_counts.flatten()
                panel_top = torch.topk(panel_flat, k=min(12, panel_flat.numel()))
                for value, index in zip(panel_top.values.cpu().tolist(), panel_top.indices.cpu().tolist()):
                    if int(value) == 0:
                        continue
                    panel = index % (block_n // 64)
                    tile_index = index // (block_n // 64)
                    tile_m = tile_index // n_blocks
                    tile_n = tile_index % n_blocks
                    tile_id = tile_m * n_blocks + tile_n
                    owner = "unknown"
                    if sm_num is not None and sm_num > 0:
                        owner = f"WG{(tile_id // sm_num) & 1}"
                    print(
                        "  panel64 "
                        f"m={tile_m} n={tile_n} panel={panel} tile_id={tile_id} owner={owner} "
                        f"mismatch={int(value)}/{block_m * 64}"
                    )
            if block_n % 32 == 0:
                panel32_counts = diff.reshape(m_blocks, block_m, n_blocks, block_n // 32, 32).sum(dim=(1, 4))
                panel32_flat = panel32_counts.flatten()
                panel32_top = torch.topk(panel32_flat, k=min(16, panel32_flat.numel()))
                for value, index in zip(panel32_top.values.cpu().tolist(), panel32_top.indices.cpu().tolist()):
                    if int(value) == 0:
                        continue
                    panel = index % (block_n // 32)
                    tile_index = index // (block_n // 32)
                    tile_m = tile_index // n_blocks
                    tile_n = tile_index % n_blocks
                    tile_id = tile_m * n_blocks + tile_n
                    owner = "unknown"
                    if sm_num is not None and sm_num > 0:
                        owner = f"WG{(tile_id // sm_num) & 1}"
                    print(
                        "  panel32 "
                        f"m={tile_m} n={tile_n} panel={panel} tile_id={tile_id} owner={owner} "
                        f"mismatch={int(value)}/{block_m * 32}"
                    )
                tile0 = diff[:block_m, :block_n]
                for panel in range(block_n // 32):
                    panel_diff = tile0[:, panel * 32 : (panel + 1) * 32]
                    row32 = panel_diff.reshape(4, 32, 32).sum(dim=(1, 2)).cpu().tolist()
                    col8 = panel_diff.reshape(block_m, 4, 8).sum(dim=(0, 2)).cpu().tolist()
                    print(f"  tile0-panel32-buckets panel={panel} row32={[int(x) for x in row32]} col8={[int(x) for x in col8]}")
        raise


def run_tilelang(args: argparse.Namespace) -> tuple[float, float]:
    _check_blockscaled_chunk_kmajor_offsets()
    out_dtype = T.bfloat16 if args.out_dtype == "bfloat16" else T.float32
    out_torch_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float32
    warp_policy = getattr(T.GemmWarpPolicy, args.warp_policy)

    kernel = tilelang_nvfp4_blockscaled_gemm(
        args.m,
        args.n,
        args.k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_stages,
        warp_policy,
        out_dtype,
        args.producer_regs,
        args.consumer_regs,
    )

    source = kernel.get_kernel_source()
    if args.dump_source:
        dump_path = Path(args.dump_source)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(source)
        print(f"TileLang CUDA source: {dump_path}")
    if args.dump_sass:
        sass_path = Path(args.dump_sass)
        sass_path.parent.mkdir(parents=True, exist_ok=True)
        saved_compile_flags = kernel.compile_flags
        device_compile_flags = _device_compile_flags()
        if device_compile_flags:
            merged_compile_flags = list(saved_compile_flags or [])
            for flag in device_compile_flags:
                if flag not in merged_compile_flags:
                    merged_compile_flags.append(flag)
            kernel.compile_flags = merged_compile_flags
        kernel.export_sass(str(sass_path))
        kernel.compile_flags = saved_compile_flags
        print(f"TileLang SASS: {sass_path}")
    if "sm120_mma_blockscaled_kblock_fulltile_package_pingpong" not in source:
        raise RuntimeError("TileLang source did not lower to the SM120 full-tile MMA helper")

    if args.input_mode == "ones":
        A = _make_ones_packed_fp4(args.m, args.k)
        B = _make_ones_packed_fp4(args.n, args.k)
    elif args.input_mode == "bf16_quantized":
        A_bf16 = _make_bf16_activation(args.m, args.k, seed=args.seed)
        B_bf16 = _make_bf16_activation(args.n, args.k, seed=args.seed + 1)
        A, SFA = tilelang_quantize_bf16_to_nvfp4_blockscaled(A_bf16)
        B, SFB = tilelang_quantize_bf16_to_nvfp4_blockscaled(B_bf16)
        SFA_semantic = unswizzle_blockscaled_chunk_kmajor_scale_words(SFA)
        SFB_semantic = unswizzle_blockscaled_chunk_kmajor_scale_words(SFB)
    else:
        A = _make_packed_fp4(args.m, args.k, seed=args.seed)
        B = _make_packed_fp4(args.n, args.k, seed=args.seed + 1)
    if args.input_mode != "bf16_quantized":
        if args.scale_mode == "constant":
            SFA_semantic = _make_constant_scale_words(args.m, args.k)
            SFB_semantic = _make_constant_scale_words(args.n, args.k)
        elif args.scale_mode == "random_binary":
            SFA_semantic = _make_binary_scale_words(args.m, args.k, seed=args.seed + 100)
            SFB_semantic = _make_binary_scale_words(args.n, args.k, seed=args.seed + 200)
        elif args.scale_mode == "random_sfa":
            SFA_semantic = _make_binary_scale_words(args.m, args.k, seed=args.seed + 100)
            SFB_semantic = _make_constant_scale_words(args.n, args.k)
        elif args.scale_mode == "random_sfb":
            SFA_semantic = _make_constant_scale_words(args.m, args.k)
            SFB_semantic = _make_binary_scale_words(args.n, args.k, seed=args.seed + 200)
        else:
            raise ValueError(f"Unsupported scale_mode={args.scale_mode!r}")
        SFA = _swizzle_scale_words_blockscaled_chunk_kmajor(SFA_semantic)
        SFB = _swizzle_scale_words_blockscaled_chunk_kmajor(SFB_semantic)
    C = torch.empty((args.m, args.n), device="cuda", dtype=out_torch_dtype)

    kernel(A, B, SFA, SFB, C)
    torch.cuda.synchronize()

    if args.profile_only and args.profile_warmup_launches < 0:
        raise ValueError("--profile-warmup-launches must be non-negative")
    if args.profile_only:
        for _ in range(args.profile_warmup_launches):
            kernel(A, B, SFA, SFB, C)
        kernel(A, B, SFA, SFB, C)
        torch.cuda.synchronize()

    if args.verify:
        _verify_tilelang_output(
            A,
            B,
            SFA_semantic,
            SFB_semantic,
            C,
            out_torch_dtype,
            "ue4m3" if args.input_mode == "bf16_quantized" else args.scale_mode,
            args.block_m,
            args.block_n,
            driver.get_num_sms(),
        )
        print("TileLang correctness: passed")

    if args.profile_only:
        launches = 2 + args.profile_warmup_launches
        print(f"TileLang profile-only: ran {launches} kernel launch(es)")
        return float("nan"), float("nan")

    latency_ms = do_bench(
        lambda: kernel(A, B, SFA, SFB, C),
        warmup=args.warmup_ms,
        rep=args.rep_ms,
        _n_warmup=args.n_warmup,
        _n_repeat=args.n_repeat,
        backend=args.backend,
        return_mode=args.return_mode,
    )
    return latency_ms, _tflops(args.m, args.n, args.k, latency_ms)


def _find_cmake(args: argparse.Namespace) -> str:
    if args.cmake:
        return args.cmake
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    env_cmake = Path(sys.executable).resolve().parent / "cmake"
    if env_cmake.exists():
        return str(env_cmake)
    raise FileNotFoundError("cmake was not found; pass --cmake")


def _find_nvcc(args: argparse.Namespace) -> str:
    if args.nvcc:
        return args.nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    default_nvcc = Path("/usr/local/cuda-12.8/bin/nvcc")
    if default_nvcc.exists():
        return str(default_nvcc)
    raise FileNotFoundError("nvcc was not found; pass --nvcc")


def _run_command(cmd: list[str], *, cwd: Path) -> str:
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.stdout:
        print(completed.stdout, end="")
    completed.check_returncode()
    return completed.stdout


def _cutlass_binary_path(args: argparse.Namespace) -> Path:
    if args.cutlass_binary:
        return Path(args.cutlass_binary)
    return Path(args.cutlass_build_dir) / "examples" / "79_blackwell_geforce_gemm" / "79a_blackwell_geforce_nvfp4_bf16_gemm"


def build_cutlass_79a(args: argparse.Namespace) -> Path:
    binary = _cutlass_binary_path(args)
    if binary.exists() and not args.rebuild_cutlass:
        return binary

    cmake = _find_cmake(args)
    nvcc = _find_nvcc(args)
    build_dir = Path(args.cutlass_build_dir)
    util_include = REPO_ROOT / "3rdparty" / "cutlass" / "tools" / "util" / "include"

    configure_cmd = [
        cmake,
        "-S",
        "3rdparty/cutlass",
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCUTLASS_NVCC_ARCHS=120a",
        "-DCUTLASS_ENABLE_EXAMPLES=ON",
        "-DCUTLASS_ENABLE_TOOLS=ON",
        "-DCUTLASS_ENABLE_LIBRARY=OFF",
        "-DCUTLASS_ENABLE_TESTS=OFF",
        "-DCUTLASS_ENABLE_PROFILER=OFF",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCMAKE_CUDA_FLAGS=-I{util_include}",
        f"-DCMAKE_CXX_FLAGS=-I{util_include}",
    ]
    _run_command(configure_cmd, cwd=REPO_ROOT)

    build_cmd = [
        cmake,
        "--build",
        str(build_dir),
        "--target",
        "79a_blackwell_geforce_nvfp4_bf16_gemm",
        "-j",
        str(args.cutlass_build_jobs),
    ]
    _run_command(build_cmd, cwd=REPO_ROOT)

    if not binary.exists():
        raise FileNotFoundError(f"CUTLASS 79a binary was not produced at {binary}")
    return binary


def run_cutlass(args: argparse.Namespace) -> tuple[float, float]:
    binary = build_cutlass_79a(args)
    output = _run_command(
        [
            str(binary),
            f"--m={args.m}",
            f"--n={args.n}",
            f"--k={args.k}",
            f"--iterations={args.cutlass_iterations}",
            "--skip-verification",
        ],
        cwd=REPO_ROOT,
    )

    latency_match = re.search(r"Avg runtime:\s*([0-9.eE+-]+)\s*ms", output)
    gflops_match = re.search(r"GFLOPS:\s*([0-9.eE+-]+)", output)
    if latency_match is None or gflops_match is None:
        raise RuntimeError("Could not parse CUTLASS latency/GFLOPS output")
    latency_ms = float(latency_match.group(1))
    return latency_ms, float(gflops_match.group(1)) / 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--warp-policy", choices=["Square"], default="Square")
    parser.add_argument("--out-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--input-mode", choices=["random", "ones", "bf16_quantized"], default="random")
    parser.add_argument(
        "--scale-mode",
        choices=["constant", "random_binary", "random_sfa", "random_sfb"],
        default="constant",
    )
    parser.add_argument("--producer-regs", type=int, default=40)
    parser.add_argument("--consumer-regs", type=int, default=224)
    parser.add_argument("--maxrregcount", type=int)
    parser.add_argument("--ptxas-verbose", action="store_true")
    parser.add_argument("--dump-source")
    parser.add_argument("--dump-sass")
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event")
    parser.add_argument("--return-mode", choices=["min", "max", "mean", "median"], default="mean")
    parser.add_argument("--warmup-ms", type=float, default=25)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--n-warmup", type=int, default=0)
    parser.add_argument("--n-repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-warmup-launches", type=int, default=0)
    parser.add_argument("--run-cutlass", action="store_true")
    parser.add_argument("--cutlass-iterations", type=int, default=20)
    parser.add_argument("--cutlass-build-dir", default="build-cutlass-sm120")
    parser.add_argument("--cutlass-build-jobs", type=int, default=8)
    parser.add_argument("--cutlass-binary")
    parser.add_argument("--rebuild-cutlass", action="store_true")
    parser.add_argument("--cmake")
    parser.add_argument("--nvcc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability < (12, 0):
        raise RuntimeError(f"SM120 or newer is required, got compute capability {capability}")

    print(f"Shape: M={args.m}, N={args.n}, K={args.k}")
    effective_scale_mode = "ue4m3_from_tilelang_bf16_quantizer" if args.input_mode == "bf16_quantized" else args.scale_mode
    print(
        f"TileLang tile: {args.block_m}x{args.block_n}x{args.block_k}, "
        f"threads={_SM120_THREADS}, policy={args.warp_policy}, output={args.out_dtype}, "
        f"input_mode={args.input_mode}, scale_mode={effective_scale_mode}"
    )
    print(
        "TileLang SM120 block-scaled path: "
        f"scale_layout={_SM120_SCALE_LAYOUT}, producer_regs={args.producer_regs}, "
        f"consumer_regs={args.consumer_regs}, maxrregcount={args.maxrregcount}"
    )

    tilelang_latency_ms, tilelang_tflops = run_tilelang(args)
    print(f"TileLang latency: {tilelang_latency_ms:.4f} ms")
    print(f"TileLang FLOPS: {tilelang_tflops:.2f} TFLOPS")

    if args.run_cutlass:
        cutlass_latency_ms, cutlass_tflops = run_cutlass(args)
        print(f"CUTLASS 79a latency: {cutlass_latency_ms:.4f} ms")
        print(f"CUTLASS 79a FLOPS: {cutlass_tflops:.2f} TFLOPS")
        print(f"TileLang / CUTLASS: {tilelang_tflops / cutlass_tflops:.3f}x")


if __name__ == "__main__":
    main()
