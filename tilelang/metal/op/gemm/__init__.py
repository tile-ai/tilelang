from __future__ import annotations

import itertools
from typing import List, Dict, Any

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.metal.target import target_is_metal

from .gemm_metal import (
    GEMM_INST_METAL,
    GEMM_INST_METAL_COOPERATIVE_TENSOR,
    GemmMetal,
    GemmMetalSimdGroup,
)


def _match_metal(target) -> bool:
    return target_is_metal(target)


register_gemm_impl(GEMM_INST_METAL, GEMM_INST_METAL, _match_metal, GemmMetalSimdGroup)
register_gemm_impl(GEMM_INST_METAL_COOPERATIVE_TENSOR, GEMM_INST_METAL_COOPERATIVE_TENSOR, _match_metal, GemmMetal)


def get_metal_gemm_configs(
    M: int,
    N: int,
    K: int,
) -> List[Dict[str, Any]]:
    """Generate Metal-optimized GEMM autotuner configs.

    Unlike CUDA GPUs, Apple Silicon prefers smaller tile sizes and fewer
    threads per block. MLX's Steel GEMM uses BM/BN ∈ {32, 64},
    BK ∈ {8, 16, 32}, and threads = WM*WN*32 ∈ {128, 256}.

    This mirrors those constraints and filters out invalid combinations
    (e.g. tile larger than problem dimension, or partition mismatch).

    Returns:
        A list of config dicts with keys: block_M, block_N, block_K,
        thread_num. Suitable for passing to AutoTuner.from_kernel().
    """
    # Apple GPU prefers smaller tiles
    block_M_candidates = [16, 32, 64]
    block_N_candidates = [16, 32, 64]
    block_K_candidates = [8, 16, 32]
    thread_num_candidates = [64, 128, 256]

    configs = []
    for bm, bn, bk, threads in itertools.product(
        block_M_candidates, block_N_candidates, block_K_candidates, thread_num_candidates
    ):
        # Skip invalid: tile larger than problem
        if bm > M or bn > N or bk > K:
            continue
        # Skip invalid: thread count must be a multiple of 32 (warp size)
        if threads % 32 != 0:
            continue
        # Skip invalid: tile must be divisible by 8 (simdgroup fragment size)
        if bm % 8 != 0 or bn % 8 != 0:
            continue
        # Skip invalid: 64-thread blocks too small for large tiles
        if threads == 64 and bm * bn > 1024:
            continue

        configs.append({
            "block_M": bm,
            "block_N": bn,
            "block_K": bk,
            "thread_num": threads,
        })

    return configs
