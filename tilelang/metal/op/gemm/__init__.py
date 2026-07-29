from __future__ import annotations

import itertools
from typing import Any

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
) -> list[dict[str, Any]]:
    """Generate Metal-optimized GEMM autotuner configs.

    Apple GPU prefers smaller tiles.  MLX's Steel GEMM uses
    BM/BN ∈ {16, 32, 64}, BK ∈ {8, 16, 32}, threads ∈ {64, 128, 256}.
    These match the six shapes instantiated in steel_gemm_fused.metal.

    Returns:
        A list of config dicts with keys: block_M, block_N, block_K,
        thread_num. Suitable for passing to AutoTuner.from_kernel().
    """
    block_M_candidates = [16, 32, 64]
    block_N_candidates = [16, 32, 64]
    block_K_candidates = [8, 16, 32]
    thread_num_candidates = [64, 128, 256]

    configs = []
    for bm, bn, bk, threads in itertools.product(block_M_candidates, block_N_candidates, block_K_candidates, thread_num_candidates):
        if bm > M or bn > N or bk > K:
            continue
        if threads == 64 and bm * bn > 1024:
            continue

        configs.append(
            {
                "block_M": bm,
                "block_N": bn,
                "block_K": bk,
                "thread_num": threads,
            }
        )

    return configs
