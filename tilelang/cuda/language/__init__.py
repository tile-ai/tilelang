"""CUDA language dialect: common TileLang plus CUDA extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.allocate import alloc_cluster_barrier, alloc_descriptor, alloc_tmem  # noqa: F401
from tilelang.language.annotations import annotate_l2_hit_ratio, annotate_min_blocks_per_sm  # noqa: F401
from tilelang.language.builtin import (  # noqa: F401
    annotate_consumer_reg_alloc,
    annotate_producer_reg_dealloc,
    create_tma_descriptor,
    deallocate_tmem,
    dec_max_nreg,
    disable_warp_group_reg_alloc,
    fence_proxy_async,
    get_warp_group_idx,
    inc_max_nreg,
    increase_descriptor_offset,
    ldg128,
    ldg256,
    ldg32,
    ldg64,
    lds128,
    lds32,
    lds64,
    match_all_sync,
    match_any_sync,
    named_barrier_arrive,
    ptx_arrive_cluster_barrier,
    ptx_mma_sm70,
    set_max_nreg,
    shuffle_elect,
    stg128,
    stg256,
    stg32,
    stg64,
    sts128,
    sts32,
    sts64,
    tma_load,
    tma_load_2sm,
    tma_store_arrive,
    tma_store_wait,
)
from tilelang.language.copy_op import copy_cluster, tma_copy, tma_gather4, tma_gather4_bytes, tma_scatter4  # noqa: F401
from tilelang.language.kernel import ClusterKernel, CUDASourceCodeKernel  # noqa: F401

from .cluster import *  # noqa: F401,F403
from .cluster import __all__ as _CLUSTER_ALL
from .intrinsics import *  # noqa: F401,F403
from .intrinsics import __all__ as _INTRINSICS_ALL
from .pdl import *  # noqa: F401,F403
from .pdl import __all__ as _PDL_ALL
from .print import *  # noqa: F401,F403
from .print import __all__ as _PRINT_ALL
from .random import *  # noqa: F401,F403
from .random import __all__ as _RANDOM_ALL
from .tir import *  # noqa: F401,F403
from .tir import __all__ as _TIR_ALL
from .warpgroup import *  # noqa: F401,F403
from .warpgroup import __all__ as _WARPGROUP_ALL

_CUDA_API_ALL = (
    "ClusterKernel",
    "CUDASourceCodeKernel",
    "alloc_cluster_barrier",
    "alloc_descriptor",
    "alloc_tmem",
    "annotate_consumer_reg_alloc",
    "annotate_l2_hit_ratio",
    "annotate_min_blocks_per_sm",
    "annotate_producer_reg_dealloc",
    "copy_cluster",
    "create_tma_descriptor",
    "deallocate_tmem",
    "dec_max_nreg",
    "disable_warp_group_reg_alloc",
    "fence_proxy_async",
    "get_warp_group_idx",
    "inc_max_nreg",
    "increase_descriptor_offset",
    "ldg128",
    "ldg256",
    "ldg32",
    "ldg64",
    "lds128",
    "lds32",
    "lds64",
    "match_all_sync",
    "match_any_sync",
    "named_barrier_arrive",
    "ptx_arrive_cluster_barrier",
    "ptx_mma_sm70",
    "set_max_nreg",
    "shuffle_elect",
    "stg128",
    "stg256",
    "stg32",
    "stg64",
    "sts128",
    "sts32",
    "sts64",
    "tma_copy",
    "tma_gather4",
    "tma_gather4_bytes",
    "tma_load",
    "tma_load_2sm",
    "tma_scatter4",
    "tma_store_arrive",
    "tma_store_wait",
)

__tilelang_dialect__ = "cuda"
__all__ = tuple(
    dict.fromkeys(
        (
            *_COMMON_ALL,
            *_CUDA_API_ALL,
            *_CLUSTER_ALL,
            *_INTRINSICS_ALL,
            *_PDL_ALL,
            *_PRINT_ALL,
            *_RANDOM_ALL,
            *_TIR_ALL,
            *_WARPGROUP_ALL,
        )
    )
)

del _CLUSTER_ALL, _COMMON_ALL, _CUDA_API_ALL, _INTRINSICS_ALL, _PDL_ALL, _PRINT_ALL, _RANDOM_ALL, _TIR_ALL, _WARPGROUP_ALL
