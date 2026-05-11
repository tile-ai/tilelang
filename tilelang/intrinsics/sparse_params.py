"""
Sparsity configuration constants shared between tilelang.utils.sparse and
tilelang.intrinsics.*mma_sp* modules.

This module only depends on tvm (not tilelang.jit) so it is safe to load
early in the import chain (before tilelang.jit is available).
"""

import tilelang.language as T
from tilelang.language.dtypes import dtype

GROUP_CONFIG: dict[dtype, tuple[int, int]] = {
    T.float: (1, 2),
    T.float16: (2, 4),
    T.bfloat16: (2, 4),
    T.int8: (2, 4),
    T.uint8: (2, 4),
    T.float8_e4m3: (2, 4),
    T.float8_e5m2: (2, 4),
}


_BITS_PER_GROUP = 4


def get_e_factor(a_dtype: dtype, meta_dtype: dtype) -> int:
    """Return how many a_dtype elements are indexed by one meta_dtype element."""
    _, group = GROUP_CONFIG[a_dtype]
    return (dtype(meta_dtype).bits // _BITS_PER_GROUP) * group


def get_e_replicate_factor(a_dtype: dtype) -> int:
    """Return how many consecutive threads share the same logical metadata value."""
    return 1 if dtype(a_dtype).bits <= 8 else 2
