"""
Sparsity configuration constants shared between tilelang.utils.sparse and
tilelang.intrinsics.*mma_sp* modules.

This module only depends on tvm (not tilelang) so it is safe to load early
in the import chain (before tilelang.jit is available).
"""

from tvm import DataType

# str/DataType → (elem, group): nonzeros per group of consecutive elements.
# 2:4 sparsity for most types; 1:2 (TF32) for float32.
SPARSE_PARAMS: dict[str, tuple[int, int]] = {
    "float32": (1, 2),
    "float16": (2, 4),
    "bfloat16": (2, 4),
    "int8": (2, 4),
    "uint8": (2, 4),
    "float8_e4m3": (2, 4),
    "float8_e5m2": (2, 4),
}

# Hardware always packs 4 bits per sparsity group for all supported types,
# including TF32 1:2 which theoretically needs only 1 bit.
_BITS_PER_GROUP = 4


def get_e_factor(a_dtype, meta_dtype) -> int:
    """Return how many a_dtype elements are indexed by one meta_dtype element."""
    _, group = SPARSE_PARAMS[a_dtype]
    return (DataType(meta_dtype).bits // _BITS_PER_GROUP) * group


def get_e_replicate_factor(a_dtype) -> int:
    """Return how many consecutive threads share the same logical metadata value."""
    return 1 if DataType(a_dtype).bits <= 8 else 2
