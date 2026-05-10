"""
Sparsity configuration constants shared between tilelang.utils.sparse and
tilelang.intrinsics.*mma_sp* modules.

This module has no tilelang/tvm imports so it is safe to load early in the
import chain (before tilelang.jit is available).
"""

# TIR dtype string → (elem, group): nonzeros per group of consecutive elements.
# 2:4 sparsity for most types; 1:2 (TF32) for float32/float.
SPARSE_PARAMS: dict[str, tuple[int, int]] = {
    "float": (1, 2),
    "float32": (1, 2),
    "float16": (2, 4),
    "bfloat16": (2, 4),
    "int8": (2, 4),
    "uint8": (2, 4),
    "float8_e4m3": (2, 4),
    "float8_e5m2": (2, 4),
}

# e_factor: how many a_dtype elements are indexed by one meta_dtype element.
# Hardware packs 4 bits per sparsity group for all types (including TF32 1:2).
E_FACTOR_MAP: dict[str, dict[str, int]] = {
    "float": {"int16": 8, "uint16": 8},
    "float32": {"int16": 8, "uint16": 8},
    "float16": {"int8": 8, "uint8": 8, "int16": 16, "uint16": 16, "int32": 32, "uint32": 32},
    "bfloat16": {"int8": 8, "uint8": 8, "int16": 16, "uint16": 16, "int32": 32, "uint32": 32},
    "int8": {"int8": 8, "uint8": 8, "int16": 16, "uint16": 16, "int32": 32, "uint32": 32},
    "uint8": {"int8": 8, "uint8": 8, "int16": 16, "uint16": 16, "int32": 32, "uint32": 32},
    "float8_e4m3": {"int8": 8, "uint8": 8, "int16": 16, "uint16": 16, "int32": 32, "uint32": 32},
    "float8_e5m2": {"int8": 8, "uint8": 8, "int16": 16, "uint16": 16, "int32": 32, "uint32": 32},
}

# How many consecutive physical threads share the same logical metadata value.
# 16/32-bit A dtypes: factor=2; 8-bit A dtypes: factor=1.
E_REPLICATE_FACTOR: dict[str, int] = {
    "float": 2,
    "float32": 2,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "float8_e4m3": 1,
    "float8_e5m2": 1,
}


def get_e_factor(a_dtype: str, meta_dtype: str) -> int:
    """Return the e_factor: how many a_dtype elements are indexed by one meta_dtype element."""
    return E_FACTOR_MAP[a_dtype][meta_dtype]
