"""Layout metadata for 2:4 semi-structured sparsity.

Shared contract between the sparse compressor (``tilelang.utils.sparse``) and the
CUDA ``mma_sp`` code generator. Depends only on the leaf ``dtypes`` module so it
can be pulled into the codegen import path without touching the language facade.
"""

from __future__ import annotations

import tilelang.language.dtypes as _dtypes
from tilelang.language.dtypes import dtype

GROUP_CONFIG: dict[dtype, tuple[int, int]] = {
    _dtypes.float: (1, 2),
    _dtypes.float16: (2, 4),
    _dtypes.bfloat16: (2, 4),
    _dtypes.int8: (2, 4),
    _dtypes.uint8: (2, 4),
    _dtypes.float8_e4m3: (2, 4),
    _dtypes.float8_e5m2: (2, 4),
}

_BITS_PER_GROUP = 4


def get_e_factor(a_dtype: dtype, meta_dtype: dtype) -> int:
    """Return how many a_dtype elements are indexed by one meta_dtype element."""
    _, group = GROUP_CONFIG[a_dtype]
    return (dtype(meta_dtype).bits // _BITS_PER_GROUP) * group


def get_e_replicate_factor(a_dtype: dtype) -> int:
    """Return how many consecutive threads share the same logical metadata value."""
    return 1 if dtype(a_dtype).bits <= 8 else 2
