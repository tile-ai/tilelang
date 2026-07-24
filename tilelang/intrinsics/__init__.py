"""Compatibility facade for backend-specific intrinsic helpers."""

from tilelang.cuda.language.intrinsics import (  # noqa: F401
    TCGEN05DescriptorParams,
    TCGEN05TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithLadderTransform,
    WGMMADescriptorParams,
    WGMMATensorCoreIntrinEmitter,
    get_ldmatrix_offset,
    get_swizzle_layout,
    make_mma_swizzle_layout,
    mma_store_index_map,
)
from tilelang.rocm.language.intrinsics import make_mfma_swizzle_layout  # noqa: F401
