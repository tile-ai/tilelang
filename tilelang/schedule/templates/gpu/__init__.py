# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
GPU-generic schedule rules.
For CUDA/ROCm/Vulkan/Metal-specific rules, use `tvm.dlight.cuda/rocm/vulkan/metal` instead
"""

from .fallback import Fallback  # noqa: F401
from .element_wise_ndim import ElementWiseNDim  # noqa: F401
from .general_reduction import GeneralReduction  # noqa: F401
from .layernorm_like import LayerNormLike  # noqa: F401
from .transpose import Transpose  # noqa: F401

from .gemv import GEMV  # noqa: F401
from .indexed_gemv import IndexedGEMV  # noqa: F401
from .matmul import Matmul  # noqa: F401


def default_schedule_rules():
    """Return the default TileLang GPU rule order with generic fallback last."""
    return [
        IndexedGEMV(),
        Matmul(),
        GEMV(),
        LayerNormLike(),  # Specialised two-reduction chain; try before GeneralReduction
        GeneralReduction(),  # Subsumes former Reduction
        Transpose(),
        ElementWiseNDim(),  # Axis-walk (preserves N-D)
        Fallback(),
    ]
