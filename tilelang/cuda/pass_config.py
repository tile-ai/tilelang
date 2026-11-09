"""CUDA-specific pass configuration keys.

Backend-owned options live next to the backend implementation so that adding a
new backend does not grow the common :mod:`tilelang.transform.pass_config`
namespace.  Common options (line directives, fast math, shared device compile
flags, and target-independent lowering controls) stay in
:class:`tilelang.transform.PassConfigKey`.

The members mirror the CUDA-specific keys registered in
``src/cuda/op/builtin.cc`` plus the Python-level CUDA compiler options.  Their
historical string values are intentionally unchanged.  Rollout is staged:

1. Introduction: both namespaces expose the members.  The legacy
   :class:`tilelang.transform.PassConfigKey` spellings keep working without a
   warning, so existing kernels are unaffected.
2. Deprecation: the legacy members emit a migration warning.  Raw string keys
   remain supported, and common members stay fully supported.
3. Removal: a separate future decision after the documented migration period.

Only the legacy enum spellings are affected; the underlying string keys and
raw-string configurations do not change.
"""

from __future__ import annotations

from enum import Enum


class PassConfigKey(str, Enum):
    """Pass configuration keys owned by the CUDA backend."""

    TL_DISABLE_WARP_SPECIALIZED = "tl.disable_warp_specialized"
    """Disable warp specialization optimization. Default: False"""

    TL_ENABLE_AUTO_WARP_SPECIALIZATION = "tl.enable_auto_warp_specialization"
    """Name of the automatic warp-specialization scheduler to run (e.g.
    "role_based"). Default: unset (disabled)."""

    TL_DISABLE_TMA_LOWER = "tl.disable_tma_lower"
    """Deprecated. Prevents plain T.copy() from auto-lowering to TMA store.

    Prefer ``T.copy(..., disable_tma=True)`` per copy instead.
    """

    TL_PTXAS_REGISTER_USAGE_LEVEL = "tl.ptxas_register_usage_level"
    """The PTXAS register usage level in [0, 10], which controls the
    aggressiveness of optimizations that affect register usage. Default: None"""

    TL_DISABLE_VECTORIZE_256 = "tl.disable_vectorize_256"
    """Disable usage of LDG/STG 256. Default: False"""

    TL_ENABLE_FP32X2_REDUCTION = "tl.enable_fp32x2_reduction"
    """Enable packed FP32x2 accumulation for CUDA sum and absolute-sum
    reductions on SM100 and newer GPUs. Per-reduction annotations can still
    disable the optimization. Setting this to False disables it globally.
    Default: True"""

    TL_DISABLE_WGMMA = "tl.disable_wgmma"
    """Disable usage of Hopper WGMMA. Default: False"""

    TL_DISABLE_SHUFFLE_ELECT = "tl.disable_shuffle_elect"
    """Disable shuffle election optimization. Default: False"""

    TL_ENABLE_LOWER_LDGSTG = "tl.enable_lower_ldgstg"
    """Enable non-predicated LDG/STG lowering for global memory access.
    When enabled, converts Ramp-based global buffer load/store to ldg/stg intrinsics.
    Default: False"""

    TL_ENABLE_LOWER_LDGSTG_PREDICATED = "tl.enable_lower_ldgstg_predicated"
    """Enable predicated LDG/STG lowering.
    When True, predicated loads (if_then_else with else=0) and
    predicated stores (IfThenElse with empty then case) are lowered to
    ldg/stg intrinsics. Default: False"""

    TL_CUDA_COMPILER = "tl.cuda_compiler"
    """CUDA device compiler for the TVM-FFI execution backend: "nvcc" or "nvrtc".

    Default: "nvcc". NVRTC requires cuda-python, CUDA headers, and the NVRTC
    library. Host IR uses the selected host codegen; target_host="c" requires
    a host C/C++ compiler, including on Windows.
    """

    CUDA_KERNELS_OUTPUT_DIR = "cuda.kernels_output_dir"
    """Output directory for generated CUDA kernels. Default: empty string"""


__all__ = ["PassConfigKey"]
