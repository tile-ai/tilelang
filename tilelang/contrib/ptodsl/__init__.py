"""TileLang compatibility helpers for generated PTODSL kernels."""

from .common import (
    as_logical_bool,
    coerce_i1,
    coerce_i8,
    coerce_i16,
    coerce_i32,
    coerce_i64,
    coerce_runtime_integer_value,
    if_then_else,
    logical_not,
    scalar_bitcast,
    scalar_cast,
    unwrap_surface_value,
    ushr,
    wrap_surface_value,
)
from .dcache_bypass import (
    read_gm_bypass_dcache,
    write_gm_bypass_dcache,
)
from .gemm import PTOBlockscaledGemmL1Template, PTOGemmL1Template
from .mixed_kernel import finalize_mixed_kernel, mixed_kernel_section
from .rng import PhiloxRNG
from .simd_inst import (
    vdiv_precise_f32,
    vexp_1ulp_ftz_false,
    vln_1ulp_ftz_false,
    vsqrt_0ulp_ftz_false,
)
from .simt import (
    scalar_div,
    scalar_rsqrt,
    simt_allreduce_max,
    simt_allreduce_min,
    simt_allreduce_sum,
    store_vector_to_list,
    vector_from_list,
    vector_to_list,
    vectorize_binary_f32x2,
    vectorize_unary_f32x2,
)
from .sync import ascend_cross_core_set_flag, ascend_cross_core_wait_flag

__all__ = [
    "PhiloxRNG",
    "PTOBlockscaledGemmL1Template",
    "PTOGemmL1Template",
    "ascend_cross_core_set_flag",
    "ascend_cross_core_wait_flag",
    "as_logical_bool",
    "coerce_i1",
    "coerce_i8",
    "coerce_i16",
    "coerce_i32",
    "coerce_i64",
    "coerce_runtime_integer_value",
    "finalize_mixed_kernel",
    "if_then_else",
    "logical_not",
    "mixed_kernel_section",
    "read_gm_bypass_dcache",
    "scalar_bitcast",
    "scalar_cast",
    "unwrap_surface_value",
    "ushr",
    "vdiv_precise_f32",
    "vexp_1ulp_ftz_false",
    "vln_1ulp_ftz_false",
    "vsqrt_0ulp_ftz_false",
    "wrap_surface_value",
    "write_gm_bypass_dcache",
    "scalar_div",
    "scalar_rsqrt",
    "simt_allreduce_max",
    "simt_allreduce_min",
    "simt_allreduce_sum",
    "store_vector_to_list",
    "vector_from_list",
    "vector_to_list",
    "vectorize_binary_f32x2",
    "vectorize_unary_f32x2",
]
