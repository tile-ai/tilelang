"""TileLang compatibility helpers for generated PTODSL kernels."""

from .allreduce import (
    simt_allreduce_max,
    simt_allreduce_min,
    simt_allreduce_sum,
)
from .common import (
    as_logical_bool,
    if_then_else,
    logical_not,
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
    fp8_byte_load,
    fp8_byte_store,
    scalar_binary_fp8,
    scalar_div,
    scalar_rsqrt,
    store_vector_to_list,
    shuffle_vec,
    vector_from_list,
    vector_to_list,
    vectorize_binary_f32x2,
    vectorize_binary_fp8,
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
    "fp8_byte_load",
    "fp8_byte_store",
    "finalize_mixed_kernel",
    "if_then_else",
    "logical_not",
    "mixed_kernel_section",
    "read_gm_bypass_dcache",
    "scalar_binary_fp8",
    "vdiv_precise_f32",
    "vexp_1ulp_ftz_false",
    "vln_1ulp_ftz_false",
    "vsqrt_0ulp_ftz_false",
    "write_gm_bypass_dcache",
    "scalar_div",
    "scalar_rsqrt",
    "simt_allreduce_max",
    "simt_allreduce_min",
    "simt_allreduce_sum",
    "store_vector_to_list",
    "shuffle_vec",
    "vector_from_list",
    "vector_to_list",
    "vectorize_binary_f32x2",
    "vectorize_binary_fp8",
    "vectorize_unary_f32x2",
]
