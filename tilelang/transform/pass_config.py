# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# TODO: Add more documentation for each pass config

from enum import Enum


class PassConfigKey(str, Enum):
    """Pass configuration keys for TileLang compiler."""
    # TileLang specific configs
    TL_SIMPLIFY = "tl.Simplify"
    """Enable/disable TileLang simplification passes. Default: True"""

    TL_DYNAMIC_ALIGNMENT = "tl.dynamic_alignment"
    """Memory alignment requirement for dynamic shapes. Default: 16"""

    TL_DISABLE_DYNAMIC_TAIL_SPLIT = "tl.disable_dynamic_tail_split"
    """Disable dynamic tail splitting optimization. Default: False"""

    TL_DISABLE_WARP_SPECIALIZED = "tl.disable_warp_specialized"
    """Disable warp specialization optimization. Default: False"""

    TL_CONFIG_INDEX_BITWIDTH = "tl.config_index_bitwidth"
    """Bitwidth for configuration indices. Default: 32"""

    TL_DISABLE_TMA_LOWER = "tl.disable_tma_lower"
    """Disable TMA (Tensor Memory Access) lowering. Default: False"""

    TL_DISABLE_SAFE_MEMORY_ACCESS = "tl.disable_safe_memory_legalize"
    """Disable safe memory access optimization. Default: False"""

    # TIR related configs
    TIR_ENABLE_EQUIV_TERMS_IN_CSE = "tir.enable_equiv_terms_in_cse_tir"
    """Enable equivalent terms in TIR Common Subexpression Elimination. Default: True"""

    TIR_DISABLE_CSE = "tir.disable_cse_tir"
    """Disable TIR Common Subexpression Elimination. Default: False"""

    TIR_SIMPLIFY = "tir.Simplify"
    """Enable/disable TIR simplification passes. Default: True"""

    TIR_DISABLE_STORAGE_REWRITE = "tir.disable_storage_rewrite"
    """Disable storage rewrite optimization. Default: False"""

    TIR_DISABLE_VECTORIZE = "tir.disable_vectorize"
    """Disable vectorization optimization. Default: False"""

    TIR_USE_ASYNC_COPY = "tir.use_async_copy"
    """Enable asynchronous memory copy operations. Default: True"""

    TIR_ENABLE_DEBUG = "tir.enable_debug"
    """Enable debug information in generated code. Default: False"""

    TIR_MERGE_STATIC_SMEM = "tir.merge_static_smem"
    """Merge static shared memory allocations. Default: True"""

    TIR_ADD_LOWER_PASS = "tir.add_lower_pass"
    """Additional lowering passes to be applied. Default: None"""

    TIR_NOALIAS = "tir.noalias"
    """Enable pointer non-aliasing assumptions. Default: True"""

    CUDA_KERNELS_OUTPUT_DIR = "cuda.kernels_output_dir"
    """Output directory for generated CUDA kernels. Default: empty string"""
