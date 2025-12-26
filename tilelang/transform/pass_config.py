# TODO: Add more documentation for each pass config

from enum import Enum


class PassConfigKey(str, Enum):
    """Pass configuration keys for TileLang compiler."""

    # TileLang specific configs
    TL_SIMPLIFY = "tl.Simplify"
    """Enable/disable TileLang simplification passes. Default: True"""

    TL_DISABLE_WARP_SPECIALIZED = "tl.disable_warp_specialized"
    """Disable warp specialization optimization. Default: False"""

    TL_ENABLE_FAST_MATH = "tl.enable_fast_math"
    """
        Enable fast math optimization. Default: False
        if enabled, --use_fast_math will be passed to nvcc
    """

    TL_PTXAS_REGISTER_USAGE_LEVEL = "tl.ptxas_register_usage_level"
    """The PTXAS register usage level in [0, 10], which controls the
    aggressiveness of optimizations that affect register usage. Default: None"""

    TL_ENABLE_PTXAS_VERBOSE_OUTPUT = "tl.enable_ptxas_verbose_output"
    """Enable ptxas verbose output. Default: False"""

    TL_DEVICE_COMPILE_FLAGS = "tl.device_compile_flags"
    """Additional device compiler flags passed to nvcc/NVRTC.

    Accepts either a string (parsed with shell-like splitting) or a list of
    strings. Typical usage is to provide extra include paths, defines or
    ptxas options, e.g.:

    - "-I/opt/include -DMY_SWITCH=1 --ptxas-options=--verbose"
    - ["-I/opt/include", "-DMY_SWITCH=1", "--ptxas-options=--verbose"]

    These flags are appended to the compiler options used in the tvm_ffi
    CUDA compile callback. Default: None
    """

    TL_CONFIG_INDEX_BITWIDTH = "tl.config_index_bitwidth"
    """Bitwidth for configuration indices. Default: 32"""

    TL_DISABLE_TMA_LOWER = "tl.disable_tma_lower"
    """Disable TMA (Tensor Memory Access) lowering. Default: False"""

    TL_DISABLE_SAFE_MEMORY_ACCESS = "tl.disable_safe_memory_legalize"
    """Disable safe memory access optimization. Default: False"""

    TL_DISABLE_VECTORIZE_256 = "tl.disable_vectorize_256"
    """Disable usage of LDG/STG 256. Default: False"""
    TL_DISABLE_WGMMA = "tl.disable_wgmma"
    """Disable usage of Hopper WGMMA. Default: False"""

    TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS = "tl.debug_merge_shared_memory_allocations"
    """Enable debug information for merge shared memory allocations. Default: False"""

    TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE = "tl.enable_aggressive_shared_memory_merge"
    """Enable aggressive merge of shared memory allocations. Default: False"""

    TL_DISABLE_SHUFFLE_ELECT = "tl.disable_shuffle_elect"
    """Disable shuffle election optimization. Default: False"""

    TL_DISABLE_THREAD_STORAGE_SYNC = "tl.disable_thread_storage_sync"
    """Disable thread storage synchronization pass. When enabled, disables the
    automatic insertion of thread synchronization barriers (e.g., __syncthreads())
    for shared memory access coordination. This can be useful for performance
    optimization in cases where manual synchronization is preferred or when
    synchronization is not needed. Default: False"""

    TL_FORCE_LET_INLINE = "tl.force_let_inline"
    """Force TileLang to inline let bindings during simplification. Default: False"""

    TL_AST_PRINT_ENABLE = "tl.ast_print_enable"
    """Enable TIR AST printing for debugging purposes. Default: False"""

    TL_LAYOUT_VISUALIZATION_ENABLE = "tl.layout_visualization_enable"
    """Enable layout inference visualization. Default: False"""

    TL_LAYOUT_VISUALIZATION_FORMATS = "tl.layout_visualization_formats"
    """Layout visualization formats.
    Acceptable values: "pdf", "png", "svg", "all"

    """

    TL_STORAGE_REWRITE_DETECT_INPLACE = "tl.storage_rewrite_detect_inplace"
    """Control StorageRewrite inplace detection.

    When False (default) StorageRewrite keeps distinct temporaries for patterns
    such as `dst[i] = f(src[i])`, avoiding implicit aliasing:

    ```
    read = T.allocate([1], T.int32, "local.var")
    write = T.allocate([1], T.int32, "local.var")
    read_buf = T.Buffer((1,), T.int32, data=read, scope="local.var")
    write_buf = T.Buffer((1,), T.int32, data=write, scope="local.var")
    write_buf[0] = read_buf[0] * 2
    f(write_buf[0])
    ```

    Setting the flag to True allows StorageRewrite to reuse the `read` buffer
    for the write when it can prove the update is safely inplace, producing IR
    like:

    ```
    read = T.allocate([1], T.int32, "local.var")
    read_buf = T.Buffer((1,), T.int32, data=read, scope="local.var")
    read_buf[0] = read_buf[0] * 2
    f(read_buf[0])
    ```

    This reduces local memory usage but introduces aliasing between the buffers.

    Usage:

    ```python
    from tilelang.transform import PassContext, PassConfigKey

    with PassContext(
        config={PassConfigKey.TL_STORAGE_REWRITE_DETECT_INPLACE.value: True}
    ):
        mod = tilelang.transform.StorageRewrite()(mod)
    ```
    """

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
