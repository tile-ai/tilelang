"""DMA address expressions must emit valid C++ statements."""

import re

import tilelang
from tilelang.ascend import language as T


def runtime_divisor_copy():
    @T.prim_func
    def kernel(
        src: T.Tensor((16, 8), "float32"),
        dst: T.Tensor((16, 8), "float32"),
        num_cols: T.int32,
    ):
        with T.Kernel(1):
            buf = T.alloc_shared((4, 8), "float32")
            for i in T.serial(4):
                row = i // num_cols
                T.copy(src[row * 4, 0], buf)
                T.copy(buf, dst[row * 4, 0])

    return kernel


def test_mte_copy_hoists_let_bindings_before_calls():
    source = tilelang.lower(runtime_divisor_copy(), target="ascend").kernel_source

    for call_name in ("asc_copy_gm2ub_align", "asc_copy_ub2gm_align"):
        call_start = source.index(call_name)
        call_end = source.index(");", call_start)
        call = source[call_start:call_end]
        assert not re.search(r"\bint(?:32|64)_t\s+\w+\s*=", call), source


def test_dma_extern_calls_with_imported_source():
    helper = """namespace user_dma {
__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t* dst, __gm__ uint8_t* src) {
    ::asc_copy_gm2ub_align(dst, src, 32);
}
}
#define asc_copy_gm2ub_align(...) static_assert(false, "imported macro reached generated DMA")
"""

    @T.prim_func
    def kernel(asc_copy_gm2ub_align: T.Tensor((32,), "uint8"), asc_load_l2_cache_mode: T.int32):
        with T.Kernel(1):
            T.import_source(helper)
            ub = T.alloc_shared((32,), "uint8")
            T.evaluate(T.call_extern("void", "user_dma::asc_copy_gm2ub_align", ub.data, asc_copy_gm2ub_align.data))
            T.evaluate(T.call_extern("void", "asc_copy_gm2ub_align", ub.data, asc_copy_gm2ub_align.data, 32))
            T.evaluate(T.call_extern("void", "::asc_copy_ub2gm_align", asc_copy_gm2ub_align.data, ub.data, asc_load_l2_cache_mode))

    source = tilelang.lower(kernel, target="ascend").kernel_source
    assert 0 <= source.rfind("#include") < source.index(helper)
    assert source.index("TL_CANN_SAVE_NAMES") < source.index(helper) < source.index("TL_CANN_RESTORE_NAMES")
    assert source.index("TL_CANN_RESTORE_NAMES") < source.index("TL_CANN_USE_CORE_NAMES")
    assert "user_dma::asc_copy_gm2ub_align(" in source
    signature, body = source.split("TL_CANN_USE_CORE_NAMES", 1)
    # Source parameter names can match either a bound function or a bound type.
    assert re.search(r"uint8_t\* asc_copy_gm2ub_align_?\d+", signature)
    assert re.search(r"int32_t asc_load_l2_cache_mode_?\d+", signature)
    assert re.search(r"(?<!:)\basc_copy_gm2ub_align\(", body)
    assert re.search(r"(?<!:)\basc_copy_ub2gm_align\(", body)
