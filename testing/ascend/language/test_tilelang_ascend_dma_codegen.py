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

    begin = "#include <tl_templates/ascend/cann_compat_begin.h>"
    end = "#include <tl_templates/ascend/cann_compat_end.h>"
    for call_name in ("asc_copy_gm2ub_align", "asc_copy_ub2gm_align"):
        call_start = source.index(call_name)
        assert source.rfind(begin, 0, call_start) > source.rfind(end, 0, call_start)
        assert end in source[call_start:]
        call_end = source.index(");", call_start)
        call = source[call_start:call_end]
        assert not re.search(r"\bint(?:32|64)_t\s+\w+\s*=", call), source
