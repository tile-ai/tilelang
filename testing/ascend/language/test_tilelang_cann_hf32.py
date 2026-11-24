"""HF32 modes must reach the CANN compatibility layer through real codegen."""

import pytest
import tilelang.ascend.language as T
from tilelang.engine.lower import lower


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, "asc_disable_hf32();"),
        ("nearest_zero", "asc_hf32_round_mode::NEAREST_AWAY"),
        ("nearest_even", "asc_hf32_round_mode::NEAREST_EVEN"),
    ],
)
def test_hf32_compat_codegen(mode, expected):
    @T.prim_func
    def kernel(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 16), "float32")
            b_l1 = T.alloc_l1((16, 16), "float32")
            a = T.alloc_l0a((16, 16), "float32")
            b = T.alloc_l0b((16, 16), "float32")
            c = T.alloc_l0c((16, 16), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.set_hf32_mode(mode)
            T.copy(a_l1, a)
            T.copy(b_l1, b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C)

    source = lower(kernel, target="ascend").kernel_source
    assert expected in source.split("TL_CANN_USE_CORE_NAMES", 1)[1]
