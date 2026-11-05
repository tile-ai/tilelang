"""Native static cost analysis uses real tensor dtypes and lexical loop domains."""

import sys

import pytest

import tilelang.language as T
from tilelang.tools import Analyzer


class _Device:
    compute_capability = "80"
    bandwidth = [750, 12080]


def test_three_dimensional_launch_and_nonunit_trip_count_reset_on_repeat():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "float32")):
        with T.Kernel(2, 3, 4, threads=32):
            shared = T.alloc_shared((4,), "float32")
            for _ in T.serial(2, 8, 2):
                T.copy(A, shared)

    analyzer = Analyzer(kernel, _Device())
    for _ in range(2):
        result = analyzer.ir_pass().calculate()
        assert result.total_global_bytes == 2 * 3 * 4 * 3 * 4 * 4
        assert result.total_flops == 0
    assert "tvm" not in sys.modules


@pytest.mark.parametrize("dtype,expected_bytes", [("bfloat16", 10), ("float4_e2m1fn", 3)])
def test_copy_byte_count_uses_native_bits_and_packed_rounding(dtype, expected_bytes):
    @T.prim_func
    def kernel(A: T.Tensor((5,), dtype), B: T.Tensor((5,), dtype)):
        T.copy(A, B)

    result = Analyzer.analysis(kernel, _Device())
    assert result.total_global_bytes == expected_bytes


def test_symbolic_work_is_explicitly_unresolved_in_static_model():
    n = T.dynamic("n")

    @T.prim_func
    def kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        T.copy(A, B)

    with pytest.raises(ValueError, match="Static performance analysis needs a concrete copy extent"):
        Analyzer.analysis(kernel, _Device())
