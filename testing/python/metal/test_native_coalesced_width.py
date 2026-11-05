"""Native hint scalar types retain the established geometry constraints."""

import pytest

import tilelang
import tilelang.language as T
from tvm_tirx import ir
from tvm_tirx.target import Target


def _copy_source(hint):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1, threads=32):
            T.copy(A, B, coalesced_width=hint)

    target = Target("metal")
    with target:
        return tilelang.lower(main, target=target).kernel_source


@pytest.mark.parametrize("width", [1, 2, 3, 4, 8, 1 << 62])
def test_scalar_annotation_matches_explicit_ir_literal(width):
    source = _copy_source(width)
    assert "main_kernel" in source
    assert source == _copy_source(ir.IntImm("int64", width))


def test_requested_width_keeps_geometry_divisor_and_upper_bound():
    assert _copy_source(3) == _copy_source(2)
    assert _copy_source(1 << 62) == _copy_source(4)


@pytest.mark.parametrize("width", [0, -1, -(1 << 62)])
@pytest.mark.parametrize("representation", ["scalar", "ir"])
def test_nonpositive_hint_is_rejected(width, representation):
    hint = width if representation == "scalar" else ir.IntImm("int64", width)
    with pytest.raises(RuntimeError, match="coalesced_width must be a positive integer"):
        _copy_source(hint)


@pytest.mark.parametrize("hint", [1.5, "2"])
def test_noninteger_hint_is_rejected(hint):
    with pytest.raises(TypeError, match="Cannot treat type .* as type `ir.Expr`"):
        _copy_source(hint)


def test_out_of_range_literal_is_rejected_without_truncation():
    with pytest.raises(RuntimeError, match="Integer control argument is outside int64"):
        _copy_source(ir.IntImm("uint64", 1 << 63))
