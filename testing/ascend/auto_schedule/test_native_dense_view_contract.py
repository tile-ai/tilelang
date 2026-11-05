"""Dense views retain an explicit layout and the actual backing pointer."""

import pytest
import tilelang.language as T
import tvm_ffi
from tilelang.language import _native
from tvm_tirx import ir, tirx, sym


@pytest.mark.parametrize("scope", ["global", "shared.dyn", "shared.l1", "local"])
@pytest.mark.parametrize(
    "shape,dtype,view_shape,view_dtype,strides",
    [
        ((8,), "int32", (8,), None, (1,)),
        ((2, 8), "float32", (4, 8), "float16", (8, 1)),
        ((2, 16), "float4_e2m1fn", (2, 8), "uint8", (8, 1)),
        ((), "float32", (), "int32", ()),
        ((1,), "float32", (), "float32", ()),
        ((0, 8), "float32", (0, 4), "float64", (4, 1)),
        ((2, 3), "float32", (6,), "float32", (1,)),
    ],
)
def test_dense_view_keeps_explicit_layout_and_scope(scope, shape, dtype, view_shape, view_dtype, strides):
    src = ir.Var("src", tirx.TensorType(shape, dtype, scope=scope, layout=None))
    view = T.view(src, view_shape, dtype=view_dtype)
    view_type = view.as_type()
    # A scalar has no stride axes; the native type canonicalizes its empty
    # layout. Ranked views retain their explicit dense strides.
    assert (view_type.layout is not None) == bool(view_shape)
    assert str(view_type.scope.name) == scope
    assert [int(value) for value in _native.tensor_strides(view)] == list(strides)
    assert int(_native.tensor_offset(view)) == 0
    assert tvm_ffi.structural_equal(view.data, src.data)


def test_dense_view_keeps_symbolic_destination_strides():
    extent = ir.Var("extent", "int32")
    src = ir.Var("src", tirx.TensorType((2, extent), "float32", scope="shared.dyn", layout=None))
    view = T.view(src, (1, 2, extent))
    assert view.as_type().layout is not None
    actual = _native.tensor_strides(view)
    assert len(actual) == 3
    analyzer = sym.Analyzer()
    assert analyzer.can_prove_equal(actual[0], 2 * extent)
    assert analyzer.can_prove_equal(actual[1], extent)
    assert int(actual[2]) == 1
