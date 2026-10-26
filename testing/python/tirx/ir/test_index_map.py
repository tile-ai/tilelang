import pytest
from tvm_tirx.ir import Var, assert_structural_equal
from tvm_tirx.tirx import IndexMap
from tvm_tirx.script import tirx as T
from tilelang._compat import ir as tlir


@pytest.mark.parametrize("dtypes", [("int32", "int32"), ("int64", "int64"), ("int32", "int64")])
@pytest.mark.parametrize("inverse_dtype", [None, "int32", "int64"])
def test_index_map_script_preserves_types(dtypes, inverse_dtype):
    row, column = [Var(name, dtype) for name, dtype in zip(("row", "column"), dtypes)]
    inverse = None
    if inverse_dtype is not None:
        inverse = tlir.index_map(lambda offset: (offset // 8, offset % 8), index_dtype=inverse_dtype)
    mapping = IndexMap([row, column], [row * 8 + column], inverse)
    source = tlir.script(mapping)
    restored = eval(source, {"T": T, "tlir": tlir})
    assert_structural_equal(restored, mapping)


def test_index_map_script_without_inputs():
    mapping = IndexMap([], [T.int32(4)], None)
    restored = eval(tlir.script(mapping), {"T": T, "tlir": tlir})
    assert_structural_equal(restored, mapping)


def test_index_map_per_axis_types_with_variadic_and_keyword_inputs():
    def mapping(row, *columns, offset):
        return [row, *columns, offset]

    dtypes = ["int32", "int64", "int32", "uint32"]
    index_map = tlir.index_map(mapping, ndim=4, index_dtype=dtypes)
    assert [str(index.ty) for index in index_map.initial_indices] == dtypes
    assert_structural_equal(index_map.initial_indices, index_map.final_indices)
    with pytest.raises(ValueError, match="one dtype per input iterator"):
        tlir.index_map(mapping, ndim=4, index_dtype=["int32"])
