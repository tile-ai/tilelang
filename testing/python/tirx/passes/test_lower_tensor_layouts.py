import pytest
from tvm_tirx.backend.cuda.axis import laneid
from tvm_tirx.script import tirx as T

pytestmark = pytest.mark.usefixtures("native_tirx_passes")

_CTA_LAUNCH_ATTRS = {
    "cuda.kernel_attrs": {},
    "cuda.launch_fields": ["grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"],
}

L_LANE = T.TileLayout(T.S[32 : 1 @ laneid])


@pytest.mark.parametrize("begin,end,step", [(0, 0, 1), (3, 1, 1), (0, 0, -1), (1, 3, -1)])
def test_empty_loops_do_not_apply_zero_extent_layouts(begin, end, step):
    from tvm_tirx import ir, tirx

    tensor_type = tirx.TensorType((0, 3), "float32")
    source = ir.Var("source", tensor_type)
    destination = ir.Var("destination", tensor_type)
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    body = ir.TensorStore(destination, [row, column], ir.TensorLoad(source, [row, column]))
    body = ir.For(column, 0, 3, 1, ir.ForKind.DEFAULT, body)
    body = ir.For(row, begin, end, step, ir.ForKind.DEFAULT, body)
    function = tirx.Function([source, destination], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "empty_layout_kernel")
    tirx.compile(function, target="c")
