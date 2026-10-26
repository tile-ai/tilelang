import pytest

from tilelang._compat import ir, tirx
from tvm_tirx.ir import prim
from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_vector_integer_cast_retains_all_lanes(lanes):
    source = ir.Var("source", tirx.TensorType([lanes], "float32"))
    output = ir.Var("output", tirx.TensorType([lanes], "int32"))
    indices = [prim.Ramp(0, 1, lanes)]
    value = prim.Cast(f"int32x{lanes}", ir.TensorLoad(source, indices))
    function = tirx.Function([source, output], ir.TensorStore(output, indices, value))
    module = ir.IRModule({"main": function})
    transformed = _tirx_passes.ForceNarrowIndexToInt32()(module)
    ir.assert_structural_equal(module, transformed)


@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_vector_index_parameter_retains_all_lanes(lanes):
    def make_function(bits):
        indices = ir.Var("indices", f"int{bits}x{lanes}")
        source = ir.Var("source", tirx.TensorType([32], "float32"))
        output = ir.Var("output", tirx.TensorType([lanes], "float32"))
        body = ir.TensorStore(output, [prim.Ramp(0, 1, lanes)], ir.TensorLoad(source, [indices]))
        return tirx.Function([indices, source, output], body)

    module = ir.IRModule({"main": make_function(64)})
    transformed = _tirx_passes.ForceNarrowIndexToInt32()(module)
    ir.assert_structural_equal(ir.IRModule({"main": make_function(32)}), transformed)
