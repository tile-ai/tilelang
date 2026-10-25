import numpy as np
import pytest
from tvm_tirx import ir, runtime, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.transform import ConfigIndexBitwidth

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def run_pass(function, width=None):
    config = {} if width is None else {"tl.config_index_bitwidth": width}
    with ir.transform.PassContext(config=config):
        return ConfigIndexBitwidth()(ir.IRModule({"main": function}))["main"]


def test_index_promotion_does_not_change_data_arithmetic_or_parameter_identity():
    source = TensorView.create("source", [3000000000], "float32")
    output = TensorView.create("output", [5], "float32")
    index = ir.Var("index", "int32")
    body = output.store(index, source.load(index * 600000000) + 1)
    function = tirx.Function([source.tensor, output.tensor], ir.For(index, 0, 5, 1, ir.ForKind.DEFAULT, body), ret_type=ir.TupleType([]))
    rewritten = run_pass(function)
    loads = []
    stmt_functor.post_order_visit(rewritten.body, lambda node: loads.append(node) if isinstance(node, ir.TensorLoad) else None)
    assert len(loads) == 1
    assert loads[0].indices[0].ty == ir.PrimType("int64")
    assert loads[0].ty == ir.PrimType("float32")
    assert loads[0].source.same_as(source.tensor)
    assert all(before.same_as(after) for before, after in zip(function.params, rewritten.params))
    assert tirx.analysis.check_well_formed(rewritten)


@pytest.mark.parametrize("width", [32, 64])
def test_explicit_index_width_preserves_symbolic_tensor_abi(width):
    extent = ir.Var("extent", "int64")
    output = TensorView.create("output", [extent], "float32")
    index = ir.Var("index", "int64")
    body = output.store(index, ir.const(2.0, "float32"))
    function = tirx.Function([extent, output.tensor], ir.For(index, 0, extent, 1, ir.ForKind.DEFAULT, body), ret_type=ir.TupleType([]))
    rewritten = run_pass(function, width)
    assert rewritten.params[0].ty == ir.PrimType(f"int{width}")
    assert rewritten.params[1].ty.shape[0].same_as(rewritten.params[0])
    assert tirx.analysis.check_well_formed(rewritten)
    ir.assert_structural_equal(rewritten, ir.load_json(ir.save_json(rewritten)))


@pytest.mark.parametrize("width", [0, 8, 16, 128])
def test_invalid_index_width_is_rejected(width):
    function = tirx.Function([], ir.Evaluate(ir.const(0)), ret_type=ir.TupleType([]))
    with pytest.raises(ValueError, match="32 or 64"):
        run_pass(function, width)


def test_automatic_index_promotion_executes_without_int8_overflow():
    source = TensorView.create("source", [256], "int32")
    output = TensorView.create("output", [6], "int32")
    index = ir.Var("index", "int8")
    body = output.store(index, source.load(index * ir.const(30, "int8")))
    loop = ir.For(index, ir.const(0, "int8"), ir.const(6, "int8"), ir.const(1, "int8"), ir.ForKind.DEFAULT, body)
    function = tirx.Function([source.tensor, output.tensor], loop, ret_type=ir.TupleType([])).with_attr("global_symbol", "safe_indices")
    rewritten = run_pass(function)
    executable = tirx.compile(rewritten, target="c")
    actual = runtime.empty((6,), "int32")
    executable(runtime.tensor(np.arange(256, dtype="int32")), actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(6, dtype="int32") * 30)
