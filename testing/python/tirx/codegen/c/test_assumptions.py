import numpy as np

from tvm_tirx import ir, runtime, tirx


def test_optimizer_assumptions_do_not_become_unresolved_c_calls():
    size = ir.Var("size", "int32")
    output = ir.Var("output", tirx.TensorType([1], "int32"))
    function = tirx.Function(
        [size, output],
        [
            ir.Evaluate(ir.Call("prim.assume", [size > 0])),
            ir.TensorStore(output, [ir.const(0)], size + 1),
        ],
    ).with_attr("global_symbol", "assumed")
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((1,), "int32")
    executable(5, actual)
    np.testing.assert_array_equal(actual.numpy(), [6])
