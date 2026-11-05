"""A For induction definition snapshots begin even when it reads mutable state."""

import numpy as np
import pytest
import tilelang
from tvm_tirx import ir, runtime, tirx
from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("factory", ["none", "tirx", "tilelang"])
def test_induction_var_does_not_reread_mutable_begin(factory):
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    state = ir.Var("state", tirx.TensorType((1,), "int32"))
    index = ir.Var("i", "int32")
    begin = ir.TensorLoad(state, [0])
    function = tirx.Function(
        [output, state],
        [
            ir.For(
                index,
                begin,
                begin + 1,
                ir.const(1),
                ir.ForKind.DEFAULT,
                [
                    ir.TensorStore(state, [0], ir.const(100)),
                    ir.TensorStore(output, [0], index),
                    ir.Break(),
                ],
            )
        ],
    ).with_attr("global_symbol", "main")
    module = ir.IRModule({"main": function})
    assert tirx.analysis.check_well_formed(module)
    if factory == "tirx":
        module = tirx.transform.StmtSimplify()(module)
    elif factory == "tilelang":
        module = tilelang.transform.Simplify()(module)
    executable = compile_packed_function(module, c_codegen="compute")
    result = runtime.tensor(np.zeros(1, dtype="int32"))
    data = runtime.tensor(np.array([3], dtype="int32"))
    executable(result, data)
    np.testing.assert_array_equal(result.numpy(), [3])
    np.testing.assert_array_equal(data.numpy(), [100])
