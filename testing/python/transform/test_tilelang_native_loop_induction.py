"""Independent finite oracles for raw native For; no Python range interpretation."""

import numpy as np
import pytest

import tilelang
from tvm_tirx import ir, runtime, tirx
from tilelang.testing.packed import compile_packed_function


def _run(function, factory):
    module = ir.IRModule({"main": function.with_attr("global_symbol", "main")})
    if factory == "tirx":
        module = tirx.transform.StmtSimplify()(module)
    elif factory == "tilelang":
        module = tilelang.transform.Simplify()(module)
    assert tirx.analysis.check_well_formed(module)
    exe = compile_packed_function(module, c_codegen="compute")
    output = runtime.tensor(np.zeros(3, dtype="int32"))
    exe(output)
    return output.numpy()


@pytest.mark.parametrize("factory", ["none", "tirx", "tilelang"])
@pytest.mark.parametrize(
    "dtype,begin,end,step,expected",
    [
        ("int8", 120, 127, 10, [4, 3, -228]),
        ("uint8", 250, 255, 10, [4, 3, 292]),
    ],
)
def test_positive_small_integer_induction_can_wrap(factory, dtype, begin, end, step, expected):
    # LLVM deliberately omits NSW/NUW for sub-32-bit arithmetic. Values are
    # signed: 120, -126, -116, -106; unsigned: 250, 4, 14, 24.
    # All satisfy the native var < end predicate. Break fixes exactly 4 trips.
    output = ir.Var("output", tirx.TensorType((3,), "int32"))
    index = ir.Var("i", dtype)
    body = [
        ir.TensorStore(output, [0], ir.TensorLoad(output, [0]) + 1),
        ir.TensorStore(output, [1], ir.TensorLoad(output, [1]) + ir.prim.Cast("int32", index < ir.const(begin, dtype))),
        ir.TensorStore(output, [2], ir.TensorLoad(output, [2]) + ir.prim.Cast("int32", index)),
        ir.If(ir.TensorLoad(output, [0]) == 4, [ir.Break()], None),
    ]
    function = tirx.Function(
        [output],
        [
            ir.For(index, ir.const(begin, dtype), ir.const(end, dtype), ir.const(step, dtype), ir.ForKind.DEFAULT, body),
        ],
    )
    np.testing.assert_array_equal(_run(function, factory), expected)


@pytest.mark.parametrize("factory", ["none", "tirx", "tilelang"])
def test_wrapping_positive_step_is_not_one_iteration_from_extent_one(factory):
    output = ir.Var("output", tirx.TensorType((3,), "int32"))
    index = ir.Var("i", "int8")
    # 120, -9, 118, -11. The interval [120,121) is not the iteration domain.
    body = [
        ir.TensorStore(output, [0], ir.TensorLoad(output, [0]) + 1),
        ir.If(
            index < ir.const(120, "int8"),
            [
                ir.TensorStore(output, [1], ir.TensorLoad(output, [1]) + 1),
            ],
            None,
        ),
        ir.TensorStore(output, [2], ir.TensorLoad(output, [2]) + ir.prim.Cast("int32", index)),
        ir.If(ir.TensorLoad(output, [0]) == 4, [ir.Break()], None),
    ]
    function = tirx.Function(
        [output],
        [
            ir.For(
                index,
                ir.const(120, "int8"),
                ir.const(121, "int8"),
                ir.const(127, "int8"),
                ir.ForKind.DEFAULT,
                body,
            )
        ],
    )
    np.testing.assert_array_equal(_run(function, factory), [4, 3, 218])
