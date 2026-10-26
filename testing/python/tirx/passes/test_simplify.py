import pytest
from tvm_tirx.script import ir as I

from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes", "tilelang_builders")


def _apply_simplify(func):
    """Helper to apply the default statement simplification."""
    input_func = func

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    return _tirx_passes.StmtSimplify()(InputModule)["main"]


def test_simplification_rewrites_for_annotation_expressions():
    from tvm_tirx import ir, tirx

    captured = ir.Var("captured", "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        ir.Evaluate(index),
        annotations={"metadata": {"nested": [captured, index]}},
    )
    function = tirx.Function([], [ir.Bind(captured, ir.const(7)), loop])
    result = _apply_simplify(function)
    assert int(result.body.seq[-1].annotations["metadata"]["nested"][0]) == 7
    assert not tirx.analysis.undefined_vars(result.body, [])


@pytest.mark.parametrize("pass_name", ["StmtSimplify", "RemoveNoOp", "LowerIntrin"])
@pytest.mark.parametrize("step", [-1, -3, -12])
@pytest.mark.parametrize("target", ["c", "llvm"])
def test_signed_loop_constraints_preserve_descending_stores(pass_name, step, target):
    import numpy as np
    from tvm_tirx import ir, runtime, tirx
    from tvm_tirx.target import Target

    if target == "llvm" and not runtime.enabled(target):
        pytest.skip("LLVM is not enabled")
    output = ir.Var("output", tirx.TensorType([16], "int32"))
    index = ir.Var("index", "int32")
    store = ir.TensorStore(output, [index], ir.prim.Select(index >= 0, index + 1, -77))
    body = ir.For(index, 9, -1, step, ir.ForKind.DEFAULT, store)
    function = tirx.Function([output], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "main").with_attr("target", Target(target))
    result = getattr(_tirx_passes, pass_name)()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    actual = runtime.tensor(np.full(16, -99, dtype="int32"))
    from tvm_tirx.tirx._compile import Executable, _build_for_target

    result = result.with_attr("target", Target({"kind": target, "host": target}))
    module = _tirx_passes.MakePackedAPI()(ir.IRModule({"main": result}))
    module = tirx.transform.LowerTVMBuiltin()(module)
    module = _tirx_passes.LowerIntrin()(module)
    module = _tirx_passes.AnnotateEntryFunc()(module)
    Executable(_build_for_target(module, Target(target)))(actual)
    expected = np.full(16, -99, dtype="int32")
    for iteration in range(9, -1, step):
        expected[iteration] = iteration + 1
    np.testing.assert_array_equal(actual.numpy(), expected)
