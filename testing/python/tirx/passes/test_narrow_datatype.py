import pytest

from tilelang._compat import ir, tirx
from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def test_narrow_pointer_valued_conditional():
    pointer = ir.PtrType(ir.PrimType("int32"))
    first = ir.Var("first", pointer)
    second = ir.Var("second", pointer)
    condition = ir.Var("condition", "bool")
    selected = ir.Var("selected", pointer)
    function = tirx.Function(
        [first, second, condition],
        [
            ir.Bind(selected, ir.Call("prim.if_then_else", [condition, first, second])),
            ir.Evaluate(ir.Call("tirx.isnullptr", [selected])),
        ],
    )
    module = ir.IRModule({"main": function})
    narrowed = _tirx_passes.NarrowDataType(32)(module)
    assert tirx.analysis.check_well_formed(narrowed["main"])
    ir.assert_structural_equal(narrowed, module)


@pytest.mark.parametrize("dtype", ["float4_e2m1fn", "float8_e4m3fn", "float16", "bfloat16"])
def test_narrow_preserves_floating_conditional_type(dtype):
    condition = ir.Var("condition", "bool")
    value = ir.Var("value", dtype)
    conditional = ir.Call("prim.if_then_else", [condition, value, ir.const(0, "int32")], ty=ir.PrimType(dtype))
    function = tirx.Function([condition, value], [ir.Evaluate(conditional)])
    rewritten = _tirx_passes.NarrowDataType(32)(ir.IRModule({"main": function}))
    result = rewritten["main"].body.seq[0].value
    assert result.ty == ir.PrimType(dtype)
    assert result.args[1].ty == result.ty
    assert result.args[2].ty == result.ty
    assert tirx.analysis.check_well_formed(rewritten)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("reverse", [False, True])
def test_inferred_floating_conditional_does_not_cast_infinity_to_integer(dtype, reverse):
    condition = ir.Var("condition", "bool")
    branches = [ir.const(0, "int32"), ir.const(float("-inf"), dtype)]
    if reverse:
        branches.reverse()
    conditional = tirx.if_then_else(condition, *branches)
    assert conditional.ty == ir.PrimType(dtype)
    function = tirx.Function([condition], [ir.Evaluate(conditional)])
    rewritten = _tirx_passes.NarrowDataType(32)(ir.IRModule({"main": function}))
    result = rewritten["main"].body.seq[0].value
    assert result.ty == ir.PrimType(dtype)
    assert all(branch.ty == result.ty for branch in result.args[1:])
    assert float(result.args[1 if reverse else 2].value) == float("-inf")
    assert tirx.analysis.check_well_formed(rewritten)
