import pytest
from tvm_tirx import ir, tirx

from tilelang._compat.tensor import TensorView
from testing.python.tirx.test_tirx_ascend_codegen import _function, _source

pytestmark = pytest.mark.usefixtures("native_tirx_passes", "ascend_codegen")


def _program(mode, build_body):
    flag = ir.Var("flag", "int32")
    dtype = "float32" if mode == "hf32" else "bfloat16"
    left = TensorView.create("left", [256], dtype, scope="shared.l0a")
    right = TensorView.create("right", [256], dtype, scope="shared.l0b")
    output = TensorView.create("output", [256], "float32", scope="shared.l0c")

    def set_mode(value):
        name = "tl.ascend_set_hf32_mode" if mode == "hf32" else "tl.ascend_set_mmad_direction"
        argument = value if mode == "hf32" else ("n" if value else "m")
        return ir.Evaluate(ir.Call(name, [argument]))

    def mad():
        arguments = [tirx.address_of(tensor.load(0)) for tensor in (output, left, right)]
        return ir.Evaluate(ir.Call("tl.ascend_mad", [*arguments, 16, 16, 16, 0, 1, 0, True]))

    body = [tensor.allocate() for tensor in (left, right, output)]
    return _function([flag], [*body, *build_body(flag, set_mode, mad)])


def _modes(function, mode):
    source = _source(function)
    marker = "tf32_mode=pto.Tf32Mode.ROUND_AWAY" if mode == "hf32" else "n_dir=True"
    return [marker in line for line in source.splitlines() if line.strip().startswith("pto.mad(")]


@pytest.mark.parametrize("mode", ["direction", "hf32"])
@pytest.mark.parametrize("terminal", ["break", "continue"])
@pytest.mark.parametrize("bounds", [(0, 3, 1), (3, -1, -2), (3, 0, -4)])
def test_pto_loop_exits_ignore_unreachable_setters(mode, terminal, bounds):
    def body(flag, set_mode, mad):
        index = ir.Var("index", "int32")
        exit_statement = ir.Break() if terminal == "break" else ir.Continue()
        loop = ir.For(index, *bounds, ir.ForKind.DEFAULT, [set_mode(1), exit_statement, set_mode(0)])
        return [loop, mad()]

    assert _modes(_program(mode, body), mode) == [True]


@pytest.mark.parametrize("mode", ["direction", "hf32"])
def test_pto_returning_branch_does_not_reach_following_mad(mode):
    def body(flag, set_mode, mad):
        return [ir.If(flag > 0, [set_mode(1), ir.Return(ir.Tuple([]))], None), mad()]

    assert _modes(_program(mode, body), mode) == [False]


@pytest.mark.parametrize("mode", ["direction", "hf32"])
def test_pto_nested_continue_is_consumed_by_its_own_loop(mode):
    def body(flag, set_mode, mad):
        outer = ir.Var("outer", "int32")
        inner = ir.Var("inner", "int32")
        nested = ir.For(inner, 0, 2, 1, ir.ForKind.DEFAULT, [set_mode(0), ir.Continue()])
        loop = ir.For(outer, 0, 2, 1, ir.ForKind.DEFAULT, [nested, set_mode(1)])
        return [loop, mad()]

    assert _modes(_program(mode, body), mode) == [True]


@pytest.mark.parametrize("mode", ["direction", "hf32"])
def test_pto_zero_trip_loop_preserves_incoming_mode(mode):
    def body(flag, set_mode, mad):
        index = ir.Var("index", "int32")
        return [ir.For(index, 0, 4, -1, ir.ForKind.DEFAULT, [set_mode(1), mad()]), mad()]

    assert _modes(_program(mode, body), mode)[-1] is False


@pytest.mark.parametrize("mode", ["direction", "hf32"])
def test_pto_constant_dead_branches_still_have_codegen_state(mode):
    def body(flag, set_mode, mad):
        return [ir.If(ir.const(False), [set_mode(1), mad()], None), mad()]

    assert _modes(_program(mode, body), mode)[-1] is False


@pytest.mark.parametrize("mode", ["direction", "hf32"])
def test_pto_modes_correlate_immutable_guards(mode):
    def body(flag, set_mode, mad):
        return [
            ir.If(ir.prim.And(flag > 0, flag < 10), set_mode(1), None),
            ir.If(flag > 0, ir.If(flag < 10, [mad(), set_mode(0)], None), None),
            mad(),
        ]

    assert _modes(_program(mode, body), mode) == [True, False]


@pytest.mark.parametrize("mode", ["direction", "hf32"])
def test_pto_ambiguous_reaching_mode_is_rejected(mode):
    def body(flag, set_mode, mad):
        return [ir.If(flag > 0, set_mode(1), None), mad()]

    with pytest.raises(Exception, match="control.flow"):
        _source(_program(mode, body))
