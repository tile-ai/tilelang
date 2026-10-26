import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

from tilelang._compat import stmt_functor
from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_operator_helpers")


def collect_proofs(body):
    return [int(proof) for proof in tvm_ffi.get_global_func("tl.testing.CollectLoopProofs")(body)]


@pytest.mark.parametrize("step", [1, 3, -1, -3])
def test_visitor_tracks_signed_loop_bounds(step):
    index = ir.Var("index", "int32")
    begin = ir.Var("begin", "int32")
    end = ir.Var("end", "int32")
    if step > 0:
        predicates = [index >= begin, index < end, begin < end]
    else:
        predicates = [index <= begin, index > end, begin > end]
    loop = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, [ir.Evaluate(predicate) for predicate in predicates])
    assert collect_proofs(ir.SeqStmt([loop, ir.Evaluate(predicates[-1])])) == [1, 1, 1, 0]


def test_visitor_does_not_guess_symbolic_step_sign():
    index = ir.Var("index", "int32")
    begin = ir.Var("begin", "int32")
    end = ir.Var("end", "int32")
    step = ir.Var("step", "int32")
    loop = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, [ir.Evaluate(begin < end), ir.Evaluate(begin > end)])
    assert collect_proofs(loop) == [0, 0]


def test_visitor_does_not_assume_mutable_bounds():
    index = ir.Var("index", "int32")
    end = ir.Call("tirx.call_extern", [ir.StringImm("get_end")], ty="int32")
    loop = ir.For(index, 0, end, 1, ir.ForKind.DEFAULT, ir.Evaluate(end > 0))
    assert collect_proofs(loop) == [0]


def test_sequential_assume_does_not_escape_its_branch():
    value = ir.Var("value", "int32")
    condition = ir.Var("condition", "bool")
    assumption = ir.Evaluate(ir.Call("prim.assume", [value >= 0]))

    def marker():
        return ir.If(value >= 0, ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("nonnegative")], ty="int32")), None)

    function = tirx.Function([value, condition], [ir.If(condition, [assumption, marker()], None), marker()])
    result = _tirx_passes.StmtSimplify()(ir.IRModule({"main": function}))["main"]
    branches = []
    stmt_functor.post_order_visit(result, lambda node: branches.append(node) if isinstance(node, ir.If) else None)
    assert len(branches) == 2
    assert sym.Analyzer().can_prove(branches[-1].condition == (value >= 0))
    assert tirx.analysis.check_well_formed(result)
