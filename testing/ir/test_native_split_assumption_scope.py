"""Host facts keep their definition order and scope through device outlining."""

import pytest
import torch
import tvm_tirx as tvm
from tvm_tirx import ir, tirx

import tilelang
import tilelang.cpu.language as C
from tilelang.testing.ir import collect_calls, collect_nodes
from tilelang.testing.packed import compile_packed_function


HOST = tvm.target.Target("c")
CUDA = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})


def _assume(condition):
    return ir.Evaluate(ir.Call("prim.assume", [condition]))


def _launch(value, *, prefix=()):
    # An opaque consumer gives each outlined kernel one real scalar capture.
    # No device compilation or execution is needed for the dominance oracle.
    use = ir.Evaluate(tirx.call_extern("assumption_scope_consumer", value, ty="void"))
    return ir.RegionStmt("tirx.device_scope", [], [], {"target": CUDA}, [*prefix, use])


def _outline(parameters, body, *, round_trip=False):
    function = tirx.Function(parameters, body, ir.PrimType("void")).with_attr(
        {"global_symbol": "main", "target": tvm.target.Target(CUDA, host=HOST)}
    )
    module = ir.IRModule({"main": function})
    assert tirx.analysis.check_well_formed(module)
    original = ir.save_json(module)
    if round_trip:
        module = ir.load_json(original)
    injected = tilelang.transform.InjectAssumes()(module)
    result = tilelang.transform.OutlineDeviceRegions()(injected)
    assert ir.save_json(module) == original
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result["main"])
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    # device_scope without an explicit launch configuration outlines a direct
    # GlobalVar call; device_entry with launch fields uses tirx.launch_kernel.
    launches = _device_calls(result["main"])
    devices = [result[launch.op] for launch in launches]
    for device in devices:
        assert not tirx.analysis.undefined_vars(device)
        assert len(device.params) == 1
    return result, devices


def _facts(function):
    return [call.args[0] for call in collect_calls(function, op="prim.assume")]


def _device_calls(function):
    return [call for call in collect_nodes(function, ir.Call) if isinstance(call.op, ir.GlobalVar)]


@pytest.mark.parametrize("position,counts", [(0, [1, 1]), (1, [0, 1]), (2, [0, 0])])
@pytest.mark.parametrize("round_trip", [False, True])
def test_only_earlier_dominating_host_facts_reach_each_launch(position, counts, round_trip):
    n = ir.Var("n", "int32")
    statements = [_launch(n), _launch(n)]
    statements.insert(position, _assume(n > 7))
    _, devices = _outline([n], statements, round_trip=round_trip)
    assert [len(_facts(device)) for device in devices] == counts
    for device in devices:
        for condition in _facts(device):
            ir.assert_structural_equal(condition, device.params[0] > 7)


@pytest.mark.parametrize("scope", ["if", "else", "for", "while", "lexical"])
def test_branch_and_nested_scope_facts_do_not_leak_to_siblings(scope):
    n, flag = ir.Var("n", "int32"), ir.Var("flag", "bool")
    inside = [_assume(n > 7), _launch(n)]
    counts = [1, 0]
    if scope == "if":
        nested = ir.If(flag, inside, [_launch(n)])
        counts = [1, 0, 0]
    elif scope == "else":
        nested = ir.If(flag, [_launch(n)], inside)
        counts = [0, 1, 0]
    elif scope == "for":
        nested = ir.For(ir.Var("i", "int32"), 0, 2, 1, ir.ForKind.DEFAULT, inside)
    elif scope == "while":
        nested = ir.While(flag, inside)
    else:
        nested = ir.RegionStmt("tl.lexical_scope", [], [], {}, inside)
    _, devices = _outline([n, flag], [nested, _launch(n)])
    assert [len(_facts(device)) for device in devices] == counts


@pytest.mark.parametrize("same_name", [False, True])
def test_host_fact_never_adds_an_unused_capture_or_matches_by_name(same_name):
    checked = ir.Var("n", "int32")
    captured = ir.Var("n" if same_name else "other", "int32")
    module, devices = _outline([checked, captured], [_assume(checked > 7), _launch(captured), _launch(checked)])
    assert [len(_facts(device)) for device in devices] == [0, 1]
    launches = _device_calls(module["main"])
    assert len(launches[0].args) == 1
    assert launches[0].args[0].same_as(module["main"].params[1])
    assert launches[1].args[0].same_as(module["main"].params[0])
    ir.assert_structural_equal(_facts(devices[1])[0], devices[1].params[0] > 7)


def _events(statement):
    if isinstance(statement, ir.SeqStmt):
        return [event for child in statement.seq for event in _events(child)]
    if isinstance(statement, ir.RegionStmt) and statement.op.name == "tl.lexical_scope":
        return _events(statement.body)
    if isinstance(statement, ir.AssertStmt):
        return ["check"]
    if isinstance(statement, ir.Evaluate) and isinstance(statement.value, ir.Call):
        if isinstance(statement.value.op, ir.GlobalVar):
            return ["launch"]
        return {"prim.assume": ["fact"], "tirx.launch_kernel": ["launch"]}.get(getattr(statement.value.op, "name", ""), [])
    return []


def test_lifted_device_check_precedes_its_optimizer_fact_and_launch():
    n = ir.Var("n", "int32")
    module, devices = _outline([n], [_launch(n, prefix=[_assume(n % 4 == 0)])])
    assert _events(module["main"].body) == ["check", "fact", "launch"]
    assert len(_facts(devices[0])) == 1
    assert not collect_nodes(devices[0], ir.AssertStmt)


@pytest.mark.parametrize(
    "begin,end,step,dtype,check_count",
    [
        (0, 1, 1, "int32", 1),
        (3, 4, 1, "int32", 1),
        (-1, 0, 1, "int32", 1),
        (0, 3, 4, "int32", 1),
        (0, 2, 1, "int32", 0),
        (1, 1, 1, "int32", 0),
        (1, 0, 1, "int32", 0),
        (0, 1, 0, "int32", 0),
        (0, 1, -1, "int32", 0),
        (120, 121, 127, "int8", 0),
        (250, 251, 10, "uint8", 0),
    ],
)
def test_only_unconditional_nonwrapping_single_iteration_allows_check_lifting(begin, end, step, dtype, check_count):
    n = ir.Var("n", "int32")
    loop = ir.For(
        ir.Var("i", dtype),
        ir.const(begin, dtype),
        ir.const(end, dtype),
        ir.const(step, dtype),
        ir.ForKind.DEFAULT,
        [_assume(n >= 3)],
    )
    module, _ = _outline([n], [_launch(n, prefix=[loop])])
    assert len(collect_nodes(module["main"], ir.AssertStmt)) == check_count


@pytest.mark.parametrize("control", ["break", "continue", "return"])
def test_single_iteration_with_early_control_flow_does_not_lift_unreachable_check(control):
    n, i = ir.Var("n", "int32"), ir.Var("i", "int32")
    stop = {"break": ir.Break, "continue": ir.Continue, "return": lambda: ir.Return(ir.Tuple([]))}[control]()
    loop = ir.For(i, 0, 1, 1, ir.ForKind.DEFAULT, [stop, _assume(n >= 3)])
    module, _ = _outline([n], [_launch(n, prefix=[loop])])
    assert not collect_nodes(module["main"], ir.AssertStmt)


def test_single_iteration_induction_value_is_substituted_only_in_the_host_check():
    n, i = ir.Var("n", "int32"), ir.Var("i", "int32")
    loop = ir.For(i, 4, 5, 1, ir.ForKind.DEFAULT, [_assume(n >= i + 3)])
    module, devices = _outline([n], [_launch(n, prefix=[loop])])
    (check,) = collect_nodes(module["main"], ir.AssertStmt)
    assert all(variable.same_as(module["main"].params[0]) for variable in collect_nodes(check.condition, ir.Var))
    expected = module["main"].params[0] >= 7
    analyzer = tvm.sym.Analyzer()
    ir.assert_structural_equal(analyzer.simplify(check.condition), analyzer.simplify(expected))
    (retained_loop,) = collect_nodes(devices[0], ir.For)
    assert any(variable.same_as(retained_loop.var) for variable in collect_nodes(_facts(devices[0])[0], ir.Var))


@pytest.mark.parametrize("simplification_rounds", [0, 1, 3])
def test_repeated_simplification_cannot_prove_a_runtime_check_from_its_own_fact(simplification_rounds):
    n = ir.Var("n", "int32")
    function = tirx.Function([n], [_assume(n >= 3), ir.Return(n + 1)], ir.PrimType("int32")).with_attr(
        {"global_symbol": "main", "target": tvm.target.Target(HOST, host=HOST)}
    )
    module = tilelang.transform.InjectAssumes()(ir.IRModule({"main": function}))
    module = tilelang.transform.OutlineDeviceRegions()(module)
    for _ in range(simplification_rounds):
        module = tilelang.transform.Simplify()(module)
    assert len(collect_nodes(module["main"], ir.AssertStmt)) == 1
    compiled = compile_packed_function(module, target=HOST)
    for value in [3, 7, 257]:
        assert compiled(value) == value + 1
    for value in [-1, 0, 2]:
        with pytest.raises(RuntimeError, match="Assume:"):
            compiled(value)


def test_public_cpu_dsl_checks_host_evaluable_device_assume_before_writing_output():
    @C.prim_func
    def main(output: C.Tensor((1,), "int32"), n: C.int32):
        with C.Kernel(1):
            C.assume(n >= 3)
            output[0] = n + 1

    compiled = tilelang.compile(main, target=HOST, target_host=HOST, execution_backend="tvm_ffi")
    output = torch.full((1,), -17, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="Assume:"):
        compiled(output, 2)
    assert output.item() == -17
    compiled(output, 3)
    assert output.item() == 4
