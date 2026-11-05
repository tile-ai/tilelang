"""Synchronous copy injection must preserve visibility on every executed path.

The oracle interprets commit groups and wait counts, including skipped branches
and loop iterations. Structural well-formedness alone cannot detect these bugs.
"""

import json

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx


@pytest.fixture(params=["PTX", "ROCm"])
def backend(request):
    return request.param


def integer(value):
    return ir.IntImm("int32", value)


def intrinsic(name, *args):
    return ir.Evaluate(ir.Call("tl.ptx_" + name, list(args), ty=ir.PrimType("void")))


def wait(count=0):
    return intrinsic("wait_group", integer(count))


def commit():
    return intrinsic("commit_group")


class Program:
    def __init__(self):
        self.source = ir.Var("A", tirx.TensorType([8], "float32", scope="global", layout="default"))
        self.shared = ir.Var("S", tirx.TensorType([8], "float32", scope="shared", layout="default"))
        self.output = ir.Var("B", tirx.TensorType([8], "float32", scope="global", layout="default"))
        self.predicate = ir.Var("condition", "bool")
        self.extent = ir.Var("extent", "int32")
        self.lane = ir.Var("i", "int32")

    def copy(self, index=0):
        index = integer(index) if isinstance(index, int) else index
        return ir.TensorStore(self.shared, [index], ir.TensorLoad(self.source, [index]))

    def consume(self, index=0):
        index = integer(index) if isinstance(index, int) else index
        return ir.TensorStore(self.output, [index], ir.TensorLoad(self.shared, [index]))

    def loop(self, end, body, begin=0):
        end = integer(end) if isinstance(end, int) else end
        return ir.For(self.lane, integer(begin), end, integer(1), ir.ForKind.DEFAULT, body)

    def function(self, body):
        return tirx.Function([self.source, self.shared, self.output, self.predicate, self.extent], body, ir.PrimType("void"))


def check_paths(backend, program, body, tmp_path, environments=((False, 0), (True, 0))):
    original = program.function(body)
    assert tirx.analysis.check_well_formed(original)
    serialized = ir.save_json(original)
    result = tvm_ffi.get_global_func("tl.testing.Inject" + backend + "AsyncCopy")(original.body, False)
    assert result["injected"]
    output = program.function(result["statement"])
    assert tirx.analysis.check_well_formed(output)
    assert ir.save_json(original) == serialized
    (tmp_path / "input.json").write_text(serialized)
    (tmp_path / "output.json").write_text(ir.save_json(output))
    (tmp_path / "output.py").write_text(output.script())
    traces = []
    for predicate, extent in environments:
        trace = visibility_trace(program, output.body, predicate, extent)
        traces.append(trace)
    (tmp_path / "traces.json").write_text(json.dumps(traces, indent=2))
    for trace in traces:
        assert not trace["violations"], trace
    return traces


def visibility_trace(program, body, predicate, extent):
    """Model the oldest groups a PTX wait must retire, rather than its presence."""
    events, violations = [], []
    uncommitted, groups = [], []
    values = {program.predicate: predicate, program.extent: extent}

    def scalar(expr):
        if isinstance(expr, ir.IntImm):
            return int(expr.value)
        if isinstance(expr, ir.Var):
            return values[expr]
        raise AssertionError(type(expr))

    def visit(stmt):
        if isinstance(stmt, ir.SeqStmt):
            for child in stmt.seq:
                visit(child)
        elif isinstance(stmt, ir.If):
            if scalar(stmt.condition):
                visit(stmt.then_case)
            elif stmt.else_case is not None:
                visit(stmt.else_case)
        elif isinstance(stmt, ir.For):
            for lane in range(scalar(stmt.begin), scalar(stmt.end), scalar(stmt.step)):
                values[stmt.var] = lane
                visit(stmt.body)
        elif isinstance(stmt, ir.Evaluate):
            if isinstance(stmt.value, ir.IntImm):
                return
            call = stmt.value
            assert isinstance(call, ir.Call)
            name = call.op.name
            if name in ("tl.ptx_cp_async", "tl.ptx_cp_async_bytes"):
                pointer = call.args[0]
                assert pointer.op.name == "tl.access_ptr"
                load = pointer.args[0]
                assert isinstance(load, ir.TensorLoad) and load.source.same_as(program.shared)
                first = scalar(load.indices[0])
                count = scalar(call.args[2])
                uncommitted.extend(range(first, first + count))
                events.append("copy")
            elif name == "tl.ptx_commit_group":
                groups.append(list(uncommitted))
                uncommitted.clear()
                events.append("commit")
            elif name == "tl.ptx_wait_group":
                count = scalar(call.args[0])
                if len(groups) > count:
                    del groups[: len(groups) - count]
                events.append("wait(" + str(count) + ")")
            else:
                raise AssertionError(name)
        elif isinstance(stmt, ir.TensorStore):
            assert stmt.dest.same_as(program.output)
            assert isinstance(stmt.value, ir.TensorLoad) and stmt.value.source.same_as(program.shared)
            events.append("consume(" + str(scalar(stmt.value.indices[0])) + ")")
            index = scalar(stmt.value.indices[0])
            if index in uncommitted or any(index in group for group in groups):
                violations.append({"events": list(events), "uncommitted": list(uncommitted), "groups": list(groups)})
        else:
            raise AssertionError(type(stmt))

    visit(body)
    return {"predicate": predicate, "extent": extent, "events": events, "violations": violations}


@pytest.mark.parametrize("branches", ["then", "else", "both"])
def test_conditional_wait_must_cover_every_path(backend, branches, tmp_path):
    p = Program()
    then = [wait()] if branches in ("then", "both") else [ir.Evaluate(integer(0))]
    otherwise = [wait()] if branches in ("else", "both") else None
    traces = check_paths(backend, p, [p.copy(), ir.If(p.predicate, then, otherwise), p.consume()], tmp_path)
    if branches == "both":
        for trace in traces:
            assert trace["events"].count("wait(0)") == 1
            assert trace["events"].count("commit") == 1


@pytest.mark.parametrize("branches", ["then", "else", "both"])
def test_conditional_commit_must_cover_every_path(backend, branches, tmp_path):
    p = Program()
    then = [commit()] if branches in ("then", "both") else [ir.Evaluate(integer(0))]
    otherwise = [commit()] if branches in ("else", "both") else None
    traces = check_paths(backend, p, [p.copy(), ir.If(p.predicate, then, otherwise), wait(), p.consume()], tmp_path)
    if branches == "both":
        for trace in traces:
            assert trace["events"].count("commit") == 1
            assert trace["events"].count("wait(0)") == 1


@pytest.mark.parametrize("loop_end", [0, 1, "symbolic"])
def test_loop_wait_requires_a_guaranteed_iteration(backend, loop_end, tmp_path):
    p = Program()
    end = p.extent if loop_end == "symbolic" else loop_end
    traces = check_paths(backend, p, [p.copy(), p.loop(end, [wait()]), p.consume()], tmp_path, environments=[(False, n) for n in (0, 1, 2)])
    if loop_end == 1:
        for trace in traces:
            assert trace["events"].count("wait(0)") == 1


@pytest.mark.parametrize("loop_end", [0, 1, "symbolic"])
def test_loop_commit_requires_a_guaranteed_iteration(backend, loop_end, tmp_path):
    p = Program()
    end = p.extent if loop_end == "symbolic" else loop_end
    traces = check_paths(
        backend, p, [p.copy(), p.loop(end, [commit()]), wait(), p.consume()], tmp_path, environments=[(False, n) for n in (0, 1, 2)]
    )
    if loop_end == 1:
        for trace in traces:
            assert trace["events"].count("commit") == 1


@pytest.mark.parametrize("count", [0, 1, 2])
def test_wait_count_must_complete_the_latest_copy(backend, count, tmp_path):
    p = Program()
    traces = check_paths(backend, p, [p.copy(), wait(count), p.consume()], tmp_path)
    for trace in traces:
        assert trace["events"].count("wait(0)") == 1


@pytest.mark.parametrize("trailing_commit", [False, True])
def test_wait_before_a_new_branch_copy_does_not_complete_that_copy(backend, trailing_commit, tmp_path):
    p = Program()
    branch = [wait(), p.copy(1)] + ([commit()] if trailing_commit else [])
    traces = check_paths(backend, p, [p.copy(), ir.If(p.predicate, branch, branch), p.consume(), p.consume(1)], tmp_path)
    for trace in traces:
        assert trace["events"].count("wait(0)") == 2


@pytest.mark.parametrize("trailing_commit", [False, True])
def test_commit_before_a_new_branch_copy_does_not_commit_that_copy(backend, trailing_commit, tmp_path):
    p = Program()
    branch = [commit(), p.copy(1)] + ([commit()] if trailing_commit else [])
    check_paths(backend, p, [p.copy(), ir.If(p.predicate, branch, branch), wait(), p.consume(), p.consume(1)], tmp_path)


@pytest.mark.parametrize("loop_end", [1, 2])
def test_wait_before_a_new_loop_copy_does_not_complete_that_copy(backend, loop_end, tmp_path):
    p = Program()
    check_paths(backend, p, [p.copy(2), p.loop(loop_end, [wait(), p.copy(p.lane)]), p.consume()], tmp_path)


def test_loop_backedge_makes_trailing_copies_visible(backend, tmp_path):
    p = Program()
    check_paths(backend, p, [p.copy(), p.loop(2, [p.consume(), p.copy()]), p.consume()], tmp_path)


def test_pure_copy_loop_keeps_one_hoisted_commit_and_wait(backend, tmp_path):
    p = Program()
    loop = p.loop(4, [p.copy(p.lane), ir.Evaluate(integer(0))])
    traces = check_paths(backend, p, [loop, *(p.consume(i) for i in range(4))], tmp_path)
    for trace in traces:
        assert trace["events"] == ["copy"] * 4 + ["commit", "wait(0)"] + ["consume(" + str(i) + ")" for i in range(4)]


def test_explicit_async_keeps_caller_commit_and_wait_ownership(backend):
    p = Program()
    body = ir.SeqStmt([p.copy(), ir.If(p.predicate, [wait()], None)])
    result = tvm_ffi.get_global_func("tl.testing.Inject" + backend + "AsyncCopy")(body, True)
    assert result["injected"]
    traces = [visibility_trace(p, result["statement"], value, 0)["events"] for value in (False, True)]
    assert traces == [["copy"], ["copy", "wait(0)"]]
