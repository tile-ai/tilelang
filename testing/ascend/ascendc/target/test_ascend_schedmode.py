"""Global core synchronization selects batch scheduling for each entry."""

import re

import pytest
from tilelang import tvm
from tvm import tirx


def _flag(operation, mode, pipe, flag_id=0):
    return tirx.Evaluate(tirx.call_intrin("void", tvm.ir.Op.get(f"tl.ascend_cross_core_{operation}_flag"), mode, pipe, flag_id))


def _barrier(core, mode=0):
    pipe = "PIPE_FIX" if core == "CUBE" else "PIPE_MTE3"
    body = tirx.SeqStmt([_flag("set", mode, pipe), _flag("wait", mode, pipe)])
    return tirx.SBlock([], [], [], core, body)


def _function(name, body):
    return tirx.PrimFunc([], body).with_attr({"global_symbol": name, "calling_conv": 2})


def _source(functions):
    build = tvm.ffi.get_global_func("target.build.tilelang_ascend_without_compile")
    return build(tvm.IRModule(functions), tvm.target.Target("ascend")).inspect_source()


@pytest.mark.parametrize("core", ["CUBE", "VECTOR"])
def test_global_core_sync_enables_batch_scheduling(core):
    source = _source({"main": _function("main", _barrier(core))})
    assert re.search(r"__schedmode__\(1\).*__global__.*void main\(", source)


def test_combined_global_core_sync_enables_batch_scheduling():
    # AIV -> local AIC, all AICs, then local AIC -> AIV.
    cube = tirx.SBlock(
        [],
        [],
        [],
        "CUBE",
        tirx.SeqStmt(
            [
                _flag("wait", 2, "PIPE_FIX", 1),
                _flag("set", 0, "PIPE_FIX", 0),
                _flag("wait", 0, "PIPE_FIX", 0),
                _flag("set", 2, "PIPE_FIX", 2),
            ]
        ),
    )
    vector = tirx.SBlock(
        [],
        [],
        [],
        "VECTOR",
        tirx.SeqStmt([_flag("set", 2, "PIPE_MTE3", 1), _flag("wait", 2, "PIPE_MTE3", 2)]),
    )
    source = _source({"main": _function("main", tirx.SeqStmt([cube, vector]))})
    assert "__schedmode__(1) __global__ __mix__(1, 2)" in source


@pytest.mark.parametrize("mode", [1, 2, 4])
def test_local_core_sync_keeps_default_scheduling(mode):
    source = _source({"main": _function("main", _barrier("VECTOR", mode))})
    assert "__schedmode__" not in source


def test_global_core_sync_flag_does_not_leak_between_entries():
    functions = {
        "sync0": _function("sync0", _barrier("CUBE")),
        "plain0": _function("plain0", tirx.Evaluate(0)),
        "sync1": _function("sync1", _barrier("VECTOR")),
        "plain1": _function("plain1", tirx.Evaluate(0)),
    }
    source = _source(functions)
    for name in functions:
        declaration = next(line for line in source.splitlines() if f"void {name}(" in line)
        assert ("__schedmode__(1)" in declaration) == name.startswith("sync")
