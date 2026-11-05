"""CPU grid proofs traverse lexical scopes without erasing their lifetimes."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_ffi.serialization import to_json_graph_str
from tvm_tirx import ir, tirx

import tilelang.cpu.transform
from tilelang.testing.packed import compile_packed_function


def _walk(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def _tensor(name, extent=16, scope="global"):
    return ir.Var(name, tirx.TensorType((extent,), "int32", scope=scope, layout="default"))


def _allocate(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _lexical(body, explicit):
    attrs = {"explicit_alloc_scope": True} if explicit else {}
    return ir.RegionStmt("tl.lexical_scope", [], [], attrs, body)


def _grid(bx, body):
    return ir.For(bx, 0, 16, 1, ir.ForKind.DEFAULT, body, annotations={"tl.cpu_grid_dim": 0})


def _materialize(out, body, target):
    function = tirx.Function([out], body).with_attr(
        {"global_symbol": "main", "target": tvm_tirx.target.Target(target), "tirx.noalias": True}
    )
    module = ir.IRModule.from_expr(function)
    assert tirx.analysis.check_well_formed(module)
    module = tilelang.cpu.transform.MaterializeCPUParallelGrid()(module)
    assert tirx.analysis.check_well_formed(module)
    assert all("tl.cpu_grid_dim" not in loop.annotations for loop in _walk(module, ir.For))
    return module


def _parallel(module):
    return [loop for loop in _walk(module, ir.For) if loop.kind == ir.ForKind.PARALLEL]


def _execute(module, expected):
    output = tvm_tirx.runtime.tensor(np.zeros(16, dtype="int32"))
    # Numerical checks use the TileLang C boundary. Parallel eligibility and
    # allocation ownership are checked on the IR independently above; this
    # scalar C host oracle does not claim LLVM or parallel-runtime coverage.
    compile_packed_function(module)(output)
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("scope_outside_grid", [False, True])
def test_once_executed_lexical_scope_keeps_parallel_grid(target, explicit, scope_outside_grid):
    out, bx = _tensor("out"), ir.Var("bx", "int32")
    store = ir.TensorStore(out, [bx], bx + 7)
    body = _lexical(_grid(bx, store), explicit) if scope_outside_grid else _grid(bx, _lexical(store, explicit))
    module = _materialize(out, body, target)
    assert len(_parallel(module)) == 1
    regions = _walk(module, ir.RegionStmt)
    assert len(regions) == 1
    assert regions[0].op.name == "tl.lexical_scope"
    assert dict(regions[0].attrs) == ({"explicit_alloc_scope": True} if explicit else {})
    if scope_outside_grid:
        assert len(_walk(regions[0].body, ir.For)) == 1
    else:
        assert len(_walk(_parallel(module)[0].body, ir.RegionStmt)) == 1
    if target == "c":
        _execute(module, np.arange(16, dtype="int32") + 7)


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("explicit", [False, True])
def test_private_allocations_remain_inside_nested_lexical_boundaries(target, explicit):
    out, bx = _tensor("out"), ir.Var("bx", "int32")
    outer = _tensor("scratch", 1, "local")
    inner = _tensor("scratch", 1, "local")
    inside = _lexical(
        [_allocate(inner), ir.TensorStore(inner, [0], bx + 2), ir.TensorStore(out, [bx], outer[0] + inner[0])],
        explicit,
    )
    body = _grid(bx, _lexical([_allocate(outer), ir.TensorStore(outer, [0], bx + 1), inside], explicit))
    module = _materialize(out, body, target)
    (parallel,) = _parallel(module)
    (outer_region,) = parallel.body.seq
    assert isinstance(outer_region, ir.RegionStmt)
    outer_bind, _, inner_region = outer_region.body.seq
    assert isinstance(outer_bind, ir.Bind) and outer_bind.var.same_as(outer)
    assert isinstance(inner_region, ir.RegionStmt)
    inner_bind = inner_region.body.seq[0]
    assert isinstance(inner_bind, ir.Bind) and inner_bind.var.same_as(inner)
    assert not inner_bind.var.same_as(outer_bind.var)
    assert len(_walk(module, ir.Bind)) == 2
    if target == "c":
        _execute(module, np.arange(16, dtype="int32") * 2 + 3)


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("reset", ["full", "conditional", "empty", "partial", "read_before_complete"])
def test_lexical_scope_preserves_reset_and_previous_iteration_analysis(target, reset):
    out, bx = _tensor("out"), ir.Var("bx", "int32")
    state = _tensor("state", 2, "local")

    def fill(value):
        j = ir.Var("j", "int32")
        return ir.For(j, 0, 2, 1, ir.ForKind.DEFAULT, ir.TensorStore(state, [j], value))

    initialize_first = ir.If(bx == 0, fill(ir.const(0, "int32")), None)
    reset_body = fill(ir.const(0, "int32"))
    if reset == "conditional":
        reset_body = ir.If(bx == 0, reset_body, None)
    elif reset == "empty":
        reset_body = ir.For(ir.Var("k", "int32"), 0, 0, 1, ir.ForKind.DEFAULT, reset_body)
    elif reset == "partial":
        reset_body = ir.TensorStore(state, [0], ir.const(0, "int32"))
    elif reset == "read_before_complete":
        j = ir.Var("j", "int32")
        reset_body = ir.For(
            j,
            0,
            2,
            1,
            ir.ForKind.DEFAULT,
            [ir.TensorStore(state, [j], ir.const(0, "int32")), ir.TensorStore(out, [bx], state[(j + 1) % 2])],
        )
    scope = _lexical(
        [initialize_first, reset_body, ir.TensorStore(state, [1], state[1] + 1), ir.TensorStore(out, [bx], state[1])],
        True,
    )
    module = _materialize(out, [_allocate(state), _grid(bx, scope)], target)
    assert bool(_parallel(module)) == (reset == "full")
    (region,) = _walk(module, ir.RegionStmt)
    assert region.op.name == "tl.lexical_scope" and region.attrs["explicit_alloc_scope"]
    if reset == "full":
        assert any(binding.var.same_as(state) for binding in _walk(_parallel(module)[0].body, ir.Bind))
    if target == "c":
        expected = np.ones(16, dtype="int32") if reset in ("full", "read_before_complete") else np.arange(1, 17, dtype="int32")
        _execute(module, expected)


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_lexical_scope_does_not_hide_cross_iteration_access(target):
    out, bx = _tensor("out"), ir.Var("bx", "int32")
    body = _grid(bx, _lexical(ir.TensorStore(out, [bx], out[(bx - 1) % 16] + 1), True))
    module = _materialize(out, body, target)
    assert not _parallel(module)
    if target == "c":
        _execute(module, np.arange(1, 17, dtype="int32"))


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_context_changing_region_is_not_lexically_transparent(target, capfd):
    out, bx = _tensor("out"), ir.Var("bx", "int32")
    launch = ir.RegionStmt("tirx.parallel_launch", [], [], {}, ir.TensorStore(out, [bx], bx))
    module = _materialize(out, _grid(bx, _lexical(launch, True)), target)
    assert not _parallel(module)
    assert "effects that cannot be proven safe" in capfd.readouterr().err


def test_repeated_materialization_preserves_scopes_and_source():
    out, bx = _tensor("out"), ir.Var("bx", "int32")
    module = _materialize(out, _grid(bx, _lexical(ir.TensorStore(out, [bx], bx), True)), "llvm")
    original = to_json_graph_str(module)
    repeated = tilelang.cpu.transform.MaterializeCPUParallelGrid()(module)
    assert tvm_ffi.structural_equal(module, repeated)
    assert to_json_graph_str(module) == original
    assert len(_walk(repeated, ir.RegionStmt)) == 1
