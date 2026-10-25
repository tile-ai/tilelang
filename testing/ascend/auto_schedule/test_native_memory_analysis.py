import pytest
import tvm_ffi
from tvm_tirx import ir, sym

import tilelang.ascend.language as T
from tilelang._compat.tensor import TensorView


def _analyze(body, physical=False):
    if isinstance(body, list):
        body = ir.SeqStmt(body)
    return tvm_ffi.get_global_func("tl.ascend.AnalyzeMemoryAccesses")(body, physical, {})


def _has(regions, tensor):
    return any(region.source.same_as(tensor.tensor) for region in regions)


def _region(regions, tensor):
    return next(region for region in regions if region.source.same_as(tensor.tensor))


def _bounds(region):
    analyzer = sym.Analyzer()
    return [(int(analyzer.simplify(axis.begin)), int(analyzer.simplify(axis.extent))) for axis in region.region]


def test_native_copy_access_roles_do_not_read_destination():
    source = TensorView.create("source", [64], "float32")
    target = TensorView.create("target", [64], "float32", scope="shared.dyn")
    result = _analyze(ir.Evaluate(T.copy(source.tensor, target.tensor)), physical=True)
    assert _has(result["reads"], source)
    assert not _has(result["reads"], target)
    assert _has(result["writes"], target)
    assert _has(result["must_writes"], target)
    assert _bounds(_region(result["writes"], target)) == [(0, 64)]


def test_fill_preserves_scalar_value_load_dependencies():
    source = TensorView.create("source", [1], "float32")
    target = TensorView.create("target", [32], "float32", scope="shared.dyn")
    result = _analyze(ir.Evaluate(T.fill(target.tensor, source.load(0))), physical=True)
    assert _has(result["reads"], source)
    assert not _has(result["reads"], target)
    assert _has(result["must_writes"], target)


@pytest.mark.parametrize("clear", [True, False])
def test_reduction_accumulation_reads_are_owned_by_native_operator(clear):
    source = TensorView.create("source", [4, 16], "float32", scope="shared.dyn")
    target = TensorView.create("target", [4], "float32", scope="shared.dyn")
    source_region = source.region([ir.Range(0, 4), ir.Range(0, 16)])
    target_region = target.region([ir.Range(0, 4)])
    operation = ir.Call("tl.tileop.reduce", [source_region, target_region, "sum", 1, clear])
    result = _analyze(ir.Evaluate(operation), physical=True)
    assert _has(result["reads"], source)
    assert _has(result["reads"], target) == (not clear)
    assert _has(result["writes"], target)


@pytest.mark.parametrize(
    "begin,end,step,expected,dense",
    [(3, 11, 1, (3, 8), True), (11, 3, -1, (4, 8), True), (3, 11, 2, (3, 7), False), (11, 3, -2, (5, 7), False)],
)
def test_signed_loop_domains_and_must_write_coverage(begin, end, step, expected, dense):
    tensor = TensorView.create("tensor", [16], "float32", scope="shared.dyn")
    index = ir.Var("index", "int32")
    loop = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, tensor.store(index, ir.const(1.0)))
    result = _analyze(loop, physical=True)
    assert _bounds(_region(result["writes"], tensor)) == [expected]
    assert bool(result["must_writes"]) == dense


def test_correlated_diagonal_write_is_not_a_full_definition():
    tensor = TensorView.create("tensor", [8, 8], "float32", scope="shared.dyn")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, tensor.store([index, index], ir.const(1.0)))
    result = _analyze(loop, physical=True)
    assert _bounds(_region(result["writes"], tensor)) == [(0, 8), (0, 8)]
    assert not result["must_writes"]


def test_bind_snapshots_resolve_indices_without_losing_input_reads():
    tensor = TensorView.create("tensor", [32], "float32", scope="shared.dyn")
    index = ir.Var("index", "int32")
    snapshot = ir.Var("snapshot", "int32")
    loop = ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, [ir.Bind(snapshot, index + 4), tensor.store(snapshot, ir.const(1.0))])
    result = _analyze(loop, physical=True)
    assert _bounds(_region(result["writes"], tensor)) == [(4, 8)]
    assert _has(result["must_writes"], tensor)
    assert any(variable.same_as(snapshot) for variable in result["write_vars"])


def test_native_tensor_definitions_are_not_content_accesses():
    extent = ir.Var("extent", "int32")
    storage = TensorView.create("storage", [extent], "float32", scope="shared.dyn")
    view = TensorView.create("view", [extent], "float32", scope="shared.dyn")
    result = _analyze(
        [
            ir.Bind(storage.tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[storage.tensor.ty])),
            ir.Bind(view.tensor, ir.Call("tirx.decl_tensor", [storage.tensor.data], ty_args=[view.tensor.ty])),
        ],
        physical=True,
    )
    assert not result["reads"]
    assert not result["writes"]
    assert not result["opaque"]
    assert any(variable.same_as(extent) for variable in result["read_vars"])
    assert any(variable.same_as(storage.tensor) for variable in result["read_vars"])
    assert all(any(variable.same_as(tensor.tensor) for variable in result["write_vars"]) for tensor in (storage, view))


@pytest.mark.parametrize("escape", ["tensor", "pointer", "integer"])
def test_native_tensor_escape_is_not_hidden_by_a_structured_access(escape):
    tensor = TensorView.create("tensor", [16], "float32", scope="shared.dyn")
    argument = tensor.tensor
    if escape == "pointer":
        argument = tensor.tensor.data
    elif escape == "integer":
        argument = ir.Call("tirx.reinterpret", [ir.Call("tl.access_ptr", [tensor.load(0), 1, 1])], ty="uint64")
    result = _analyze(
        [
            ir.Evaluate(ir.Call("tirx.call_extern", ["consume", argument], ty="int32")),
            ir.Evaluate(tensor.load(0)),
        ],
        physical=True,
    )
    assert _has(result["reads"], tensor)
    assert any(variable.same_as(tensor.tensor) for variable in result["opaque"])


@pytest.mark.parametrize("control", ["break", "continue", "return"])
def test_control_transfer_prevents_full_write_proofs(control):
    tensor = TensorView.create("tensor", [16], "float32", scope="shared.dyn")
    transfer = {"break": ir.Break, "continue": ir.Continue, "return": lambda: ir.Return([])}[control]()
    body = [ir.Evaluate(T.fill(tensor.tensor, 0.0)), transfer]
    result = _analyze(body, physical=True)
    assert _has(result["writes"], tensor)
    assert not result["must_writes"]


@pytest.mark.parametrize("strides,opaque", [([16, 1], False), ([32, 1], True)])
def test_typed_pointer_footprints_require_proven_contiguous_layout(strides, opaque):
    tensor = TensorView.create("tensor", [4, 16], "float32", scope="shared.dyn", strides=strides, elem_offset=4)
    pointer = ir.Call("tl.access_ptr", [tensor.load([0, 0]), 32, 2])
    result = _analyze(ir.Evaluate(pointer), physical=True)
    assert _has(result["writes"], tensor)
    assert not result["must_writes"]
    assert any(variable.same_as(tensor.tensor) for variable in result["opaque"]) == opaque


def test_symbolic_pointer_mask_is_conservative_and_a_scalar_dependency():
    tensor = TensorView.create("tensor", [16], "float32", scope="shared.dyn")
    mask = ir.Var("mask", "int32")
    pointer = ir.Call("tl.access_ptr", [tensor.load(0), 16, mask])
    result = _analyze(ir.Evaluate(pointer))
    assert _has(result["reads"], tensor)
    assert _has(result["writes"], tensor)
    assert any(variable.same_as(mask) for variable in result["read_vars"])


def test_region_thread_parameter_bounds_are_used_for_accesses():
    tensor = TensorView.create("tensor", [32], "float32", scope="shared.dyn")
    thread = ir.Var("thread", "int32")
    region = ir.RegionStmt("tirx.launch_thread", ["threadIdx.x", 32], [thread], {}, tensor.store(thread, ir.const(1.0)))
    result = _analyze(region, physical=True)
    assert _bounds(_region(result["writes"], tensor)) == [(0, 32)]
    assert _has(result["must_writes"], tensor)


@pytest.mark.parametrize("begin,end,step,expected", [(5, 16, 3, 4), (16, 5, -3, 4), (5, 16, -3, 0), (16, 5, 3, 0)])
def test_latency_trip_count_obeys_signed_end_exclusive_loops(begin, end, step, expected):
    index = ir.Var("index", "int32")
    loop = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, ir.Evaluate(index))
    assert tvm_ffi.get_global_func("tl.ascend.EstimateLoopTripCount")(loop, {}) == expected


def test_known_symbolic_trip_count_bound_is_not_capped_by_fallback():
    index = ir.Var("index", "int32")
    extent = ir.Var("extent", "int32")
    loop = ir.For(index, 0, extent, 1, ir.ForKind.DEFAULT, ir.Evaluate(index))
    assert tvm_ffi.get_global_func("tl.ascend.EstimateLoopTripCount")(loop, {extent: ir.Range(0, 1001)}) == 1000
