import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang._compat.tensor import TensorView
from testing.ascend.auto_schedule.test_native_core_assignment import _module, _root


def _dependencies(body, params=(), attrs=None):
    module = _module(body, params, attrs=attrs)
    module = tvm_ffi.get_global_func("tl.transform.MaterializeScheduleUnits")()(module)
    assert tirx.analysis.check_well_formed(module)
    return tvm_ffi.get_global_func("tl.ascend.AnalyzeScheduleDependencies")(_root(module))


@pytest.mark.parametrize("offset", [0, 16])
def test_declared_views_keep_their_allocation_storage_identity(offset):
    storage = TensorView.create("storage", [64], "float32", scope="shared.dyn")
    view = TensorView.create("view", [4, 4], "float32", scope="shared.dyn")
    pointer = storage.pointer_at(offset) if offset else storage.data
    result = _dependencies([storage.allocate(), view.declare(pointer), storage.store(offset + 5, 1), ir.Evaluate(view.load([1, 1]))])
    dependency = next(item for item in result if item["producer"] == 2 and item["consumer"] == 3)
    assert dependency["storage"].same_as(storage.tensor)
    assert dependency["distance"] == 0


def test_pointer_and_tensor_alias_chains_keep_storage_identity():
    storage = TensorView.create("storage", [64], "float32", scope="shared.dyn")
    view = TensorView.create("view", [16], "float32", scope="shared.dyn")
    second = TensorView.create("second", [8], "float32", scope="shared.dyn")
    pointer = ir.Var("pointer", storage.data.ty)
    result = _dependencies(
        [
            storage.allocate(),
            ir.Bind(pointer, storage.data),
            view.declare(pointer),
            second.declare(view.data),
            storage.store(3, 1),
            ir.Evaluate(second.load(3)),
        ]
    )
    assert any(item["producer"] == 4 and item["consumer"] == 5 and item["storage"].same_as(storage.tensor) for item in result)


def test_independent_allocations_do_not_alias_by_type():
    first = TensorView.create("first", [64], "float32", scope="shared.dyn")
    second = TensorView.create("second", [64], "float32", scope="shared.dyn")
    assert not _dependencies([first.allocate(), second.allocate(), first.store(3, 1), ir.Evaluate(second.load(3))])


@pytest.mark.parametrize("begin,end,step", [(0, 10, 2), (10, 0, -2)])
def test_cross_iteration_distance_counts_iterations_not_index_deltas(begin, end, step):
    storage = TensorView.create("storage", [16], "float32", scope="shared.dyn")
    index = ir.Var("index", "int32")
    loop = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, [storage.store(index, 1), ir.Evaluate(storage.load(index - step))])
    result = _dependencies([storage.allocate(), loop], attrs={"tl.manual_multi_buffer": {storage.tensor: 4}})
    dependencies = [item for item in result if item["loop"] is not None and item["producer"] == 0 and item["consumer"] == 1]
    assert len(dependencies) == 1
    assert dependencies[0]["distance"] == 1


def test_disjoint_native_regions_have_no_dependency():
    storage = TensorView.create("storage", [16], "float32", scope="shared.dyn")
    assert not _dependencies([storage.allocate(), storage.store(0, 1), ir.Evaluate(storage.load(8))])


@pytest.mark.parametrize("force", [False, True])
def test_native_conflict_hints_accept_boolean_metadata(force):
    storage = TensorView.create("storage", [16], "float32", scope="shared.dyn")
    attributes = {"tl.root_conflict_hints": [[storage.tensor, storage.tensor, -1, force]]}
    result = _dependencies([storage.allocate(), storage.store(0, 1), ir.Evaluate(storage.load(0 if not force else 8))], attrs=attributes)
    assert bool(result) is force
