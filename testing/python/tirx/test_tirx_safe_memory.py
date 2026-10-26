from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tilelang._compat import ir
from tvm_tirx.ir import prim
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts")


def hint(tensor, value):
    return ir.Evaluate(ir.Call("tl.safe_value_hint", [tensor.tensor, ir.const(value, tensor.dtype)]))


def lower(body, parameters, target="c", config=None):
    function = (
        tirx.Function(parameters, body, ret_type=ir.TupleType([]))
        .with_attr("target", Target(target))
        .with_attr("global_symbol", "safe_memory_test")
    )
    with PassContext(config=config or {}):
        module = tvm_ffi.get_global_func("tl.transform.LegalizeSafeMemoryAccess")()(ir.IRModule({"main": function}))
    assert ir.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def collect(value, node_type):
    result = []
    stmt_functor.post_order_visit(value, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_global_load_and_store_bounds_execute(target, *, compile_tilelang):
    source = TensorView.create("source", [8], "float32")
    output = TensorView.create("output", [8], "float32")
    index = ir.Var("index", "int32")
    body = [hint(source, 5), ir.For(index, 0, 14, 1, ir.ForKind.DEFAULT, output.store(index - 2, source.load(index - 3)))]
    module = lower(body, [source.tensor, output.tensor], target)
    executable = compile_tilelang(module, target=target)
    actual = runtime.empty((8,), "float32")
    executable(runtime.tensor(np.arange(8, dtype="float32")), actual)
    np.testing.assert_array_equal(actual.numpy(), [5, 0, 1, 2, 3, 4, 5, 6])


def test_scope_fill_values_are_inherited_and_do_not_escape(*, compile_tilelang):
    source = TensorView.create("source", [1], "float32")
    output = TensorView.create("output", [3], "float32")
    inner = ir.RegionStmt(
        "tl.scope",
        [],
        [],
        {"safe_value_map": {source.tensor: ir.const(7.0)}},
        output.store(0, source.load(-1)),
    )
    outer = ir.RegionStmt(
        "tl.scope",
        [],
        [],
        {"safe_value_map": {source.tensor: ir.const(9.0)}},
        [inner, output.store(1, source.load(-1))],
    )
    module = lower([outer, output.store(2, source.load(-1))], [source.tensor, output.tensor])
    actual = runtime.empty((3,), "float32")
    compile_tilelang(module, target="c")(runtime.tensor(np.ones((1,), dtype="float32")), actual)
    np.testing.assert_array_equal(actual.numpy(), [7, 9, 0])


def test_pure_assumptions_remove_only_dominated_bounds_checks(*, compile_tilelang):
    source = TensorView.create("source", [8], "float32")
    output = TensorView.create("output", [2], "float32")
    index = ir.Var("index", "int32")
    enabled = ir.Var("enabled", "bool")
    condition = prim.And(index >= 0, index < 8)
    guarded = ir.If(enabled, [ir.Evaluate(ir.Call("prim.assume", [condition])), output.store(0, source.load(index))], None)
    module = lower([guarded, output.store(1, source.load(index))], [source.tensor, output.tensor, index, enabled])
    checks = [call for call in collect(module, ir.Call) if call.op.name == "prim.if_then_else"]
    assert len(checks) == 1
    actual = runtime.tensor(np.full((2,), -1, dtype="float32"))
    compile_tilelang(module, target="c")(runtime.tensor(np.arange(8, dtype="float32")), actual, 11, False)
    np.testing.assert_array_equal(actual.numpy(), [-1, 0])


def test_local_scalar_assumption_does_not_survive_a_write(*, compile_tilelang):
    source = TensorView.create("source", [8], "float32")
    output = TensorView.create("output", [2], "float32")
    scalar = TensorView.create("scalar", [1], "int32", scope="local")
    index = ir.Var("index", "int32")
    body = [
        scalar.allocate(),
        scalar.store(0, index),
        ir.Evaluate(ir.Call("prim.assume", [prim.And(scalar.load(0) >= 0, scalar.load(0) < 8)])),
        output.store(0, source.load(scalar.load(0))),
        scalar.store(0, 11),
        output.store(1, source.load(scalar.load(0))),
    ]
    function = tirx.Function([source.tensor, output.tensor, index], body)
    simplified = tvm_ffi.get_global_func("tl.transform.Simplify")(False)(ir.IRModule({"main": function}))["main"]
    module = lower(simplified.body, list(simplified.params))
    actual = runtime.empty((2,), "float32")
    compile_tilelang(module, target="c")(runtime.tensor(np.arange(8, dtype="float32")), actual, 3)
    np.testing.assert_array_equal(actual.numpy(), [3, 0])


@pytest.mark.parametrize("size,index", [(8, -1), (8, 8), (8, 13), (0, 0)])
def test_constant_out_of_bounds_and_zero_size_are_guarded(size, index, *, compile_tilelang):
    source = TensorView.create("source", [size], "int32")
    output = TensorView.create("output", [1], "int32")
    module = lower([hint(source, 9), output.store(0, source.load(index))], [source.tensor, output.tensor])
    executable = compile_tilelang(module, target="c")
    actual = runtime.empty((1,), "int32")
    executable(runtime.tensor(np.arange(size, dtype="int32")), actual)
    np.testing.assert_array_equal(actual.numpy(), [9])


def test_store_with_conditional_value_still_checks_destination(*, compile_tilelang):
    output = TensorView.create("output", [8], "int32")
    index = ir.Var("index", "int32")
    value = ir.Call("prim.if_then_else", [index < 5, ir.const(7), ir.const(3)])
    module = lower(ir.For(index, 0, 16, 1, ir.ForKind.DEFAULT, output.store(index, value)), [output.tensor])
    actual = runtime.empty((8,), "int32")
    compile_tilelang(module, target="c")(actual)
    np.testing.assert_array_equal(actual.numpy(), [7] * 5 + [3] * 3)


def atomic_index(counter):
    pointer = ir.Call("tl.access_ptr", [counter.load(0), 1, 3])
    return ir.Call("tl.atomic_add_ret_elem_op", [pointer, 1, 0], ty="int32")


def test_annotated_pointer_keeps_its_actual_offset():
    tensor = TensorView.create("tensor", [32], "float32")
    offset = ir.Var("offset", "int32")
    pointer = ir.Call(
        "tirx.address_of",
        [tensor.load(offset)],
        attrs={
            "tl.access_region": ir.TensorRegion(tensor.tensor, [ir.Range(0, 32)], ir.TensorRegionType()),
            "tl.access_extent": ir.const(4),
            "tl.access_mask": ir.const(3),
        },
    )
    atomic = ir.Call("tl.atomic_addx4_elem_op", [pointer, prim.Broadcast(ir.const(1.0), 4), 0])
    module = lower(ir.Evaluate(atomic), [tensor.tensor, offset], "cuda")
    addresses = [call for call in collect(module, ir.Call) if call.op.name == "tirx.address_of"]
    assert len(addresses) == 1
    ir.assert_structural_equal(addresses[0].args[0].indices[0], offset)
    repeated = tvm_ffi.get_global_func("tl.transform.LegalizeSafeMemoryAccess")()(module)
    ir.assert_structural_equal(module, repeated)


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("selected", [False, True])
def test_mutable_indices_evaluate_once_and_stay_in_lazy_branch(target, selected, *, compile_tilelang):
    source = TensorView.create("source", [8], "int32")
    counter = TensorView.create("counter", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    flag = ir.Var("flag", "bool")
    value = ir.Call("prim.if_then_else", [flag, source.load(atomic_index(counter)), ir.const(99)])
    module = lower(output.store(0, value), [source.tensor, counter.tensor, output.tensor, flag], target)
    module = tvm_ffi.get_global_func("tl.cpu.transform.LowerCPUAtomics")()(module)
    executable = compile_tilelang(module, target=target)
    actual = runtime.empty((1,), "int32")
    count = runtime.tensor(np.array([7], dtype="int32"))
    executable(runtime.tensor(np.arange(8, dtype="int32") * 10), count, actual, selected)
    np.testing.assert_array_equal(actual.numpy(), [70 if selected else 99])
    np.testing.assert_array_equal(count.numpy(), [8 if selected else 7])


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("later_index", [-1, 0, 3, 4])
def test_independent_bounds_guard_effectful_indices(target, direction, later_index, *, compile_tilelang):
    source = TensorView.create("source", [4, 4], "int32")
    counter = TensorView.create("counter", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    later = ir.Var("later", "int32")
    index = atomic_index(counter)
    if direction == "load":
        body = [hint(source, 37), output.store(0, source.load([index, later]))]
    else:
        body = source.store([index, later], ir.const(99))
    module = lower(body, [source.tensor, counter.tensor, output.tensor, later], target)
    module = tvm_ffi.get_global_func("tl.cpu.transform.LowerCPUAtomics")()(module)
    executable = compile_tilelang(module, target=target)
    original = np.arange(16, dtype="int32").reshape(4, 4)
    values = runtime.tensor(original)
    count = runtime.tensor(np.array([1], dtype="int32"))
    actual = runtime.empty((1,), "int32")
    executable(values, count, actual, later_index)
    valid = 0 <= later_index < 4
    np.testing.assert_array_equal(count.numpy(), [2 if valid else 1])
    if direction == "load":
        np.testing.assert_array_equal(actual.numpy(), [original[1, later_index] if valid else 37])
    else:
        if valid:
            original[1, later_index] = 99
        np.testing.assert_array_equal(values.numpy(), original)


@pytest.mark.parametrize("base", [4, 6, -1])
def test_vector_access_requires_every_lane_in_bounds(base, *, compile_tilelang):
    source = TensorView.create("source", [8], "float32")
    output = TensorView.create("output", [4], "float32")
    body = [hint(source, 7), output.store(prim.Ramp(0, 1, 4), source.load(prim.Ramp(base, 1, 4)))]
    module = lower(body, [source.tensor, output.tensor], "llvm")
    actual = runtime.empty((4,), "float32")
    compile_tilelang(module, target="llvm")(runtime.tensor(np.arange(8, dtype="float32")), actual)
    np.testing.assert_array_equal(actual.numpy(), [4, 5, 6, 7] if base == 4 else [7] * 4)


def test_narrow_unsigned_index_does_not_truncate_shape(*, compile_tilelang):
    source = TensorView.create("source", [256], "int32")
    output = TensorView.create("output", [1], "int32")
    index = ir.Var("index", "uint8")
    module = lower(output.store(0, source.load(index)), [source.tensor, output.tensor, index])
    actual = runtime.empty((1,), "int32")
    compile_tilelang(module, target="c")(runtime.tensor(np.arange(256, dtype="int32")), actual, 255)
    np.testing.assert_array_equal(actual.numpy(), [255])


def test_safe_value_hints_follow_aliases_and_lexical_scope(*, compile_tilelang):
    source = TensorView.create("source", [4], "int32")
    alias = TensorView.create("alias", [4], "int32")
    output = TensorView.create("output", [2], "int32")
    inner = ir.RegionStmt("tl.scope", [], [], None, [hint(source, 11), output.store(0, alias.load(4))])
    body = [alias.declare(source.data), hint(source, 7), inner, output.store(1, alias.load(4))]
    module = lower(body, [source.tensor, output.tensor])
    actual = runtime.empty((2,), "int32")
    compile_tilelang(module, target="c")(runtime.tensor(np.arange(4, dtype="int32")), actual)
    np.testing.assert_array_equal(actual.numpy(), [11, 7])


def test_disabled_safety_still_consumes_metadata():
    source = TensorView.create("source", [4], "int32")
    output = TensorView.create("output", [1], "int32")
    module = lower(
        [hint(source, 7), output.store(0, source.load(4))], [source.tensor, output.tensor], config={"tl.disable_safe_memory_legalize": True}
    )
    assert not collect(module, ir.If)
    assert not any(call.op.name == "tl.safe_value_hint" for call in collect(module, ir.Call))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,elements", [("float32", 4), ("float16", 8)])
@pytest.mark.parametrize("safe_value", [0.0, 3.0, -0.0])
@pytest.mark.parametrize("predicate", [False, True])
def test_async_copy_safe_values_fill_the_entire_transfer(dtype, elements, safe_value, predicate, *, compile_tilelang):
    device = runtime.cuda(0)
    if not device.exist or int(device.compute_version.split(".")[0]) < 8:
        pytest.skip("Asynchronous copies require CUDA compute capability 8.0 or newer")
    root = Path(__file__).resolve().parents[3]
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {
                "cuda": {"compiler": "nvcc", "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty' / 'cutlass' / 'include'}"]}
            },
        },
        host="c",
    )
    source = TensorView.create("source", [elements], dtype)
    output = TensorView.create("output", [elements * 2], dtype)
    shared = TensorView.create("shared", [elements * 2], dtype, scope="shared", data_alignment=16)
    thread = ir.Var("thread", "int32")
    lane = ir.Var("lane", "int32")
    source_pointer = ir.Call("tl.access_ptr", [source.load(thread * elements), elements, 1])
    destination_pointer = ir.Call("tl.access_ptr", [shared.load(thread * elements), elements, 2])
    copy = ir.Evaluate(ir.Call("tl.ptx_cp_async", [destination_pointer, source_pointer, elements, ir.const(predicate, "bool")]))
    body = [
        shared.allocate(),
        hint(source, safe_value),
        copy,
        ir.Evaluate(ir.Call("tirx.ptx.cp_async_commit_group", [ir.StringImm("async"), ir.StringImm("commit_group"), ir.StringImm("")])),
        ir.Evaluate(ir.Call("tirx.ptx.cp_async_wait_group", [0, ir.StringImm("async"), ir.StringImm("wait_group"), ir.StringImm("")])),
        ir.For(lane, 0, elements, 1, ir.ForKind.DEFAULT, output.store(thread * elements + lane, shared.load(thread * elements + lane))),
    ]
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(2)], [thread], None, body)
    module = lower(body, [source.tensor, output.tensor], target)
    module = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(module)
    executable = compile_tilelang(module, target=target)
    actual = runtime.empty((elements * 2,), dtype, device=device)
    values = np.arange(elements, dtype=dtype) + 1
    executable(runtime.tensor(values, device=device), actual)
    expected = np.concatenate([values, np.full(elements, safe_value, dtype=dtype)]) if predicate else np.zeros(elements * 2, dtype=dtype)
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(np.signbit(actual.numpy()), np.signbit(expected))


@pytest.mark.parametrize("explicit_region", [False, True])
@pytest.mark.parametrize("safe_value", [0, 7])
def test_async_fallback_snapshots_destination_and_predicate(explicit_region, safe_value, *, compile_tilelang):
    device = runtime.cuda(0)
    if not device.exist or int(device.compute_version.split(".")[0]) < 8:
        pytest.skip("Asynchronous copies require CUDA compute capability 8.0 or newer")
    root = Path(__file__).resolve().parents[3]
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {
                "cuda": {"compiler": "nvcc", "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty' / 'cutlass' / 'include'}"]}
            },
        },
        host="c",
    )
    source = TensorView.create("source", [2], "float32")
    output = TensorView.create("output", [4], "float32")
    destination_counter = TensorView.create("destination_counter", [1], "int32")
    predicate_counter = TensorView.create("predicate_counter", [1], "int32")
    shared = TensorView.create("shared", [4], "float32", scope="shared", data_alignment=16)
    thread = ir.Var("thread", "int32")
    lane = ir.Var("lane", "int32")
    source_base = source.region([ir.Range.from_min_extent(0, 1)]) if explicit_region else source.load(0)
    source_pointer = ir.Call("tl.access_ptr", [source_base, 4, 1])
    destination_pointer = ir.Call("tl.access_ptr", [shared.load(atomic_index(destination_counter)), 4, 2])
    predicate = atomic_index(predicate_counter) == 0
    copy = ir.Evaluate(ir.Call("tl.ptx_cp_async", [destination_pointer, source_pointer, 4, predicate]))
    body = [
        shared.allocate(),
        hint(source, safe_value),
        copy,
        ir.Evaluate(ir.Call("tirx.ptx.cp_async_commit_group", [ir.StringImm("async"), ir.StringImm("commit_group"), ir.StringImm("")])),
        ir.Evaluate(ir.Call("tirx.ptx.cp_async_wait_group", [0, ir.StringImm("async"), ir.StringImm("wait_group"), ir.StringImm("")])),
        ir.For(lane, 0, 4, 1, ir.ForKind.DEFAULT, output.store(lane, shared.load(lane))),
    ]
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, body)
    parameters = [source.tensor, output.tensor, destination_counter.tensor, predicate_counter.tensor]
    module = lower(body, parameters, target)
    module = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(module)
    executable = compile_tilelang(module, target=target)
    actual = runtime.empty((4,), "float32", device=device)
    destination_count = runtime.tensor(np.array([0], dtype="int32"), device=device)
    predicate_count = runtime.tensor(np.array([0], dtype="int32"), device=device)
    executable(runtime.tensor(np.ones(2, dtype="float32"), device=device), actual, destination_count, predicate_count)
    np.testing.assert_array_equal(actual.numpy(), np.full(4, safe_value, dtype="float32"))
    np.testing.assert_array_equal(destination_count.numpy(), [1])
    np.testing.assert_array_equal(predicate_count.numpy(), [1])


def test_mutable_safe_value_requires_explicit_binding():
    source = TensorView.create("source", [1], "int32")
    counter = TensorView.create("counter", [1], "int32")
    annotation = ir.Evaluate(ir.Call("tl.safe_value_hint", [source.tensor, counter.load(0)]))
    with pytest.raises(ValueError, match="Safe-value hints must be pure"):
        lower(annotation, [source.tensor, counter.tensor])
