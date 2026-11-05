"""Native allocation placement retains scoped identity and host availability."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import IRModule, codegen, ir, runtime, tirx
from tvm_tirx.target import Target

import tilelang
from tilelang.backend.module import create_backend_context
from tilelang.engine.lower import device_codegen, is_device_call
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.testing.packed import build_packed_module, compile_packed_function


def tensor(name, shape=(4,), scope="local", **kwargs):
    return ir.Var(name, tirx.TensorType(shape, "int32", scope=scope, **kwargs))


def allocate(value, attrs=None):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], attrs=attrs, ty_args=[value.ty]))


def function(params, statements):
    return tirx.Function(params, statements).with_attr("global_symbol", "main")


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def run(name, value):
    before = ir.save_json(value)
    assert tirx.analysis.check_well_formed(value)
    transform = getattr(tilelang.transform, name)()
    result = transform(IRModule({"main": value}))
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(transform(result), result)
    return result["main"]


def plan(value):
    return run("PlanAndUpdateBufferAllocationLocation", value)


def hoist(value):
    return run("HoistGlobalBufferAllocations", value)


def launch(body, name="blockIdx.x"):
    index = ir.Var(name.rsplit(".", 1)[0], "int32")
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(name), ir.IntImm("int32", 1)], [index], None, body)


def test_first_use_includes_pointer_alias():
    scratch = tensor("scratch")
    pointer = ir.Var("pointer", ir.PtrType(scratch.ty.dtype, scratch.ty.scope))
    # A pointer alias is the first use even though all memory accesses follow.
    value = function([], [allocate(scratch), ir.Evaluate(7), ir.Bind(pointer, scratch.data), ir.Evaluate(pointer)])
    result = plan(value)
    assert isinstance(result.body.seq[0], ir.Evaluate)
    assert result.body.seq[1].var.same_as(scratch)
    assert result.body.seq[2].var.same_as(pointer)


def test_loop_header_use_is_outside_the_loop():
    scratch = tensor("extent", (1,), scope="local.var")
    index = ir.Var("i", "int32")
    value = function([], [allocate(scratch), ir.Evaluate(7), ir.For(index, 0, ir.TensorLoad(scratch, [0]), 1, ir.ForKind.DEFAULT, [])])
    result = plan(value)
    assert result.body.seq[1].var.same_as(scratch)
    assert not objects(result.body.seq[2].body, ir.Bind)


def test_type_shape_use_keeps_its_allocation():
    count = tensor("count", (1,))
    values = tensor("values", (ir.TensorLoad(count, [0]),))
    value = function([], [allocate(count), ir.Evaluate(7), allocate(values), ir.Evaluate(values.data)])
    result = plan(value)
    assert result.body.seq[1].var.same_as(count)
    assert result.body.seq[2].var.same_as(values)


@pytest.mark.parametrize("scope", ["shared.barrier", "shared.cluster_barrier", "metal.cooperative_tensor"])
def test_special_storage_keeps_its_owner(scope):
    scratch = tensor("scratch", scope=scope)
    value = function([], [allocate(scratch), ir.Evaluate(7), ir.Evaluate(scratch.data)])
    ir.assert_structural_equal(plan(value), value)


def test_dynamic_and_explicitly_annotated_allocations_do_not_move():
    n = ir.Var("n", "int32")
    scratch = tensor("scratch", (n,))
    value = function([n], [allocate(scratch), ir.Evaluate(7), ir.Evaluate(scratch.data)])
    ir.assert_structural_equal(plan(value), value)
    scratch = tensor("scratch")
    value = function([], [allocate(scratch, {"volatile": True}), ir.Evaluate(7), ir.Evaluate(scratch.data)])
    ir.assert_structural_equal(plan(value), value)


def test_loop_carried_storage_executes_with_one_identity():
    output = tensor("output", (1,), "global")
    scratch = tensor("state", (1,))
    i = ir.Var("i", "int32")
    value = function(
        [output],
        [
            allocate(scratch),
            ir.Evaluate(7),
            ir.TensorStore(scratch, [0], 1),
            ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [ir.TensorStore(scratch, [0], ir.TensorLoad(scratch, [0]) * 2)]),
            ir.TensorStore(output, [0], ir.TensorLoad(scratch, [0])),
        ],
    )
    result = plan(value)
    for original in (value, result):
        executable = compile_packed_function(original)
        out = runtime.tensor(np.array([-1], dtype="int32"))
        executable(out)
        np.testing.assert_array_equal(out.numpy(), [16])


def test_allocation_does_not_leave_explicit_scope():
    scratch = tensor("scratch")
    region = ir.RegionStmt("tl.lexical_scope", [], [], None, [allocate(scratch), ir.Evaluate(7), ir.Evaluate(scratch.data)])
    result = plan(function([], [region]))
    assert len(result.body.seq) == 1
    assert isinstance(result.body.seq[0], ir.RegionStmt)
    assert result.body.seq[0].body.seq[1].var.same_as(scratch)


def test_global_workspace_and_shape_alias_are_host_captures():
    n = ir.Var("n", "int32")
    count = ir.Var("count", "int32")
    scratch = tensor("scratch", (count,), "global")
    local = tensor("local")
    region = launch([ir.Bind(count, n + 1), allocate(scratch), allocate(local), ir.TensorStore(scratch, [0], 7)])
    result = hoist(function([n], [region]))
    assert result.body.seq[0].var.same_as(count)
    assert result.body.seq[1].var.same_as(scratch)
    device = result.body.seq[2]
    assert len([bind for bind in objects(device, ir.Bind) if bind.var.same_as(count)]) == 0
    assert len([bind for bind in objects(device, ir.Bind) if bind.var.same_as(scratch)]) == 0
    assert len([bind for bind in objects(device, ir.Bind) if bind.var.same_as(local)]) == 1
    assert any(var.same_as(scratch) for var in tirx.analysis.undefined_vars(device))


def test_global_hoisting_stays_inside_host_iteration():
    index = ir.Var("host_i", "int32")
    scratch = tensor("scratch", (index + 1,), "global")
    region = launch([allocate(scratch), ir.TensorStore(scratch, [0], 7)])
    value = function([], [ir.For(index, 0, 3, 1, ir.ForKind.DEFAULT, [region])])
    result = hoist(value)
    assert len(result.body.seq) == 1
    assert isinstance(result.body.seq[0], ir.For)
    assert result.body.seq[0].body.seq[0].var.same_as(scratch)


def test_global_shape_from_later_parameter_is_host_available():
    n = ir.Var("n", "int32")
    output = tensor("output", (1,), "global")
    source = tensor("source", (n,), "global")
    scratch = tensor("scratch", (n,), "global")
    result = hoist(function([output, source], [launch([allocate(scratch), ir.TensorStore(output, [0], 7)])]))
    assert result.body.seq[0].var.same_as(scratch)


@pytest.mark.parametrize("dependency", ["thread", "memory", "division"])
def test_global_workspace_rejects_device_or_unsafe_shape(dependency):
    source = tensor("source", (1,), "global")
    index = ir.Var("block", "int32")
    denominator = ir.Var("denominator", "int32")
    count = ir.Var("count", "int32")
    size = {"thread": index + 1, "memory": ir.TensorLoad(source, [0]), "division": 4 // denominator}[dependency]
    scratch = tensor("scratch", (count,), "global")
    region = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("blockIdx.x"), ir.IntImm("int32", 1)],
        [index],
        None,
        [
            ir.Bind(count, size),
            allocate(scratch),
            ir.TensorStore(scratch, [0], 7),
        ],
    )
    value = function([source, denominator], [region])
    before = ir.save_json(value)
    with pytest.raises(ValueError, match="Global workspace"):
        hoist(value)
    assert ir.save_json(value) == before


def test_host_workspace_survives_native_split_and_cpu_execution():
    output = tensor("output", (4,), "global")
    scratch = tensor("scratch", (4,), "global")
    index = ir.Var("i", "int32")
    target = Target("c")
    region = ir.RegionStmt(
        "tirx.device_scope",
        [],
        [],
        {"target": target},
        [
            allocate(scratch),
            ir.For(
                index,
                0,
                4,
                1,
                ir.ForKind.DEFAULT,
                [
                    ir.TensorStore(scratch, [index], index + 3),
                    ir.TensorStore(output, [index], ir.TensorLoad(scratch, [index])),
                ],
            ),
        ],
    )
    result = hoist(function([output], [region]))
    # Start at the already-hoisted boundary. Replaying frontend annotation
    # would wrap the host allocation in a new device region. Split here, then
    # use the production compute codegen and packed host module combination.
    module = ir.transform.Sequential(
        [
            tirx.transform.BindTarget(target.with_host(target)),
            tilelang.transform.PrepareTensorLayouts(),
            tirx.transform.LowerTensorLayouts(),
            tirx.transform.FlattenTensor(),
            tilelang.transform.SplitHostDevice(),
        ]
    )(IRModule({"main": result}))
    host_mod = tirx.transform.Filter(lambda func: not is_device_call(func))(module)
    device_mod = tirx.transform.Filter(is_device_call)(module)
    host_runtime = build_packed_module(host_mod, target=target)
    context = create_backend_context(target, target, "tvm_ffi")
    host_runtime.import_module(device_codegen(device_mod, context))
    executable = codegen.Executable(host_runtime)
    executable.jit(cc="c++", options=["-std=c++17", "-I" + TILELANG_TEMPLATE_PATH])
    out = runtime.tensor(np.zeros(4, dtype="int32"))
    executable(out)
    np.testing.assert_array_equal(out.numpy(), np.arange(4, dtype="int32") + 3)
