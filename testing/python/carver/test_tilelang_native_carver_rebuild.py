"""Independent semantic oracles for rebuilding a native Carver stage graph."""

import ctypes

import pytest
import torch
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.carver._native_analysis import NativeSchedule
from tilelang.carver._workload import IterAxis, Stage, emit_stage


def i64(value):
    return ir.const(value, "int64")


def axis(name, size):
    return IterAxis(ir.Var(name, ir.PrimType("int64")), ir.Range(i64(0), i64(size)), "S")


def function(parameters, statements):
    result = tirx.Function(parameters, ir.SeqStmt(statements), ir.NoneType())
    result = result.with_attr("global_symbol", "main").with_attr("tirx.noalias", True)
    tirx.analysis.check_well_formed(result)
    return result


def execute(func, *arguments):
    target = Target("c")
    kernel = tilelang.compile(func, target=target, target_host=target, execution_backend="tvm_ffi")
    kernel(*arguments)


def test_late_scalar_bind_keeps_its_memory_snapshot():
    a = ir.Var("A", tirx.TensorType((i64(1),), "int32"))
    c = ir.Var("C", tirx.TensorType((i64(1),), "int32"))
    scalar = ir.Var("snapshot", ir.PrimType("int32"))
    first, second = axis("i", 1), axis("j", 1)
    write = Stage(0, "write", (first,), ir.TensorStore(a, [first.var], a[first.var] + 10))
    read = Stage(1, "read", (second,), ir.TensorStore(c, [second.var], scalar + a[second.var]))
    original = function([a, c], [emit_stage(write), ir.Bind(scalar, a[i64(0)]), emit_stage(read)])
    source, output = torch.tensor([2], dtype=torch.int32), torch.tensor([-100], dtype=torch.int32)
    execute(original, source, output)
    torch.testing.assert_close(output, torch.tensor([24], dtype=torch.int32), atol=0, rtol=0)
    schedule = NativeSchedule(original)
    try:
        schedule.reindex(schedule.get_child_blocks(schedule.root)[1], ("read", 0))
    except ValueError:
        assert schedule.function.same_as(original), "Rejected rewrites must leave the original function intact"
    else:
        source, output = torch.tensor([2], dtype=torch.int32), torch.tensor([-100], dtype=torch.int32)
        execute(schedule.function, source, output)
        torch.testing.assert_close(output, torch.tensor([24], dtype=torch.int32), atol=0, rtol=0)


def test_block_axis_domain_does_not_expand_the_actual_loop():
    a = ir.Var("A", tirx.TensorType((i64(8),), "int32"))
    c = ir.Var("C", tirx.TensorType((i64(8),), "int32"))
    spatial = axis("i", 8)
    stage = Stage(0, "partial", (spatial,), ir.TensorStore(c, [spatial.var], a[spatial.var] + 1))
    loop = emit_stage(stage)
    partial_loop = ir.For(loop.var, loop.begin, i64(4), loop.step, loop.kind, loop.body)
    original = function([a, c], [partial_loop])
    source, output = torch.arange(8, dtype=torch.int32), torch.full((8,), -100, dtype=torch.int32)
    expected = torch.tensor([1, 2, 3, 4, -100, -100, -100, -100], dtype=torch.int32)
    execute(original, source, output)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    schedule = NativeSchedule(original)
    try:
        schedule.reindex(schedule.get_child_blocks(schedule.root)[0], ("read", 0))
    except ValueError:
        assert schedule.function.same_as(original), "Rejected rewrites must leave the original function intact"
    else:
        output = torch.full((8,), -100, dtype=torch.int32)
        execute(schedule.function, source, output)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("operation", ["inline", "reindex"])
def test_internal_view_alias_is_not_a_disjoint_tensor(operation):
    a = ir.Var("A", tirx.TensorType((i64(4),), "int32"))
    alias = ir.Var("view", a.ty)
    temporary = ir.Var("snapshot", a.ty)
    first, second = axis("i", 4), axis("j", 4)
    view_bind = ir.Bind(alias, ir.Call("tirx.decl_tensor", [tirx.tensor_data_ptr(a)], ty_args=[a.ty]))
    if operation == "inline":
        alloc = ir.Bind(temporary, ir.Call("tirx.alloc_tensor", [], ty_args=[temporary.ty]))
        copy = Stage(0, "snapshot", (first,), ir.TensorStore(temporary, [first.var], a[(first.var + 3) % 4]))
        write = Stage(1, "write", (second,), ir.TensorStore(alias, [second.var], temporary[second.var]))
        original = function([a], [view_bind, alloc, emit_stage(copy), emit_stage(write)])
        expected = torch.tensor([4, 1, 2, 3], dtype=torch.int32)
    else:
        write = Stage(0, "inplace", (first,), ir.TensorStore(alias, [first.var], a[(first.var + 3) % 4]))
        original = function([a], [view_bind, emit_stage(write)])
        expected = torch.tensor([4, 4, 4, 4], dtype=torch.int32)
    source = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    execute(original, source)
    torch.testing.assert_close(source, expected, atol=0, rtol=0)
    schedule = NativeSchedule(original)
    try:
        block = schedule.get_child_blocks(schedule.root)[0]
        if operation == "inline":
            schedule.compute_inline(block)
        else:
            schedule.reindex(block, ("read", 0))
    except ValueError:
        assert schedule.function.same_as(original), "Rejected rewrites must leave the original function intact"
    else:
        source = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        execute(schedule.function, source)
        torch.testing.assert_close(source, expected, atol=0, rtol=0)


def test_block_axis_without_a_loop_does_not_create_iterations():
    @T.prim_func
    def original(A: T.Tensor((4,), "int32"), C: T.Tensor((4,), "int32")):
        T.func_attr({"tirx.noalias": True})
        with T.sblock("once"):
            vi = T.axis.spatial(4, 0)
            C[vi] = A[vi] + 1

    source, output = torch.arange(4, dtype=torch.int32), torch.full((4,), -100, dtype=torch.int32)
    expected = torch.tensor([1, -100, -100, -100], dtype=torch.int32)
    execute(original, source, output)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    schedule = NativeSchedule(original)
    try:
        schedule.reindex(schedule.get_child_blocks(schedule.root)[0], ("read", 0))
    except ValueError:
        assert schedule.function.same_as(original), "Rejected rewrites must leave the original function intact"
    else:
        output = torch.full((4,), -100, dtype=torch.int32)
        execute(schedule.function, source, output)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_reindex_keeps_reads_after_an_external_memory_write():
    a = ir.Var("A", tirx.TensorType((i64(4),), "int32"))
    c = ir.Var("C", tirx.TensorType((i64(4),), "int32"))
    spatial = axis("i", 4)
    # Match the actual C ABI: void* memset(void*, int, size_t).
    pointer = tirx.reinterpret(tirx.tensor_data_ptr(a), ty=ir.PtrType("void"))
    size_type = f"uint{ctypes.sizeof(ctypes.c_size_t) * 8}"
    clear = tirx.call_extern("memset", pointer, ir.const(0, "int32"), ir.const(16, size_type), ty=pointer.ty)
    body = ir.SeqStmt([ir.Evaluate(clear), ir.TensorStore(c, [spatial.var], a[spatial.var] + 1)])
    stage = Stage(0, "clear_then_read", (spatial,), body)
    original = function([a, c], [emit_stage(stage)])
    source, output = torch.arange(4, dtype=torch.int32), torch.full((4,), -100, dtype=torch.int32)
    expected = torch.ones(4, dtype=torch.int32)
    execute(original, source, output)
    torch.testing.assert_close(source, torch.zeros(4, dtype=torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    schedule = NativeSchedule(original)
    try:
        schedule.reindex(schedule.get_child_blocks(schedule.root)[0], ("read", 0))
    except ValueError:
        assert schedule.function.same_as(original), "Rejected rewrites must leave the original function intact"
    else:
        source, output = torch.arange(4, dtype=torch.int32), torch.full((4,), -100, dtype=torch.int32)
        execute(schedule.function, source, output)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("use_alias", [False, True])
def test_block_permutation_keeps_cross_coordinate_recurrence_order(use_alias):
    source = ir.Var("A", tirx.TensorType([i64(2), i64(2)], "int32"))
    alias = ir.Var("view", source.ty)
    first, second = axis("i", 2), axis("j", 2)
    previous = (first.var * 2 + second.var + 3) % 4
    body = ir.TensorStore(alias if use_alias else source, [first.var, second.var], source[previous // 2, previous % 2])
    stage = Stage(0, "cyclic_inplace", (first, second), body)
    statements = [ir.Bind(alias, ir.Call("tirx.decl_tensor", [source.data], ty_args=[source.ty]))] if use_alias else []
    original = function([source], [*statements, emit_stage(stage)])
    expected = torch.full((2, 2), 4, dtype=torch.int32)
    data = torch.arange(1, 5, dtype=torch.int32).reshape(2, 2)
    execute(original, data)
    torch.testing.assert_close(data, expected, atol=0, rtol=0)
    schedule = NativeSchedule(original)
    try:
        schedule.transform_block_layout(
            schedule.get_child_blocks(schedule.root)[0], tirx.IndexMap.from_func(lambda i, j: (j, i), index_dtype="int64")
        )
    except ValueError:
        assert schedule.function.same_as(original)
    else:
        data = torch.arange(1, 5, dtype=torch.int32).reshape(2, 2)
        execute(schedule.function, data)
        torch.testing.assert_close(data, expected, atol=0, rtol=0)


def test_block_permutation_checks_physical_stride_overlap():
    from tvm_tirx.tirx.layout import Iter

    layout = tirx.TileLayout.from_iters([Iter(i64(2), i64(1), "m"), Iter(i64(2), i64(1), "m")])
    output = ir.Var("C", tirx.TensorType([i64(2), i64(2)], "int32", layout=layout))
    first, second = axis("i", 2), axis("j", 2)
    body = ir.TensorStore(output, [first.var, second.var], ir.prim.Cast("int32", first.var * 2 + second.var))
    original = function([output], [emit_stage(Stage(0, "overlap", (first, second), body))])
    storage = torch.full((3,), -100, dtype=torch.int32)
    execute(original, storage.as_strided((2, 2), (1, 1)))
    expected = torch.tensor([0, 2, 3], dtype=torch.int32)
    torch.testing.assert_close(storage, expected, atol=0, rtol=0)
    schedule = NativeSchedule(original)
    try:
        schedule.transform_block_layout(
            schedule.get_child_blocks(schedule.root)[0], tirx.IndexMap.from_func(lambda i, j: (j, i), index_dtype="int64")
        )
    except ValueError:
        assert schedule.function.same_as(original)
    else:
        storage = torch.full((3,), -100, dtype=torch.int32)
        execute(schedule.function, storage.as_strided((2, 2), (1, 1)))
        torch.testing.assert_close(storage, expected, atol=0, rtol=0)
