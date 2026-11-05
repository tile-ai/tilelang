"""Native shared arenas: physical byte ranges, alias lifetime and execution."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import IRModule, ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

import tilelang


def tensor(name, count=16, dtype="int32", scope="shared.dyn", alignment=16, layout=None):
    return ir.Var(name, tirx.TensorType([count], dtype, scope=scope, data_alignment=alignment, layout=layout))


def allocate(value, attrs=None):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], attrs=attrs, ty_args=[value.ty]))


def store(value, data=1, index=0):
    return ir.TensorStore(value, [index], ir.const(data, value.ty.dtype) if isinstance(data, (int, float)) else data)


def load(value, index=0):
    return ir.TensorLoad(value, [index])


def function(statements, params=(), attrs=None):
    fields = {"global_symbol": "main", "target": Target("c")}
    fields.update(attrs or {})
    return tirx.Function(list(params), statements, ret_type=ir.PrimType("void"), attrs=ir.DictAttrs(fields))


def merge(value, *, reuse=True, aggressive=False, static=False, alignment=16):
    with ir.transform.PassContext(config={"tirx.merge_static_smem": static}):
        result = tilelang.transform.MergeSharedMemoryAllocations(
            enable_aggressive_merge=aggressive, align_bytes=alignment, disable_reuse=not reuse
        )(IRModule({"main": value}))["main"]
    assert tirx.analysis.check_well_formed(IRModule({"main": result}))
    return result


def bindings(value):
    result = []
    tvm_ffi.structural_walk(value, (ir.Bind, result.append), order="pre")
    return result


def allocations(value):
    return [x for x in bindings(value) if isinstance(x.value, ir.Call) and x.value.op.name == "tirx.alloc_tensor"]


def offset(value, var):
    bind = next(x for x in bindings(value) if x.var.same_as(var))
    assert bind.value.op.name == "tirx.decl_tensor"
    pointer = bind.value.args[0]
    assert pointer.op.name == "tirx.ptr_byte_offset"
    return int(pointer.args[1])


@pytest.mark.parametrize("reuse", [False, True])
def test_sequential_reuse_preserves_tensor_identity_and_byte_size(reuse):
    a, b = tensor("a", 17), tensor("b", 17)
    output = tensor("output", 2, scope="global")
    original = function(
        [allocate(a), store(a, 7, 16), store(output, load(a, 16)), allocate(b), store(b, 9, 16), store(output, load(b, 16), 1)], [output]
    )
    before = ir.save_json(original)
    result = merge(original, reuse=reuse)
    arena = allocations(result)
    assert len(arena) == 1 and str(arena[0].var.ty.dtype) == "uint8"
    assert int(arena[0].var.ty.shape[0]) == (80 if reuse else 160)
    assert offset(result, a) == 0
    assert offset(result, b) == (0 if reuse else 80)
    assert ir.save_json(original) == before
    assert tvm_ffi.structural_equal(merge(result), result)


@pytest.mark.parametrize("aggressive", [False, True])
def test_all_live_allocations_conflict_even_when_another_pair_can_reuse(aggressive):
    a, b, c = tensor("a", 64), tensor("b", 32), tensor("c", 32)
    output = tensor("output", 3, scope="global")
    result = merge(
        function(
            [
                allocate(a),
                store(a, 1),
                allocate(b),
                store(b, 2),
                store(output, load(b)),
                allocate(c),
                store(c, 3),
                store(output, load(c), 1),
                store(output, load(a), 2),
            ],
            [output],
        ),
        aggressive=aggressive,
    )
    assert offset(result, b) == offset(result, c)
    assert offset(result, b) >= offset(result, a) + 64 * 4
    assert int(allocations(result)[0].var.ty.shape[0]) == 96 * 4


@pytest.mark.parametrize("aggressive", [False, True])
def test_branch_arms_are_not_ordered_between_threads(aggressive):
    a, b = tensor("same"), tensor("same")
    predicate = ir.Var("predicate", "bool")
    output = tensor("output", 1, scope="global")
    branch = ir.If(predicate, [allocate(a), store(a, 3), store(output, load(a))], [allocate(b), store(b, 5), store(output, load(b))])
    result = merge(function([branch], [predicate, output]), aggressive=aggressive)
    assert abs(offset(result, a) - offset(result, b)) >= 64


@pytest.mark.parametrize("aggressive", [False, True])
def test_loop_carried_storage_cannot_reuse_iteration_temporary(aggressive):
    a, b = tensor("carried"), tensor("scratch")
    output = tensor("output", 1, scope="global")
    i = ir.Var("i", "int32")
    loop = ir.For(i, 0, 3, 1, ir.ForKind.DEFAULT, [allocate(b), store(b, load(a) + 1), store(a, load(b) * 2)])
    result = merge(function([allocate(a), store(a, 1), loop, store(output, load(a))], [output]), aggressive=aggressive)
    assert abs(offset(result, a) - offset(result, b)) >= 64


@pytest.mark.parametrize("aggressive", [False, True])
def test_aggressive_reuse_only_for_fresh_iteration_allocations(aggressive):
    a, b = tensor("a"), tensor("b")
    output = tensor("output", 2, scope="global")
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        0,
        2,
        1,
        ir.ForKind.DEFAULT,
        [allocate(a), store(a, 1), store(output, load(a)), allocate(b), store(b, 2), store(output, load(b), 1)],
    )
    result = merge(function([loop], [output]), aggressive=aggressive)
    assert (offset(result, a) == offset(result, b)) == aggressive


def test_pointer_and_cross_dtype_view_keep_backing_alive():
    a, b = tensor("a", 8), tensor("b", 8)
    pointer = ir.Var("pointer", ir.PtrType("uint8", scope="shared.dyn"))
    view = tensor("view", 16, dtype="uint8")
    shifted = ir.Call("tirx.ptr_byte_offset", [a.data, ir.const(4, "int64")], ty=pointer.ty)
    decl = ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])
    output = tensor("output", 1, dtype="uint8", scope="global")
    original = function(
        [
            allocate(a),
            ir.Bind(pointer, shifted),
            ir.Bind(view, decl),
            allocate(b),
            store(b, 3),
            store(view, 7, 15),
            store(output, load(view, 15)),
        ],
        [output],
    )
    result = merge(original)
    assert abs(offset(result, a) - offset(result, b)) >= 32
    view_bind = next(x for x in bindings(result) if x.var.same_as(view))
    assert view_bind.value.same_as(decl)
    assert view_bind.var.ty.same_as(view.ty)


@pytest.mark.parametrize(
    "dtype,count,bytes_per_allocation",
    [
        ("int4", 7, 4),
        ("uint4", 9, 5),
        ("float4_e2m1fn", 7, 4),
        ("int4x3", 7, 11),
        ("int8x3", 7, 21),
        ("float16", 7, 14),
        ("bool", 7, 7),
    ],
)
def test_storage_size_uses_physical_packing(dtype, count, bytes_per_allocation):
    a = tensor("a", count, dtype, alignment=1)
    b = tensor("b", count, dtype, alignment=1)
    # Keep the objects as operands without constructing unsupported arithmetic
    # for packed/vector dtypes. Physical sizes are an independent byte oracle.
    result = merge(function([allocate(a), ir.Evaluate(a.data), allocate(b), ir.Evaluate(b.data)]), reuse=False, alignment=1)
    assert int(allocations(result)[0].var.ty.shape[0]) == 2 * bytes_per_allocation
    assert offset(result, b) == bytes_per_allocation


def test_alignment_map_and_types_both_constrain_offsets():
    a, b = tensor("ordinary", 3, dtype="int8", alignment=16), tensor("swizzled", 17, alignment=64)
    result = merge(
        function(
            [allocate(a), ir.Evaluate(a.data), allocate(b), ir.Evaluate(b.data)],
            attrs={"tl.smem_alignment_map": {"swizzled": ir.IntImm("int32", 1024)}},
        ),
        reuse=False,
    )
    assert offset(result, b) % 1024 == 0
    assert allocations(result)[0].var.ty.data_alignment == 1024


def test_dynamic_shape_alias_is_expanded_and_later_parameter_symbols_remain_valid():
    n = ir.Var("n", "int32")
    size = ir.Var("size", "int32")
    a, b = tensor("a", size), tensor("b", 3)
    carrier = tensor("carrier", n, scope="global")
    original = function([ir.Bind(size, n + 1), allocate(a), ir.Evaluate(a.data), allocate(b), ir.Evaluate(b.data)], [carrier])
    result = merge(original, reuse=False)
    free = tirx.analysis.undefined_vars(allocations(result)[0].var.ty)
    assert any(var.same_as(n) for var in free)
    assert not any(var.same_as(size) for var in free)
    assert next(x for x in bindings(result) if x.var.same_as(a)).var.ty.same_as(a.ty)


def test_device_index_cannot_be_hoisted_into_arena_size():
    index = ir.Var("index", "int32")
    a, b = tensor("a", index + 1), tensor("b")
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [allocate(a), ir.Evaluate(a.data), allocate(b), ir.Evaluate(b.data)])
    with pytest.raises(ValueError, match="device-local variable"):
        merge(function([loop]))


def test_unconsumed_placement_annotations_are_not_silently_discarded():
    a, b = tensor("a"), tensor("b")
    with pytest.raises(ValueError, match="allocation annotations"):
        merge(function([allocate(a, ir.DictAttrs({"must_be_consumed": 1})), ir.Evaluate(a.data), allocate(b)]))


@pytest.mark.parametrize("bad", [0, 3, -1, 1 << 28])
def test_invalid_alignment_is_rejected(bad):
    a, b = tensor("a"), tensor("b")
    with pytest.raises(ValueError, match="power of two"):
        merge(function([allocate(a), allocate(b)]), alignment=bad)


def test_storage_overflow_is_rejected_before_narrowing():
    a, b = tensor("a", 1 << 62), tensor("b")
    with pytest.raises((ValueError, OverflowError), match="overflows|int64|represent"):
        merge(function([allocate(a), allocate(b)]))


@pytest.mark.parametrize("enable", [False, True])
def test_static_merge_is_a_separate_policy(enable):
    a, b = tensor("a", scope="shared"), tensor("b", scope="shared")
    result = merge(function([allocate(a), store(a), allocate(b), store(b)]), static=enable)
    assert len(allocations(result)) == (1 if enable else 2)


@pytest.mark.parametrize("reuse", [False, True])
def test_merged_arena_executes_through_tilelang_c_emitter(reuse, tmp_path):
    from testing.python.cpu.test_native_codegen_c import build_own_c

    a, b = tensor("a", 17), tensor("b", 17)
    output = tensor("output", 2, scope="global")
    source = function(
        [allocate(a), store(a, 7, 16), store(output, load(a, 16)), allocate(b), store(b, 9, 16), store(output, load(b, 16), 1)], [output]
    )
    transformed = merge(source, reuse=reuse)
    module, code = build_own_c(transformed, tmp_path, retain_allocations=True)
    import tvm_tirx

    result = tvm_tirx.runtime.tensor(np.zeros(2, dtype="int32"))
    module(result)
    np.testing.assert_array_equal(result.numpy(), np.array([7, 9], dtype="int32"))
    assert "buf_dyn_shmem" in code and "__shared__" not in code


def test_direct_memory_shape_cannot_move_before_the_allocation_scope():
    values = tensor("values", 1, scope="global")
    a, b = tensor("a", load(values) + 1), tensor("b")
    with pytest.raises(ValueError, match="memory reads|unsafe"):
        merge(function([allocate(a), allocate(b)], [values]))


def test_potentially_trapping_size_is_not_speculated_out_of_a_branch():
    numerator, denominator = ir.Var("n", "int32"), ir.Var("d", "int32")
    a, b = tensor("a", numerator // denominator), tensor("b")
    branch = ir.If(denominator != 0, [allocate(a), ir.Evaluate(a.data), allocate(b), ir.Evaluate(b.data)], None)
    with pytest.raises(ValueError, match="trapping|unsafe"):
        merge(function([branch], [numerator, denominator]))


def test_async_copy_keeps_destination_live_until_completion():
    a, b = tensor("a"), tensor("b")
    source = tensor("source", 16, scope="global")
    asynchronous = ir.Call("tl.ptx_cp_async_bytes", [a.data, source.data, ir.const(4)], ty_args=[ir.PrimType("void")])
    result = merge(function([allocate(a), ir.Evaluate(asynchronous), allocate(b), store(b)], [source]))
    assert abs(offset(result, a) - offset(result, b)) >= 64


def test_tma_smem_alignment_fallback_does_not_depend_on_a_hopper_target_name():
    a, b = tensor("small", 3, dtype="uint8"), tensor("tma", 16)
    descriptor = tensor("desc", 1, dtype="uint64", scope="local")
    initialize = ir.Call(
        "tl.initialize_wgmma_descriptor", [descriptor.data, b.data, ir.const(0), ir.const(0), ir.const(0)], ty_args=[ir.PrimType("void")]
    )
    for arch in ["sm_90a", "sm_100a", "sm_120a"]:
        result = merge(
            function(
                [allocate(a), ir.Evaluate(a.data), allocate(b), allocate(descriptor), ir.Evaluate(initialize)],
                attrs={"target": Target({"kind": "cuda", "arch": arch})},
            ),
            reuse=False,
        )
        assert offset(result, b) % 1024 == 0
        assert allocations(result)[0].var.ty.data_alignment == 1024


def test_pointer_select_keeps_all_possible_backings_live():
    a, b, scratch = tensor("a"), tensor("b"), tensor("scratch")
    predicate = ir.Var("predicate", "bool")
    pointer = ir.Var("pointer", a.data.ty)
    view = tensor("view")
    selected = ir.Call("prim.if_then_else", [predicate, a.data, b.data], ty=pointer.ty)
    result = merge(
        function(
            [
                allocate(a),
                allocate(b),
                ir.Bind(pointer, selected),
                ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
                allocate(scratch),
                store(scratch, 3),
                store(view, 9),
            ],
            [predicate],
        )
    )
    assert abs(offset(result, a) - offset(result, scratch)) >= 64
    assert abs(offset(result, b) - offset(result, scratch)) >= 64


# Regression cases from independent review ARENA-REVIEW-001/002.
def test_native_allreduce_volatile_scratch_can_be_merged():
    combine = ir.LambdaExpr(["int8"] * 4, lambda a, b, c, d: (a + c, prim.max(b, d)))

    @T.function
    def main(source: T.Tensor((32,), "int8"), sums: T.Tensor((32,), "int8"), maxima: T.Tensor((32,), "int8")):
        with T.launch_thread("blockIdx.x", 1):  # noqa: SIM117 -- Distinct nested IR definition scopes.
            with T.launch_thread("threadIdx.x", 32) as tx:
                sum_value = T.alloc_tensor(ty_args=[T.Tensor((1,), "int8", scope="local")])
                max_value = T.alloc_tensor(ty_args=[T.Tensor((1,), "int8", scope="local")])
                T.gpu_thread_allreduce(
                    combine,
                    (T.int8(0), T.int8(-128)),
                    (source[tx], source[tx]),
                    True,
                    (sum_value[0], max_value[0]),
                    (tx,),
                )
                sums[tx] = sum_value[0]
                maxima[tx] = max_value[0]

    original = main.with_attr("target", Target("cuda"))
    lowered = tirx.transform.LowerThreadAllreduce()(IRModule({"main": original}))["main"]
    assert tirx.analysis.check_well_formed(lowered)
    scratch = [allocation for allocation in allocations(lowered) if allocation.value.attrs and allocation.value.attrs.get("tirx.volatile")]
    assert len(scratch) == 2
    result = merge(lowered, static=True)
    assert len(allocations(result)) == 1
    assert all(next(bind for bind in bindings(result) if bind.var.same_as(item.var)).value.attrs["tirx.volatile"] for item in scratch)


def integer_pointer_roundtrip():
    a, temporary = tensor("source_storage"), tensor("temporary")
    output = tensor("output", 1, scope="global")
    encoded = ir.Var("encoded_address", "uint64")
    pointer = ir.Var("decoded_address", a.data.ty)
    view = tensor("view")
    program = function(
        [
            allocate(a),
            store(a, 7),
            ir.Bind(encoded, tirx.reinterpret(a.data, ty="uint64")),
            allocate(temporary),
            store(temporary, 9),
            ir.Bind(pointer, tirx.reinterpret(encoded, ty=pointer.ty)),
            ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
            store(output, load(view)),
        ],
        [output],
    )
    assert tirx.analysis.check_well_formed(program)
    return program, a, temporary


def test_pointer_to_integer_to_pointer_retains_backing_lifetime():
    original, a, temporary = integer_pointer_roundtrip()
    result = merge(original)
    assert abs(offset(result, a) - offset(result, temporary)) >= 64


def test_pointer_integer_roundtrip_preserves_c_execution(tmp_path):
    from testing.python.cpu.test_native_codegen_c import build_own_c
    import tvm_tirx

    original, _, _ = integer_pointer_roundtrip()
    actual = []
    for label, function in [("original", original), ("merged", merge(original))]:
        output_dir = tmp_path / label
        output_dir.mkdir()
        module, _ = build_own_c(function, output_dir, retain_allocations=True)
        output = tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32"))
        module(output)
        actual.append(int(output.numpy()[0]))
    assert actual == [7, 7]
