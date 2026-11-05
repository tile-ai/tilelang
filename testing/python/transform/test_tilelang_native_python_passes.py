"""Native Python lowering passes preserve control flow, types and definitions."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

import tilelang.language as T
from tilelang.analysis._visitor import tilelang_metadata
from tilelang.testing.packed import compile_packed_function
from tilelang.transform.add_bufstore_wrapper import AddWrapperForSingleBufStore
from tilelang.transform.decouple_type_cast import DecoupleTypeCast
from tilelang.transform.hoist_broadcast_values import HoistBroadcastValues


def nodes(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def transform(factory, function):
    module = ir.IRModule.from_expr(function)
    result = factory()(module)
    function = next(iter(result.functions.values()))
    assert tirx.analysis.check_well_formed(function)
    return function


def scopes(function):
    return [
        node
        for node in nodes(function, ir.RegionStmt)
        if node.op.name == "tl.attr" and tilelang_metadata(node)["key"] == "lexical_alloc_scope"
    ]


def compile_with_scopes(function, target="c", *, vectorize=False):
    """Consume real scopes and execute the logical loop contract.

    Nonzero vector-loop begins belong to the pipeline's LegalizeVectorizedLoop
    stage. Disable the generic vectorizer for the domain oracle; a separate
    zero-begin case below checks the real vectorized path. TileLang owns the
    lexical scopes and packed ABI, so compile through its native consumers.
    """
    if "global_symbol" not in function.attrs:
        function = function.with_attr("global_symbol", "main")
    module = ir.IRModule.from_expr(function)
    module = tvm_ffi.get_global_func("tl.transform.LowerFrontendScopes")()(module)
    assert tirx.analysis.check_well_formed(module)
    module = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(vectorize)(module)
    return compile_packed_function(module, target=target, c_codegen="compute" if target == "c" else "host")


def test_single_fragment_store_wrapper_preserves_location():
    @T.prim_func
    def kernel(A: T.Tensor((1,), "float32")):
        fragment = T.alloc_fragment((1,), "float32")
        fragment[0] = A[0]
        A[0] = fragment[0]

    old_stores = nodes(kernel, ir.TensorStore)
    result = transform(AddWrapperForSingleBufStore, kernel)
    loops = nodes(result, ir.For)
    assert len(loops) == 2
    for loop, store in zip(loops, old_stores):
        assert loop.kind == ir.ForKind.PARALLEL
        assert int(loop.begin) == 0 and int(loop.end) == 1 and int(loop.step) == 1
        assert tvm_ffi.structural_equal(loop.loc, store.loc)


def test_wrapper_checks_all_indices_of_the_same_fragment():
    @T.prim_func
    def kernel(A: T.Tensor((1,), "float32")):
        fragment = T.alloc_fragment((2,), "float32")
        A[0] = fragment[1] + fragment[0]

    with pytest.raises(ValueError, match="Fragment buffer access with non-zero index"):
        transform(AddWrapperForSingleBufStore, kernel)


def test_wrapper_does_not_change_explicit_ownership_or_launch_dependent_store():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            fragment = T.alloc_fragment((32,), "float32")
            for i in T.Parallel(32):
                fragment[i] = A[i]
            tid = T.get_thread_binding()
            A[tid] = fragment[0]

    result = transform(AddWrapperForSingleBufStore, kernel)
    assert len(nodes(result, ir.For)) == 1
    assert tvm_ffi.structural_equal(kernel, result)


def _broadcast_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "float32")):
        value: T.float32x4 = T.Broadcast(T.float32(1.25), 4) + T.Broadcast(T.float32(2.5), 4)
        A[T.Ramp(0, 1, 4)] = value + T.Broadcast(T.float32(4.0), 4)

    # Vectorization consumes already-folded immediates. Construct that native
    # pass boundary explicitly, retaining one flat vector Bind whose uses must
    # remain dominated after the three scalar constants are hoisted.
    location = nodes(kernel, ir.TensorStore)[0].loc
    first = prim.Broadcast(ir.const(1.25, "float32"), ir.const(4), loc=location)
    second = prim.Broadcast(ir.const(2.5, "float32"), ir.const(4), loc=location)
    third = prim.Broadcast(ir.const(4.0, "float32"), ir.const(4), loc=location)
    value = ir.Var("value", first.ty, loc=location)
    body = [
        ir.Bind(value, first + second, loc=location),
        ir.TensorStore(
            kernel.params[0],
            [prim.Ramp(ir.const(0), ir.const(1), ir.const(4))],
            value + third,
            loc=location,
        ),
    ]
    return kernel.with_body(ir.SeqStmt(body, loc=kernel.body.loc), loc=kernel.loc)


def test_broadcast_hoist_keeps_flat_bind_dominance_and_executes():
    original = _broadcast_kernel()
    result = transform(HoistBroadcastValues, original)
    assert len([bind for bind in nodes(result, ir.Bind) if bind.var.name == "broadcast_var"]) == 3
    value_binds = [bind for bind in nodes(result, ir.Bind) if bind.var.name == "value"]
    assert len(value_binds) == 1
    assert not tirx.analysis.undefined_vars(result)
    for function in (original, result):
        output = tvm_tirx.runtime.tensor(np.zeros(4, dtype="float32"))
        compile_packed_function(function, c_codegen="compute")(output)
        np.testing.assert_array_equal(output.numpy(), np.full(4, 7.75, dtype="float32"))
    for statement in nodes(result, (ir.Bind, ir.TensorStore)):
        assert not isinstance(statement.loc, ir.UnknownLoc)
    assert tvm_ffi.structural_equal(result.body.loc, original.body.loc)


@pytest.mark.parametrize("start", [0, 2, -3, "symbolic"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("target", ["llvm", "c"])
def test_decouple_rmw_preserves_logical_domain_and_ordinal_scratch(start, index_dtype, target):
    tensor = ir.Var("A", T.Tensor((8,), "float32").as_type())
    i = ir.Var("i", ir.PrimType(index_dtype))
    begin = ir.Var("start", i.ty) if start == "symbolic" else ir.const(start, index_dtype)
    extent = ir.const(4, index_dtype)
    ordinal = i - begin
    value = prim.Cast("float32", prim.Cast("float64", ir.TensorLoad(tensor, [ordinal])) + prim.Cast("float64", i))
    body = ir.TensorStore(tensor, [ordinal], value)
    loop = ir.For(i, begin, begin + extent, ir.const(1, index_dtype), ir.ForKind.VECTORIZED, body)
    original = tirx.Function([tensor, *([begin] if start == "symbolic" else [])], loop)
    result = transform(DecoupleTypeCast, original)
    allocation_scopes = scopes(result)
    assert len(allocation_scopes) == 1
    allocations = [bind for bind in nodes(result, ir.Bind) if isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.alloc_tensor"]
    assert len(allocations) == 1
    scratch = allocations[0].var
    assert list(map(int, scratch.ty.shape)) == [4]
    loops = nodes(result, ir.For)
    assert len(loops) == 3
    for stage in loops:
        assert tvm_ffi.structural_equal(stage.begin, begin)
        assert tvm_ffi.structural_equal(stage.end, begin + extent)
        assert stage.var.ty == i.ty
        assert int(stage.step) == 1
    compute_store = nodes(loops[1], ir.TensorStore)[0]
    compute_load = nodes(loops[1], ir.TensorLoad)[0]
    assert compute_store.dest.same_as(scratch) and compute_load.source.same_as(scratch)
    assert tvm_ffi.structural_equal(compute_store.indices[0], ordinal)
    assert tirx.analysis.check_well_formed(result)
    source = np.arange(8, dtype="float32") + 0.5
    actual_start = 5 if start == "symbolic" else start
    expected = source.copy()
    expected[:4] += np.arange(actual_start, actual_start + 4, dtype="float32")
    for function in (original, result):
        output = tvm_tirx.runtime.tensor(source.copy())
        compile_with_scopes(function, target)(output, *([actual_start] if start == "symbolic" else []))
        np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("step", [2, "symbolic"])
def test_decouple_nonunit_step_keeps_original_loop(step):
    tensor = ir.Var("A", T.Tensor((16,), "float32").as_type())
    i = ir.Var("i", ir.PrimType("int32"))
    step_value = ir.Var("step", i.ty) if step == "symbolic" else ir.const(step)
    body = ir.TensorStore(tensor, [i], prim.Cast("float32", prim.Cast("float64", ir.TensorLoad(tensor, [i])) + ir.const(1, "float64")))
    loop = ir.For(i, ir.const(0), ir.const(8), step_value, ir.ForKind.VECTORIZED, body)
    original = tirx.Function([tensor, *([step_value] if step == "symbolic" else [])], loop)
    assert tvm_ffi.structural_equal(transform(DecoupleTypeCast, original), original)


def test_decouple_bind_chain_and_distinct_rmw_indices():
    @T.prim_func
    def kernel(A: T.Tensor((16,), "float16")):
        for i in T.vectorized(8):
            left = T.Cast("float32", A[i])
            right = T.Cast("float32", A[i + 8])
            A[i] = T.Cast("float16", left + right)

    result = transform(DecoupleTypeCast, kernel)
    allocations = [bind for bind in nodes(result, ir.Bind) if isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.alloc_tensor"]
    assert len(allocations) == 2
    assert len(nodes(result, ir.For)) == 4
    assert len(scopes(result)) == 1
    assert not tirx.analysis.undefined_vars(result)
    source = np.arange(16, dtype="float16") + np.float16(0.5)
    expected = source.copy()
    expected[:8] += source[8:]
    for function in (kernel, result):
        output = tvm_tirx.runtime.tensor(source.copy())
        compile_with_scopes(function, vectorize=True)(output)
        np.testing.assert_array_equal(output.numpy(), expected)


def test_decouple_nested_guards_preserve_short_circuit():
    @T.prim_func
    def kernel(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float16")):
        for i in T.vectorized(16):
            if i != 0:  # noqa: SIM102 - nested guards exercise conditional evaluation
                if 16 // i > 2:
                    B[i] = T.Cast("float16", A[i])

    result = transform(DecoupleTypeCast, kernel)
    assert len(scopes(result)) == 1
    copy_out = nodes(result, ir.For)[-1]
    condition = nodes(copy_out, ir.If)[0].condition
    assert isinstance(condition, ir.Call) and condition.op.name == "prim.if_then_else"
    assert int(condition.args[2]) == 0
    source = np.arange(16, dtype="float32") + 0.5
    expected = np.full(16, -1, dtype="float16")
    expected[1:6] = source[1:6].astype("float16")
    for function in (kernel, result):
        output = tvm_tirx.runtime.tensor(np.full(16, -1, dtype="float16"))
        compile_with_scopes(function, vectorize=True)(tvm_tirx.runtime.tensor(source), output)
        np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("guard", ["memory", "effect", "conditional_load"])
def test_decouple_keeps_stateful_guards_and_conditional_loads(guard):
    @T.prim_func
    def kernel(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float16")):
        for i in T.vectorized(8):
            if guard == "memory":
                if B[i] > T.float16(0):
                    B[i] = T.Cast("float16", A[i])
            elif guard == "effect":
                if T.call_extern("bool", "guard", i):
                    B[i] = T.Cast("float16", A[i])
            else:
                B[i] = T.Cast("float16", T.if_then_else(i < 8, A[i], A[i + 8]))

    result = transform(DecoupleTypeCast, kernel)
    assert tvm_ffi.structural_equal(result, kernel)
    assert not scopes(result)


@pytest.mark.parametrize("pure", [False, True])
def test_dtype_first_external_calls_keep_native_result_type_and_source(pure):
    external = T.call_pure_extern if pure else T.call_extern

    @T.prim_func
    def kernel(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        for i in T.serial(4):
            B[i] = external("float32", "floorf", A[i])

    call = [value for value in nodes(kernel, ir.Call) if value.op.name in {"tirx.call_extern", "tirx.call_pure_extern"}][0]
    assert call.ty == ir.PrimType("float32")
    assert not isinstance(call.loc, ir.UnknownLoc)
    values = np.array([-1.5, -0.25, 2.75, 8.0], dtype="float32")
    source = tvm_tirx.runtime.tensor(values)
    output = tvm_tirx.runtime.tensor(np.zeros_like(values))
    compile_packed_function(kernel)(source, output)
    np.testing.assert_array_equal(output.numpy(), np.floor(values))
