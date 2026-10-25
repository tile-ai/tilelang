from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_reducer_frontend_accepts_native_tensor_load_contributions(dtype):
    import tilelang.language as T

    @T.prim_func
    def kernel(source: T.Tensor((32,), dtype), output: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            accumulator = T.alloc_reducer((1,), T.float32)
            result = T.alloc_fragment((1,), T.float32)
            T.reducer_init(accumulator)
            for coordinate in T.Parallel(32):
                T.reducer_update(accumulator[0], source[coordinate])
            T.finalize_reducer(accumulator, result)
            T.copy(result, output)

    updates = [call for call in nodes(kernel, ir.Call) if call.op.name == "tl.reducer_update"]
    assert len(updates) == 1
    assert updates[0].args[1].ty == ir.PrimType("float32")


def apply(name, module):
    result = tvm_ffi.get_global_func(f"tl.transform.{name}")()(module)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def tile(name, *arguments, attrs=None):
    return ir.Evaluate(ir.Call(f"tl.tileop.{name}", list(arguments), attrs=attrs or {}))


def allocate(reducer, operation="sum", legacy=False):
    call = ir.Call("tirx.alloc_tensor", [], attrs={"tl.reducer_op": operation, "tl.reducer_legacy": legacy}, ty_args=[reducer.tensor.ty])
    return ir.Bind(reducer.tensor, call)


def wrap(body, parameters=(), target="c"):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([]))
    function = function.with_attr("target", Target(target)).with_attr("global_symbol", "reducer_test")
    return ir.IRModule({"main": function})


def nodes(root, node_type):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


def cuda_target():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    root = Path(__file__).resolve().parents[3]
    return Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {
                "cuda": {"compiler": "nvcc", "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty' / 'cutlass' / 'include'}"]}
            },
        },
        host="c",
    )


def combine(operation, first, second):
    return {
        "sum": lambda: first + second,
        "max": lambda: prim.Max(first, second),
        "min": lambda: prim.Min(first, second),
        "bitand": lambda: first & second,
        "bitor": lambda: first | second,
        "bitxor": lambda: first ^ second,
    }[operation]()


def identity(operation, dtype):
    if operation in ("sum", "bitor", "bitxor"):
        return ir.const(0, dtype)
    if operation == "bitand":
        return ir.const(-1, dtype)
    if operation == "max":
        return ir.const(-float("inf") if dtype.startswith("float") else np.iinfo(dtype).min, dtype)
    return ir.const(float("inf") if dtype.startswith("float") else np.iinfo(dtype).max, dtype)


@pytest.mark.parametrize(
    "operation,dtype",
    [
        ("sum", "float32"),
        ("max", "float32"),
        ("min", "float32"),
        ("sum", "int32"),
        ("bitand", "int32"),
        ("bitor", "int32"),
        ("bitxor", "int32"),
    ],
)
@pytest.mark.parametrize("baseline", [False, True])
def test_reducer_wide_and_narrow_execute(operation, dtype, baseline, native_layouts, tilelang_builders):
    target = cuda_target()
    source = TensorView.create("source", [2, 32], dtype)
    output = TensorView.create("output", [2], dtype)
    reducer = TensorView.create("reducer", [2], dtype, scope="local.reducer")
    destination = TensorView.create("destination", [2], dtype, scope="local.fragment")
    seed = TensorView.create("seed", [1], dtype, scope="local")
    thread = ir.Var("thread", "int32")
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    layout = native_layouts.Fragment(
        [2, 32], forward_thread_fn=lambda row, column: row * 32 + column, forward_index_fn=lambda row, column: 0
    )
    update = ir.Evaluate(ir.Call("tl.reducer_update", [reducer.load(row), source.load([row, column])]))
    loop = ir.For(
        row,
        0,
        2,
        1,
        ir.ForKind.PARALLEL,
        ir.For(column, 0, 32, 1, ir.ForKind.PARALLEL, update),
        annotations={"parallel_loop_layout": layout},
    )
    body = [
        allocate(reducer, operation),
        destination.allocate(),
        seed.allocate(),
        seed.store(0, 3),
        tile("reducer_init", reducer.tensor, seed.load(0)),
        loop,
        seed.store(0, 99),
        tile("finalize_reducer_v2", reducer.tensor, destination.tensor),
        tile("copy", destination.tensor, output.tensor),
    ]
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(64)], [thread], None, body)
    module = wrap(body, [source.tensor, output.tensor], target)
    with target, PassContext(config={"tl.reducer_force_baseline": baseline}):
        module = apply("CanonicalizeLegacyReducer", module)
        module = apply("VerifyReducerEpoch", module)
        module = apply("LayoutInference", module)
        hints = [
            call
            for call in nodes(module, ir.Call)
            if isinstance(call.op, ir.Op) and call.op.name == "tl.layout_hint" and call.args[0].same_as(reducer.tensor)
        ]
        assert len(hints) == 1
        assert isinstance(hints[0].attrs["layout"], native_layouts.PartialFragment)
        if not baseline:
            assert [loop.var.name for loop in nodes(module, ir.For)] == ["column", "row"]
        module = apply("ReducerPlanAndMaterialize", module)
        calls = [call for call in nodes(module, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == "tl.tileop.finalize_reducer"]
        assert len(calls) == 1
        assert int(calls[0].args[2]) == (64 if baseline else 32)
        module = apply("LowerTileOp", module)
        module = apply("LowerAccessPtr", module)
        module = apply("VerifyReducerConsumed", module)
    executable = tirx.compile(module, target=target)
    values = (np.arange(64).reshape(2, 32) % 15 + 1).astype(dtype)
    actual = runtime.empty((2,), dtype, device=runtime.cuda(0))
    executable(runtime.tensor(values, device=runtime.cuda(0)), actual)
    reductions = {
        "sum": np.add,
        "max": np.maximum,
        "min": np.minimum,
        "bitand": np.bitwise_and,
        "bitor": np.bitwise_or,
        "bitxor": np.bitwise_xor,
    }
    expected = reductions[operation](reductions[operation].reduce(values, axis=1), np.array(3, dtype=dtype))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("operation", ["sum", "max", "min"])
def test_legacy_reducer_canonicalizes_without_legacy_nodes(operation):
    reducer = TensorView.create("reducer", [1], "int32", scope="local.fragment")
    output = TensorView.create("output", [1], "int32")
    index = ir.Var("index", "int32")
    update = reducer.store(0, combine(operation, reducer.load(0), index))
    loop = ir.For(index, 0, 8, 1, ir.ForKind.PARALLEL, update)
    body = [
        allocate(reducer, operation, legacy=True),
        tile("fill", reducer.tensor, identity(operation, "int32")),
        loop,
        tile("finalize_reducer", reducer.tensor),
        tile("copy", reducer.tensor, output.tensor),
    ]
    result = apply("CanonicalizeLegacyReducer", wrap(body, [output.tensor]))
    result = apply("VerifyReducerEpoch", result)
    calls = [call.op.name for call in nodes(result, ir.Call) if isinstance(call.op, ir.Op)]
    assert "tl.tileop.reducer_init" in calls
    assert "tl.reducer_update" in calls
    assert "tl.tileop.finalize_reducer_v2" in calls
    assert "tl.tileop.finalize_reducer" not in calls


@pytest.mark.parametrize(
    "case,match",
    [
        ("no_init", "never initialized"),
        ("no_finalize", "never finalized"),
        ("double_init", "double reducer_init"),
        ("update_before_init", "after reducer_init"),
        ("update_after_finalize", "before finalize_reducer"),
        ("update_outside_parallel", "outside any T.Parallel"),
        ("plain_load", "Ordinary access"),
        ("plain_store", "Ordinary access"),
        ("finalize_inner_loop", "same iteration"),
        ("init_inside_parallel", "inside a T.Parallel"),
    ],
)
def test_reducer_lifecycle_rejects_invalid_programs(case, match):
    reducer = TensorView.create("reducer", [1], "int32", scope="local.reducer")
    destination = TensorView.create("destination", [1], "int32", scope="local.fragment")
    index = ir.Var("index", "int32")
    init = tile("reducer_init", reducer.tensor)
    update = ir.Evaluate(ir.Call("tl.reducer_update", [reducer.load(0), index]))
    loop = ir.For(index, 0, 8, 1, ir.ForKind.PARALLEL, update)
    finalize = tile("finalize_reducer_v2", reducer.tensor, destination.tensor)
    variants = {
        "no_init": [],
        "no_finalize": [init],
        "double_init": [init, init, finalize],
        "update_before_init": [loop, init, finalize],
        "update_after_finalize": [init, finalize, loop],
        "update_outside_parallel": [init, ir.Evaluate(ir.Call("tl.reducer_update", [reducer.load(0), ir.const(1)])), finalize],
        "plain_load": [init, destination.store(0, reducer.load(0)), finalize],
        "plain_store": [init, reducer.store(0, 0), finalize],
        "finalize_inner_loop": [init, ir.For(index, 0, 2, 1, ir.ForKind.DEFAULT, finalize)],
        "init_inside_parallel": [ir.For(index, 0, 2, 1, ir.ForKind.PARALLEL, init), finalize],
    }
    body = [allocate(reducer), destination.allocate(), *variants[case]]
    with pytest.raises(ValueError, match=match):
        apply("VerifyReducerEpoch", wrap(body))


def test_reducer_requires_allocation_metadata():
    reducer = TensorView.create("reducer", [1], "int32", scope="local.reducer")
    with pytest.raises(ValueError, match="tl.reducer_op"):
        apply("VerifyReducerEpoch", wrap([reducer.allocate()]))


def test_legacy_nonidentity_sum_requires_explicit_seed():
    reducer = TensorView.create("reducer", [1], "int32", scope="local.fragment")
    body = [allocate(reducer, legacy=True), tile("fill", reducer.tensor, ir.const(3)), tile("finalize_reducer", reducer.tensor)]
    with pytest.raises(ValueError, match="non-zero"):
        apply("CanonicalizeLegacyReducer", wrap(body))


@pytest.mark.parametrize("target_kind", ["c", "cuda"])
@pytest.mark.parametrize("unroll", [False, True])
@pytest.mark.parametrize("operation", ["sum", "max", "bitxor"])
def test_reducer_seed_survives_reopened_and_duplicated_epochs(target_kind, unroll, operation, tilelang_builders):
    target = cuda_target() if target_kind == "cuda" else Target("c")
    device = runtime.cuda(0) if target_kind == "cuda" else runtime.cpu(0)
    source = TensorView.create("source", [3, 2, 32], "int32")
    output = TensorView.create("output", [3, 2], "int32")
    reducer = TensorView.create("reducer", [2], "int32", scope="local.reducer")
    destination = TensorView.create("destination", [2], "int32", scope="local.fragment")
    seed = TensorView.create("seed", [1], "int32", scope="local")
    epoch = ir.Var("epoch", "int32")
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    publish = ir.Var("publish", "int32")
    update = ir.Evaluate(ir.Call("tl.reducer_update", [reducer.load(row), source.load([epoch, row, column])]))
    reduction = ir.For(row, 0, 2, 1, ir.ForKind.PARALLEL, ir.For(column, 0, 32, 1, ir.ForKind.PARALLEL, update))
    body = [
        seed.store(0, epoch + 3),
        tile("reducer_init", reducer.tensor, seed.load(0)),
        reduction,
        seed.store(0, 99),
        tile("finalize_reducer_v2", reducer.tensor, destination.tensor),
        ir.For(publish, 0, 2, 1, ir.ForKind.PARALLEL, output.store([epoch, publish], destination.load(publish))),
    ]
    loop = ir.For(
        epoch, 0, 3, 1, ir.ForKind.UNROLLED if unroll else ir.ForKind.DEFAULT, body, {"pragma_unroll_explicit": True} if unroll else {}
    )
    body = [allocate(reducer, operation), destination.allocate(), seed.allocate(), loop]
    if target_kind == "cuda":
        thread = ir.Var("thread", "int32")
        body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(64)], [thread], None, body)
    module = wrap(body, [source.tensor, output.tensor], target)
    with target:
        module = apply("VerifyReducerEpoch", module)
        if unroll:
            module = apply("UnrollLoop", module)
        for name in ("LayoutInference", "ReducerPlanAndMaterialize", "LowerTileOp", "LowerAccessPtr", "VerifyReducerConsumed"):
            module = apply(name, module)
    executable = tirx.compile(module, target=target)
    values = (np.arange(192).reshape(3, 2, 32) % 19).astype("int32")
    actual = runtime.empty((3, 2), "int32", device=device)
    executable(runtime.tensor(values, device=device), actual)
    reduction = {"sum": np.add, "max": np.maximum, "bitxor": np.bitwise_xor}[operation]
    expected = reduction(reduction.reduce(values, axis=-1), np.arange(3, 6, dtype="int32")[:, None])
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("target_kind", ["c", "cuda"])
@pytest.mark.parametrize("operation", ["sum", "max", "min"])
def test_reducer_zero_updates_still_publishes_seed(target_kind, operation, tilelang_builders):
    target = cuda_target() if target_kind == "cuda" else Target("c")
    device = runtime.cuda(0) if target_kind == "cuda" else runtime.cpu(0)
    output = TensorView.create("output", [2], "int32")
    reducer = TensorView.create("reducer", [2], "int32", scope="local.reducer")
    destination = TensorView.create("destination", [2], "int32", scope="local.fragment")
    body = [
        allocate(reducer, operation),
        destination.allocate(),
        tile("reducer_init", reducer.tensor, ir.const(7)),
        tile("finalize_reducer_v2", reducer.tensor, destination.tensor),
        tile("copy", destination.tensor, output.tensor),
    ]
    if target_kind == "cuda":
        thread = ir.Var("thread", "int32")
        body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(64)], [thread], None, body)
    module = wrap(body, [output.tensor], target)
    with target:
        for name in (
            "VerifyReducerEpoch",
            "LayoutInference",
            "ReducerPlanAndMaterialize",
            "LowerTileOp",
            "LowerAccessPtr",
            "VerifyReducerConsumed",
        ):
            module = apply(name, module)
    executable = tirx.compile(module, target=target)
    actual = runtime.empty((2,), "int32", device=device)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.full(2, 7, dtype="int32"))
