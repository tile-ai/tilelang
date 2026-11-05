"""Native reducer epochs preserve capture, multiplicity, scopes and physical plans."""

import os
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape, dtype="float32", scope="local"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def region(value):
    return ir.TensorRegion(value, [ir.Range(0, extent) for extent in value.ty.shape], ir.TensorRegionType())


def tileop(name, *operands, metadata=None):
    return ir.Evaluate(
        tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"), ir.Op.get("tl.tileop." + name), list(operands), metadata or {}, ir.UnknownLoc()
        )
    )


def annotation(values):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def attribute(node, key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": node, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def decoded(call):
    count = int(call.attrs["operand_count"])
    return list(call.args[:count]), tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[count], call.attrs["annotation_schema"])


def annotations(value):
    result = []
    for call in objects(value, ir.Call):
        if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation":
            result.append(tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"]))
    return result


def update(acc, indices, value):
    return ir.Evaluate(ir.Call("tl.reducer_update", [ir.TensorLoad(acc, indices), value]))


def loop(variable, extent, body, parallel=True):
    return ir.For(variable, integer(0), integer(extent), integer(1), ir.ForKind.PARALLEL if parallel else ir.ForKind.DEFAULT, body)


def function(params, body, target="llvm"):
    target = tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"} if target == "llvm" else {"kind": "cuda", "arch": "sm_90"})
    return tirx.Function(params, body, ir.NoneType()).with_attr("target", target).with_attr("global_symbol", "main")


def run_pass(name, value):
    with value.attrs["target"]:
        return tvm_ffi.get_global_func("tl.transform." + name)()(ir.IRModule({"main": value}))["main"]


def objects(value, cls):
    result = []
    tvm_ffi.structural_walk(value, (cls, result.append), order="pre")
    return result


def lower(value):
    original = ir.save_json(value)
    stages = {}
    if value.attrs["target"].kind.name in {"llvm", "c"}:
        # Match the CPU pipeline: consume source allocation origins before
        # the reducer passes and the eventual FinalizeAnnotations boundary.
        with value.attrs["target"]:
            value = tvm_ffi.get_global_func("tl.cpu.transform.LowerCPUKernelLaunch")()(ir.IRModule({"main": value}))["main"]
        stages["LowerCPUKernelLaunch"] = value
    for name in (
        "CanonicalizeLegacyReducer",
        "VerifyReducerEpoch",
        "LayoutInference",
        "ReducerPlanAndMaterialize",
        "LowerTileOp",
        "VerifyReducerConsumed",
    ):
        value = run_pass(name, value)
        stages[name] = value
        assert tirx.analysis.check_well_formed(value), name
        assert not tirx.analysis.undefined_vars(value), name
    assert not any(v.ty.scope.name == "local.reducer" for v in objects(value, ir.Var) if isinstance(v.ty, tirx.TensorType))
    assert not any("reducer" in call.op.name for call in objects(value, ir.Call) if isinstance(call.op, ir.Op))
    return value, stages, original


def execute(value, arrays, name):
    artifact_dir = os.environ.get("TILELANG_OPS_ARTIFACT_DIR")
    if artifact_dir:
        directory = Path(artifact_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (name + ".json")).write_text(ir.save_json(value))
    from tilelang.testing.packed import compile_packed_function

    # BindTarget preserves an existing target. Retarget this execution-only
    # copy explicitly; the original lowered IR and its assertions are kept.
    module = ir.IRModule({"main": value.with_attr("target", tvm_tirx.target.Target("c"))})
    module = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("c")(module)
    executable = compile_packed_function(module)
    tensors = [tvm_tirx.runtime.tensor(a) for a in arrays]
    executable(*tensors)
    return [a.numpy() for a in tensors]


def row_reduction(op="sum", dtype="float32", seed=None, serial_tiles=False):
    a, b = tensor("A", [3, 8], dtype, "global"), tensor("B", [3], dtype, "global")
    acc, dst = tensor("acc", [3], dtype, "local.reducer"), tensor("dst", [3], dtype, "local.fragment")
    i, j, tile = (ir.Var(name, "int32") for name in ("i", "j", "tile"))
    body = [allocate(acc), allocate(dst), annotation({"reducer_info_v2": {acc: {"op": op}}})]
    init_args = [region(acc)]
    if seed is not None:
        init_args.append(ir.FloatImm(dtype, seed) if dtype.startswith("float") else integer(seed, dtype))
    body.append(tileop("reducer_init", *init_args))
    column = tile * 4 + j if serial_tiles else j
    updates = loop(i, 3, [loop(j, 4 if serial_tiles else 8, [update(acc, [i], ir.TensorLoad(a, [i, column]))])])
    body.append(loop(tile, 2, [updates], parallel=False) if serial_tiles else updates)
    body.extend([tileop("finalize_reducer_v2", region(acc), region(dst)), tileop("copy", region(dst), region(b))])
    return function([a, b], body)


@pytest.mark.parametrize("serial_tiles", [False, True])
@pytest.mark.parametrize("op,seed", [("sum", None), ("sum", 5.0), ("max", None), ("min", None)])
def test_cpu_deferred_reducer_numeric(op, seed, serial_tiles):
    value = row_reduction(op, seed=seed, serial_tiles=serial_tiles)
    before = ir.save_json(value)
    result, _, _ = lower(value)
    assert before == ir.save_json(value)
    inputs = np.arange(24, dtype="float32").reshape(3, 8) - 11
    actual = execute(result, [inputs, np.zeros(3, dtype="float32")], f"rows-{op}-{seed}-{serial_tiles}")[-1]
    expected = {"sum": np.sum, "max": np.max, "min": np.min}[op](inputs, axis=1)
    if seed is not None:
        expected += seed
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("op", ["sum", "bitand", "bitor", "bitxor"])
def test_cpu_integer_reducers(op):
    value, _, _ = lower(row_reduction(op, "int32"))
    inputs = np.arange(24, dtype="int32").reshape(3, 8) * 3 + 5
    expected = (
        np.sum(inputs, axis=1)
        if op == "sum"
        else {"bitand": np.bitwise_and, "bitor": np.bitwise_or, "bitxor": np.bitwise_xor}[op].reduce(inputs, axis=1)
    )
    actual = execute(value, [inputs, np.zeros(3, dtype="int32")], "int-" + op)[-1]
    np.testing.assert_array_equal(actual, expected)


def test_seed_is_captured_before_its_source_changes():
    a, b = tensor("A", [8], scope="global"), tensor("B", [1], scope="global")
    acc, dst, seed = tensor("acc", [1], scope="local.reducer"), tensor("dst", [1]), tensor("seed", [1])
    i = ir.Var("i", "int32")
    value = function(
        [a, b],
        [
            allocate(acc),
            allocate(dst),
            allocate(seed),
            annotation({"reducer_info_v2": {acc: {"op": "sum"}}}),
            ir.TensorStore(seed, [0], ir.FloatImm("float32", 5)),
            tileop("reducer_init", region(acc), ir.TensorLoad(seed, [0])),
            ir.TensorStore(seed, [0], ir.FloatImm("float32", 100)),
            loop(i, 8, [update(acc, [0], ir.TensorLoad(a, [i]))]),
            tileop("finalize_reducer_v2", region(acc), region(dst)),
            tileop("copy", region(dst), region(b)),
        ],
    )
    result, stages, _ = lower(value)
    captures = [b for b in objects(stages["ReducerPlanAndMaterialize"], ir.Bind) if b.var.name == "acc_seed"]
    assert len(captures) == 1
    inputs = np.arange(1, 9, dtype="float32")
    actual = execute(result, [inputs, np.zeros(1, dtype="float32")], "seed-capture")[-1]
    np.testing.assert_array_equal(actual, [41.0])


def epoch_fixture():
    acc, dst = tensor("acc", [1], scope="local.reducer"), tensor("dst", [1])
    i = ir.Var("i", "int32")
    prefix = [allocate(acc), allocate(dst), annotation({"reducer_info_v2": {acc: {"op": "sum"}}})]
    init = tileop("reducer_init", region(acc))
    updates = loop(i, 8, [update(acc, [0], ir.FloatImm("float32", 1))])
    finalize = tileop("finalize_reducer_v2", region(acc), region(dst))
    return acc, dst, prefix, init, updates, finalize


@pytest.mark.parametrize(
    "problem,match",
    [
        ("uninitialized", "never initialized"),
        ("unfinalized", "never finalized"),
        ("early_update", "before T.reducer_init"),
        ("late_update", "after T.finalize_reducer"),
        ("double_init", "double T.reducer_init"),
        ("double_finalize", "double T.finalize_reducer"),
        ("store", "illegal store"),
        ("load", "illegal read"),
        ("alias", "illegal use"),
        ("update_outside_parallel", "outside any T.Parallel"),
        ("init_in_parallel", "may not appear inside"),
        ("finalize_in_loop", "not in the scope"),
        ("different_branch", "not in the scope"),
        ("seed_dtype", "seed dtype"),
    ],
)
def test_invalid_epoch_has_specific_diagnostic(problem, match):
    acc, _, prefix, init, updates, finalize = epoch_fixture()
    i = ir.Var("outer", "int32")
    bodies = {
        "uninitialized": [],
        "unfinalized": [init, updates],
        "early_update": [updates, init, finalize],
        "late_update": [init, finalize, updates],
        "double_init": [init, init, finalize],
        "double_finalize": [init, finalize, finalize],
        "store": [init, ir.TensorStore(acc, [0], ir.FloatImm("float32", 1)), finalize],
        "load": [init, ir.Evaluate(ir.TensorLoad(acc, [0])), finalize],
        "alias": [init, ir.Bind(tensor("alias", [1], scope="local.reducer"), acc), finalize],
        "update_outside_parallel": [init, update(acc, [0], ir.FloatImm("float32", 1)), finalize],
        "init_in_parallel": [loop(i, 2, [init]), finalize],
        "finalize_in_loop": [init, loop(i, 2, [finalize], parallel=False)],
        "different_branch": [ir.If(integer(1, "bool"), [init], [finalize])],
        "seed_dtype": [tileop("reducer_init", region(acc), integer(5)), finalize],
    }
    with pytest.raises((ValueError, TypeError), match=match):
        run_pass("VerifyReducerEpoch", function([], prefix + bodies[problem]))


@pytest.mark.parametrize("conditional", [False, True])
def test_epoch_nested_in_serial_loop_and_conditional_refinement(conditional):
    _, _, prefix, init, updates, finalize = epoch_fixture()
    i = ir.Var("epoch", "int32")
    finish = ir.If(i < 2, [finalize], None) if conditional else finalize
    value = function([], prefix + [loop(i, 3, [init, updates, finish], parallel=False)])
    run_pass("VerifyReducerEpoch", value)


def test_consumed_verifier_rejects_first_class_reducer_leftovers():
    _, _, prefix, init, updates, finalize = epoch_fixture()
    with pytest.raises(ValueError, match="survived past"):
        run_pass("VerifyReducerConsumed", function([], prefix + [init, updates, finalize]))


@pytest.mark.parametrize("legacy_scope", ["local.fragment", "local.reducer"])
def test_legacy_zero_sum_is_canonicalized_and_executes(legacy_scope):
    a, b = tensor("A", [8], scope="global"), tensor("B", [1], scope="global")
    acc = tensor("legacy", [1], scope=legacy_scope)
    i = ir.Var("i", "int32")
    key = "reducer_info" if legacy_scope == "local.fragment" else "reducer_info_v2"
    value = function(
        [a, b],
        [
            allocate(acc),
            annotation({key: {acc: {"op": "sum", "rep": "all"}}}),
            tileop("fill", region(acc), ir.FloatImm("float32", 0)),
            loop(i, 8, [ir.TensorStore(acc, [0], ir.TensorLoad(acc, [0]) + ir.TensorLoad(a, [i]))]),
            tileop("finalize_reducer", region(acc)),
            tileop("copy", region(acc), region(b)),
        ],
    )
    result, stages, _ = lower(value)
    assert "reducer_info_v2" in stages["CanonicalizeLegacyReducer"].script()
    values = np.arange(1, 9, dtype="float32")
    actual = execute(result, [values, np.zeros(1, dtype="float32")], "legacy-" + legacy_scope)[-1]
    np.testing.assert_array_equal(actual, [36.0])


def test_legacy_nonzero_sum_fill_is_rejected_instead_of_changing_multiplicity():
    acc = tensor("legacy", [1], scope="local.fragment")
    value = function(
        [],
        [
            allocate(acc),
            annotation({"reducer_info": {acc: {"op": "sum", "rep": "all"}}}),
            tileop("fill", region(acc), ir.FloatImm("float32", 5)),
            tileop("finalize_reducer", region(acc)),
        ],
    )
    with pytest.raises(ValueError, match="physical thread count"):
        run_pass("CanonicalizeLegacyReducer", value)


def test_public_dsl_reducer_keeps_seed_and_fragment_semantics():
    from tilelang import language as T

    @T.prim_func
    def kernel(A: T.Tensor((8,), "float32"), B: T.Tensor((1,), "float32")):
        seed = T.alloc_local((1,), "float32")
        acc = T.alloc_reducer((1,), "float32", op="sum")
        result = T.alloc_fragment((1,), "float32")
        seed[0] = 5.0
        T.reducer_init(acc, seed[0])
        seed[0] = 100.0
        for i in T.Parallel(8):
            T.reducer_update(acc[0], A[i])
        T.finalize_reducer(acc, result)
        T.copy(result, B)

    target = tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"})
    result, _, _ = lower(kernel.with_attr("target", target).with_attr("global_symbol", "main"))
    actual = execute(result, [np.arange(1, 9, dtype="float32"), np.zeros(1, dtype="float32")], "public-dsl-reducer")[-1]
    np.testing.assert_array_equal(actual, [41.0])
