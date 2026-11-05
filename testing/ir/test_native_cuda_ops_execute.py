"""Real TileLang CUDA operators, NVCC, runtime loading and numerical oracles."""

import json

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.layout import Fragment

from test_native_cuda_copy import lower
from test_native_cuda_copy_execute import build_cuda_copy
from test_native_cuda_ops import cuda_function, gemm
from test_native_reducer_contracts import allocate, annotation, integer, region, run_pass, tensor, tileop


def execute(value, arrays, directory, *, already_lowered=False):
    assert tvm_tirx.cuda(0).exist, "A CUDA device is required for this execution suite"
    if not already_lowered:
        value = lower(value, "batch7-" + directory.name)
    value = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": value}))["main"]
    value = run_pass("LegalizeSafeMemoryAccess", value)
    module, source = build_cuda_copy(value, directory.resolve())
    device = tvm_tirx.cuda(0)
    tensors = [tvm_tirx.runtime.tensor(array, device) for array in arrays]
    module(*tensors)
    return [value.numpy() for value in tensors], source


def record(directory, case, **metadata):
    (directory / "numeric.json").write_text(json.dumps({"case": case, "seed": 7421, "passed": True, **metadata}, indent=2))


def sync():
    return ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")]))


@pytest.mark.parametrize("scope,size,value", [("global", 257, 3.25), ("shared", 128, 0.0), ("shared", 257, -0.0)])
def test_fill_native_gpu(scope, size, value, tmp_path):
    output = tensor("B", [size], scope="global")
    filled = output if scope == "global" else tensor("S", [size], scope=scope)
    body = [] if scope == "global" else [allocate(filled)]
    body.append(tileop("fill", region(filled), ir.FloatImm("float32", value)))
    if scope != "global":
        body.extend([sync(), tileop("copy", region(filled), region(output), metadata={"prefer_instruction": "sync"})])
    result, source = execute(cuda_function([output], body), [np.ones(size, "float32")], tmp_path)
    expected = np.full(size, value, "float32")
    np.testing.assert_array_equal(result[-1].view("uint32"), expected.view("uint32"))
    if scope == "shared" and size == 128:
        assert "st_bulk_shared" in source
    record(tmp_path, "fill", scope=scope, size=size, value=value, oracle="bitwise float32 equality, including sign of zero")


@pytest.mark.parametrize("kind,scalar", [("atomicadd", False), ("atomicadd", True), ("atomicmax", False), ("atomicmin", False)])
def test_atomic_native_gpu(kind, scalar, tmp_path):
    size = 257
    a, b = tensor("A", [size], scope="global"), tensor("B", [size], scope="global")
    source = ir.FloatImm("float32", 3) if scalar else region(a)
    value = cuda_function([a, b], [tileop(kind, source, region(b), metadata={"memory_order": integer(5)})], blocks=2)
    rng = np.random.default_rng(7421)
    inputs = rng.integers(-8, 9, size).astype("float32")
    initial = np.ones(size, "float32")
    results, _ = execute(value, [inputs, initial], tmp_path)
    expected = (
        initial + 2 * (3 if scalar else inputs)
        if kind == "atomicadd"
        else (np.maximum if kind == "atomicmax" else np.minimum)(initial, inputs)
    )
    np.testing.assert_array_equal(results[-1], expected)
    record(tmp_path, kind, scalar_source=scalar, shape=[size], concurrent_blocks=2, oracle="exact float32 integer RMW values")


@pytest.mark.parametrize("kind,columns", [("sum", 16), ("sum", 256), ("max", 64), ("min", 64)])
def test_reduce_native_gpu(kind, columns, tmp_path):
    rows = 8
    a, b = tensor("A", [rows, columns], scope="global"), tensor("B", [rows], scope="global")
    fragment = tensor("F", [rows, columns], scope="local.fragment")
    reduced = tensor("R", [rows], scope="local.fragment")
    body = [
        allocate(fragment),
        allocate(reduced),
        tileop("copy", region(a), region(fragment), metadata={"prefer_instruction": "sync"}),
        tileop("reduce", region(fragment), region(reduced), ir.StringImm(kind), integer(1), integer(1, "bool")),
        tileop("copy", region(reduced), region(b), metadata={"prefer_instruction": "sync"}),
    ]
    inputs = np.random.default_rng(7421).integers(-4, 5, (rows, columns)).astype("float32")
    results, source = execute(cuda_function([a, b], body), [inputs, np.zeros(rows, "float32")], tmp_path)
    expected = {"sum": np.sum, "min": np.min, "max": np.max}[kind](inputs, axis=1)
    np.testing.assert_array_equal(results[-1], expected)
    assert "__global__" in source
    record(tmp_path, "reduce-" + kind, shape=[rows, columns], oracle="exact float32 integer reduction")


@pytest.mark.parametrize("transpose_b", [False, True])
def test_mma_gemm_native_gpu(transpose_b, tmp_path):
    # This native operator test needs the production instruction registry.
    # Full BackendModule/adapter registration has its own integration tests;
    # its eager loading of unrelated adapters is not part of this fixture.
    import tilelang.cuda.op  # noqa: F401

    m = n = k = 32
    bshape = [n, k] if transpose_b else [k, n]
    a, b, c = tensor("A", [m, k], "float16", "global"), tensor("B", bshape, "float16", "global"), tensor("C", [m, n], scope="global")
    sa, sb = tensor("SA", [m, k], "float16", "shared"), tensor("SB", bshape, "float16", "shared")
    acc = tensor("acc", [m, n], scope="local.fragment")
    body = [
        allocate(sa),
        allocate(sb),
        allocate(acc),
        tileop("copy", region(a), region(sa), metadata={"prefer_instruction": "sync"}),
        tileop("copy", region(b), region(sb), metadata={"prefer_instruction": "sync"}),
        sync(),
        gemm(sa, sb, acc, m, n, k, transpose_b=transpose_b),
        tileop("copy", region(acc), region(c), metadata={"prefer_instruction": "sync"}),
    ]
    rng = np.random.default_rng(7421)
    av = rng.integers(-3, 4, (m, k)).astype("float16")
    bv = rng.integers(-3, 4, bshape).astype("float16")
    results, source = execute(cuda_function([a, b, c], body), [av, bv, np.zeros((m, n), "float32")], tmp_path)
    expected = av.astype("float32") @ (bv.T if transpose_b else bv).astype("float32")
    np.testing.assert_array_equal(results[-1], expected)
    assert "mma" in source
    record(
        tmp_path,
        "gemm-mma",
        shape=[m, n, k],
        transpose_b=transpose_b,
        oracle="exact float32 accumulation of small integer float16 matrices",
    )


@pytest.mark.parametrize("partial,offset_layout", [(True, False), (False, True)])
def test_shared_fill_preserves_region_and_view_origin(partial, offset_layout, tmp_path):
    shape = [4, 16]
    output = tensor("B", shape, scope="global")
    if offset_layout:
        memory = tirx.TileLayout(tirx.layout.S[(4, 16) : (32, 1)] + 8)
        shared = ir.Var("S", tirx.TensorType(shape, "float32", scope="shared", layout=memory))
    else:
        shared = tensor("S", shape, scope="shared")
    selected = (
        ir.TensorRegion(shared, [ir.Range.from_min_extent(1, 2), ir.Range.from_min_extent(4, 8)], ir.TensorRegionType())
        if partial
        else region(shared)
    )
    body = [
        allocate(shared),
        tileop("fill", region(shared), ir.FloatImm("float32", 7)),
        sync(),
        tileop("fill", selected, ir.FloatImm("float32", 0)),
        sync(),
        tileop("copy", region(shared), region(output), metadata={"prefer_instruction": "sync"}),
    ]
    results, source = execute(cuda_function([output], body), [np.empty(shape, "float32")], tmp_path)
    expected = np.full(shape, 7 if partial else 0, "float32")
    if partial:
        expected[1:3, 4:12] = 0
    np.testing.assert_array_equal(results[-1], expected)
    assert "st_bulk_shared" not in source
    record(tmp_path, "fill-view", partial=partial, offset_layout=offset_layout, oracle="exact values and untouched region")


@pytest.mark.parametrize("batch,clear,dtype", [(2, True, "float32"), (1, False, "float32"), (2, True, "float16")])
def test_reduce_batch_and_existing_accumulator_native_gpu(batch, clear, dtype, tmp_path):
    rows, columns = 8, 256
    threads = 128
    a, b = tensor("A", [rows, columns], dtype, "global"), tensor("B", [rows], dtype, "global")
    fragment, reduced = tensor("F", [rows, columns], dtype, "local.fragment"), tensor("R", [rows], dtype, "local.fragment")
    # Exercise the cross-warp batch path explicitly: all 128 threads reduce
    # each row, and every thread owns eight output elements. Automatic layouts
    # may select a warp-only reduction or fewer per-thread outputs.
    layout = Fragment(
        [rows, columns],
        forward_fn=lambda i, j: (j % threads, i * (columns // threads) + j // threads),
    )
    body = [allocate(fragment), allocate(reduced), annotation({"layout_map": {fragment: layout}})]
    if not clear:
        body.append(tileop("fill", region(reduced), ir.FloatImm(dtype, 3)))
    body.extend(
        [
            tileop("copy", region(a), region(fragment), metadata={"prefer_instruction": "sync"}),
            tileop(
                "reduce",
                region(fragment),
                region(reduced),
                ir.StringImm("sum"),
                integer(1),
                integer(int(clear), "bool"),
                metadata={"batch": integer(batch)},
            ),
            tileop("copy", region(reduced), region(b), metadata={"prefer_instruction": "sync"}),
        ]
    )
    inputs = np.random.default_rng(7421).integers(-2, 3, (rows, columns)).astype(dtype)
    results, source = execute(cuda_function([a, b], body, threads=threads), [inputs, np.zeros(rows, dtype)], tmp_path)
    expected = (inputs.astype("float32").sum(axis=1) + (0 if clear else 3)).astype(dtype)
    np.testing.assert_array_equal(results[-1], expected)
    if batch > 1:
        assert "run_batch" in source
    assert "__shared__" in source
    record(tmp_path, "reduce-batch", batch=batch, clear=clear, dtype=dtype, oracle="exact small integer sum plus prior accumulator")


@pytest.mark.parametrize(
    "name,shape,axis,reverse",
    [
        ("cumsum", [129], 0, False),
        ("cumsum", [129], 0, True),
        ("cummax", [129], 0, False),
        ("cummax", [129], 0, True),
        ("cumsum", [8, 32], 0, False),
        ("cummax", [8, 32], 1, True),
    ],
)
def test_scan_native_gpu(name, shape, axis, reverse, tmp_path):
    a, b = tensor("A", shape, scope="global"), tensor("B", shape, scope="global")
    source, destination = tensor("S", shape, scope="shared"), tensor("D", shape, scope="shared")
    body = [
        allocate(source),
        allocate(destination),
        tileop("copy", region(a), region(source), metadata={"prefer_instruction": "sync"}),
        sync(),
        tileop(name, region(source), region(destination), integer(axis), integer(int(reverse))),
        sync(),
        tileop("copy", region(destination), region(b), metadata={"prefer_instruction": "sync"}),
    ]
    inputs = np.random.default_rng(7421).integers(-4, 5, shape).astype("float32")
    results, source_code = execute(cuda_function([a, b], body), [inputs, np.zeros(shape, "float32")], tmp_path)
    values = np.flip(inputs, axis=axis) if reverse else inputs
    expected = np.cumsum(values, axis=axis) if name == "cumsum" else np.maximum.accumulate(values, axis=axis)
    if reverse:
        expected = np.flip(expected, axis=axis)
    np.testing.assert_array_equal(results[-1], expected)
    assert "tl::Cum" in source_code
    record(tmp_path, name, shape=shape, axis=axis, reverse=reverse, oracle="numpy cumulative exact integer float32 values")


@pytest.mark.parametrize("scope,shape", [("global", [7, 17]), ("shared", [7, 17]), ("global", [2, 7, 17])])
def test_transpose_native_gpu(scope, shape, tmp_path):
    dest_shape = [*shape[:-2], shape[-1], shape[-2]]
    a, b = tensor("A", shape, scope="global"), tensor("B", dest_shape, scope="global")
    source, destination = (a, b) if scope == "global" else (tensor("S", shape, scope="shared"), tensor("D", dest_shape, scope="shared"))
    body = []
    if scope == "shared":
        body.extend(
            [
                allocate(source),
                allocate(destination),
                tileop("copy", region(a), region(source), metadata={"prefer_instruction": "sync"}),
                sync(),
            ]
        )
    body.append(tileop("transpose", region(source), region(destination)))
    if scope == "shared":
        body.extend([sync(), tileop("copy", region(destination), region(b), metadata={"prefer_instruction": "sync"})])
    inputs = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    results, _ = execute(cuda_function([a, b], body), [inputs, np.zeros(dest_shape, "float32")], tmp_path)
    np.testing.assert_array_equal(results[-1], inputs.swapaxes(-1, -2))
    record(tmp_path, "transpose", shape=shape, scope=scope, oracle="numpy last-two-axis permutation")


@pytest.mark.parametrize("name,seed,repetitions", [("sum", 5.0, 2), ("max", None, 1), ("min", None, 1)])
def test_deferred_reducer_finalize_native_gpu(name, seed, repetitions, tmp_path):
    from test_native_reducer_contracts import annotation, loop, lower as lower_reducer, update

    rows, columns = 8, 256
    a, b = tensor("A", [rows, columns], scope="global"), tensor("B", [rows], scope="global")
    acc, result = tensor("acc", [rows], scope="local.reducer"), tensor("R", [rows], scope="local.fragment")
    i, j, rep = [ir.Var(var, "int32") for var in ("i", "j", "rep")]
    init = [region(acc)] + ([] if seed is None else [ir.FloatImm("float32", seed)])
    updates = loop(i, rows, [loop(j, columns, [update(acc, [i], ir.TensorLoad(a, [i, j]))])])
    body = [
        allocate(acc),
        allocate(result),
        annotation({"reducer_info_v2": {acc: {"op": name}}}),
        tileop("reducer_init", *init),
        loop(rep, repetitions, [updates], parallel=False),
        tileop("finalize_reducer_v2", region(acc), region(result)),
        tileop("copy", region(result), region(b), metadata={"prefer_instruction": "sync"}),
    ]
    lowered, stages, _ = lower_reducer(cuda_function([a, b], body))
    for stage_name, stage in stages.items():
        (tmp_path / (stage_name + ".json")).write_text(ir.save_json(stage))
    inputs = np.random.default_rng(7421).integers(-4, 5, (rows, columns)).astype("float32")
    results, _ = execute(lowered, [inputs, np.zeros(rows, "float32")], tmp_path, already_lowered=True)
    expected = {"sum": np.sum, "max": np.max, "min": np.min}[name](inputs, axis=1)
    if name == "sum":
        expected = expected * repetitions + (0 if seed is None else seed)
    np.testing.assert_array_equal(results[-1], expected)
    record(
        tmp_path,
        "finalize-" + name,
        shape=[rows, columns],
        seed_value=seed,
        repetitions=repetitions,
        oracle="exact reduction and one seed contribution per result",
    )
