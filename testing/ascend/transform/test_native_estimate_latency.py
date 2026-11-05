"""Native Ascend latency estimates preserve measured cost and task semantics.

Numeric costs below come from test_estimate_latency.py's pre-migration A5
model. They are model-contract checks, not new NPU timing measurements.
"""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape, dtype="bfloat16", scope="global"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope))


def region(value, shape=None, starts=None):
    shape = value.ty.shape if shape is None else shape
    starts = [0] * len(shape) if starts is None else starts
    return ir.TensorRegion(value, [ir.Range.from_min_extent(start, extent) for start, extent in zip(starts, shape)], ir.TensorRegionType())


def tilecall(name, *operands, metadata=None):
    return tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
        ir.PrimType("void"), ir.Op.get("tl.tileop." + name), list(operands), metadata or {}, ir.UnknownLoc()
    )


def attribute(node, key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": node, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def task(body, metadata=None):
    return attribute(metadata or {}, "tl.ascend_task", integer(1), body)


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def fields(reg):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(reg.args[0], reg.attrs["annotation_schema"])


def estimate(params, body):
    root = attribute(None, "tl.sblock", {"name": "tilelang_root", "no_realize": True}, body)
    before = tirx.Function(params, [root]).with_attr("target", tvm_tirx.target.Target("ascend"))
    assert tirx.analysis.check_well_formed(before)
    source = ir.save_json(before)
    pass_ = tvm_ffi.get_global_func("tl.transform.EstimateLatency")()
    result = pass_(ir.IRModule({"main": before}))["main"]
    assert ir.save_json(before) == source
    assert tirx.analysis.check_well_formed(result)
    for call in objects(result, ir.Call):
        call.validate()
    ir.assert_structural_equal(pass_(ir.IRModule({"main": result}))["main"], result)
    return result


def costs(result, kind="tl.ascend_task"):
    return [
        (int(f["node"]["latency"]), int(f["node"]["ii"]))
        for reg in objects(result, ir.RegionStmt)
        if reg.op.name == "tl.attr" and (f := fields(reg))["key"] == kind
    ]


def gemm_program(m=64, n=64, k=24, dtype="float32", hf32=True, dynamic=False, blockscaled=False, mode_sequence=None):
    a, b = tensor("a", [64, 64], dtype, "shared.l0a"), tensor("b", [64, 64], dtype, "shared.l0b")
    c = tensor("c", [64, 64], "float32", "shared.l0c")
    params = [a, b, c]
    effective_k = k
    if dynamic:
        tail = ir.Var("tail", "int32")
        params.append(tail)
        effective_k = ir.prim.Max(ir.prim.Min(tail, integer(k)), integer(0))
    args = [
        region(a, [m, effective_k]),
        region(b, [n, effective_k]),
        region(c, [m, n]),
        integer(False, "bool"),
        integer(True, "bool"),
        integer(m),
        integer(n),
        integer(k),
        integer(0),
        integer(True, "bool"),
        integer(0),
        integer(0),
        integer(0),
    ]
    name = "gemm"
    if blockscaled:
        sfa, sfb = tensor("sfa", [64, 2], "uint16", "shared.l0a.sf"), tensor("sfb", [64, 2], "uint16", "shared.l0b.sf")
        params += [sfa, sfb]
        args += [region(sfa), region(sfb), integer(0)]
        name = "gemm_blockscaled"
    gemm = ir.Evaluate(tilecall(name, *args))
    mode = ir.Evaluate(ir.Call("tl.ascend_set_hf32_mode", [integer(1 if hf32 else 0)], ty="void"))
    body = [task([mode]), task([gemm])] if dtype == "float32" else [task([gemm])]
    if mode_sequence is not None:
        body, extra = mode_sequence(mode, gemm)
        params += extra
    return params, body


@pytest.mark.parametrize("dynamic", [False, True])
def test_hf32_gemm_uses_effective_k_region(dynamic):
    assert costs(estimate(*gemm_program(dynamic=dynamic)))[-1] == (115, 48)
    assert costs(estimate(*gemm_program(k=32)))[-1] == (131, 64)


def test_fp32_gemm_mode_and_dense_fp8():
    assert costs(estimate(*gemm_program(hf32=False)))[-1] == (451, 384)
    assert costs(estimate(*gemm_program(16, 16, 64, "float8_e4m3fn")))[-1] == (10, 2)


@pytest.mark.parametrize("dtype,k,rounded", [("float32", 17, 24), ("bfloat16", 17, 32), ("float8_e4m3fn", 33, 64)])
def test_gemm_compute_group_rounding(dtype, k, rounded):
    assert costs(estimate(*gemm_program(k=k, dtype=dtype)))[-1] == costs(estimate(*gemm_program(k=rounded, dtype=dtype)))[-1]


@pytest.mark.parametrize(
    "mn,dtype,expected",
    [(16, "float8_e4m3fn", (31, 6)), (64, "float8_e4m3fn", (57, 32)), (16, "float4_e2m1fn", (28, 3)), (64, "float4_e2m1fn", (41, 16))],
)
def test_blockscaled_compute_cost_matches_frozen_model(mn, dtype, expected):
    assert costs(estimate(*gemm_program(mn, mn, 64, dtype, blockscaled=True)))[-1] == expected


def test_gemm_mn_fractal_rounding():
    assert costs(estimate(*gemm_program(24, 17, 32)))[-1] == costs(estimate(*gemm_program(32, 32, 32)))[-1]


def test_unknown_branch_hf32_mode_is_conservative_but_same_guard_correlates():
    def program(mode, gemm):
        cond = ir.Var("cond", "bool")
        return [ir.If(cond, [task([mode])], None), task([gemm])], [cond]

    assert costs(estimate(*gemm_program(mode_sequence=program)))[-1] == (451, 384)

    def correlated(mode, gemm):
        cond = ir.Var("cond", "bool")
        return [ir.If(cond, [task([mode])], None), ir.If(cond, [task([gemm])], None)], [cond]

    assert costs(estimate(*gemm_program(mode_sequence=correlated)))[-1] == (115, 48)


@pytest.mark.parametrize(
    "dtype,scope,expected",
    [
        ("float32", "shared.dyn", (570, 514)),
        ("float32", "global", (712, 514)),
        ("bfloat16", "shared.dyn", (314, 258)),
        ("bfloat16", "global", (456, 258)),
    ],
)
def test_fixpipe_destination_bytes(dtype, scope, expected):
    a, b = tensor("a", [128, 128], "float32", "shared.l0c"), tensor("b", [128, 128], dtype, scope)
    result = estimate([a, b], [task([ir.Evaluate(tilecall("ascend_copy", region(a), region(b)))])])
    assert costs(result) == [expected]


@pytest.mark.parametrize("size,expected", [(16, (196, 64)), (128, (518, 328))])
def test_gm_to_l1_cost(size, expected):
    a, b = tensor("a", [size, size]), tensor("b", [size, size], scope="shared.l1")
    assert costs(estimate([a, b], [task([ir.Evaluate(tilecall("ascend_copy", region(a), region(b)))])])) == [expected]


def test_composed_sequential_copy_overlaps_only_per_issue_interval():
    a, b = tensor("a", [16, 16]), tensor("b", [16, 16], scope="shared.l1")
    c, d = tensor("c", [16, 16]), tensor("d", [16, 16], scope="shared.l1")
    body = [ir.Evaluate(tilecall("ascend_copy", region(a), region(b))), ir.Evaluate(tilecall("ascend_copy", region(c), region(d)))]
    assert costs(estimate([a, b, c, d], [task(body)])) == [(260, 128)]


@pytest.mark.parametrize("rows,cols,expected", [(64, 128, [(418, 328), (465, 285)]), (1, 64, [(93, 13), (183, 10)])])
def test_aiv_mte_copy_model(rows, cols, expected):
    a, b, ub = tensor("a", [rows, cols]), tensor("b", [rows, cols]), tensor("ub", [rows, cols], scope="shared.dyn")
    body = [
        task([ir.Evaluate(tilecall("ascend_copy", region(a), region(ub)))]),
        task([ir.Evaluate(tilecall("ascend_copy", region(ub), region(b)))]),
    ]
    assert costs(estimate([a, b, ub], body)) == expected


def test_explicit_cost_metadata_is_preserved_and_invalid_cost_rejected():
    a, b = tensor("a", [16, 16]), tensor("b", [16, 16], scope="shared.l1")
    body = [ir.Evaluate(tilecall("ascend_copy", region(a), region(b)))]
    result = estimate([a, b], [task(body, {"latency": integer(17), "ii": integer(5), "trace": "preserved"})])
    assert costs(result) == [(17, 5)]
    assert any(
        fields(reg)["node"].get("trace") == "preserved" for reg in objects(result, ir.RegionStmt) if fields(reg)["key"] == "tl.ascend_task"
    )
    with pytest.raises(Exception, match="greater than or equal to ii"):
        estimate([a, b], [task(body, {"latency": integer(0)})])


def test_flat_scheduled_scalar_bind_metadata_is_rewritten_without_scoping_away_var():
    a = tensor("a", [1], "int32")
    x = ir.Var("x", "int32")
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"stage": integer(-1, "int64"), "task": {"trace": "x"}})
    bind = ir.Bind(
        x,
        ir.Call(
            "tl.ascend.scheduled_bind",
            [ir.IntImm("int32", 7), payload],
            attrs={"schema_version": 1, "operand_count": 1, "annotation_schema": schema},
        ),
    )
    result = estimate([a], [bind, task([ir.TensorStore(a, [0], x)])])
    (out_bind,) = objects(result, ir.Bind)
    assert out_bind.var.same_as(x)
    metadata = tvm_ffi.get_global_func("tl.DecodeAnnotations")(out_bind.value.args[1], out_bind.value.attrs["annotation_schema"])
    assert metadata["task"]["trace"] == "x"
    assert int(metadata["task"]["latency"]) == int(metadata["task"]["ii"]) == 1


def test_vector_block_user_latency_is_consumed_from_native_payload():
    a = tensor("a", [64], "int32", "shared.dyn")
    i = ir.Var("i", "int32")
    loop = ir.For(i, integer(0), integer(64), integer(1), ir.ForKind.PARALLEL, [ir.TensorStore(a, [i], i + 1)])
    block = attribute(None, "tl.sblock", {"name": "SIMT_VF", "no_realize": True, "annotations": {"tl.vf_latency": integer(77)}}, [loop])
    assert costs(estimate([a], [task([block])])) == [(77, 77)]
