"""Vector groups preserve scalar assumptions and private atomic contributions."""

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, target, tirx

import tilelang
from tilelang.analysis._visitor import tilelang_metadata
from tilelang.testing.ir import collect_calls, collect_nodes


def _pack(function):
    before = ir.save_json(function)
    result = tilelang.transform.VectorizeLoop()(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    for call in collect_nodes(result, ir.Call):
        call.validate()
    return result


@pytest.mark.parametrize("begin", [0, 5])
def test_assumption_and_check_keep_every_lane_as_a_scalar_condition(begin):
    source, output = [ir.Var(name, tirx.TensorType((4,), "int32")) for name in ("source", "output")]
    limit, lane = ir.Var("limit", "int32"), ir.Var("lane", "int32")
    loop = ir.For(
        lane,
        begin,
        begin + 4,
        1,
        ir.ForKind.VECTORIZED,
        [ir.Evaluate(ir.Call("prim.assume", [lane < limit])), ir.TensorStore(output, [lane - begin], source[lane - begin])],
    )
    function = tirx.Function([source, output, limit], loop).with_attr("tirx.noalias", True)
    function = tilelang.transform.InjectAssumes()(ir.IRModule({"main": function}))["main"]
    result = _pack(function)
    assert not collect_nodes(result, ir.For)
    assumptions = collect_calls(result, op="prim.assume")
    markers = [
        tilelang_metadata(node)
        for node in collect_nodes(result, ir.RegionStmt)
        if tilelang_metadata(node).get("key") == "tl.assume_requires_runtime_check"
    ]
    assert len(assumptions) == len(markers) == 1
    predicates = [assumptions[0].args[0], markers[0]["node"]]
    for predicate in predicates:
        assert predicate.ty == ir.PrimType("bool")
        for size in (begin, begin + 3, begin + 4, begin + 7):
            concrete = tvm_ffi.structural_map(
                predicate, (ir.Var, lambda var, size=size: ir.const(size) if var.same_as(limit) else var), order="post"
            )
            assert bool(int(sym.Analyzer().simplify(concrete))) == all(index < size for index in range(begin, begin + 4))


@pytest.mark.parametrize("injected", [False, True])
@pytest.mark.parametrize("begin", [0, 5])
def test_assumption_local_definitions_are_fresh_per_copy(injected, begin):
    source, output = [ir.Var(name, tirx.TensorType((4,), "int32")) for name in ("source", "output")]
    lane = ir.Var("lane", "int32")
    # Identical names must not confuse the two bound identities with the free
    # parameter, including when the marker and optimizer fact share a value.
    limit, first, second = [ir.Var("local", "int32") for _ in range(3)]
    condition = ir.prim.Let(first, lane + 1, ir.prim.Let(second, first + 2, second < limit))
    body = [ir.Evaluate(ir.Call("prim.assume", [condition])), ir.TensorStore(output, [lane - begin], source[lane - begin])]
    function = tirx.Function([source, output, limit], ir.For(lane, begin, begin + 4, 1, ir.ForKind.VECTORIZED, body)).with_attr(
        "tirx.noalias", True
    )
    assert tirx.analysis.check_well_formed(function)
    if injected:
        before = ir.save_json(function)
        transformed = tilelang.transform.InjectAssumes()(ir.IRModule({"main": function}))["main"]
        assert ir.save_json(function) == before
        assert tirx.analysis.check_well_formed(transformed)
        function = transformed
    result = _pack(function)
    assert not collect_nodes(result, ir.For)
    assert result.params[2].same_as(limit)
    # Both serialization and repeated pass application must retain unique
    # local definitions and the caller's free-variable identity.
    assert tirx.analysis.check_well_formed(ir.load_json(ir.save_json(result)))
    _pack(result)
    assumptions = collect_calls(result, op="prim.assume")
    assert len(assumptions) == 1
    predicates = [assumptions[0].args[0]]
    if injected:
        markers = [
            tilelang_metadata(node)
            for node in collect_nodes(result, ir.RegionStmt)
            if tilelang_metadata(node).get("key") == "tl.assume_requires_runtime_check"
        ]
        assert len(markers) == 1
        predicates.append(markers[0]["node"])
    for predicate in predicates:
        for size in range(begin, begin + 9):
            concrete = tvm_ffi.structural_map(
                predicate, (ir.Var, lambda var, size=size: ir.const(size) if var.same_as(limit) else var), order="post"
            )
            assert bool(int(sym.Analyzer().simplify(concrete))) == all(index + 3 < size for index in range(begin, begin + 4))


@pytest.mark.parametrize(
    "storage,predicate,expect_packed",
    [
        ("local", "positive_divisor", True),
        ("local", "unknown_divisor", False),
        ("local", "readonly", True),
        ("local", "may_alias", False),
        ("shared", "positive_divisor", False),
        ("parameter_local", "positive_divisor", False),
    ],
)
def test_atomic_group_requires_private_storage_and_a_stable_nontrapping_predicate(storage, predicate, expect_packed):
    source, output = [ir.Var(name, tirx.TensorType((4,), "float32", data_alignment=64)) for name in ("source", "output")]
    values = ir.Var("values", tirx.TensorType((4,), "float32", scope="local" if "local" in storage else storage))
    limit, divisor, lane = [ir.Var(name, "int32") for name in ("limit", "divisor", "lane")]
    indices = ir.Var("indices", tirx.TensorType((1,), "int32"))
    params = [source, output, limit, divisor, indices]
    prefix = []
    if storage == "parameter_local":
        params.append(values)
    else:
        prefix.append(ir.Bind(values, ir.Call("tirx.alloc_tensor", [], ty_args=[values.ty])))
    if predicate == "unknown_divisor":
        condition = limit // divisor >= 0
    elif predicate in ("readonly", "may_alias"):
        condition = indices[0] >= 0
    else:
        condition = limit // 4 >= 0
    pointer = ir.Call("tl.access_ptr", [output[lane], ir.const(1), ir.const(3)])
    atomic = ir.Evaluate(ir.Call("tl.atomic_add_elem_op", [pointer, values[lane], ir.const(3)]))
    body = [ir.TensorStore(values, [lane], source[lane]), ir.If(condition, [atomic], [])]
    function = tirx.Function(params, prefix + [ir.For(lane, 0, 4, 1, ir.ForKind.VECTORIZED, body)]).with_attrs(
        {"target": target.Target({"kind": "cuda", "arch": "sm_90"}), "tirx.noalias": predicate != "may_alias"}
    )
    result = _pack(function)
    packed = collect_calls(result, op="tl.atomic_addx4_elem_op")
    assert bool(packed) == expect_packed
    assert bool(collect_nodes(result, ir.For)) != expect_packed
    if packed:
        assert int(packed[0].args[2]) == 3
