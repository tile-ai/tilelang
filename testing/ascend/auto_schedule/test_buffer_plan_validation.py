"""Invalid version plans fail at PrepareMultiBuffer, before synchronization."""

import pytest
from tvm_tirx import ir, tirx
from tilelang.ascend import transform
from tilelang._compat.tensor import TensorView
from testing.ascend._ir import kernel, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


def _tensor(name, scope):
    return TensorView.create(name, (64,), "float32", scope=scope).tensor


@pytest.mark.parametrize(
    "case, message",
    [
        ("nested-owner", "nested owner loops"),
        ("external-read", "outside its annotated owner loops"),
        ("self-reading-fill", "only T.fill accesses"),
        ("l1-fill", "Broadcast fill for L1 storage"),
        ("storage-bound", "unnormalized loop bound"),
    ],
)
def test_invalid_storage_plans(case, message):
    shared = _tensor("ub", "shared.l1" if case == "l1-fill" else "shared.dyn")
    output = _tensor("out", "global")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    annotations = {"multi_buffer_eligible": [shared]}
    body = unit(ir.TensorStore(shared, [0], ir.const(1, "float32")))
    if case in ("nested-owner", "storage-bound"):
        bound = ir.prim.Cast("int32", shared[0]) if case == "storage-bound" else 2
        child = ir.For(
            inner, 0, bound, 1, ir.ForKind.DEFAULT, body, annotations={"multi_buffer_eligible": [shared]} if case == "nested-owner" else {}
        )
        body = unit(child, core=None)
    loop = unit(ir.For(outer, 0, 4, 1, ir.ForKind.DEFAULT, body, annotations=annotations), core=None)
    if case == "external-read":
        loop = seq(loop, unit(ir.TensorStore(output, [0], shared[0])))
    if case in ("self-reading-fill", "l1-fill"):
        fill = ir.Evaluate(
            ir.Call(
                "tl.tileop.fill", [region(shared, 2), shared[0] if case == "self-reading-fill" else ir.const(0, "float32")], ty="handle"
            )
        )
        loop = seq(unit(fill), loop)
    before = kernel(
        loop,
        buffers=[shared],
        params=[output],
        annotations={
            "tl.buffer_versions_map": {shared: 2},
            "tl.buffer_version_mode": {shared: "counter"},
        },
    )
    assert tirx.analysis.check_well_formed(before)
    with pytest.raises(Exception, match=message):
        transform.PrepareMultiBuffer()(before)


@pytest.mark.parametrize(
    "tail, partial", [(False, False), (False, True), (True, False)], ids=["initial-fill", "partial-fill", "final-fill"]
)
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_external_fill_is_marked_for_broadcast(tail, partial, versions):
    shared = _tensor("ub", "shared.dyn")
    index = ir.Var("index", "int32")
    fill = unit(
        ir.Evaluate(ir.Call("tl.tileop.fill", [region(shared, 2, shape=[32 if partial else 64]), ir.const(0, "float32")], ty="handle"))
    )
    loop = unit(
        ir.For(
            index,
            0,
            4,
            1,
            ir.ForKind.DEFAULT,
            unit(ir.TensorStore(shared, [0], ir.const(1, "float32"))),
            annotations={"multi_buffer_eligible": [shared]},
        ),
        core=None,
    )
    before = kernel(
        seq(loop, fill) if tail else seq(fill, loop), buffers=[shared], annotations={"tl.buffer_versions_map": {shared: versions}}
    )
    after = transform.PrepareMultiBuffer()(before)
    assert tirx.analysis.check_well_formed(after)
    broadcast = [
        scope.attrs["tl.ascend_task"]["tl.multi_buffer_broadcast_fill"]
        for scope in nodes(after, ir.RegionStmt)
        if "tl.ascend_task" in scope.attrs and "tl.multi_buffer_broadcast_fill" in scope.attrs["tl.ascend_task"]
    ]
    assert len(broadcast) == 1 and shared in broadcast[0]


def test_bound_data_and_sf_cannot_have_different_owners():
    data = _tensor("data", "shared.l0a")
    scale = _tensor("sf", "shared.l0a.sf")
    loops = []
    for tensor in (data, scale):
        index = ir.Var("index", "int32")
        body = unit(ir.TensorStore(tensor, [0], ir.prim.Cast("float32", index)), core=2)
        loops.append(unit(ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, body, annotations={"multi_buffer_eligible": [tensor]}), core=None))
    before = kernel(
        seq(*loops),
        buffers=[data, scale],
        annotations={
            "tl.l0_sf_bindings": {scale: data},
            "tl.buffer_versions_map": {data: 2, scale: 2},
        },
    )
    assert tirx.analysis.check_well_formed(before)
    with pytest.raises(Exception, match="same owner loops"):
        transform.PrepareMultiBuffer()(before)
