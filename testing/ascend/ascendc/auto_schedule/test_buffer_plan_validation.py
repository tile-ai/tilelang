"""Invalid version plans fail at PrepareMultiBuffer, before synchronization."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import copy, kernel, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


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
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.l1" if case == "l1-fill" else "shared.dyn")
    out = tirx.decl_buffer((1,), "float32", name="out")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    annotations = {"multi_buffer_eligible": [ub.data]}
    body = unit(tirx.BufferStore(ub, tirx.const(1, "float32"), [0]))
    if case in ("nested-owner", "storage-bound"):
        child = tirx.For(
            j,
            0,
            tirx.Cast("int32", ub[0]) if case == "storage-bound" else 2,
            tirx.ForKind.SERIAL,
            body,
            annotations={"multi_buffer_eligible": [ub.data]} if case == "nested-owner" else {},
        )
        body = unit(child, core=None)
    loop = unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations=annotations), core=None)
    if case == "external-read":
        loop = seq(loop, unit(tirx.BufferStore(out, ub[0], [0])))
    if case in ("self-reading-fill", "l1-fill"):
        fill = tirx.Evaluate(
            tirx.Call(
                "handle",
                tvm.ir.Op.get("tl.tileop.fill"),
                [region(ub, 2), ub[0] if case == "self-reading-fill" else tirx.const(0, "float32")],
            )
        )
        loop = seq(unit(fill), loop)
    before = kernel(
        loop,
        buffers=[ub],
        params=[out],
        annotations={"tl.buffer_versions_map": {ub.data: 2}, "tl.buffer_version_mode": {ub.data: "counter"}},
    )
    with pytest.raises(tvm.error.InternalError, match=message):
        transform.PrepareMultiBuffer()(before)


@pytest.mark.parametrize(
    "tail, partial", [(False, False), (False, True), (True, False)], ids=["initial-fill", "partial-fill", "final-fill"]
)
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_external_fill_is_marked_for_broadcast(tail, partial, versions):
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    fill = unit(
        tirx.Evaluate(
            tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"), [region(ub, 2, shape=[32 if partial else 64]), tirx.const(0, "float32")])
        )
    )
    loop = unit(
        tirx.For(
            i,
            0,
            4,
            tirx.ForKind.SERIAL,
            unit(tirx.BufferStore(ub, tirx.const(1, "float32"), [0])),
            annotations={"multi_buffer_eligible": [ub.data]},
        ),
        core=None,
    )
    before = kernel(seq(loop, fill) if tail else seq(fill, loop), buffers=[ub], annotations={"tl.buffer_versions_map": {ub.data: versions}})
    after = transform.PrepareMultiBuffer()(before)
    broadcast = [
        node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.ascend_task" and "tl.multi_buffer_broadcast_fill" in node.node
    ]
    assert len(broadcast) == 1 and ub.data in broadcast[0].node["tl.multi_buffer_broadcast_fill"]


def test_bound_data_and_sf_cannot_have_different_owners():
    data = tirx.decl_buffer((1,), "int32", name="data", scope="shared.l0a")
    sf = tirx.decl_buffer((1,), "int32", name="sf", scope="shared.l0a.sf")
    loops = []
    for buf in (data, sf):
        i = tirx.Var("i", "int32")
        loops.append(
            unit(
                tirx.For(
                    i,
                    0,
                    4,
                    tirx.ForKind.SERIAL,
                    unit(tirx.BufferStore(buf, i, [0]), core=2),
                    annotations={"multi_buffer_eligible": [buf.data]},
                ),
                core=None,
            )
        )
    before = kernel(
        seq(*loops),
        buffers=[data, sf],
        annotations={
            "tl.l0_sf_bindings": {sf.data: data.data},
            "tl.buffer_versions_map": {data.data: 2, sf.data: 2},
        },
    )
    with pytest.raises(tvm.error.InternalError, match="same owner loops"):
        transform.PrepareMultiBuffer()(before)


@pytest.mark.parametrize("contract", ["initialization", "read-modify-write", "partial-region"])
def test_explicit_write_owner_trusts_the_data_contract(contract):
    a, c = [tirx.decl_buffer((64,), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    write = copy(a, x)
    if contract == "read-modify-write":
        write = copy(x, x)
    elif contract == "partial-region":
        write = copy(a, x, src_shape=[16], dst_shape=[16])
    body = seq(unit(write, guard=i > 0 if contract == "initialization" else None), unit(copy(x, c)))
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None),
        buffers=[x],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    # Deliberately invalid data contracts: test annotation trust only, never
    # execute these or treat their acceptance as synchronization evidence.
    after = transform.PrepareMultiBuffer()(before)
    (owner,) = nodes(after, tirx.For)
    assert owner.annotations["tl.multi_buffer_counter_map"][x.data].dtype == "int64"


@pytest.mark.parametrize(
    "case, message",
    [
        ("opaque", "opaque pointer"),
        ("mutable-guard", "unsnapshotted guard"),
        ("loop-break", "loop_break"),
        ("cross-stage", "multiple schedule stages"),
        ("changing-view", "loop-invariant buffer view geometry"),
    ],
)
def test_unsupported_write_counter_structure(case, message):
    a, c = [tirx.decl_buffer((64,), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((256,), "float32", name="x", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    view = tirx.decl_buffer((64,), "float32", data=x.data, scope="shared.dyn", elem_offset=i * 64 if case == "changing-view" else 0)
    write = copy(a, view)
    if case == "opaque":
        write = seq(write, tirx.Evaluate(tirx.call_extern("int32", "opaque_write", x.data)))
    elif case == "loop-break":
        write = seq(write, tirx.Evaluate(tirx.call_intrin("handle", tvm.ir.Op.get("tl.loop_break"))))
    body = seq(
        unit(write, guard=x[0] > 0 if case == "mutable-guard" else i % 2 == 0),
        unit(copy(view, c), stage=1 if case == "cross-stage" else 0),
    )
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None),
        buffers=[x],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    with pytest.raises(tvm.error.InternalError, match=message):
        transform.PrepareMultiBuffer()(before)
