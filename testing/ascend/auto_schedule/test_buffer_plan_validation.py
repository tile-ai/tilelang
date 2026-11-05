"""Invalid version plans fail at PrepareMultiBuffer, before synchronization."""

import pytest
from tilelang.ascend import transform
from tvm_tirx import ir, tirx
from testing.ascend._ir import attribute_fields, attributes, fill, kernel, seq, serial, tensor
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
    ub = tensor((64,), "float32", name="ub", scope="shared.l1" if case == "l1-fill" else "shared.dyn")
    out = tensor((1,), "float32", name="out")
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    annotations = {"multi_buffer_eligible": [ub]}
    body = unit(ir.TensorStore(ub, [0], ir.const(1, "float32")))
    if case in ("nested-owner", "storage-bound"):
        child = serial(
            j,
            0,
            tirx.Cast("int32", ub[0]) if case == "storage-bound" else 2,
            body,
            annotations={"multi_buffer_eligible": [ub]} if case == "nested-owner" else {},
        )
        body = unit(child, core=None)
    loop = unit(serial(i, 0, 4, body, annotations=annotations), core=None)
    if case == "external-read":
        loop = seq(loop, unit(ir.TensorStore(out, [0], ub[0])))
    if case in ("self-reading-fill", "l1-fill"):
        initialization = fill(ub, ub[0] if case == "self-reading-fill" else ir.const(0, "float32"))
        loop = seq(unit(initialization), loop)
    before = kernel(
        loop,
        buffers=[ub],
        params=[out],
        annotations={"tl.buffer_versions_map": {ub: 2}, "tl.buffer_version_mode": {ub: "counter"}},
        allocation_stage=0,
    )
    with pytest.raises(RuntimeError, match=message):
        transform.PrepareMultiBuffer()(before)


@pytest.mark.parametrize(
    "tail, partial", [(False, False), (False, True), (True, False)], ids=["initial-fill", "partial-fill", "final-fill"]
)
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_external_fill_is_marked_for_broadcast(tail, partial, versions):
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    i = ir.Var("i", "int32")
    initialization = unit(fill(ub, ir.const(0, "float32"), shape=[32 if partial else 64]))
    loop = unit(
        serial(
            i,
            0,
            4,
            unit(ir.TensorStore(ub, [0], ir.const(1, "float32"))),
            annotations={"multi_buffer_eligible": [ub]},
        ),
        core=None,
    )
    before = kernel(
        seq(loop, initialization) if tail else seq(initialization, loop),
        buffers=[ub],
        annotations={"tl.buffer_versions_map": {ub: versions}},
        allocation_stage=0,
    )
    after = transform.PrepareMultiBuffer()(before)
    broadcast = [
        attribute_fields(node)["node"]
        for node in attributes(after, "tl.ascend_task")
        if "tl.multi_buffer_broadcast_fill" in attribute_fields(node)["node"]
    ]
    assert len(broadcast) == 1 and ub in broadcast[0]["tl.multi_buffer_broadcast_fill"]


def test_bound_data_and_sf_cannot_have_different_owners():
    data = tensor((1,), "int32", name="data", scope="shared.l0a")
    sf = tensor((1,), "int32", name="sf", scope="shared.l0a.sf")
    loops = []
    for buf in (data, sf):
        i = ir.Var("i", "int32")
        loops.append(
            unit(
                serial(
                    i,
                    0,
                    4,
                    unit(ir.TensorStore(buf, [0], i), core=2),
                    annotations={"multi_buffer_eligible": [buf]},
                ),
                core=None,
            )
        )
    before = kernel(
        seq(*loops),
        buffers=[data, sf],
        annotations={
            "tl.l0_sf_bindings": {sf: data},
            "tl.buffer_versions_map": {data: 2, sf: 2},
        },
        allocation_stage=0,
    )
    with pytest.raises(RuntimeError, match="same owner loops"):
        transform.PrepareMultiBuffer()(before)
