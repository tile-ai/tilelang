"""AnnotateMultiBufferEligible selects complete, write-first storage owners."""

import pytest
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize(
    "case, expected",
    [
        ("write-first", True),
        ("read-first", False),
        ("same-guard", True),
        ("disjoint-guards", False),
        ("exhaustive-writes", True),
        ("manual-stage-order", True),
        ("vf-write-first", True),
        ("vf-read-first", False),
    ],
)
def test_write_first_ownership(case, expected):
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((4,), "int32", name="out", scope="shared.dyn" if case.startswith("vf-") else "global")
    enabled = tirx.Var("enabled", "bool")
    i = tirx.Var("i", "int32")
    write = tirx.BufferStore(ub, i, [0])
    read = tirx.BufferStore(out, ub[0], [i])
    if case.startswith("vf-"):
        atomic_body = seq(write, read) if expected else seq(read, write)
        body = unit(tirx.SBlock([], [], [], "SIMD_VF", atomic_body))
    elif case == "read-first":
        body = seq(unit(read), unit(write))
    elif case == "same-guard":
        body = seq(unit(write, guard=enabled), unit(read, guard=enabled))
    elif case == "disjoint-guards":
        body = seq(unit(write, guard=enabled), unit(read, guard=tirx.Not(enabled)))
    elif case == "exhaustive-writes":
        body = seq(unit(write, guard=enabled), unit(write, guard=tirx.Not(enabled)), unit(read))
    elif case == "manual-stage-order":
        body = seq(unit(read, stage=1), unit(write, stage=0))
    else:
        body = seq(unit(write), unit(read))
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body)
    before = kernel(
        unit(loop, core=None),
        buffers=[ub, out] if case.startswith("vf-") else [ub],
        params=[enabled] if case.startswith("vf-") else [out, enabled],
    )
    after = transform.AnnotateMultiBufferEligible()(before)
    (owner,) = nodes(after, tirx.For)
    assert (ub.data in owner.annotations.get("multi_buffer_eligible", [])) == expected


def test_nested_alias_accesses_have_one_complete_owner():
    base = tirx.decl_buffer((4,), "int32", name="base", scope="shared.dyn")
    alias = tirx.decl_buffer((2, 2), "int32", name="view", data=base.data)
    out = tirx.decl_buffer((4,), "int32", name="out")
    outer, inner = tirx.Var("outer", "int32"), tirx.Var("inner", "int32")
    inner_loop = tirx.For(inner, 0, 2, tirx.ForKind.SERIAL, unit(tirx.BufferStore(out, alias[0, inner], [inner])))
    outer_loop = tirx.For(outer, 0, 4, tirx.ForKind.SERIAL, seq(unit(tirx.BufferStore(base, outer, [0])), unit(inner_loop, core=None)))
    before = kernel(unit(outer_loop, core=None), buffers=[base], params=[out])
    after = transform.AnnotateMultiBufferEligible()(before)
    owners = [loop for loop in nodes(after, tirx.For) if base.data in loop.annotations.get("multi_buffer_eligible", [])]
    assert len(owners) == 1 and owners[0].loop_var.same_as(outer)


@pytest.mark.parametrize("manual", [False, True], ids=["explicit-owner", "manual-ring"])
def test_explicit_storage_policy_overrides_automatic_claims(manual, capfd):
    ub = tirx.decl_buffer((2,), "int32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((4,), "int32", name="out")
    loops = []
    for index in range(2):
        i = tirx.Var(f"i{index}", "int32")
        loops.append(
            unit(
                tirx.For(
                    i,
                    0,
                    4,
                    tirx.ForKind.SERIAL,
                    seq(unit(tirx.BufferStore(ub, i, [0])), unit(tirx.BufferStore(out, ub[0], [i]))),
                    annotations={"multi_buffer_eligible": [ub.data]} if index == 0 else {},
                ),
                core=None,
            )
        )
    before = kernel(seq(*loops), buffers=[ub], params=[out], annotations={"tl.manual_multi_buffer": {ub.data: 2}} if manual else {})
    capfd.readouterr()
    after = transform.AnnotateMultiBufferEligible()(before)
    claims = [ub.data in loop.annotations.get("multi_buffer_eligible", []) for loop in nodes(after, tirx.For)]
    assert claims == ([False, False] if manual else [True, False])
    if manual:
        assert "already manually multi-buffered" in capfd.readouterr().err


@pytest.mark.parametrize("control", ["assume", "bound"])
def test_nested_control_reads_belong_to_the_enclosing_owner(control):
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((4,), "int32", name="out")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    child = tirx.For(j, 0, ub[0] if control == "bound" else 1, tirx.ForKind.SERIAL, unit(tirx.BufferStore(out, ub[0], [i])))
    if control == "assume":
        child = tirx.AttrStmt(ub[0] >= 0, "tl.assume", 1, child)
    body = seq(unit(tirx.BufferStore(ub, i + 1, [0])), unit(child, core=None))
    before = kernel(unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body), core=None), buffers=[ub], params=[out])
    after = transform.AnnotateMultiBufferEligible()(before)
    owners = [loop.loop_var for loop in nodes(after, tirx.For) if ub.data in loop.annotations.get("multi_buffer_eligible", [])]
    assert len(owners) == 1 and owners[0].same_as(i)


def test_row_writes_and_whole_buffer_read_belong_to_outer_owner():
    a = tirx.decl_buffer((4, 2, 64), "float32", name="A")
    out = tirx.decl_buffer((4, 2, 64), "float32", name="out")
    ub = tirx.decl_buffer((2, 64), "float32", name="ub", scope="shared.dyn")
    i, row = tirx.Var("i", "int32"), tirx.Var("row", "int32")
    load_row = copy(a, ub, src_indices=[i, row, 0], src_shape=[1, 1, 64], dst_indices=[row, 0], dst_shape=[1, 64])
    rows = tirx.For(row, 0, 2, tirx.ForKind.SERIAL, unit(load_row))
    body = seq(unit(rows, core=None), unit(copy(ub, out, dst_indices=[i, 0, 0], dst_shape=[1, 2, 64])))
    before = kernel(unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body), core=None), buffers=[ub], params=[a, out])
    after = transform.AnnotateMultiBufferEligible()(before)
    owners = [loop.loop_var for loop in nodes(after, tirx.For) if ub.data in loop.annotations.get("multi_buffer_eligible", [])]
    assert len(owners) == 1 and owners[0].same_as(i)


@pytest.mark.parametrize(
    "versions, mode, explicit",
    [
        pytest.param(1, None, False, id="opt-out-inferred"),
        pytest.param(1, None, True, id="opt-out-explicit"),
        pytest.param(1, "auto", False, id="single-auto-inferred"),
        pytest.param(1, "auto", True, id="single-auto-explicit"),
        pytest.param(1, "iteration", True, id="single-iteration"),
        pytest.param(1, "counter", True, id="single-counter"),
        pytest.param(2, None, False, id="fixed-ring"),
        pytest.param(None, "auto", False, id="automatic-ring"),
    ],
)
def test_version_policy_controls_storage_and_alias_eligibility(versions, mode, explicit, capfd):
    ub = tirx.decl_buffer((2,), "int32", name="ub", scope="shared.dyn")
    alias = tirx.decl_buffer((1, 2), "int32", name="view", data=ub.data)
    out = tirx.decl_buffer((2, 4), "int32", name="out")
    owners = []
    for index in range(2):
        i = tirx.Var(f"owner{index}", "int32")
        write = tirx.BufferStore(ub, i, [0]) if index == 0 else tirx.BufferStore(alias, i, [0, 0])
        read = alias[0, 0] if index == 0 else ub[0]
        owners.append(
            unit(
                tirx.For(
                    i,
                    0,
                    4,
                    tirx.ForKind.SERIAL,
                    seq(unit(write), unit(tirx.BufferStore(out, read, [index, i]))),
                    annotations={"multi_buffer_eligible": [ub.data]} if explicit else {},
                ),
                core=None,
            )
        )
    annotations = {}
    if versions is not None:
        annotations["tl.buffer_versions_map"] = {alias.data: versions}
    if mode is not None:
        annotations["tl.buffer_version_mode"] = {alias.data: mode}
    before = kernel(seq(*owners), buffers=[ub], params=[out], annotations=annotations)
    capfd.readouterr()
    after = transform.AnnotateMultiBufferEligible()(before)
    diagnostic = capfd.readouterr().err
    warning = "Ignoring explicit 'multi_buffer_eligible' claim"
    if explicit and versions == 1 and mode is None:
        assert diagnostic.count(warning) == 1
        assert "storage ub" in diagnostic
        assert "T.annotate_buffer_versions({buf: 1})" in diagnostic
        assert '(1, "auto")' in diagnostic
    else:
        assert warning not in diagnostic
    loops = nodes(after, tirx.For)
    assert len(loops) == 2
    expected = versions != 1 or mode is not None
    assert all((ub.data in loop.annotations.get("multi_buffer_eligible", [])) == expected for loop in loops)


@pytest.mark.parametrize("placement,expected", [("inner", "inner"), ("outer", "outer"), ("hoisted", None)])
def test_l0_sf_group_selects_one_complete_owner(placement, expected):
    data = tirx.decl_buffer((1,), "int32", name="data", scope="shared.l0a")
    sf = tirx.decl_buffer((1,), "int32", name="sf", scope="shared.l0a.sf")
    out = tirx.decl_buffer((4,), "int32", name="out")
    outer, inner = tirx.Var("outer", "int32"), tirx.Var("inner", "int32")
    sf_write = unit(tirx.BufferStore(sf, 2, [0]), core=2)
    inner_body = seq(unit(tirx.BufferStore(data, inner, [0]), core=2), unit(tirx.BufferStore(out, data[0] + sf[0], [inner]), core=2))
    if placement == "inner":
        inner_body = seq(sf_write, inner_body)
    outer_body = unit(tirx.For(inner, 0, 4, tirx.ForKind.SERIAL, inner_body), core=None)
    if placement == "outer":
        outer_body = seq(sf_write, outer_body)
    body = unit(tirx.For(outer, 0, 4, tirx.ForKind.SERIAL, outer_body), core=None)
    if placement == "hoisted":
        body = seq(sf_write, body)
    before = kernel(body, buffers=[data, sf], params=[out], annotations={"tl.l0_sf_bindings": {sf.data: data.data}})
    after = transform.AnnotateMultiBufferEligible()(before)
    for storage in (data.data, sf.data):
        owners = [loop.loop_var.name for loop in nodes(after, tirx.For) if storage in loop.annotations.get("multi_buffer_eligible", [])]
        assert owners == ([expected] if expected else [])


@pytest.mark.parametrize("conflict", [False, True])
def test_sf_version_override_applies_to_the_allocation_group(conflict):
    from tilelang import tvm

    data = tirx.decl_buffer((1,), "int32", name="data", scope="shared.l0a")
    sf = tirx.decl_buffer((1,), "int32", name="sf", scope="shared.l0a.sf")
    versions = {sf.data: 2, **({data.data: 3} if conflict else {})}
    before = kernel(
        unit(tirx.Evaluate(0)),
        buffers=[data, sf],
        annotations={"tl.l0_sf_bindings": {sf.data: data.data}, "tl.buffer_versions_map": versions},
    )
    if conflict:
        with pytest.raises(tvm.error.InternalError, match="Conflicting.*bound L0 data/SF group"):
            transform.AnnotateMultiBufferEligible()(before)
    else:
        after = transform.AnnotateMultiBufferEligible()(before)
        root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
        assert int(root.annotations["tl.buffer_versions_map"][data.data]) == 2
