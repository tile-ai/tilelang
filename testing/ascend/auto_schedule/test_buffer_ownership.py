"""AnnotateMultiBufferEligible selects complete, write-first storage owners."""

from tvm_tirx import ir
from testing.ascend._ir import annotation, attribute, loop_annotations, sblock, sblock_annotations, sblock_metadata, sblocks, tensor

import pytest
from tilelang.ascend import transform
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
    ub = tensor((1,), "int32", name="ub", scope="shared.dyn")
    out = tensor((4,), "int32", name="out", scope="shared.dyn" if case.startswith("vf-") else "global")
    enabled = ir.Var("enabled", "bool")
    i = ir.Var("i", "int32")
    write = ir.TensorStore(ub, [0], i)
    read = ir.TensorStore(out, [i], ub[0])
    if case.startswith("vf-"):
        atomic_body = seq(write, read) if expected else seq(read, write)
        body = unit(sblock("SIMD_VF", atomic_body))
    elif case == "read-first":
        body = seq(unit(read), unit(write))
    elif case == "same-guard":
        body = seq(unit(write, guard=enabled), unit(read, guard=enabled))
    elif case == "disjoint-guards":
        body = seq(unit(write, guard=enabled), unit(read, guard=ir.prim.Not(enabled)))
    elif case == "exhaustive-writes":
        body = seq(unit(write, guard=enabled), unit(write, guard=ir.prim.Not(enabled)), unit(read))
    elif case == "manual-stage-order":
        body = seq(unit(read, stage=1), unit(write, stage=0))
    else:
        body = seq(unit(write), unit(read))
    loop = ir.For(i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, body)
    before = kernel(
        unit(loop, core=None),
        buffers=[ub, out] if case.startswith("vf-") else [ub],
        params=[enabled] if case.startswith("vf-") else [out, enabled],
        allocation_stage=0,
    )
    after = transform.AnnotateMultiBufferEligible()(before)
    (owner,) = nodes(after, ir.For)
    assert (ub in loop_annotations(owner).get("multi_buffer_eligible", [])) == expected


def test_nested_alias_accesses_have_one_complete_owner():
    base = tensor((4,), "int32", name="base", scope="shared.dyn")
    alias = tensor((2, 2), "int32", name="view", scope=base.ty.scope)
    declaration = unit(ir.Bind(alias, ir.Call("tirx.decl_tensor", [base.data], ty_args=[alias.ty])), core=7)
    out = tensor((4,), "int32", name="out")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    inner_loop = ir.For(
        inner, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, unit(ir.TensorStore(out, [inner], alias[0, inner]))
    )
    outer_loop = ir.For(
        outer,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        seq(unit(ir.TensorStore(base, [0], outer)), unit(inner_loop, core=None)),
    )
    before = kernel(seq(declaration, unit(outer_loop, core=None)), buffers=[base], params=[out], allocation_stage=0)
    after = transform.AnnotateMultiBufferEligible()(before)
    owners = [loop for loop in nodes(after, ir.For) if base in loop_annotations(loop).get("multi_buffer_eligible", [])]
    assert len(owners) == 1 and owners[0].var.same_as(outer)


def test_ss_packed_publication_has_one_shared_owner():
    words = tensor((2,), "uint32", name="words", scope="shared.ss")
    packed = tensor((1,), "uint64", name="packed", scope=words.ty.scope)
    pointer = ir.Call("tirx.reinterpret", [words.data], ty=ir.PtrType("uint64", words.ty.scope))
    declaration = unit(ir.Bind(packed, ir.Call("tirx.decl_tensor", [pointer], ty_args=[packed.ty])), core=7)
    out = tensor((4,), "uint64", name="out")
    i = ir.Var("i", "int32")
    tasks = [unit(ir.TensorStore(words, [sid], ir.prim.Cast("uint32", i)), core=core) for sid, core in enumerate((1, 4))]
    tasks.append(unit(ir.TensorStore(out, [i], packed[0]), core=2))
    loop = ir.For(i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, seq(*tasks))
    before = kernel(
        seq(declaration, unit(loop, core=None)), buffers=[words], params=[out], annotations={"tl.split_subcores": True}, allocation_stage=0
    )
    after = transform.AnnotateMultiBufferEligible()(before)
    (owner,) = nodes(after, ir.For)
    claims = loop_annotations(owner).get("multi_buffer_eligible", [])
    assert len(claims) == 1 and claims[0].same_as(words)


@pytest.mark.parametrize("manual", [False, True], ids=["explicit-owner", "manual-ring"])
def test_explicit_storage_policy_overrides_automatic_claims(manual, capfd):
    ub = tensor((2,), "int32", name="ub", scope="shared.dyn")
    out = tensor((4,), "int32", name="out")
    loops = []
    for index in range(2):
        i = ir.Var(f"i{index}", "int32")
        loops.append(
            unit(
                ir.For(
                    i,
                    ir.const(0),
                    ir.const(0) + ir.const(4),
                    ir.const(1),
                    ir.ForKind.DEFAULT,
                    seq(unit(ir.TensorStore(ub, [0], i)), unit(ir.TensorStore(out, [i], ub[0]))),
                    annotations={"multi_buffer_eligible": [ub]} if index == 0 else {},
                ),
                core=None,
            )
        )
    before = kernel(
        seq(*loops), buffers=[ub], params=[out], annotations={"tl.manual_multi_buffer": {ub: 2}} if manual else {}, allocation_stage=0
    )
    capfd.readouterr()
    after = transform.AnnotateMultiBufferEligible()(before)
    claims = [ub in loop_annotations(loop).get("multi_buffer_eligible", []) for loop in nodes(after, ir.For)]
    assert claims == ([False, False] if manual else [True, False])
    if manual:
        assert "already manually multi-buffered" in capfd.readouterr().err


@pytest.mark.parametrize("control", ["assume", "bound"])
def test_nested_control_reads_belong_to_the_enclosing_owner(control):
    ub = tensor((1,), "int32", name="ub", scope="shared.dyn")
    out = tensor((4,), "int32", name="out")
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    child = unit(
        ir.For(
            j,
            ir.const(0),
            ir.const(0) + (ub[0] if control == "bound" else 1),
            ir.const(1),
            ir.ForKind.DEFAULT,
            unit(ir.TensorStore(out, [i], ub[0])),
        ),
        core=None,
    )
    if control == "assume":
        child = attribute(ub[0] >= 0, "tilelang_assume", 1, child)
    body = seq(unit(ir.TensorStore(ub, [0], i + 1)), child)
    before = kernel(
        unit(ir.For(i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, body), core=None),
        buffers=[ub],
        params=[out],
        allocation_stage=0,
    )
    after = transform.AnnotateMultiBufferEligible()(before)
    owners = [loop.var for loop in nodes(after, ir.For) if ub in loop_annotations(loop).get("multi_buffer_eligible", [])]
    assert len(owners) == 1 and owners[0].same_as(i)


def test_row_writes_and_whole_buffer_read_belong_to_outer_owner():
    a = tensor((4, 2, 64), "float32", name="A")
    out = tensor((4, 2, 64), "float32", name="out")
    ub = tensor((2, 64), "float32", name="ub", scope="shared.dyn")
    i, row = ir.Var("i", "int32"), ir.Var("row", "int32")
    load_row = copy(a, ub, src_indices=[i, row, 0], src_shape=[1, 1, 64], dst_indices=[row, 0], dst_shape=[1, 64])
    rows = ir.For(row, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, unit(load_row))
    body = seq(unit(rows, core=None), unit(copy(ub, out, dst_indices=[i, 0, 0], dst_shape=[1, 2, 64])))
    before = kernel(
        unit(ir.For(i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, body), core=None),
        buffers=[ub],
        params=[a, out],
        allocation_stage=0,
    )
    after = transform.AnnotateMultiBufferEligible()(before)
    owners = [loop.var for loop in nodes(after, ir.For) if ub in loop_annotations(loop).get("multi_buffer_eligible", [])]
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
    ub = tensor((2,), "int32", name="ub", scope="shared.dyn")
    alias = tensor((1, 2), "int32", name="view", scope=ub.ty.scope)
    declaration = unit(ir.Bind(alias, ir.Call("tirx.decl_tensor", [ub.data], ty_args=[alias.ty])), core=7)
    out = tensor((2, 4), "int32", name="out")
    owners = []
    for index in range(2):
        i = ir.Var(f"owner{index}", "int32")
        write = ir.TensorStore(ub, [0], i) if index == 0 else ir.TensorStore(alias, [0, 0], i)
        read = alias[0, 0] if index == 0 else ub[0]
        owners.append(
            unit(
                ir.For(
                    i,
                    ir.const(0),
                    ir.const(0) + ir.const(4),
                    ir.const(1),
                    ir.ForKind.DEFAULT,
                    seq(unit(write), unit(ir.TensorStore(out, [index, i], read))),
                    annotations={"multi_buffer_eligible": [ub]} if explicit else {},
                ),
                core=None,
            )
        )
    annotations = {}
    if versions is not None:
        annotations["tl.buffer_versions_map"] = {alias: versions}
    if mode is not None:
        annotations["tl.buffer_version_mode"] = {alias: mode}
    # Alias-valued policy operands follow the SSA declaration they reference.
    before = kernel(seq(declaration, annotation(annotations), *owners), buffers=[ub], params=[out], allocation_stage=0)
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
    loops = nodes(after, ir.For)
    assert len(loops) == 2
    expected = versions != 1 or mode is not None
    assert all((ub in loop_annotations(loop).get("multi_buffer_eligible", [])) == expected for loop in loops)


@pytest.mark.parametrize("placement,expected", [("inner", "inner"), ("outer", "outer"), ("hoisted", None)])
def test_l0_sf_group_selects_one_complete_owner(placement, expected):
    data = tensor((1,), "int32", name="data", scope="shared.l0a")
    sf = tensor((1,), "int32", name="sf", scope="shared.l0a.sf")
    out = tensor((4,), "int32", name="out")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    sf_write = unit(ir.TensorStore(sf, [0], 2), core=2)
    inner_body = seq(unit(ir.TensorStore(data, [0], inner), core=2), unit(ir.TensorStore(out, [inner], data[0] + sf[0]), core=2))
    if placement == "inner":
        inner_body = seq(sf_write, inner_body)
    outer_body = unit(ir.For(inner, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, inner_body), core=None)
    if placement == "outer":
        outer_body = seq(sf_write, outer_body)
    body = unit(ir.For(outer, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, outer_body), core=None)
    if placement == "hoisted":
        body = seq(sf_write, body)
    before = kernel(body, buffers=[data, sf], params=[out], annotations={"tl.l0_sf_bindings": {sf: data}}, allocation_stage=0)
    after = transform.AnnotateMultiBufferEligible()(before)
    for storage in (data, sf):
        owners = [loop.var.name for loop in nodes(after, ir.For) if storage in loop_annotations(loop).get("multi_buffer_eligible", [])]
        assert owners == ([expected] if expected else [])


@pytest.mark.parametrize("conflict", [False, True])
def test_sf_version_override_applies_to_the_allocation_group(conflict):

    data = tensor((1,), "int32", name="data", scope="shared.l0a")
    sf = tensor((1,), "int32", name="sf", scope="shared.l0a.sf")
    versions = {sf: 2, **({data: 3} if conflict else {})}
    before = kernel(
        unit(ir.Evaluate(0)),
        buffers=[data, sf],
        annotations={"tl.l0_sf_bindings": {sf: data}, "tl.buffer_versions_map": versions},
        allocation_stage=0,
    )
    if conflict:
        with pytest.raises((ValueError, RuntimeError), match="Conflicting.*bound L0 data/SF group"):
            transform.AnnotateMultiBufferEligible()(before)
    else:
        after = transform.AnnotateMultiBufferEligible()(before)
        root = next(block for block in sblocks(after) if sblock_metadata(block)["name"] == "tilelang_root")
        assert int(sblock_annotations(root)["tl.buffer_versions_map"][data]) == 2
