"""High-value direct-TIR regressions for ``LegalizePTOVMIMemoryAccess``.

The pass consumes post-``LegalizeNegativeIndex`` IR: an automatic VMI address
is the original scalar-origin ``BufferLoad`` and an explicit address is a
``tl.access_ptr`` call, either direct or reached through one direct handle
``Bind``/``Let``.  These cases pin descriptor construction, physical
envelopes, negative-index/group candidates, explicit preservation and logical
coverage before the pass is wired into the pipeline, where the existing
frontend tests cannot observe them.

These are host-only IR tests covering the contracts that must remain in-tree.
Broader exploratory matrices and replay scripts are kept outside the
repository.
"""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import transform as ascend
from tvm import tirx
from tvm.target import Target
from testing.ascend._ir import calls, nodes

_PTO_TARGET = Target({"kind": "ascend", "keys": ["pto", "ascend"]})


# ---------------------------------------------------------------------------
# IR builders
# ---------------------------------------------------------------------------
def _int(value, dtype="int32"):
    return tirx.IntImm(dtype, int(value))


def _buffer(shape, dtype="float32", *, name="work", scope="shared", strides=None, elem_offset=0):
    kwargs = {"name": name, "scope": scope}
    if strides is not None:
        kwargs["strides"] = tuple(_int(stride, "int64") if stride > 2**31 - 1 else _int(stride) for stride in strides)
    if elem_offset:
        kwargs["elem_offset"] = _int(elem_offset)
    return tirx.decl_buffer(shape, dtype, **kwargs)


def _load(buffer, indices):
    return tirx.BufferLoad(buffer, list(indices))


def _access_ptr(buffer, indices, extent, mask, dtype="int32"):
    return tirx.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.access_ptr"),
        _load(buffer, indices),
        extent if isinstance(extent, tirx.PrimExpr) else _int(extent, dtype),
        _int(mask),
    )


def _annotation(value):
    if isinstance(value, str):
        return tirx.StringImm(value)
    return value


def _annotation_text(annotations, key):
    value = annotations[key]
    return value.value if isinstance(value, tirx.StringImm) else str(value)


def _vload(source, offset=0, *, elem="float32", size=64, **annotations):
    attrs = {"size": size}
    attrs.update({key: _annotation(value) for key, value in annotations.items() if value is not None})
    return tirx.call_intrin(
        f"{elem}x{size}",
        tvm.ir.Op.get("tl.vmi.vload"),
        source,
        offset if isinstance(offset, tirx.PrimExpr) else _int(offset),
        annotations=attrs,
    )


def _vstore(value, destination, offset=0, mask=None, *, dintlv=False, **annotations):
    args = list(value) if dintlv else [value]
    args.extend([destination, offset if isinstance(offset, tirx.PrimExpr) else _int(offset)])
    if mask is not None:
        args.append(mask)
    attrs = {key: _annotation(val) for key, val in annotations.items() if val is not None}
    if dintlv:
        attrs["dist_mode"] = _annotation("dintlv")
    return tirx.call_intrin("void", tvm.ir.Op.get("tl.vmi.vstore"), *args, annotations=attrs or None)


def _vec(name, elem="float32", lanes=64):
    return tirx.Var(name, f"{elem}x{lanes}")


def _mask(name, lanes=64):
    return tirx.Var(name, f"boolx{lanes}")


def _evaluate(call):
    return tirx.Evaluate(call)


def _module(body, params=(), target=_PTO_TARGET):
    mod = tvm.IRModule({"main": tirx.PrimFunc(list(params), body)})
    if target is not None:
        mod = tirx.transform.BindTarget(target)(mod)
    return mod


def _run(body, params=(), target=_PTO_TARGET):
    return ascend.LegalizePTOVMIMemoryAccess()(_module(body, params, target))["main"].body


def _legalize_negative(body):
    return tilelang.transform.LegalizeNegativeIndex()(_module(body))["main"].body


def _vmi_calls(ir, name="tl.vmi.vload"):
    return calls(ir, name)


def _descriptor(call, address_index=0):
    pointer = call.args[address_index]
    assert isinstance(pointer, tirx.Call), pointer
    assert isinstance(pointer.op, tvm.ir.Op) and pointer.op.name == "tl.access_ptr", pointer
    base = pointer.args[0]
    assert isinstance(base, tirx.BufferLoad), base
    return base, pointer


def _logical_offset(buffer, indices):
    """Element offset from the buffer's logical origin (elem_offset cancels)."""
    values = list(buffer.offset_of(list(indices)))
    origin = list(buffer.offset_of([_int(0, str(dim.dtype)) for dim in buffer.shape]))
    assert len(values) == len(origin) == 1
    return values[0] - origin[0]


def _constant(value):
    assert isinstance(value, tirx.IntImm), value
    return int(value.value)


def _is_dintlv(call):
    mode = call.annotations.get("dist_mode")
    return mode is not None and _annotation_text(call.annotations, "dist_mode") == "dintlv"


def _assert_no_raw_address(body):
    for name in ("tl.vmi.vload", "tl.vmi.vstore"):
        for call in _vmi_calls(body, name):
            address_index = 0 if name.endswith("vload") else (2 if _is_dintlv(call) else 1)
            address = call.args[address_index]
            assert not isinstance(address, (tirx.BufferLoad, tirx.Buffer, tirx.BufferRegion)), address


def _assert_plan(call, *, address_index, b, e, d, rw):
    base, pointer = _descriptor(call, address_index)
    assert _constant(_logical_offset(base.buffer, base.indices)) == b
    assert _constant(pointer.args[1]) == e
    assert _constant(pointer.args[2]) == rw
    assert _constant(call.args[address_index + 1]) == d
    return base, pointer


# ---------------------------------------------------------------------------
# Ordinary descriptor construction and envelopes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "size,annotations,extent",
    [
        (64, {}, 64),
        (80, {}, 128),
        (128, {"dist_mode": "brc"}, 1),
    ],
    ids=["load64", "load-tail80", "brc128"],
)
def test_continuous_broadcast_and_load_tail(size, annotations, extent):
    work = _buffer((512,))
    call = _vload(_load(work, [_int(64)]), size=size, **annotations)

    body = _run(_evaluate(call))

    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=64, e=extent, d=0, rw=1)
    assert out.dtype == call.dtype, "the VMI result dtype is preserved"
    assert _constant(out.annotations["size"]) == size
    for key, value in annotations.items():
        assert _annotation_text(out.annotations, key) == str(value)
    _assert_no_raw_address(body)


def test_store_masks_and_dintlv():
    work = _buffer((512,))
    value = _vec("value")
    mask = _mask("mask")
    call = _vstore(value, _load(work, [_int(0)]), mask=mask)

    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body, "tl.vmi.vstore")
    _assert_plan(out, address_index=1, b=0, e=64, d=0, rw=2)
    assert out.args[0].same_as(value)
    assert out.args[3].same_as(mask), "the user mask is preserved and never shrinks the store extent"
    _assert_no_raw_address(body)

    even = _vec("even")
    odd = _vec("odd")
    call = _vstore((even, odd), _load(work, [_int(0)]), mask=mask, dintlv=True)
    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body, "tl.vmi.vstore")
    _assert_plan(out, address_index=2, b=0, e=128, d=0, rw=2)
    assert out.args[0].same_as(even) and out.args[1].same_as(odd)
    assert out.args[4].same_as(mask)
    assert _annotation_text(out.annotations, "dist_mode") == "dintlv"

    call = _vload(_load(work, [_int(0)]), size=64, dist_mode="dintlv")
    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=0, e=128, d=0, rw=1)


def test_group_and_block_envelopes():
    work = _buffer((512,))
    # A negative group stride is relocated by the first candidate: the
    # descriptor starts at the lowest address and the instruction offset
    # carries the displacement.
    call = _vload(_load(work, [_int(192)]), size=128, group=2, stride=-128)
    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body)
    base, _ = _assert_plan(out, address_index=0, b=64, e=192, d=128, rw=1)
    assert _constant(_logical_offset(base.buffer, base.indices)) + _constant(out.args[1]) == 192

    # group=1 has no inter-group displacement: a dynamic stride is inert.
    k = tirx.Var("k", "int32")
    call = _vload(_load(work, [_int(0)]), size=64, group=1, stride=k)
    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=0, e=64, d=0, rw=1)
    assert _annotation_text(out.annotations, "stride") == "k"

    with pytest.raises(tvm.error.InternalError, match="stride"):
        _run(_evaluate(_vload(_load(work, [_int(0)]), size=64, group=2, stride=k)))

    # The block stride is expressed in 32-byte blocks: E=(K-1)*S*B+B.
    call = _vload(_load(work, [_int(0)]), size=64, block_stride=2)
    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=0, e=120, d=0, rw=1)

    with pytest.raises(tvm.error.InternalError, match="block"):
        _run(_evaluate(_vload(_load(work, [_int(0)]), size=64, block_stride=k)))
    with pytest.raises(tvm.error.InternalError, match="block"):
        _run(_evaluate(_vload(_load(work, [_int(0)]), size=64, block_stride=-1)))


def test_negative_index_and_group_candidates():
    # Negative indices are normalized by the preceding pass before descriptor
    # construction consumes the scalar origin.
    work = _buffer((512,))
    call = _vload(_load(work, [_int(-64)]), size=64)
    normalized = _legalize_negative(_evaluate(call))
    (normalized_load,) = [node for node in nodes(normalized, tirx.BufferLoad) if node.buffer.data.same_as(work.data)]
    assert _constant(normalized_load.indices[0]) == 448
    body = _run(normalized)
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=448, e=64, d=0, rw=1)

    work2 = _buffer((4, 64))
    call = _vload(_load(work2, [_int(2), _int(-32)]), size=64, dist_mode="brc")
    body = _run(_legalize_negative(_evaluate(call)))
    (out,) = _vmi_calls(body)
    # The normalized row-major address is 2*64+32, never 2*64-32.
    _assert_plan(out, address_index=0, b=160, e=1, d=0, rw=1)

    # A padded view relocates to its origin and declares the padded prefix.
    padded = _buffer((4, 64), strides=(128, 1), elem_offset=16)
    call = _vload(_load(padded, [_int(2), _int(0)]), size=64, group=2, stride=-64)
    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body)
    base, pointer = _descriptor(out, 0)
    assert all(_constant(index) == 0 for index in base.indices)
    assert _constant(pointer.args[1]) == 320
    assert _constant(out.args[1]) == 256
    assert _constant(_logical_offset(padded, [_int(2), _int(0)])) == 256, "B+D preserves the source address"
    assert _constant(_logical_offset(base.buffer, base.indices)) + _constant(out.args[1]) == 256

    with pytest.raises(tvm.error.InternalError, match="below"):
        _run(_evaluate(_vload(_load(padded, [_int(0), _int(0)]), size=64, group=2, stride=-64)))

    # A tight two-dimensional buffer falls through to the whole-buffer origin.
    tight = _buffer((2, 128))
    call = _vload(_load(tight, [_int(1), _int(64)]), size=64, group=2, stride=-64)
    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body)
    base, pointer = _descriptor(out, 0)
    assert all(_constant(index) == 0 for index in base.indices)
    assert _constant(pointer.args[1]) == 256
    assert _constant(out.args[1]) == 192
    assert _constant(_logical_offset(base.buffer, base.indices)) + _constant(out.args[1]) == 192


# ---------------------------------------------------------------------------
# Explicit declarations and direct handle bindings
# ---------------------------------------------------------------------------
def test_explicit_descriptor_preserved_and_diagnosed():
    work = _buffer((512,))
    pointer = _access_ptr(work, [_int(32)], 128, 3)
    call = _vload(pointer, 16, size=64)

    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(pointer), "an explicit pointer keeps its original operand"
    assert _constant(out.args[1]) == 16
    assert _constant(pointer.args[1]) == 128
    assert _constant(pointer.args[2]) == 3, "rw_mask=3 is preserved"

    handle = tirx.Var("p", "handle")
    bound = tirx.SeqStmt([tirx.Bind(handle, pointer), _evaluate(_vload(handle, 16, size=64))])
    body = _run(bound)
    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(handle), "a directly bound handle stays the pointer operand"
    assert _constant(out.args[1]) == 16

    let_handle = tirx.Var("let_p", "handle")
    body = _run(_evaluate(tirx.Let(let_handle, pointer, _vload(let_handle, 16, size=64))))
    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(let_handle), "a Let-bound handle stays the pointer operand"
    assert _constant(out.args[1]) == 16
    let = body.value if isinstance(body, tirx.Evaluate) else body
    assert isinstance(let, tirx.Let) and let.value.same_as(pointer)

    # The Let definition is lexical to its body.
    outside = tirx.SeqStmt(
        [
            _evaluate(tirx.Let(let_handle, pointer, _vload(let_handle, 16, size=64))),
            _evaluate(_vload(let_handle, 16, size=64)),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="no recoverable definition"):
        _run(outside)

    alias = tirx.Var("q", "handle")
    nested = tirx.SeqStmt(
        [
            tirx.Bind(handle, pointer),
            tirx.Bind(alias, handle),
            _evaluate(_vload(alias, 16, size=64)),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="multi-level"):
        _run(nested)

    with pytest.raises(tvm.error.InternalError, match="exceeds"):
        _run(_evaluate(_vload(_access_ptr(work, [_int(0)], 1, 1), 0, size=64)))
    with pytest.raises(tvm.error.InternalError, match="read access"):
        _run(_evaluate(_vload(_access_ptr(work, [_int(0)], 128, 2), 0, size=64)))
    with pytest.raises(tvm.error.InternalError, match="write access"):
        _run(_evaluate(_vstore(_vec("v"), _access_ptr(work, [_int(0)], 128, 1))))
    with pytest.raises(tvm.error.InternalError, match="scalar integer"):
        _run(_evaluate(_vload(_load(work, [_int(0)]), tirx.Var("bad", "handle"), size=64)))


# ---------------------------------------------------------------------------
# Logical coverage, not only physical containment
# ---------------------------------------------------------------------------
def test_logical_coverage_counterexamples():
    work = _buffer((2, 128))
    body = _run(_evaluate(_vload(_load(work, [_int(0), _int(64)]), size=64)))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=64, e=64, d=0, rw=1)

    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(_evaluate(_vload(_load(work, [_int(0), _int(64)]), size=128)))

    body = _run(_evaluate(_vload(_load(work, [_int(0), _int(64)]), size=192)))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=64, e=192, d=0, rw=1)

    padded = _buffer((2, 64), strides=(128, 1))
    with pytest.raises(tvm.error.InternalError, match="tight row-major"):
        _run(_evaluate(_vload(_load(padded, [_int(0), _int(0)]), size=64)))

    permuted = _buffer((2, 64), strides=(1, 2))
    pointer = _access_ptr(permuted, [_int(0), _int(0)], 128, 1)
    body = _run(_evaluate(_vload(pointer, 0, size=128)))
    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(pointer)
    assert _constant(pointer.args[1]) == 128

    broken = _buffer((2, 64), strides=(1, 3))
    with pytest.raises(tvm.error.InternalError, match="dense permutation"):
        _run(_evaluate(_vload(_access_ptr(broken, [_int(0), _int(0)], 128, 1), 0, size=128)))


# ---------------------------------------------------------------------------
# FP4: prove in logical units, then pack the scalar offset once
# ---------------------------------------------------------------------------
def test_fp4_logical_units_and_packing():
    fp4 = _buffer((256,), dtype="float4_e2m1fn", name="fp4")
    pointer = _access_ptr(fp4, [_int(32)], 128, 1)
    call = _vload(pointer, 16, elem="float4_e2m1fn", size=64)
    alloc = tirx.AllocBuffer(fp4, annotations={})

    body = _run(tirx.SeqStmt([alloc, _evaluate(call)]))
    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(pointer), "the descriptor base/extent stay in logical units"
    assert _constant(pointer.args[1]) == 128
    assert _constant(out.args[1]) == 8, "the independent scalar offset is packed once"

    mod = _module(tirx.SeqStmt([alloc, _evaluate(call)]))
    mod = ascend.LegalizePTOVMIMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = ascend.RewriteFp4ToFp4x2()(mod)
    # The production pipeline simplifies right after the FP4 rewrite
    # (tilelang/ascend/pipeline.py), so a host-only check reproduces that step.
    mod = tirx.transform.Simplify()(mod)
    (packed,) = _vmi_calls(mod)
    packed_ptr = packed.args[0]
    assert isinstance(packed_ptr, tirx.Call) and packed_ptr.op.name == "tirx.tvm_access_ptr"
    assert _constant(packed_ptr.args[2]) == 16
    assert _constant(packed_ptr.args[3]) == 64
    assert _constant(packed.args[1]) == 8

    with pytest.raises(tvm.error.InternalError, match="even FP4"):
        _run(
            tirx.SeqStmt(
                [
                    tirx.AllocBuffer(fp4, annotations={}),
                    _evaluate(_vload(pointer, 17, elem="float4_e2m1fn", size=64)),
                ]
            )
        )

    d = tirx.Var("d", "int32")
    guarded = tirx.IfThenElse(
        tirx.And(d >= 0, d <= 64),
        _evaluate(_vload(pointer, d, elem="float4_e2m1fn", size=64)),
        _evaluate(0),
    )
    body = _run(tirx.SeqStmt([tirx.AllocBuffer(fp4, annotations={}), guarded]))
    (out,) = _vmi_calls(body)
    assert isinstance(out.args[1], tirx.FloorDiv)
    assert out.args[1].a.same_as(d) and _constant(out.args[1].b) == 2


# ---------------------------------------------------------------------------
# Branch, For and explicit context
# ---------------------------------------------------------------------------
def _bounded_2d(body_fn, lower=0, upper=1):
    k = tirx.Var("k", "int32")
    condition = tirx.And(k >= lower, k <= upper) if lower is not None else tirx.IntImm("bool", 1)
    return k, tirx.IfThenElse(condition, body_fn(k), _evaluate(0))


def _access_k(k, work, *, size=64):
    return _vload(_load(work, [_int(0), k * 64]), size=size)


def test_branch_for_and_explicit_context():
    work = _buffer((2, 128))
    k, body = _bounded_2d(lambda k: _evaluate(_access_k(k, work)), 0, 1)
    after = _run(body, params=[k])
    (out,) = _vmi_calls(after)
    base, pointer = _descriptor(out, 0)
    assert tvm.ir.structural_equal(base.indices[1], k * 64) and _constant(pointer.args[1]) == 64

    k, body = _bounded_2d(lambda k: _evaluate(_access_k(k, work)), 0, 2)
    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(body, params=[k])

    k = tirx.Var("k", "int32")
    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(_evaluate(_access_k(k, work)), params=[k])

    k = tirx.Var("k", "int32")
    outside = tirx.SeqStmt(
        [
            tirx.IfThenElse(
                tirx.And(k >= 0, k <= 1),
                _evaluate(_vload(_load(work, [_int(0), _int(0)]), size=64)),
                _evaluate(0),
            ),
            _evaluate(_access_k(k, work)),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(outside, params=[k])

    i = tirx.Var("i", "int32")
    loop = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, _evaluate(_access_k(i, work)))
    after = _run(loop)
    (out,) = _vmi_calls(after)
    assert tvm.ir.structural_equal(_descriptor(out, 0)[0].indices[1], i * 64)

    i = tirx.Var("i2", "int32")
    loop = tirx.For(i, 0, 3, tirx.ForKind.SERIAL, _evaluate(_access_k(i, work)))
    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(loop)

    # A fact from one branch never reaches the next sibling branch.
    k = tirx.Var("k", "int32")
    siblings = tirx.SeqStmt(
        [
            tirx.IfThenElse(k == 0, _evaluate(_vload(_load(work, [_int(0), _int(0)]), size=64)), _evaluate(0)),
            tirx.IfThenElse(k < 2, _evaluate(_access_k(k, work)), _evaluate(0)),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(siblings, params=[k])

    k = tirx.Var("k", "int32")
    siblings = tirx.SeqStmt(
        [
            tirx.IfThenElse(k == 0, _evaluate(_vload(_load(work, [_int(0), _int(0)]), size=64)), _evaluate(0)),
            tirx.IfThenElse(tirx.And(k >= 0, k < 2), _evaluate(_access_k(k, work)), _evaluate(0)),
        ]
    )
    after = _run(siblings, params=[k])
    assert len(_vmi_calls(after)) == 2

    # The explicit path sees the same branch context and rejects outside it.
    k = tirx.Var("k", "int32")
    pointer = _access_ptr(work, [_int(0), k * 64], 64, 1)
    inside = tirx.IfThenElse(tirx.And(k >= 0, k <= 1), _evaluate(_vload(pointer, 0, size=64)), _evaluate(0))
    after = _run(inside, params=[k])
    assert _vmi_calls(after)[0].args[0].same_as(pointer)
    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(_evaluate(_vload(pointer, 0, size=64)), params=[k])


# ---------------------------------------------------------------------------
# While and mutable-index default analyzer behavior
# ---------------------------------------------------------------------------
def test_while_uses_existing_context_only():
    work = _buffer((2, 128))
    body = _run(tirx.While(tirx.IntImm("bool", 1), _evaluate(_vload(_load(work, [_int(0), _int(0)]), size=64))))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=0, e=64, d=0, rw=1)

    k = tirx.Var("k", "int32")
    outer_if = tirx.IfThenElse(
        tirx.And(k >= 0, k <= 1),
        tirx.While(tirx.IntImm("bool", 1), _evaluate(_access_k(k, work))),
        _evaluate(0),
    )
    after = _run(outer_if, params=[k])
    assert len(_vmi_calls(after)) == 1

    k = tirx.Var("k", "int32")
    while_only = tirx.While(
        tirx.And(k >= 0, k < 2),
        _evaluate(_access_k(k, work)),
    )
    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(while_only, params=[k])

    k = tirx.Var("k", "int32")
    inner_if = tirx.While(
        tirx.IntImm("bool", 1),
        tirx.IfThenElse(tirx.And(k >= 0, k <= 1), _evaluate(_access_k(k, work)), _evaluate(0)),
    )
    after = _run(inner_if, params=[k])
    assert len(_vmi_calls(after)) == 1


def test_mutable_index_is_not_a_stable_context_fact():
    work = _buffer((2, 128))
    index = tirx.decl_buffer((1,), "int32", name="index", scope="local")
    guard = tirx.And(index[0] >= 0, index[0] <= 1)
    body = tirx.IfThenElse(guard, _evaluate(_vload(_load(work, [_int(0), index[0] * 64]), size=64)), _evaluate(0))
    # The default analyzer does not turn a mutable buffer read into a stable
    # fact, so this access stays unproven; no write tracking is added here.
    with pytest.raises(tvm.error.InternalError, match="logical pointer-region"):
        _run(body)


# ---------------------------------------------------------------------------
# Target isolation
# ---------------------------------------------------------------------------
def test_passes_through_non_target_and_non_vmi():
    work = _buffer((64,))
    body = _evaluate(_vload(_load(work, [_int(0)]), size=64))

    for target in (None, Target({"kind": "cuda"})):
        mod = _module(body, target=target)
        after = ascend.LegalizePTOVMIMemoryAccess()(mod)
        assert after["main"].body.same_as(mod["main"].body)

    pointer = _access_ptr(work, [_int(0)], 64, 1)
    mod = _module(_evaluate(pointer))
    after = ascend.LegalizePTOVMIMemoryAccess()(mod)
    assert after["main"].body.same_as(mod["main"].body)
