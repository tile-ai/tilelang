"""High-value direct-TIR regressions for ``LegalizePTOVMIMemoryAccess``.

The pass consumes post-``LegalizeNegativeIndex`` IR: an automatic VMI address
is the original scalar-origin ``BufferLoad`` and an explicit address is a
``tl.access_ptr`` call, either direct or reached through one direct handle
``Bind``/``Let``.  These cases pin descriptor construction, physical
envelopes, negative-index/group candidates and explicit preservation in the
C++ legalization pass that the frontend tests cannot inspect directly.

These are host-only IR tests covering the contracts that must remain in-tree.
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


def _vstore(value, destination, offset=0, mask=None, **annotations):
    args = [value, destination, offset if isinstance(offset, tirx.PrimExpr) else _int(offset)]
    if mask is not None:
        args.append(mask)
    attrs = {key: _annotation(val) for key, val in annotations.items() if val is not None}
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


def _assert_no_raw_address(body):
    names = (
        "tl.vmi.vload",
        "tl.vmi.vstore",
        "tl.vmi.vgather",
        "tl.vmi.vscatter",
    )
    for name in names:
        for call in _vmi_calls(body, name):
            if name.endswith("vload"):
                address_index = 0
            elif name.endswith("vstore") or name.endswith("vscatter"):
                address_index = 1
            else:
                address_index = 0
            address = call.args[address_index]
            assert not isinstance(address, (tirx.BufferLoad, tirx.Buffer, tirx.BufferRegion)), address


def _assert_plan(call, *, address_index, b, e, d, rw):
    base, pointer = _descriptor(call, address_index)
    assert _constant(_logical_offset(base.buffer, base.indices)) == b
    assert _constant(pointer.args[1]) == e
    assert _constant(pointer.args[2]) == rw
    assert _constant(call.args[address_index + 1]) == d
    return base, pointer


def _indexed_gather(op_name, source, offset=0, *, offsets=None, mask=None, elem="float32", lanes=64, **annotations):
    offsets = offsets if offsets is not None else tirx.Var("offsets", f"uint32x{lanes}")
    mask = mask if mask is not None else tirx.Var("mask", f"boolx{lanes}")
    attrs = {key: _annotation(value) for key, value in annotations.items() if value is not None}
    return tirx.call_intrin(
        f"{elem}x{lanes}",
        tvm.ir.Op.get(op_name),
        source,
        offset if isinstance(offset, tirx.PrimExpr) else _int(offset),
        offsets,
        mask,
        annotations=attrs or None,
    )


def _vgather(source, offset=0, **kwargs):
    return _indexed_gather("tl.vmi.vgather", source, offset, **kwargs)


def _vscatter(value, destination, offset=0, *, offsets=None, mask=None, lanes=64, **annotations):
    offsets = offsets if offsets is not None else tirx.Var("offsets", f"uint32x{lanes}")
    mask = mask if mask is not None else tirx.Var("mask", f"boolx{lanes}")
    attrs = {key: _annotation(value) for key, value in annotations.items() if value is not None}
    return tirx.call_intrin(
        "void",
        tvm.ir.Op.get("tl.vmi.vscatter"),
        value,
        destination,
        offset if isinstance(offset, tirx.PrimExpr) else _int(offset),
        offsets,
        mask,
        annotations=attrs or None,
    )


def _sblock(body, *buffers, init=None):
    return tirx.SBlock([], [], [], "root", body, init=init, alloc_buffers=list(buffers))


def _assert_owner_plan(call, *, address_index, owner, e, d, rw):
    base, pointer = _descriptor(call, address_index)
    assert base.buffer.same_as(owner), (base.buffer, owner)
    assert all(_constant(index) == 0 for index in base.indices)
    assert str(pointer.args[1].dtype) == "int64", pointer.args[1]
    assert _constant(pointer.args[1]) == e
    assert _constant(pointer.args[2]) == rw
    assert _constant(call.args[address_index + 1]) == d
    return base, pointer


def _view(owner, shape, *, name="view", strides=None, elem_offset=0, dtype=None, scope=None):
    kwargs = {
        "name": name,
        "data": owner.data,
        "scope": owner.scope() if scope is None else scope,
    }
    if strides is not None:
        kwargs["strides"] = tuple(value if isinstance(value, tirx.PrimExpr) else _int(value) for value in strides)
    if isinstance(elem_offset, tirx.PrimExpr) or elem_offset:
        kwargs["elem_offset"] = elem_offset if isinstance(elem_offset, tirx.PrimExpr) else _int(elem_offset)
    return tirx.decl_buffer(shape, dtype or owner.dtype, **kwargs)


# ---------------------------------------------------------------------------
# Ordinary descriptor construction and envelopes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "size,extent",
    [
        (64, 64),
        (80, 128),
    ],
    ids=["load64", "load-tail80"],
)
@pytest.mark.pto
def test_ordinary_load_envelope(size, extent):
    work = _buffer((512,))
    call = _vload(_load(work, [_int(64)]), size=size)

    body = _run(_evaluate(call))

    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=64, e=extent, d=0, rw=1)
    assert out.dtype == call.dtype, "the VMI result dtype is preserved"
    assert _constant(out.annotations["size"]) == size
    _assert_no_raw_address(body)


@pytest.mark.pto
def test_store_mask_keeps_full_extent():
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


@pytest.mark.pto
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

    # A single logical lane still uses the block-stride envelope, not the
    # ordinary single-element broadcast shortcut.
    call = _vload(_load(work, [_int(0)]), size=1, block_stride=2)
    body = _run(_evaluate(call))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=0, e=120, d=0, rw=1)

    with pytest.raises(tvm.error.InternalError, match="block"):
        _run(_evaluate(_vload(_load(work, [_int(0)]), size=64, block_stride=k)))
    with pytest.raises(tvm.error.InternalError, match="block"):
        _run(_evaluate(_vload(_load(work, [_int(0)]), size=64, block_stride=-1)))


@pytest.mark.pto
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
@pytest.mark.pto
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
# Ordinary single-element broadcast loads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "float16",
        "bfloat16",
        "float32",
        "float8_e4m3fn",
        "float8_e5m2",
    ],
)
@pytest.mark.pto
def test_single_element_load_uses_scalar_broadcast_envelope(dtype):
    work = _buffer((2,), dtype=dtype)
    call = _vload(_load(work, [_int(1)]), elem=dtype, size=1)

    body = _run(_evaluate(call))

    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=1, e=1, d=0, rw=1)
    assert out.dtype == call.dtype
    assert _constant(out.annotations["size"]) == 1


@pytest.mark.pto
def test_single_element_explicit_descriptor_is_not_shrunk():
    work = _buffer((512,))
    pointer = _access_ptr(work, [_int(32)], 64, 1)
    call = _vload(pointer, 0, size=1)

    body = _run(_evaluate(call))

    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(pointer)
    assert _constant(pointer.args[1]) == 64
    assert _constant(out.args[1]) == 0

    small = _buffer((2,))
    explicit_one = _access_ptr(small, [_int(1)], 1, 1)
    body = _run(_evaluate(_vload(explicit_one, 0, size=1)))
    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(explicit_one)
    assert _constant(explicit_one.args[1]) == 1


# ---------------------------------------------------------------------------
# Descriptor construction without a local logical-region proof
# ---------------------------------------------------------------------------
@pytest.mark.pto
def test_descriptor_construction_without_logical_coverage():
    work = _buffer((2, 128))
    body = _run(_evaluate(_vload(_load(work, [_int(0), _int(64)]), size=64)))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=64, e=64, d=0, rw=1)

    body = _run(_evaluate(_vload(_load(work, [_int(0), _int(64)]), size=128)))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=64, e=128, d=0, rw=1)

    body = _run(_evaluate(_vload(_load(work, [_int(0), _int(64)]), size=192)))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=64, e=192, d=0, rw=1)

    padded = _buffer((2, 64), strides=(128, 1))
    body = _run(_evaluate(_vload(_load(padded, [_int(0), _int(0)]), size=64)))
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=0, e=64, d=0, rw=1)

    permuted = _buffer((2, 64), strides=(1, 2))
    pointer = _access_ptr(permuted, [_int(0), _int(0)], 128, 1)
    body = _run(_evaluate(_vload(pointer, 0, size=128)))
    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(pointer)
    assert _constant(pointer.args[1]) == 128

    broken = _buffer((2, 64), strides=(1, 3))
    pointer = _access_ptr(broken, [_int(0), _int(0)], 128, 1)
    body = _run(_evaluate(_vload(pointer, 0, size=128)))
    (out,) = _vmi_calls(body)
    assert out.args[0].same_as(pointer)
    assert _constant(pointer.args[1]) == 128


# ---------------------------------------------------------------------------
# FP4: prove in logical units, then pack the scalar offset once
# ---------------------------------------------------------------------------
@pytest.mark.pto
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


@pytest.mark.pto
def test_branch_for_and_while_context():
    work = _buffer((2, 128))
    k, body = _bounded_2d(lambda k: _evaluate(_access_k(k, work)), 0, 1)
    after = _run(body, params=[k])
    (out,) = _vmi_calls(after)
    base, pointer = _descriptor(out, 0)
    assert tvm.ir.structural_equal(base.indices[1], k * 64) and _constant(pointer.args[1]) == 64

    k, body = _bounded_2d(lambda k: _evaluate(_access_k(k, work)), 0, 2)
    after = _run(body, params=[k])
    (out,) = _vmi_calls(after)
    assert tvm.ir.structural_equal(_descriptor(out, 0)[0].indices[1], k * 64)

    k = tirx.Var("k", "int32")
    after = _run(_evaluate(_access_k(k, work)), params=[k])
    (out,) = _vmi_calls(after)
    assert tvm.ir.structural_equal(_descriptor(out, 0)[0].indices[1], k * 64)

    i = tirx.Var("i", "int32")
    loop = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, _evaluate(_access_k(i, work)))
    after = _run(loop)
    (out,) = _vmi_calls(after)
    assert tvm.ir.structural_equal(_descriptor(out, 0)[0].indices[1], i * 64)

    i = tirx.Var("i2", "int32")
    loop = tirx.For(i, 0, 3, tirx.ForKind.SERIAL, _evaluate(_access_k(i, work)))
    after = _run(loop)
    (out,) = _vmi_calls(after)
    assert tvm.ir.structural_equal(_descriptor(out, 0)[0].indices[1], i * 64)

    body = _run(
        tirx.While(
            tirx.IntImm("bool", 1),
            _evaluate(_vload(_load(work, [_int(0), _int(0)]), size=64)),
        )
    )
    (out,) = _vmi_calls(body)
    _assert_plan(out, address_index=0, b=0, e=64, d=0, rw=1)

    k = tirx.Var("k_while", "int32")
    outer_if = tirx.IfThenElse(
        tirx.And(k >= 0, k <= 1),
        tirx.While(tirx.IntImm("bool", 1), _evaluate(_access_k(k, work))),
        _evaluate(0),
    )
    assert len(_vmi_calls(_run(outer_if, params=[k]))) == 1

    k = tirx.Var("k_while_only", "int32")
    while_only = tirx.While(
        tirx.And(k >= 0, k < 2),
        _evaluate(_access_k(k, work)),
    )
    assert len(_vmi_calls(_run(while_only, params=[k]))) == 1

    k = tirx.Var("k_inner_if", "int32")
    inner_if = tirx.While(
        tirx.IntImm("bool", 1),
        tirx.IfThenElse(
            tirx.And(k >= 0, k <= 1),
            _evaluate(_access_k(k, work)),
            _evaluate(0),
        ),
    )
    assert len(_vmi_calls(_run(inner_if, params=[k]))) == 1


# ---------------------------------------------------------------------------
# Mutable-index descriptor construction
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_mutable_index_keeps_dynamic_descriptor():
    work = _buffer((2, 128))
    index = tirx.decl_buffer((1,), "int32", name="index", scope="local")
    guard = tirx.And(index[0] >= 0, index[0] <= 1)
    body = tirx.IfThenElse(guard, _evaluate(_vload(_load(work, [_int(0), index[0] * 64]), size=64)), _evaluate(0))
    after = _run(body)
    (out,) = _vmi_calls(after)
    base, pointer = _descriptor(out, 0)
    assert tvm.ir.structural_equal(base.indices[1], index[0] * 64)
    assert _constant(pointer.args[1]) == 64


# ---------------------------------------------------------------------------
# Indexed gather/scatter: real owner descriptors
# ---------------------------------------------------------------------------
@pytest.mark.pto
def test_indexed_uses_active_owner_descriptor():
    work = _buffer((256,))
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    # The real-pipeline tests assert B0/E256/D64 for gather and scatter; one
    # direct-TIR case pins the non-address operands and an annotation that the
    # pipeline does not inspect.
    call = _vgather(_load(work, [_int(64)]), offsets=offsets, mask=mask, pmode="zero")
    body = _run(_sblock(_evaluate(call), work))
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    _assert_owner_plan(out, address_index=0, owner=work, e=256, d=64, rw=1)
    assert out.dtype == call.dtype, "the gather result dtype is preserved"
    assert out.args[2].same_as(offsets) and out.args[3].same_as(mask)
    assert _annotation_text(out.annotations, "pmode") == "zero"
    _assert_no_raw_address(body)

    # A direct explicit pointer is reconstructed from the same owner; its old
    # lanes extent is not retained as the descriptor extent.
    pointer = _access_ptr(work, [_int(64)], 64, 1)
    call = _vgather(pointer, offsets=offsets, mask=mask)
    body = _run(_sblock(_evaluate(call), work))
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    _assert_owner_plan(out, address_index=0, owner=work, e=256, d=64, rw=1)
    _assert_no_raw_address(body)


def _assert_typed_owner_plan(call, *, address_index, owner, dtype, e, d, rw):
    base, pointer = _descriptor(call, address_index)
    assert not base.buffer.same_as(owner)
    assert base.buffer.data.same_as(owner.data)
    assert base.buffer.dtype == owner.dtype or str(base.buffer.dtype) == dtype
    assert str(base.buffer.dtype) == dtype
    assert all(_constant(index) == 0 for index in base.indices)
    assert str(pointer.args[1].dtype) == "int64"
    assert _constant(pointer.args[1]) == e
    assert _constant(pointer.args[2]) == rw
    assert _constant(call.args[address_index + 1]) == d
    return base, pointer


@pytest.mark.parametrize(
    ("owner_dtype", "view_dtype"),
    [
        ("uint8", "uint16"),
        ("uint16", "uint8"),
        ("int16", "uint16"),
    ],
    ids=["widen-elements", "narrow-elements", "same-width-reinterpret"],
)
@pytest.mark.pto
def test_indexed_dtype_view_uses_access_element_units(owner_dtype, view_dtype):
    owner_bits = int(owner_dtype[-1]) if owner_dtype[-1] == "8" else 16
    view_bits = int(view_dtype[-1]) if view_dtype[-1] == "8" else 16
    owner = _buffer((2048 // owner_bits,), dtype=owner_dtype)
    view = _view(owner, (2048 // view_bits,), dtype=view_dtype)
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")
    call = _vgather(_load(view, [_int(3)]), offsets=offsets, mask=mask, elem=view_dtype)

    body = _run(_sblock(_evaluate(call), owner))

    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    _assert_typed_owner_plan(out, address_index=0, owner=owner, dtype=view_dtype, e=2048 // view_bits, d=3, rw=1)
    assert out.dtype == call.dtype
    assert out.args[2].same_as(offsets) and out.args[3].same_as(mask)


@pytest.mark.pto
def test_indexed_dtype_view_elem_offset_is_counted_once():
    owner = _buffer((256,), dtype="uint8")
    view = _view(owner, (64,), dtype="uint16", elem_offset=8)
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    for explicit in (False, True):
        source = _access_ptr(view, [_int(3)], 64, 1) if explicit else _load(view, [_int(3)])
        call = _vgather(source, offsets=offsets, mask=mask, elem="uint16")
        body = _run(_sblock(_evaluate(call), owner))
        (out,) = _vmi_calls(body, "tl.vmi.vgather")
        base, _ = _assert_typed_owner_plan(out, address_index=0, owner=owner, dtype="uint16", e=128, d=11, rw=1)
        assert base.buffer.elem_offset == 0


@pytest.mark.pto
def test_scatter_value_matches_access_view_dtype():
    owner = _buffer((256,), dtype="uint8")
    view = _view(owner, (128,), dtype="uint16")
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")
    value = _vec("value", "uint16")
    call = _vscatter(value, _load(view, [_int(3)]), offsets=offsets, mask=mask)

    body = _run(_sblock(_evaluate(call), owner))

    (out,) = _vmi_calls(body, "tl.vmi.vscatter")
    _assert_typed_owner_plan(out, address_index=1, owner=owner, dtype="uint16", e=128, d=3, rw=2)
    assert out.args[0].same_as(value)

    wrong_value = _vec("wrong_value", "uint8")
    wrong = _vscatter(wrong_value, _load(view, [_int(3)]), offsets=offsets, mask=mask)
    with pytest.raises(tvm.error.InternalError, match="destination element dtype"):
        _run(_sblock(_evaluate(wrong), owner))


@pytest.mark.pto
def test_indexed_dtype_view_rejects_unsafe_layout_and_range():
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")
    owner = _buffer((256,), dtype="uint8")

    too_large = _view(owner, (129,), dtype="uint16")
    with pytest.raises(tvm.error.InternalError, match="view range"):
        _run(_sblock(_evaluate(_vgather(_load(too_large, [_int(0)]), offsets=offsets, mask=mask, elem="uint16")), owner))

    padded = _view(owner, (64,), dtype="uint16", strides=(128,))
    with pytest.raises(tvm.error.InternalError, match="tight row-major"):
        _run(_sblock(_evaluate(_vgather(_load(padded, [_int(0)]), offsets=offsets, mask=mask, elem="uint16")), owner))

    unaligned = tirx.decl_buffer((256,), "uint8", name="unaligned_owner", scope="shared", data_alignment=1)
    words = _view(unaligned, (128,), dtype="uint16")
    with pytest.raises(tvm.error.InternalError, match="aligned"):
        _run(_sblock(_evaluate(_vgather(_load(words, [_int(0)]), offsets=offsets, mask=mask, elem="uint16")), unaligned))

    huge = _buffer((2**62,), dtype="uint16", name="huge_owner")
    bytes_view = _view(huge, (1,), dtype="uint8")
    with pytest.raises(tvm.error.InternalError, match="owner bit capacity"):
        _run(_sblock(_evaluate(_vgather(_load(bytes_view, [_int(0)]), offsets=offsets, mask=mask)), huge))

    owner32 = _buffer((64,), dtype="uint32")
    view64 = _view(owner32, (32,), dtype="uint64")
    with pytest.raises(tvm.error.InternalError, match="byte-addressable"):
        _run(_sblock(_evaluate(_vgather(_load(view64, [_int(0)]), offsets=offsets, mask=mask, elem="uint64")), owner32))


@pytest.mark.pto
def test_indexed_dtype_view_requires_exact_dynamic_capacity_conversion():
    n = tirx.Var("n", "int32")
    owner = tirx.decl_buffer((n,), "uint8", name="dynamic_bytes", scope="shared")
    view = tirx.decl_buffer((tirx.FloorDiv(n, _int(2)),), "uint16", name="dynamic_words", scope="shared", data=owner.data)
    call = _vgather(_load(view, [_int(0)]), offsets=_vec("offsets", "uint32"), mask=_mask("mask"), elem="uint16")

    bounded = tirx.IfThenElse(tirx.And(n > 0, n <= 256), _evaluate(call), _evaluate(0))
    with pytest.raises(tvm.error.InternalError, match="without truncating bits"):
        _run(_sblock(bounded, owner), params=[n])

    divisible = tirx.IfThenElse(tirx.And(tirx.And(n > 0, n <= 256), tirx.FloorMod(n, _int(2)) == 0), _evaluate(call), _evaluate(0))
    body = _run(_sblock(divisible, owner), params=[n])
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base, pointer = _descriptor(out, 0)
    assert base.buffer.data.same_as(owner.data)
    assert str(base.buffer.dtype) == "uint16"
    expected = tirx.FloorDiv(tirx.Cast("int64", n), _int(2, "int64"))
    tvm.ir.assert_structural_equal(pointer.args[1], expected)
    tvm.ir.assert_structural_equal(base.buffer.shape[0], expected)


@pytest.mark.pto
def test_indexed_owner_extent_is_int64_and_preserves_source_types():
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    # A small static capacity is int64; the source int32 shape/stride nodes
    # and dtypes stay untouched.
    shape32 = _int(64, "int32")
    stride32 = _int(1, "int32")
    small = tirx.decl_buffer((shape32,), "float32", name="small_owner", scope="shared", strides=(stride32,))
    call = _vgather(_load(small, [_int(0)]), offsets=offsets, mask=mask)
    (out,) = _vmi_calls(_run(_sblock(_evaluate(call), small)), "tl.vmi.vgather")
    _, pointer = _assert_owner_plan(out, address_index=0, owner=small, e=64, d=0, rw=1)
    assert isinstance(pointer.args[1], tirx.IntImm)
    assert small.shape[0].same_as(shape32) and small.strides[0].same_as(stride32)
    assert str(small.shape[0].dtype) == str(small.strides[0].dtype) == "int32"

    # An int64 shape is already in the target domain.
    n64 = tirx.Var("n64", "int64")
    owner64 = tirx.decl_buffer((n64,), "float32", name="owner64", scope="shared")
    call64 = _vgather(_load(owner64, [_int(0)]), offsets=offsets, mask=mask)
    guarded = tirx.IfThenElse(n64 > _int(0, "int64"), _evaluate(call64), _evaluate(0))
    (out64,) = _vmi_calls(_run(_sblock(guarded, owner64), params=[n64]), "tl.vmi.vgather")
    _, pointer64 = _descriptor(out64, 0)
    assert owner64.shape[0].same_as(n64) and pointer64.args[1].same_as(n64)

    # Static capacity still uses the __int128 overflow check.
    huge = _int(2**32, "int64")
    overflowing = tirx.decl_buffer((huge, huge), "float32", name="overflowing", scope="shared")
    call = _vgather(_load(overflowing, [_int(0), _int(0)]), offsets=offsets, mask=mask)
    with pytest.raises(tvm.error.InternalError, match="overflows int64"):
        _run(_sblock(_evaluate(call), overflowing))


@pytest.mark.pto
def test_indexed_owner_requirements():
    work = _buffer((512,), name="owner")
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    def gather(source, **kwargs):
        return _vgather(source, offsets=offsets, mask=mask, **kwargs)

    # No allocation: a buffer parameter or a bare declaration is not an owner.
    with pytest.raises(tvm.error.InternalError, match="no active allocation"):
        _run(_evaluate(gather(_load(work, [_int(32)]))))

    # A source view may be padded and have a non-zero elem_offset; the owner
    # only supplies identity, layout and capacity.  The padded prefix is
    # counted exactly once in the complete source offset (S=304).
    view = _view(work, (4, 64), strides=(256, 1), elem_offset=16)
    body = _run(_sblock(_evaluate(gather(_load(view, [_int(1), _int(32)]))), work))
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    _assert_owner_plan(out, address_index=0, owner=work, e=512, d=304, rw=1)

    # Same Buffer object registered twice is one allocation, not ambiguity.
    body = _run(_sblock(_evaluate(gather(_load(work, [_int(32)]))), work, work))
    assert len(_vmi_calls(body, "tl.vmi.vgather")) == 1

    # Distinct allocations that share data are ambiguous, even with identical layout.
    alias = _view(work, (512,), name="alias")
    with pytest.raises(tvm.error.InternalError, match="ambiguous"):
        _run(_sblock(_evaluate(gather(_load(work, [_int(32)]))), work, alias))

    # Owner layout must be tight row-major; a source view may still be padded.
    padded_owner = _buffer((2, 256), name="padded_owner", strides=(512, 1))
    padded_view = _view(padded_owner, (512,), name="padded_view")
    with pytest.raises(tvm.error.InternalError, match="tight row-major"):
        _run(_sblock(_evaluate(gather(_load(padded_view, [_int(32)]))), padded_owner))

    # The owner origin must be provably zero and its capacity positive.
    offset_owner = _buffer((512,), name="offset_owner", elem_offset=16)
    offset_view = _view(offset_owner, (512,), name="offset_view")
    with pytest.raises(tvm.error.InternalError, match="elem_offset"):
        _run(_sblock(_evaluate(gather(_load(offset_view, [_int(32)]))), offset_owner))

    empty_owner = _buffer((0,), name="empty_owner")
    empty_view = _view(empty_owner, (1,), name="empty_view")
    with pytest.raises(tvm.error.InternalError, match="non-positive"):
        _run(_sblock(_evaluate(gather(_load(empty_view, [_int(0)]))), empty_owner))

    # The instruction base offset must be provably inside the owner.
    far_view = _view(work, (1024,), name="far_view")
    with pytest.raises(tvm.error.InternalError, match="inside the owner"):
        _run(_sblock(_evaluate(gather(_load(far_view, [_int(600)]))), work))


@pytest.mark.pto
def test_indexed_owner_tight_stride_width():
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    # The expected tight strides are inner shape products.  Computing them in
    # the shape dtype wraps an int32 product (67108865 * 64 becomes 64), which
    # would accept the overlapping layout below and reject the mathematically
    # tight stride.  The comparison must also keep the wide stride wide.
    wide = _buffer((2, 67108865, 64), name="wide_owner", strides=(2**32 + 64, 64, 1))
    call = _vgather(_load(wide, [_int(0)] * 3), offsets=offsets, mask=mask)
    body = _run(_sblock(_evaluate(call), wide))
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base, pointer = _descriptor(out, 0)
    assert base.buffer.same_as(wide)
    assert str(pointer.args[1].dtype) == "int64", pointer.args[1]
    assert _constant(pointer.args[1]) == 2 * 67108865 * 64
    assert _constant(out.args[1]) == 0

    overlap = _buffer((2, 67108865, 64), name="overlap_owner", strides=(64, 64, 1))
    call = _vgather(_load(overlap, [_int(0)] * 3), offsets=offsets, mask=mask)
    with pytest.raises(tvm.error.InternalError, match="tight row-major"):
        _run(_sblock(_evaluate(call), overlap))


@pytest.mark.pto
def test_indexed_owner_scope_and_order():
    work = _buffer((256,))
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")
    call = _vgather(_load(work, [_int(64)]), offsets=offsets, mask=mask)
    true = tirx.IntImm("bool", 1)

    # AllocBuffer is visible only to later siblings in the same statement list.
    declared = tirx.SeqStmt([tirx.AllocBuffer(work, annotations={}), _evaluate(call)])
    body = _run(declared)
    assert len(_vmi_calls(body, "tl.vmi.vgather")) == 1
    before_declaration = tirx.SeqStmt([_evaluate(call), tirx.AllocBuffer(work, annotations={})])
    with pytest.raises(tvm.error.InternalError, match="no active allocation"):
        _run(before_declaration)

    branch_decl = tirx.IfThenElse(
        true,
        tirx.SeqStmt([tirx.AllocBuffer(work, annotations={}), _evaluate(call)]),
        _evaluate(0),
    )
    body = _run(branch_decl)
    assert len(_vmi_calls(body, "tl.vmi.vgather")) == 1
    branch_leak = tirx.SeqStmt([branch_decl, _evaluate(call)])
    with pytest.raises(tvm.error.InternalError, match="no active allocation"):
        _run(branch_leak)

    # If the folded branch is itself only a declaration, the declaration is
    # still lexical to the original If.  Registering the visited declaration as
    # an outer sibling would incorrectly legalize the following gather/bind.
    lone_pointer = _access_ptr(work, [_int(64)], 64, 1)

    def folded(stmt, truth):
        if truth:
            return tirx.IfThenElse(tirx.IntImm("bool", 1), stmt, _evaluate(0))
        return tirx.IfThenElse(tirx.IntImm("bool", 0), _evaluate(0), stmt)

    for truth in (1, 0):
        lone_alloc = folded(tirx.AllocBuffer(work, annotations={}), truth)
        with pytest.raises(tvm.error.InternalError, match="no active allocation"):
            _run(tirx.SeqStmt([lone_alloc, _evaluate(call)]))

        lone_handle = tirx.Var("lone_handle", "handle")
        lone_bind = folded(tirx.Bind(lone_handle, lone_pointer), truth)
        with pytest.raises(tvm.error.InternalError, match="no recoverable definition"):
            _run(
                _sblock(
                    tirx.SeqStmt([lone_bind, _evaluate(_vgather(lone_handle, offsets=offsets, mask=mask))]),
                    work,
                )
            )


@pytest.mark.pto
def test_indexed_dynamic_owner_capacity():
    n = tirx.Var("n", "int32")
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    def run_guarded(owner, indices):
        call = _vgather(_load(owner, indices), offsets=offsets, mask=mask)
        guarded = tirx.IfThenElse(n > 0, _evaluate(call), _evaluate(0))
        body = _run(_sblock(guarded, owner), params=[n])
        (out,) = _vmi_calls(body, "tl.vmi.vgather")
        return out

    # One-dimensional dynamic owner (n > 0).  The full (1,n)/(n,1)/(1,1,n)
    # The surrounding dynamic owner cases cover the full shape matrix.
    owner = tirx.decl_buffer((n,), "float32", name="owner", scope="shared")
    out = run_guarded(owner, [_int(0)])
    base, pointer = _descriptor(out, 0)
    assert base.buffer.same_as(owner)
    # The source int32 Var is unchanged; the emitted capacity is widened.
    assert owner.shape[0].same_as(n) and str(owner.shape[0].dtype) == "int32"
    extent = pointer.args[1]
    assert isinstance(extent, tirx.Cast) and extent.value.same_as(n)
    assert str(extent.dtype) == "int64"
    assert _constant(pointer.args[2]) == 1 and _constant(out.args[1]) == 0

    # Dynamic outer dimensions no longer require a length-one coverage proof.
    # The independent capacity representation checks still apply.
    m = tirx.Var("m", "int32")
    owner_nm = tirx.decl_buffer((n, m), "float32", name="owner_nm", scope="shared")
    call_nm = _vgather(_load(owner_nm, [_int(0), _int(0)]), offsets=offsets, mask=mask)
    bounded_outer = tirx.IfThenElse(tirx.And(n == 2, m == 128), _evaluate(call_nm), _evaluate(0))
    body = _run(_sblock(bounded_outer, owner_nm), params=[n, m])
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base_nm, _ = _descriptor(out, 0)
    assert base_nm.buffer.same_as(owner_nm)

    dynamic_outer = tirx.IfThenElse(tirx.And(n > 1, m > 0), _evaluate(call_nm), _evaluate(0))
    body = _run(_sblock(dynamic_outer, owner_nm), params=[n, m])
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base_nm, pointer_nm = _descriptor(out, 0)
    assert base_nm.buffer.same_as(owner_nm)
    assert str(pointer_nm.args[1].dtype) == "int64"

    one_outer = tirx.IfThenElse(tirx.And(n == 1, m > 0), _evaluate(call_nm), _evaluate(0))
    body = _run(_sblock(one_outer, owner_nm), params=[n, m])
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base_nm, _ = _descriptor(out, 0)
    assert base_nm.buffer.same_as(owner_nm)

    # (n, 64) keeps its innermost static axis inside the capacity product, so
    # the outer symbolic axis is the one consumed whole.  The capacity must be
    # widened before multiplication: an int32 n * 64 overflows and the
    # mathematical value 4294967360 is the real owner capacity.
    owner_n64 = tirx.decl_buffer((n, 64), "float32", name="owner_n64", scope="shared")
    call_n64 = _vgather(_load(owner_n64, [_int(0), _int(0)]), offsets=offsets, mask=mask)
    capacity_guard = tirx.IfThenElse(n == 67108865, _evaluate(call_n64), _evaluate(0))
    body = _run(_sblock(capacity_guard, owner_n64), params=[n])
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base_n64, pointer_n64 = _descriptor(out, 0)
    assert base_n64.buffer.same_as(owner_n64)
    assert str(pointer_n64.args[1].dtype) == "int64", pointer_n64.args[1]
    folded = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(pointer_n64.args[1], {n: _int(67108865)}))
    assert isinstance(folded, tirx.IntImm) and int(folded.value) == 67108865 * 64, folded
    assert _constant(out.args[1]) == 0


@pytest.mark.pto
def test_indexed_offset_capacity_comparison_width():
    work = _buffer((256,), name="work")
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    def run(offset, indices=None):
        call = _vgather(_load(work, indices or [_int(0)]), offset, offsets=offsets, mask=mask)
        return _run(_sblock(_evaluate(call), work))

    # The static owner capacity is int64 and must not be used to narrow D:
    # wide int64 and uint32 offsets are compared in a value-preserving domain
    # and rejected through the normal inside-owner diagnostic instead of
    # throwing while constructing an out-of-range int32 literal.  The full
    # The explicit offset checks cover the relevant width cases.
    for offset in (_int(2**32 + 64, "int64"), _int(2**32 - 1, "uint32")):
        with pytest.raises(tvm.error.InternalError, match="provably inside the owner"):
            run(offset)

    # Signed and unsigned offsets that fit keep their original expression and
    # width in the emitted instruction offset, including a symbolic uint32.
    for dtype in ("int64", "uint32"):
        (out,) = _vmi_calls(run(_int(64, dtype)), "tl.vmi.vgather")
        assert str(out.args[1].dtype) == dtype, out.args[1]
        assert _constant(out.args[1]) == 64

    d32u = tirx.Var("d32u", "uint32")
    call = _vgather(_load(work, [_int(0)]), d32u, offsets=offsets, mask=mask)
    guarded = tirx.IfThenElse(d32u < _int(256, "uint32"), _evaluate(call), _evaluate(0))
    (out,) = _vmi_calls(_run(_sblock(guarded, work), params=[d32u]), "tl.vmi.vgather")
    assert isinstance(out.args[1], tirx.Var) and out.args[1].same_as(d32u), out.args[1]


@pytest.mark.pto
def test_indexed_bound_pointer_snapshot():
    work = _buffer((256,))
    index = _buffer((1,), "int32", name="index", scope="shared")
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    # A saved scalar variable is a stable snapshot leaf: later mutation of the
    # buffer it was read from does not change the rebuilt descriptor offset.
    k = tirx.Var("k", "int32")
    handle = tirx.Var("p", "handle")
    pointer = _access_ptr(work, [k], 64, 1)
    branch = tirx.IfThenElse(
        tirx.And(k >= 0, k <= 192),
        tirx.SeqStmt(
            [
                tirx.Bind(handle, pointer),
                tirx.BufferStore(index, _int(64), [_int(0)]),
                _evaluate(_vgather(handle, offsets=offsets, mask=mask)),
            ]
        ),
        _evaluate(0),
    )
    body = _run(_sblock(tirx.SeqStmt([tirx.Bind(k, _load(index, [_int(0)])), branch]), work, index))
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base, descriptor = _descriptor(out, 0)
    assert base.buffer.same_as(work)
    assert _constant(descriptor.args[1]) == 256
    assert _constant(descriptor.args[2]) == 1
    from tvm import arith

    assert arith.Analyzer().simplify(out.args[1]).same_as(k), out.args[1]

    # A bound pointer whose coordinates contain a mutable read is not a stable
    # snapshot: rebuilding it would re-execute that read at the use point.
    mutable_pointer = _access_ptr(work, [_load(index, [_int(0)])], 64, 1)
    mutable = tirx.Var("mutable", "handle")
    mutable_branch = tirx.SeqStmt(
        [
            tirx.Bind(mutable, mutable_pointer),
            tirx.BufferStore(index, _int(64), [_int(0)]),
            _evaluate(_vgather(mutable, offsets=offsets, mask=mask)),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="stable snapshot"):
        _run(_sblock(mutable_branch, work, index))

    # A direct address is evaluated at the use point, so it is not a snapshot;
    # the normal sign and owner-base proofs still apply.
    direct = _vgather(
        _load(work, [tirx.FloorMod(_load(index, [_int(0)]), _int(256))]),
        offsets=offsets,
        mask=mask,
    )
    body = _run(_sblock(_evaluate(direct), work, index))
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base, descriptor = _descriptor(out, 0)
    assert base.buffer.same_as(work)
    assert _constant(descriptor.args[1]) == 256
    assert isinstance(out.args[1], tirx.FloorMod)

    # Ordinary bound pointers are not subject to the indexed snapshot check.
    ordinary = tirx.Var("ordinary", "handle")
    ordinary_pointer = _access_ptr(work, [_load(index, [_int(0)])], 64, 1)
    ordinary_body = tirx.SeqStmt([tirx.Bind(ordinary, ordinary_pointer), _evaluate(_vload(ordinary, 0, size=64))])
    body = _run(_sblock(ordinary_body, work, index))
    (out,) = _vmi_calls(body, "tl.vmi.vload")
    assert out.args[0].same_as(ordinary), "ordinary explicit pointer behavior is unchanged"

    # A view without explicit strides derives its strides from shape.  A shape
    # read that enters the linear address is copied into the rebuilt S/D and
    # must be snapshot-stable even though it is not in starts or strides.
    dynamic_inner = tirx.FloorMod(_load(index, [_int(0)]), _int(64)) + _int(1)
    implicit_view = tirx.decl_buffer((4, dynamic_inner), "float32", name="implicit_view", scope="shared", data=work.data)
    shape_handle = tirx.Var("shape_handle", "handle")
    shape_pointer = _access_ptr(implicit_view, [_int(1), _int(0)], 1, 1)
    shape_body = tirx.SeqStmt(
        [
            tirx.Bind(shape_handle, shape_pointer),
            tirx.BufferStore(index, _int(16), [_int(0)]),
            _evaluate(_vgather(shape_handle, offsets=offsets, mask=mask)),
        ]
    )
    with pytest.raises(tvm.error.InternalError, match="stable snapshot"):
        _run(_sblock(shape_body, work, index))

    # An unrelated shape expression outside the actual address is not copied
    # and must not be rejected just because it reads memory.
    irrelevant_view = tirx.decl_buffer((dynamic_inner, 64), "float32", name="irrelevant_view", scope="shared", data=work.data)
    safe_handle = tirx.Var("safe_handle", "handle")
    safe_pointer = _access_ptr(irrelevant_view, [_int(0), _int(32)], 1, 1)
    safe_body = tirx.SeqStmt(
        [
            tirx.Bind(safe_handle, safe_pointer),
            tirx.BufferStore(index, _int(16), [_int(0)]),
            _evaluate(_vgather(safe_handle, offsets=offsets, mask=mask)),
        ]
    )
    body = _run(_sblock(safe_body, work, index))
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base, descriptor = _descriptor(out, 0)
    assert base.buffer.same_as(work)
    assert _constant(descriptor.args[1]) == 256
    assert _constant(out.args[1]) == 32


@pytest.mark.pto
def test_indexed_owner_capacity_stability():
    index = _buffer((1,), "int32", name="index", scope="shared")
    offsets = _vec("offsets", "uint32")
    mask = _mask("mask")

    def run_shape(shape, name):
        owner = tirx.decl_buffer((shape,), "float32", name=name, scope="shared")
        body = _evaluate(_vgather(_load(owner, [_int(0)]), offsets=offsets, mask=mask))
        return _run(_sblock(body, owner))

    # The allocation shape is evaluated when the owner is allocated.  Rebuilding
    # E at the gather would re-read index[0] after the store below, so the
    # descriptor would claim a capacity unrelated to the real allocation.
    mutable_shape = tirx.FloorMod(_load(index, [_int(0)]), _int(64)) + _int(1)
    owner = tirx.decl_buffer((mutable_shape,), "float32", name="owner", scope="shared")
    gather = _vgather(_load(owner, [_int(0)]), offsets=offsets, mask=mask)
    mutable_body = tirx.SeqStmt([tirx.BufferStore(index, _int(0), [_int(0)]), _evaluate(gather)])
    with pytest.raises(
        tvm.error.InternalError,
        match=r"owner capacity is not a stable snapshot: shape dimension 0 .*scalar Var",
    ):
        _run(_sblock(mutable_body, owner, index))

    # A scalar saved before the allocation is a stable snapshot leaf.  Mutating
    # index afterwards must not change E, and E must still reference the saved
    # Var instead of re-expanding its mutable definition.
    saved = tirx.Var("saved_shape", "int32")
    saved_owner = tirx.decl_buffer((saved,), "float32", name="saved_owner", scope="shared")
    saved_gather = _vgather(_load(saved_owner, [_int(0)]), offsets=offsets, mask=mask)
    saved_body = tirx.SeqStmt(
        [
            tirx.Bind(saved, tirx.FloorMod(_load(index, [_int(0)]), _int(64)) + _int(1)),
            tirx.IfThenElse(
                tirx.And(saved > 0, saved <= 64),
                _sblock(
                    tirx.SeqStmt(
                        [
                            tirx.BufferStore(index, _int(0), [_int(0)]),
                            _evaluate(saved_gather),
                        ]
                    ),
                    saved_owner,
                ),
                _evaluate(0),
            ),
        ]
    )
    body = _run(_sblock(saved_body, index))
    (out,) = _vmi_calls(body, "tl.vmi.vgather")
    base, descriptor = _descriptor(out, 0)
    assert base.buffer.same_as(saved_owner)
    extent = descriptor.args[1]
    assert isinstance(extent, tirx.Cast) and extent.value.same_as(saved), extent
    assert str(extent.dtype) == "int64", extent
    assert _constant(out.args[1]) == 0

    # The raw shape is scanned before the analyzer folds it: a read that would
    # fold away (index[0] * 0) is still re-evaluated at the use point.
    foldable_shape = tirx.Mul(_load(index, [_int(0)]), _int(0)) + _int(64)
    with pytest.raises(
        tvm.error.InternalError,
        match=r"owner capacity is not a stable snapshot: shape dimension 0 .*scalar Var",
    ):
        run_shape(foldable_shape, "foldable_owner")

    # Direct Calls are rejected conservatively, including pure calls; a scalar
    # Var is the supported snapshot.
    with pytest.raises(tvm.error.InternalError, match="executes a call"):
        run_shape(tirx.call_pure_extern("int32", "capacity_value"), "call_owner")

    # A pure call is stable once its value is saved in a scalar Var: E must use
    # that Var instead of re-expanding the call at the gather.  A stable integer
    # expression may also combine the saved Var with comparisons, Select, and
    # FloorMod that are evaluated from the saved snapshot at the use point.
    capacity = tirx.Var("saved_capacity", "int32")
    capacity_owner = tirx.decl_buffer((capacity,), "float32", name="capacity_owner", scope="shared")
    stable_owner = tirx.decl_buffer(
        (tirx.Select(capacity > 0, tirx.FloorMod(capacity, _int(64)) + _int(1), _int(64)),),
        "float32",
        name="stable_owner",
        scope="shared",
    )
    capacity_body = tirx.SeqStmt(
        [
            tirx.Bind(
                capacity,
                tirx.FloorMod(tirx.call_pure_extern("int32", "capacity_value"), _int(64)) + _int(1),
            ),
            tirx.IfThenElse(
                tirx.And(capacity > 0, capacity <= 64),
                _sblock(
                    tirx.SeqStmt(
                        [
                            _evaluate(
                                _vgather(
                                    _load(capacity_owner, [_int(0)]),
                                    offsets=offsets,
                                    mask=mask,
                                )
                            ),
                            _evaluate(
                                _vgather(
                                    _load(stable_owner, [_int(0)]),
                                    offsets=offsets,
                                    mask=mask,
                                )
                            ),
                        ]
                    ),
                    capacity_owner,
                    stable_owner,
                ),
                _evaluate(0),
            ),
        ]
    )
    body = _run(_sblock(capacity_body, index))
    capacity_out, stable_out = _vmi_calls(body, "tl.vmi.vgather")
    capacity_base, capacity_descriptor = _descriptor(capacity_out, 0)
    assert capacity_base.buffer.same_as(capacity_owner)
    capacity_extent = capacity_descriptor.args[1]
    assert isinstance(capacity_extent, tirx.Cast) and capacity_extent.value.same_as(capacity)
    assert str(capacity_extent.dtype) == "int64"
    stable_base, stable_descriptor = _descriptor(stable_out, 0)
    assert stable_base.buffer.same_as(stable_owner)
    stable_extent = stable_descriptor.args[1]
    assert str(stable_extent.dtype) == "int64", stable_extent
    stable_nodes = []
    tirx.stmt_functor.post_order_visit(stable_extent, stable_nodes.append)
    assert any(isinstance(node, tirx.Select) for node in stable_nodes)
    assert any(isinstance(node, tirx.FloorMod) for node in stable_nodes)
    assert not any(isinstance(node, (tirx.BufferLoad, tirx.Call)) for node in stable_nodes)


# ---------------------------------------------------------------------------
# Target isolation
# ---------------------------------------------------------------------------
@pytest.mark.pto
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


@pytest.mark.pto
def test_shared_scalar_definitions_with_fixed_vmi_addresses():
    work = _buffer((128,))
    i = tirx.Var("i", "int64")
    scalars = [tirx.Var(f"a{level}", "int64") for level in range(65)]
    seq = [tirx.Bind(scalars[0], i), tirx.Bind(scalars[1], i + _int(1, "int64"))]
    for level in range(2, len(scalars)):
        seq.append(tirx.Bind(scalars[level], scalars[level - 1] + scalars[level - 2]))
    value = _vec("value")
    seq.extend([tirx.Bind(value, _vload(_load(work, [_int(0)]))), _evaluate(_vstore(value, _load(work, [_int(64)])))])
    body = tirx.For(i, _int(0, "int64"), _int(2, "int64"), tirx.ForKind.SERIAL, tirx.SeqStmt(seq))

    result = _run(_sblock(body, work))
    load = _vmi_calls(result)[0]
    store = _vmi_calls(result, "tl.vmi.vstore")[0]
    _assert_plan(load, address_index=0, b=0, e=64, d=0, rw=1)
    _assert_plan(store, address_index=1, b=64, e=64, d=0, rw=2)


@pytest.mark.pto
def test_non_vmi_unsigned_cast_is_not_analyzed():
    i = tirx.Var("i", "int32")
    signed = tirx.Var("signed", "int32")
    unsigned = tirx.Var("unsigned", "uint32")
    body = tirx.For(
        i,
        0,
        64,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.Bind(signed, i - 1),
                tirx.Bind(unsigned, tirx.Cast("uint32", signed)),
                tirx.Evaluate(unsigned),
            ]
        ),
    )
    mod = _module(body)
    tvm.ir.assert_structural_equal(ascend.LegalizePTOVMIMemoryAccess()(mod), mod)


@pytest.mark.parametrize("binding", ["bind", "let", "alias"])
@pytest.mark.parametrize("memory_uses_cast", [False, True])
@pytest.mark.pto
def test_wrapping_cast_with_vmi(binding, memory_uses_cast):
    work = _buffer((64,))
    i, signed = tirx.Var("i", "int32"), tirx.Var("signed", "int32")
    unsigned = tirx.Var("unsigned", "uint32")
    alias = tirx.Var("alias", "uint32")
    cast = tirx.Cast("uint32", signed)
    index = tirx.Cast("int32", alias if binding == "alias" else unsigned) if memory_uses_cast else _int(0)
    load = _vload(_load(work, [index]), size=64)
    if binding == "let":
        tail = _evaluate(tirx.Let(unsigned, cast, load))
    else:
        seq = [tirx.Bind(unsigned, cast)]
        if binding == "alias":
            seq.append(tirx.Bind(alias, unsigned + tirx.IntImm("uint32", 0)))
        seq.append(_evaluate(load))
        tail = tirx.SeqStmt(seq)
    body = tirx.For(i, 0, 64, tirx.ForKind.SERIAL, tirx.SeqStmt([tirx.Bind(signed, i - 1), tail]))
    if memory_uses_cast:
        with pytest.raises(tvm.error.InternalError, match="potentially wrapping integer cast"):
            _run(_sblock(body, work))
    else:
        result = _run(_sblock(body, work))
        assert len(_vmi_calls(result)) == 1
        _assert_no_raw_address(result)


@pytest.mark.pto
def test_nonwrapping_cast_keeps_descriptor_proof():
    work = _buffer((128,))
    i, unsigned = tirx.Var("i", "int32"), tirx.Var("unsigned", "uint32")
    body = tirx.For(
        i,
        0,
        65,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.Bind(unsigned, tirx.Cast("uint32", i)),
                _evaluate(_vload(_load(work, [tirx.Cast("int32", unsigned)]), size=64)),
            ]
        ),
    )
    _assert_no_raw_address(_run(_sblock(body, work)))


@pytest.mark.parametrize("field", ["offset", "extent", "shape", "stride", "elem_offset"])
@pytest.mark.pto
def test_wrapping_cast_in_descriptor_metadata_rejected(field):
    i = tirx.Var("i", "int32")
    unsigned = tirx.Var("unsigned", "uint32")
    value = tirx.Cast("int64", unsigned)
    kwargs = {}
    if field == "stride":
        kwargs["strides"] = [value]
    if field == "elem_offset":
        kwargs["elem_offset"] = value
    work = tirx.decl_buffer([value if field == "shape" else 64], "float32", scope="shared", **kwargs)
    pointer = _access_ptr(work, [_int(0)], value if field == "extent" else 64, 1)
    load = _vload(pointer, offset=value if field == "offset" else 0)
    body = tirx.For(
        i,
        0,
        64,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.Bind(unsigned, tirx.Cast("uint32", i - 1)),
                _evaluate(load),
            ]
        ),
    )
    with pytest.raises(tvm.error.InternalError, match="potentially wrapping integer cast"):
        _run(_sblock(body, work))


@pytest.mark.parametrize("context", ["if", "assert", "assume", "select", "call"])
@pytest.mark.pto
def test_wrapping_cast_condition_does_not_install_false_facts(context):
    i = tirx.Var("i", "int32")
    unsigned = tirx.Var("unsigned", "uint32")
    cast = tirx.Cast("uint32", i - 1)
    condition = cast < tirx.IntImm("uint32", 64)
    work = _buffer((64,))
    # The same condition at runtime cannot justify narrowing the cast globally.
    bad = _evaluate(_vload(_load(work, [tirx.Cast("int64", unsigned)]), size=1))
    if context == "if":
        tail = tirx.IfThenElse(condition, bad, _evaluate(0))
    elif context == "assert":
        tail = tirx.SeqStmt([tirx.AssertStmt(condition, tirx.StringImm("range")), bad])
    elif context == "assume":
        tail = tirx.AttrStmt(condition, "tl.assume", _int(0), bad)
    else:
        guarded = (
            tirx.Select(condition, unsigned, tirx.IntImm("uint32", 0))
            if context == "select"
            else tirx.if_then_else(condition, unsigned, tirx.IntImm("uint32", 0))
        )
        tail = _evaluate(_vload(_load(work, [tirx.Cast("int64", guarded)]), size=1))
    body = tirx.For(
        i,
        0,
        64,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.Bind(unsigned, cast),
                tail,
            ]
        ),
    )
    with pytest.raises(tvm.error.InternalError, match="potentially wrapping integer cast"):
        _run(_sblock(body, work))


@pytest.mark.parametrize("access_position", ["then", "else", "after"])
@pytest.mark.pto
def test_wrapping_condition_keeps_branch_assertions_scoped(access_position):
    work = _buffer((1, 128))
    i = tirx.Var("i", "int32")
    condition = tirx.Cast("uint32", i - 1) < _int(2, "uint32")
    assertion = tirx.AssertStmt(i < 2, tirx.StringImm("range"))
    # A fixed descriptor makes proving 64 * i + 64 <= 128 depend on i < 2.
    pointer = _access_ptr(work, [_int(0), _int(0)], 128, 2)
    store = _evaluate(_vstore(_vec("value"), pointer, offset=64 * i))
    # Only a store in the assertion's own branch may use its i < 2 fact.
    # At i == 3 the assertion does not execute and the store exceeds the descriptor.
    if access_position == "then":
        tail = tirx.IfThenElse(condition, tirx.SeqStmt([assertion, store]), None)
    elif access_position == "else":
        tail = tirx.IfThenElse(condition, assertion, store)
    else:
        tail = tirx.SeqStmt([tirx.IfThenElse(condition, assertion, None), store])
    body = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, tail)
    if access_position == "then":
        _assert_no_raw_address(_run(_sblock(body, work)))
        return
    with pytest.raises(tvm.error.InternalError, match="cannot verify the explicit access_ptr descriptor"):
        _run(_sblock(body, work))


@pytest.mark.parametrize("scope", ["for_min", "for_extent", "block_min", "block_extent", "attr", "assume", "while"])
@pytest.mark.parametrize("access_position", ["inside", "after"])
@pytest.mark.pto
def test_body_assertions_stay_in_their_statement_scope(scope, access_position):
    work = _buffer((1, 128))
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    execute_body = tirx.Var("execute_body", "bool")
    unsafe = tirx.Cast("int32", tirx.Cast("uint32", i - 1))
    assertion = tirx.AssertStmt(i < 2, tirx.StringImm("range"))
    # A fixed descriptor makes proving 64 * i + 64 <= 128 depend on i < 2.
    pointer = _access_ptr(work, [_int(0), _int(0)], 128, 2)
    store = _evaluate(_vstore(_vec("value"), pointer, offset=64 * i))
    scoped_body = tirx.SeqStmt([assertion, store]) if access_position == "inside" else assertion
    if scope == "for_min":
        wrapper = tirx.For(j, unsafe, 0, tirx.ForKind.SERIAL, scoped_body)
    elif scope == "for_extent":
        # Keep the wrapping cast in an expression whose runtime value is zero.
        wrapper = tirx.For(j, 0, tirx.Mul(unsafe, _int(0)), tirx.ForKind.SERIAL, scoped_body)
    elif scope.startswith("block"):
        minimum = unsafe if scope == "block_min" else _int(0)
        extent = _int(1) if scope == "block_min" else tirx.Add(tirx.Sub(unsafe, unsafe), _int(1))
        axis = tirx.IterVar(tvm.ir.Range.from_min_extent(minimum, extent), j, tirx.IterVar.DataPar)
        block = tirx.SBlock([axis], [], [], "guarded", scoped_body)
        wrapper = tirx.SBlockRealize([minimum], execute_body, block)
    elif scope == "attr":
        wrapper = tirx.AttrStmt(i, "test_scope", unsafe, scoped_body)
    elif scope == "assume":
        condition = tirx.Cast("uint32", i - 1) < _int(2, "uint32")
        wrapper = tirx.AttrStmt(condition, "tl.assume", _int(0), scoped_body)
    else:
        wrapper = tirx.While(execute_body, scoped_body)
    tail = wrapper if access_position == "inside" else tirx.SeqStmt([wrapper, store])
    body = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, tail)
    # An assertion may prove a store in its own scope. It must not prove a
    # subsequent store, especially when a zero-trip loop or false predicate
    # skips the assertion entirely (i == 3 would make the store exceed the descriptor).
    if access_position == "inside":
        result = _run(_sblock(body, work), params=[execute_body])
        assert len(_vmi_calls(result, "tl.vmi.vstore")) == 1
        _assert_no_raw_address(result)
    else:
        with pytest.raises(tvm.error.InternalError, match="cannot verify the explicit access_ptr descriptor"):
            _run(_sblock(body, work), params=[execute_body])


@pytest.mark.pto
def test_cast_safe_in_guarded_scope_keeps_proof():
    work = _buffer((128,))
    i, unsigned = tirx.Var("i", "int32"), tirx.Var("unsigned", "uint32")
    body = tirx.For(
        i,
        0,
        65,
        tirx.ForKind.SERIAL,
        tirx.IfThenElse(
            i >= 1,
            tirx.SeqStmt(
                [
                    tirx.Bind(unsigned, tirx.Cast("uint32", i - 1)),
                    _evaluate(_vload(_load(work, [tirx.Cast("int32", unsigned)]), size=64)),
                ]
            ),
            _evaluate(0),
        ),
    )
    _assert_no_raw_address(_run(_sblock(body, work)))


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16"])
@pytest.mark.pto
def test_narrowing_and_signedness_cast_cannot_prove_small_access(dtype):
    work = _buffer((64,))
    i, small = tirx.Var("i", "int32"), tirx.Var("small", dtype)
    source = i - 1 if dtype.startswith("uint") else i * 2048
    body = tirx.For(
        i,
        0,
        64,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.Bind(small, tirx.Cast(dtype, source)),
                _evaluate(_vload(_load(work, [tirx.Cast("int64", small)]), size=1)),
            ]
        ),
    )
    with pytest.raises(tvm.error.InternalError, match="potentially wrapping integer cast"):
        _run(_sblock(body, work))


@pytest.mark.pto
def test_wrapping_let_does_not_pollute_outer_scalar_proof():
    work = _buffer((64,))
    i = tirx.Var("i", "int32")
    outer, inner = tirx.Var("offset", "uint32"), tirx.Var("offset", "uint32")
    harmless = tirx.Let(inner, tirx.Cast("uint32", i - 1), _vload(_load(work, [_int(0)]), size=64))
    body = tirx.For(
        i,
        0,
        64,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.Bind(outer, tirx.IntImm("uint32", 0)),
                _evaluate(harmless),
                _evaluate(_vload(_load(work, [tirx.Cast("int32", outer)]), size=64)),
            ]
        ),
    )
    result = _run(_sblock(body, work))
    assert len(_vmi_calls(result)) == 2
    _assert_no_raw_address(result)


@pytest.mark.pto
def test_cast_guarded_through_scalar_alias_keeps_proof():
    work = _buffer((128,))
    i, signed = tirx.Var("i", "int32"), tirx.Var("signed", "int32")
    unsigned = tirx.Var("unsigned", "uint32")
    body = tirx.For(
        i,
        0,
        65,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.Bind(signed, i - 1),
                tirx.IfThenElse(
                    signed >= 0,
                    tirx.SeqStmt(
                        [
                            tirx.Bind(unsigned, tirx.Cast("uint32", signed)),
                            _evaluate(_vload(_load(work, [tirx.Cast("int32", unsigned)]), size=64)),
                        ]
                    ),
                    _evaluate(0),
                ),
            ]
        ),
    )
    _assert_no_raw_address(_run(_sblock(body, work)))
