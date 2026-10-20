"""PTO VMI value provenance and scalar-only consumer boundaries.

The codegen records every ``Bind``/``Let`` definition and classifies value
origins (scalar, VMI data/mask/pair/void, handle, other vector, unknown) with a
pure query that never prints, emits, or mutates the persistent definition
table.  These tests pin:

* lane-1 VMI data stays VMI data through aliases and ``vbrc(group=1)``: a VMI
  data source is passed to ``vbrc`` directly, without the signless integer
  bridge (no ``pto.cast``/``vinterpret_cast`` wrapper);
* every registered VMI producer has an explicit return contract
  (data/mask/pair/void, with ``pair_get`` extracting lane-1 data from a pair);
* unknown VMI origins stay unknown through aliases and are diagnosed at
  scalar consumers instead of ever defaulting to scalar;
* true scalar integer bridges (literal/dynamic, width/signedness) are
  preserved;
* lexical ``Let`` scoping, same-name/different-Var identity, per-function
  definition-table reset, definition-cycle detection, and linear-time
  classification of deep scalar expressions;
* grouped ``vbrc`` accepts numeric local.var reads without changing their
  classification.

Direct codegen (``target.build.tilelang_pto_without_compile`` +
``inspect_source``) is used where the standard lowering pipeline would inline
the alias/``Let`` structure under test.  Fatal diagnostics surface as
``tvm.error.InternalError``.  Backend compilation evidence is included;
device numerics live in ``test_pto_vmi_value_provenance_run.py``.

The "unknown VMI op" contract is exercised with a uniquely named test-only op
registered through ``ir.RegisterOp``.  It is never added to the product
contract table; the codegen must diagnose its result as unknown.
"""

from __future__ import annotations

import re

import pytest
import torch  # noqa: F401  (torch/torch_npu must import before tilelang/ptodsl)
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower

tirx = tvm.tirx
_Op = tvm.ir.Op.get
_BUILD = tvm.ffi.get_global_func("target.build.tilelang_pto_without_compile")
_INTERNAL_ERROR = tvm.error.InternalError

VMI_DATA_OPS = [
    "vci",
    "vbrc",
    "vzip",
    "vadd",
    "vsub",
    "vmul",
    "vdiv",
    "vmax",
    "vmin",
    "vand",
    "vor",
    "vxor",
    "vshl",
    "vshr",
    "vabs",
    "vneg",
    "vrelu",
    "vexp",
    "vln",
    "vsqrt",
    "vnot",
    "vadds",
    "vmuls",
    "vmaxs",
    "vmins",
    "vshls",
    "vshrs",
    "vsel",
    "vselr",
    "vcadd",
    "vcmax",
    "vcmin",
    "vcvt",
    "vinterpret_cast",
    "vexpdif",
    "vaxpy",
    "vlrelu",
    "vprelu",
    "vmula",
    "vdhist",
    "vchist",
    "vgather",
    "vgatherb",
]
VMI_MASK_OPS = ["create_mask", "vcmp", "vcmps"]
VMI_PAIR_OPS = ["vintlv", "vdintlv", "vunzip", "vmull"]
VMI_VOID_OPS = ["vstore", "vscatter"]
UNKNOWN_VMI_OP_NAME = "tl.vmi.provenance_test_unknown_result"


# ---------------------------------------------------------------------------
# Lowering / direct codegen helpers
# ---------------------------------------------------------------------------
def _target():
    return normalize_pto_target("pto")


def _lowered(func):
    target = _target()
    with target:
        return _lower(func, target=target)


def _kernel_source(func):
    return _lowered(func).kernel_source


def _compile(func):
    return tilelang.compile(func, target="pto", out_idx=-1)


def _source_of(module):
    target = _target()
    with target:
        return _BUILD(module, target).inspect_source()


def _raw_source(body, params=()):
    """Codegen a raw PrimFunc without the TileLang lowering pipeline."""
    func = tirx.PrimFunc(list(params), body).with_attr("global_symbol", "main")
    return _source_of(tvm.IRModule({"main": func}))


def _codegen_device_fn(artifact, fn, global_symbol="main"):
    gvar = tvm.ir.GlobalVar(global_symbol)
    fn = fn.with_attr("global_symbol", global_symbol)
    module = tvm.IRModule({gvar: fn})
    target = _target()
    with target:
        module = tilelang.transform.LowerIntrin()(module)
        module = tilelang.transform.HoistBroadcastValues()(module)
    return _source_of(module)


def _lines(source, needle):
    return [line.strip() for line in source.splitlines() if needle in line]


def _one(source, needle):
    hits = _lines(source, needle)
    assert len(hits) == 1, f"expected one line containing {needle!r}, got {hits!r}"
    return hits[0]


def _expr(line):
    """Strip an optional ``name = `` SSA assignment prefix from a source line."""
    match = re.match(r"^[A-Za-z_]\w* = (.*)$", line)
    return match.group(1) if match else line


def _expect_error(pattern, build):
    with pytest.raises(_INTERNAL_ERROR, match=re.escape(pattern)):
        build()


_UNKNOWN_VMI_OP = None


def _unknown_vmi_op():
    """Register the test-only VMI op once with no provenance contract.

    The product table is deliberately not extended; this op only proves that a
    ``tl.vmi.*`` result without an explicit return contract stays unknown.  The
    registration entry point and attribute write fail loudly if unavailable.
    """
    global _UNKNOWN_VMI_OP
    if _UNKNOWN_VMI_OP is None:
        register = tvm.ffi.get_global_func("ir.RegisterOp")
        register(UNKNOWN_VMI_OP_NAME, "Test-only VMI op without a contract.")
        op = tvm.ir.Op.get(UNKNOWN_VMI_OP_NAME)
        # Pure call effect, as required by the codegen's purity check.
        op.set_attr("TCallEffectKind", 0)
        _UNKNOWN_VMI_OP = op
    return _UNKNOWN_VMI_OP


# ---------------------------------------------------------------------------
# Real PTO kernels (standard lowering + codegen or full compile)
# ---------------------------------------------------------------------------
def _reduce_broadcast(dtype, reduce_name):
    reduce = getattr(T.vmi, reduce_name)

    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, work[0:128])
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                r = reduce(x, mask)
                y = T.vmi.vbrc(r, size=128, group=1)
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, C)

    return main


def _scalar_bridge(dtype, literal):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), scale: T.int32, C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((128,), dtype)
            around = T.alloc_shared((128,), dtype)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                base = T.int32(7) if literal else scale
                y = T.vmi.vbrc(base, size=128)
                z = T.vmi.vci(base, size=128)
                T.vmi.vstore(T.vmi.vadd(y, z, mask), around[0], mask)
            T.copy(around, C)

    return main


def _reinterpret_kernel():
    @T.prim_func
    def main(A: T.Tensor((128,), "uint32"), C: T.Tensor((128,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((128,), "uint32")
            out = T.alloc_shared((128,), "int32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                y = T.reinterpret("int32x128", x)
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, C)

    return main


def _vci_from_vmi():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), C: T.Tensor((128,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((128,), "int32")
            out = T.alloc_shared((128,), "int32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                r = T.vmi.vcmax(x, mask)
                y = T.vmi.vci(r, size=128)
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, C)

    return main


def _scalar_load_after_vmi_store():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), C: T.Tensor((128,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((128,), "int32")
            out = T.alloc_shared((128,), "int32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                T.vmi.vstore(x, work[0], mask)
                s = work[0]
                y = T.vmi.vbrc(s, size=128)
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, C)

    return main


def _local_var_vmi_store(value_is_vmi):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), C: T.Tensor((128,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((128,), "int32")
            out = T.alloc_shared((128,), "int32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                slot = T.alloc_var("int32")
                value = T.vmi.vcmax(x, mask) if value_is_vmi else work[0]
                slot = value
                T.vmi.vstore(T.vmi.vbrc(slot, size=128), out[0], mask)
            T.copy(out, C)

    return main


def _mutable_register(dtype, lanes, group):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((128,), dtype)
            dst = T.alloc_shared((128,), dtype)
            T.copy(A, src)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                slot = T.vmi.alloc_var(dtype, size=lanes)
                slot = T.vmi.vload(src[0], size=lanes)
                value = T.vmi.vbrc(slot, size=128, group=group)
                T.vmi.vstore(value, dst[0], mask)
            T.copy(dst, C)

    return main


# ---------------------------------------------------------------------------
# Raw producer/consumer helpers (direct codegen contracts)
# ---------------------------------------------------------------------------
def _vbrc(arg, group=1):
    annotations = {"size": 128}
    if group is not None:
        annotations["group"] = group
    return tirx.call_intrin("int32x128", _Op("tl.vmi.vbrc"), arg, annotations=annotations)


def _vmi_data_producer(op_name):
    op = _Op(f"tl.vmi.{op_name}")
    if op_name in ("vci", "vbrc"):
        return tirx.call_intrin("int32", op, tirx.IntImm("int32", 0)), ()
    if op_name in ("vgather", "vgatherb"):
        ptr = tirx.Var("ptr", "handle")
        offsets = tirx.Var("offsets", "int32x128")
        mask = tirx.Var("mask", "boolx128")
        call = tirx.call_intrin("int32", op, ptr, tirx.IntImm("int32", 0), offsets, mask)
        return call, (ptr, offsets, mask)
    return tirx.call_intrin("int32", op), ()


def _device_fn(func):
    artifact = _lowered(func)
    assert len(artifact.device_mod.functions) == 1
    gvar = next(iter(artifact.device_mod.functions.keys()))
    return artifact, gvar, artifact.device_mod.functions[gvar]


def _find_vbrc_call(fn):
    found = []

    def visit(node):
        if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == "tl.vmi.vbrc":
            found.append(node)

    tirx.stmt_functor.post_order_visit(fn.body, visit)
    assert len(found) == 1, f"expected one vbrc call, got {len(found)}"
    return found[0]


def _find_binding(fn, var):
    found = []

    def visit(node):
        if isinstance(node, tirx.Bind) and node.var.same_as(var):
            found.append(node)

    tirx.stmt_functor.post_order_visit(fn.body, visit)
    assert len(found) == 1, f"expected one binding for {var}, got {len(found)}"
    return found[0]


def _transform(fn, preorder):
    body = tirx.stmt_functor.ir_transform(fn.body, preorder, None)
    return fn.with_body(body)


def _replace_vbrc_source(fn, make_source, drop_producer=False):
    """Replace the vbrc source with ``make_source(producer_bind, source_var)``."""
    call = _find_vbrc_call(fn)
    source_var = call.args[0]
    producer_bind = _find_binding(fn, source_var)

    def preorder(node):
        if isinstance(node, tirx.Bind):
            if node.var.same_as(source_var):
                if drop_producer:
                    return tirx.Evaluate(tirx.IntImm("int32", 0))
                return node
            if (
                isinstance(node.value, tirx.Call)
                and getattr(node.value.op, "name", "") == "tl.vmi.vbrc"
                and node.value.args[0].same_as(source_var)
            ):
                value = node.value
                new_call = tirx.Call(
                    value.dtype,
                    value.op,
                    [make_source(producer_bind, source_var)] + list(value.args[1:]),
                    value.annotations,
                    value.span,
                )
                return tirx.Bind(node.var, new_call)
        return None

    return _transform(fn, preorder)


def _alias_chain_source(fn):
    call = _find_vbrc_call(fn)
    source_var = call.args[0]
    alias_a = tirx.Var("alias_a", source_var.dtype)
    alias_b = tirx.Var("alias_b", source_var.dtype)

    def preorder(node):
        if isinstance(node, tirx.Bind):
            if node.var.same_as(source_var):
                return tirx.SeqStmt([node, tirx.Bind(alias_a, node.var), tirx.Bind(alias_b, alias_a)])
            if (
                isinstance(node.value, tirx.Call)
                and getattr(node.value.op, "name", "") == "tl.vmi.vbrc"
                and node.value.args[0].same_as(source_var)
            ):
                value = node.value
                new_call = tirx.Call(
                    value.dtype,
                    value.op,
                    [alias_b] + list(value.args[1:]),
                    value.annotations,
                    value.span,
                )
                return tirx.Bind(node.var, new_call)
        return None

    return _transform(fn, preorder)


def _let_source(fn, nested):
    def make_source(producer_bind, source_var):
        let_var = tirx.Var("let_t", source_var.dtype)
        body = tirx.Let(let_var, producer_bind.value, let_var)
        if nested:
            inner_var = tirx.Var("let_u", source_var.dtype)
            body = tirx.Let(let_var, producer_bind.value, tirx.Let(inner_var, let_var, inner_var))
        return body

    return _replace_vbrc_source(fn, make_source, drop_producer=True)


def _direct_call_source(fn):
    return _replace_vbrc_source(fn, lambda producer_bind, source_var: producer_bind.value, drop_producer=True)


# ---------------------------------------------------------------------------
# A. Core lane-1 VMI data -> vbrc(group=1), scalar bridges, reinterpret
# ---------------------------------------------------------------------------
def _assert_direct_vbrc(source, producer_name):
    producer_line = _one(source, f"pto.vmi.{producer_name}(")
    bound = producer_line.split("=", 1)[0].strip()
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    assert _expr(vbrc_line) == f"pto.vmi.vbrc({bound}, group=1, size=128)"
    assert source.count(f"pto.vmi.{producer_name}(") == 1


def test_int32_vcmax_vbrc_uses_vmi_source_directly():
    source = _kernel_source(_reduce_broadcast("int32", "vcmax"))
    _assert_direct_vbrc(source, "vcmax")


def test_uint32_vcmin_vbrc_uses_vmi_source_directly():
    source = _kernel_source(_reduce_broadcast("uint32", "vcmin"))
    _assert_direct_vbrc(source, "vcmin")


def test_float32_vcmax_vbrc_control_uses_vmi_source_directly():
    source = _kernel_source(_reduce_broadcast("float32", "vcmax"))
    _assert_direct_vbrc(source, "vcmax")


@pytest.mark.parametrize(
    ("dtype", "reduce_name"),
    [("int32", "vcmax"), ("uint32", "vcmin"), ("float32", "vcmax")],
)
def test_reduce_broadcast_compiles_through_backend(dtype, reduce_name):
    _compile(_reduce_broadcast(dtype, reduce_name))


def test_scalar_literal_bridge_is_preserved():
    source = _kernel_source(_scalar_bridge("int32", literal=True))
    assert _expr(_one(source, "pto.vmi.vbrc(pto.si32(7), size=128)")) == ("pto.vmi.vbrc(pto.si32(7), size=128)")
    assert _expr(_one(source, "pto.vmi.vci(")).startswith("pto.vmi.vinterpret_cast(pto.vmi.vci(pto.i32(7), size=128), to_dtype=pto.si32)")


def test_scalar_dynamic_bridge_is_preserved():
    source = _kernel_source(_scalar_bridge("int32", literal=False))
    # Dynamic integer scalar sources take the signless bridge and recover the
    # authored signedness on the VMI result.
    assert _expr(_one(source, "pto.vmi.vbrc(")) == (
        "pto.vmi.vinterpret_cast(pto.vmi.vbrc(pto.cast(base, pto.i32), size=128), to_dtype=pto.si32)"
    )
    assert _expr(_one(source, "pto.vmi.vci(")) == (
        "pto.vmi.vinterpret_cast(pto.vmi.vci(pto.cast(base, pto.i32), size=128), to_dtype=pto.si32)"
    )


@pytest.mark.parametrize("literal", [True, False])
def test_scalar_bridge_compiles_through_backend(literal):
    _compile(_scalar_bridge("int32", literal=literal))


def test_vmi_reinterpret_uses_vinterpret_cast():
    source = _kernel_source(_reinterpret_kernel())
    assert "pto.vmi.vinterpret_cast(x, to_dtype=pto.si32)" in _one(source, "pto.vmi.vstore(")
    assert "vbitcast" not in source


# ---------------------------------------------------------------------------
# Pure SIMD register arrays stay on ordinary vector paths
# ---------------------------------------------------------------------------
def _pure_simd_array_reinterpret():
    @T.prim_func
    def main(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "int32")):
        with T.Kernel(1):
            src = T.alloc_shared((64,), "float32")
            dst = T.alloc_shared((64,), "int32")
            T.copy(X, src)
            with T.SimdVF():
                regs = T.simd.alloc_local((2,), "float32")
                regs[0] = T.simd.vld(src[0])
                y = T.reinterpret("int32x64", regs[0])
                T.simd.vsts(dst[0], y)
            T.copy(dst, Y)

    return main


def _pure_simd_array_cast():
    @T.prim_func
    def main(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "float16")):
        with T.Kernel(1):
            src = T.alloc_shared((64,), "float32")
            dst = T.alloc_shared((64,), "float16")
            T.copy(X, src)
            with T.SimdVF():
                regs = T.simd.alloc_local((2,), "float32")
                regs[0] = T.simd.vld(src[0])
                y = T.Cast("float16x64", regs[0])
                T.simd.vsts(dst[0], y, dist="PK_B32")
            T.copy(dst, Y)

    return main


def test_pure_simd_register_array_cast_uses_ordinary_cast():
    # The array element reaches T.Cast through a pure SIMD allocation.  It must
    # not enter the VMI cast guard and must keep the ordinary float cast path.
    func = _pure_simd_array_cast()
    assert "tl.vmi." not in func.script(), func.script()
    source = _kernel_source(func)
    assert "pto.vmi." not in source
    cast_line = _one(source, "pto.cast(regs_tl_slot_0, pto.f16)")
    assert cast_line.startswith("y = pto.cast(")
    assert _one(source, "pto.vsts(").startswith("pto.vsts(y,")


def test_pure_simd_register_array_reinterpret_compiles_through_backend():
    # The pure SIMD program has no VMI frontend input.  Its register array only
    # shares shape/dtype with a VMI register array; that must not classify it
    # as VMI data, so the bit reinterpret stays on the ordinary vector path and
    # still compiles.
    func = _pure_simd_array_reinterpret()
    assert "tl.vmi." not in func.script(), func.script()
    source = _compile(func).get_kernel_source()
    assert "pto.vbitcast(" in source
    assert "pto.vmi." not in source
    assert "pto.vbitcast(regs_tl_slot_0" in _one(source, "pto.vbitcast(")
    assert _one(source, "pto.vsts(").startswith("pto.vsts(y,")


# ---------------------------------------------------------------------------
# B. Grouped-vbrc mutable local.var register compatibility
# ---------------------------------------------------------------------------
def test_mutable_register_grouped_vbrc_is_preserved():
    # Grouped vbrc accepts this numeric local.var read without classifying it
    # as VMI data. Its classification stays other-vector, and the source is
    # emitted directly without a scalar cast.
    source = _kernel_source(_mutable_register("float32", 4, 4))
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    assert "pto.vmi.vbrc(slot, group=4, size=128)" in vbrc_line
    assert "pto.cast(slot" not in vbrc_line
    assert "slot = pto.vmi.vreg(4, pto.f32)" in source
    assert "slot = pto.vmi.vload(" in source


# ---------------------------------------------------------------------------
# Explicit mutable VMI data slots (allocation marker)
# ---------------------------------------------------------------------------
_MARKER = "tl.vmi.alloc_kind"


def _mutable_slot(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, work[0:128])
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                slot = T.vmi.alloc_var(dtype, size=1)
                slot = T.vmi.vcmax(x, mask)
                y = T.vmi.vbrc(slot, size=128, group=1)
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, C)

    return main


def _mutable_slot_branch(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), scale: T.int32, C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((128,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                slot = T.vmi.alloc_var(dtype, size=1)
                if scale > 0:
                    slot = T.vmi.vcmax(x, mask)
                else:
                    slot = T.vmi.vcmin(x, mask)
                T.vmi.vstore(T.vmi.vbrc(slot, size=128, group=1), out[0], mask)
            T.copy(out, C)

    return main


def _mutable_slot_runtime_carry(dtype, marked=True):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((128,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                mask1 = T.vmi.create_mask(1, size=1)
                x = T.vmi.vload(work[0], size=128)
                if marked:
                    slot = T.vmi.alloc_var(dtype, size=1)
                else:
                    slot = T.alloc_var(dtype)
                # Preloop VMI value, read and updated inside the runtime loop.
                slot = T.vmi.vcmax(x, mask)
                for _ in T.serial(2):
                    old = slot
                    slot = T.vmi.vadd(old, old, mask1)
                T.vmi.vstore(T.vmi.vbrc(slot, size=128, group=1), out[0], mask)
            T.copy(out, C)

    return main


def _mutable_slot_in_explicit_loop(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, work[0:128])
            for _ in T.unroll(2, explicit=True):
                with T.SimdVF():
                    mask = T.vmi.create_mask(128, size=128)
                    x = T.vmi.vload(work[0], size=128)
                    slot = T.vmi.alloc_var(dtype, size=1)
                    slot = T.vmi.vcmax(x, mask)
                    T.vmi.vstore(T.vmi.vbrc(slot, size=128, group=1), out[0], mask)
            T.copy(out, C)

    return main


def _allocations(fn):
    found = []
    tirx.stmt_functor.post_order_visit(
        fn.body,
        lambda node: found.append(node) if isinstance(node, tirx.AllocBuffer) else None,
    )
    return found


def _marked_allocations(fn):
    return [node for node in _allocations(fn) if _MARKER in node.annotations]


@pytest.mark.parametrize(
    ("dtype", "pto_type"),
    [("int32", "pto.si32"), ("uint32", "pto.ui32"), ("float32", "pto.f32")],
)
def test_single_lane_mutable_slot_compiles_through_backend(dtype, pto_type):
    kernel = _compile(_mutable_slot(dtype))
    source = kernel.get_kernel_source()
    assert f"slot = pto.vmi.vreg(1, {pto_type})" in source
    assert "slot = pto.vmi.vcmax(x, mask)" in source
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    assert "pto.vmi.vbrc(slot, group=1, size=128)" in vbrc_line
    assert "pto.cast(" not in vbrc_line
    assert source.count("pto.vmi.vcmax(") == 1


@pytest.mark.parametrize(("dtype", "pto_type"), [("int32", "pto.si32"), ("float32", "pto.f32")])
def test_single_lane_mutable_slot_annotation_reaches_device_ir(dtype, pto_type):
    artifact, _, fn = _device_fn(_mutable_slot(dtype))
    marked = _marked_allocations(fn)
    assert len(marked) == 1, marked
    (allocation,) = marked
    assert str(allocation.annotations[_MARKER]) == "data"
    assert allocation.buffer.scope() == "local.var"
    assert str(allocation.buffer.dtype) == dtype
    assert f"slot = pto.vmi.vreg(1, {pto_type})" in artifact.kernel_source


def test_multi_lane_mutable_slot_annotation_reaches_device_ir():
    artifact, _, fn = _device_fn(_mutable_register("float32", 4, 4))
    marked = _marked_allocations(fn)
    assert len(marked) == 1, marked
    (allocation,) = marked
    assert str(allocation.annotations[_MARKER]) == "data"
    assert str(allocation.buffer.dtype) == "float32x4"
    assert "slot = pto.vmi.vreg(4, pto.f32)" in artifact.kernel_source


def test_explicit_loop_slot_markers_reach_device_ir():
    artifact, _, fn = _device_fn(_mutable_slot_in_explicit_loop("int32"))
    marked = _marked_allocations(fn)
    assert len(marked) == 2, marked
    assert all(str(node.annotations[_MARKER]) == "data" for node in marked)
    assert not marked[0].buffer.data.same_as(marked[1].buffer.data)
    assert artifact.kernel_source.count("= pto.vmi.vreg(1, pto.si32)") == 2
    assert "pto.cast(slot" not in artifact.kernel_source


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_mutable_slot_branch_updates_compile_through_backend(dtype):
    source = _compile(_mutable_slot_branch(dtype)).get_kernel_source()
    assert "if tl.as_logical_bool(0 < scale):" in source
    assert "slot = pto.vmi.vcmax(x, mask)" in source
    assert "slot = pto.vmi.vcmin(x, mask)" in source


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_marked_slot_runtime_loop_carry_compiles_through_backend(dtype):
    source = _compile(_mutable_slot_runtime_carry(dtype)).get_kernel_source()
    assert "old = slot" in source
    assert "slot = pto.vmi.vadd(old, old, mask1)" in source
    assert "pto.cast(" not in _one(source, "pto.vmi.vbrc(")


# ---------------------------------------------------------------------------
# VMI register arrays: allocation marker and value provenance
# ---------------------------------------------------------------------------
def _marked_array_buffer(shape=(2,), dtype="int32x64", name="regs"):
    return tirx.decl_buffer(shape, dtype, name=name, scope="local")


def _marked_array_allocation(regs):
    return tirx.AllocBuffer(regs, annotations={_MARKER: "data_array"})


def _vmi_array_alias_and_reinterpret():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((64,), "float32")
            out = T.alloc_shared((64,), "int32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                regs = T.vmi.alloc_local((2,), T.vmi.vreg(64, "float32"))
                regs[0] = T.vmi.vload(work[0], size=64)
                alias = regs[0]
                regs[1] = alias
                y = T.reinterpret("int32x64", regs[1])
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, Y)

    return main


def test_vmi_register_array_alias_and_reinterpret_stay_vmi_data():
    source = _compile(_vmi_array_alias_and_reinterpret()).get_kernel_source()
    assert "regs_tl_slot_0 = None" in source
    assert "regs_tl_slot_1 = None" in source
    assert re.search(r"regs_tl_slot_0 = pto\.vmi\.vload\(", source), source
    assert re.search(r"regs_tl_slot_1 = (regs_tl_slot_0|alias)\b", source), source
    reinterpret_line = _one(source, "pto.vmi.vinterpret_cast(")
    assert "regs_tl_slot_1" in reinterpret_line
    assert "to_dtype=pto.si32" in reinterpret_line
    assert "pto.cast(" not in reinterpret_line
    assert "vbitcast" not in source


def test_marked_register_array_same_dtype_cast_is_a_transparent_alias():
    regs = _marked_array_buffer()
    body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.BufferStore(regs, _vmi_producer("int32x64"), [0]),
            tirx.BufferStore(regs, tirx.Cast("int32x64", tirx.BufferLoad(regs, [0])), [1]),
        ]
    )
    source = _raw_source(body)
    assert "regs_tl_slot_0 = pto.vmi.vcmax()" in source
    assert "regs_tl_slot_1 = regs_tl_slot_0" in source


def test_marked_register_array_generic_cast_is_rejected():
    regs = _marked_array_buffer()
    body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.Evaluate(tirx.Cast("float32x64", tirx.BufferLoad(regs, [0]))),
        ]
    )
    _expect_error(
        "PTO integer cast cannot consume VMI data",
        lambda: _raw_source(body),
    )


def _vmi_array_multidim_update(dtype):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), C: T.Tensor((320,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), dtype)
            out = T.alloc_shared((320,), dtype)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                regs = T.vmi.alloc_local((2, 2), T.vmi.vreg(64, dtype))
                for i in T.unroll(2, explicit=True):
                    for j in T.unroll(2, explicit=True):
                        regs[i, j] = T.vmi.vload(work[64 * (2 * i + j)], size=64)
                slot = T.vmi.alloc_var(dtype, size=1)
                slot = T.vmi.vcmax(regs[1, 0], mask)
                for _ in T.serial(3):
                    regs[1, 1] = T.vmi.vadds(regs[1, 1], 1, mask)
                T.vmi.vstore(regs[0, 0], out[0], mask)
                T.vmi.vstore(regs[0, 1], out[64], mask)
                T.vmi.vstore(regs[1, 0], out[128], mask)
                T.vmi.vstore(regs[1, 1], out[192], mask)
                T.vmi.vstore(T.vmi.vbrc(slot, size=64, group=1), out[256], mask)
            T.copy(out, C)

    return main


@pytest.mark.parametrize(
    ("dtype", "pto_type"),
    [("int32", "pto.si32"), ("uint32", "pto.ui32")],
)
def test_vmi_register_array_multidim_unroll_and_runtime_update_compile(dtype, pto_type):
    source = _compile(_vmi_array_multidim_update(dtype)).get_kernel_source()
    for index in range(4):
        assert f"regs_tl_slot_{index} = None" in source
    # The array is initialized as independent register names, never as one
    # multi-lane vreg allocation.
    assert "pto.vmi.vreg(4" not in source
    assert f"slot = pto.vmi.vreg(1, {pto_type})" in source
    assert source.count("pto.vmi.vload(") == 4
    assert "regs_tl_slot_2" in _one(source, "slot = pto.vmi.vcmax(")
    assert re.search(r"for \w+ in range\(0, 3, 1\):", source), source
    vadds_line = _one(source, "pto.vmi.vadds(")
    assert vadds_line.startswith("regs_tl_slot_3 = pto.vmi.vadds(regs_tl_slot_3, 1,")
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    assert "pto.vmi.vbrc(slot, group=1, size=64)" in vbrc_line
    assert "pto.cast(slot" not in vbrc_line
    assert "pto.cast(regs_tl_slot" not in source


def test_vmi_register_array_markers_reach_device_ir_after_flatten():
    artifact, _, fn = _device_fn(_vmi_array_multidim_update("uint32"))
    marked = [node for node in _marked_allocations(fn) if str(node.annotations[_MARKER]) == "data_array"]
    assert len(marked) == 1, marked
    (allocation,) = marked
    buffer = allocation.buffer
    assert buffer.scope() == "local"
    assert str(buffer.dtype) == "uint32x64"
    assert [int(dim) for dim in buffer.shape] == [4]

    stores = []
    loads = []

    def visit(node):
        if isinstance(node, tirx.BufferStore) and node.buffer.data.same_as(buffer.data):
            stores.append(node)
        elif isinstance(node, tirx.BufferLoad) and node.buffer.data.same_as(buffer.data):
            loads.append(node)

    tirx.stmt_functor.post_order_visit(fn.body, visit)
    assert sorted({int(node.indices[0]) for node in stores}) == [0, 1, 2, 3]
    assert sorted({int(node.indices[0]) for node in loads}) == [0, 1, 2, 3]
    assert "regs_tl_slot_3 = pto.vmi.vadds(" in artifact.kernel_source


# ---------------------------------------------------------------------------
# Direct codegen contracts for the allocation marker
# ---------------------------------------------------------------------------
def _marked_slot_buffer(dtype="int32"):
    return tirx.decl_buffer((1,), dtype, name="slot", scope="local.var")


def _marked_slot_allocation(slot):
    return tirx.AllocBuffer(slot, annotations={_MARKER: "data"})


def _vmi_producer(dtype="int32"):
    return tirx.call_intrin(dtype, _Op("tl.vmi.vcmax"))


def test_marked_slot_store_load_and_initialization():
    slot = _marked_slot_buffer()
    body = tirx.SeqStmt(
        [
            _marked_slot_allocation(slot),
            tirx.BufferStore(slot, _vmi_producer(), [0]),
            tirx.Evaluate(_vbrc(tirx.BufferLoad(slot, [0]))),
        ]
    )
    source = _raw_source(body)
    assert "slot = pto.vmi.vreg(1, pto.si32)" in source
    assert "slot = pto.vmi.vcmax()" in source
    assert source.count("pto.vmi.vcmax(") == 1
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    assert "pto.vmi.vbrc(slot, group=1, size=128)" in vbrc_line
    assert "pto.cast(" not in vbrc_line


def test_marked_slot_store_rejects_scalar_rhs():
    slot = _marked_slot_buffer()
    body = tirx.SeqStmt(
        [
            _marked_slot_allocation(slot),
            tirx.BufferStore(slot, tirx.IntImm("int32", 1), [0]),
        ]
    )
    _expect_error(
        "PTO VMI data slot store requires a VMI data RHS; got scalar",
        lambda: _raw_source(body),
    )


def test_marked_slot_rejects_a_different_dtype_view():
    slot = _marked_slot_buffer()
    view = tirx.decl_buffer((1,), "float32", name="slot_view", data=slot.data, scope="local.var")
    body = tirx.SeqStmt(
        [
            _marked_slot_allocation(slot),
            tirx.BufferStore(view, tirx.FloatImm("float32", 0.0), [0]),
        ]
    )
    _expect_error(
        "a shared data Var must keep the allocation dtype",
        lambda: _raw_source(body),
    )


def test_marked_slot_rejects_nonzero_index():
    slot = _marked_slot_buffer()
    body = tirx.SeqStmt(
        [
            _marked_slot_allocation(slot),
            tirx.BufferStore(slot, _vmi_producer(), [1]),
        ]
    )
    _expect_error("PTO VMI data slot store expects index 0", lambda: _raw_source(body))

    load_body = tirx.SeqStmt(
        [
            _marked_slot_allocation(slot),
            tirx.Evaluate(_vbrc(tirx.BufferLoad(slot, [1]))),
        ]
    )
    _expect_error("PTO VMI data slot load expects index 0", lambda: _raw_source(load_body))


def test_unmarked_scalar_slot_still_rejects_a_vmi_store():
    slot = tirx.decl_buffer((1,), "int32", name="plain", scope="local.var")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(slot),
            tirx.BufferStore(slot, _vmi_producer(), [0]),
        ]
    )
    _expect_error(
        "PTO local.var integer store expects a scalar; got VMI data",
        lambda: _raw_source(body),
    )


def test_marked_register_array_rejects_a_different_dtype_view():
    regs = _marked_array_buffer()
    view = tirx.decl_buffer((2,), "float32x64", name="regs_view", data=regs.data, scope="local")
    body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.Evaluate(_vbrc(tirx.BufferLoad(view, [0]))),
        ]
    )
    _expect_error(
        "a shared data Var must keep the allocation dtype",
        lambda: _raw_source(body),
    )


def _unknown_vmi_call(dtype="int32x64"):
    return tirx.call_intrin(dtype, _unknown_vmi_op())


@pytest.mark.parametrize(
    ("make_rhs", "kind_name"),
    [
        (lambda: tirx.call_intrin("int32x64", _Op("tl.vmi.vintlv")), "VMI pair"),
        (lambda: _unknown_vmi_call(), "unknown"),
        (
            lambda: tirx.Add(
                tirx.call_intrin("int32x64", _Op("tl.vmi.vcmax")),
                tirx.call_intrin("int32x64", _Op("tl.vmi.vcmax")),
            ),
            "unknown",
        ),
    ],
)
def test_marked_register_array_store_rejects_non_data_rhs(make_rhs, kind_name):
    regs = _marked_array_buffer()
    body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.BufferStore(regs, make_rhs(), [0]),
        ]
    )
    _expect_error(
        f"PTO VMI register array store requires a VMI data RHS; got {kind_name}",
        lambda: _raw_source(body),
    )


def test_marked_register_array_store_rejects_an_ordinary_vector_rhs():
    regs = _marked_array_buffer()
    plain = tirx.decl_buffer((2,), "int32x64", name="plain_regs", scope="local")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(plain),
            _marked_array_allocation(regs),
            tirx.BufferStore(regs, tirx.BufferLoad(plain, [0]), [0]),
        ]
    )
    _expect_error(
        "PTO VMI register array store requires a VMI data RHS; got other vector",
        lambda: _raw_source(body),
    )


def test_marked_register_array_cannot_be_used_as_an_address():
    regs = _marked_array_buffer()
    body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.Evaluate(tirx.address_of(tirx.BufferLoad(regs, [0]))),
        ]
    )
    _expect_error(
        "PTO VMI register array `regs` cannot be used as a memory address",
        lambda: _raw_source(body),
    )


def test_marked_register_array_requires_constant_indices():
    regs = _marked_array_buffer()
    index = tirx.Var("i", "int32")
    store_body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.BufferStore(regs, _vmi_producer("int32x64"), [index]),
        ]
    )
    _expect_error(
        "PTO VMI local register buffer `regs` requires a compile-time constant index",
        lambda: _raw_source(store_body),
    )

    load_body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.Evaluate(_vbrc(tirx.BufferLoad(regs, [index]), group=64)),
        ]
    )
    _expect_error(
        "PTO VMI local register buffer `regs` requires a compile-time constant index",
        lambda: _raw_source(load_body),
    )


def test_marked_register_array_cannot_use_simt_backing():
    regs = _marked_array_buffer()
    simt_body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.Evaluate(tirx.IntImm("int32", 0)),
        ]
    )
    body = tirx.SBlock([], [], [], "SIMT_VF", simt_body)
    _expect_error(
        "PTO VMI register array `regs` cannot use SIMT local backing",
        lambda: _raw_source(body),
    )


def test_vmi_allocation_table_resets_between_functions():
    regs = _marked_array_buffer()
    plain_slot = tirx.decl_buffer((1,), "int32", name="plain_slot", data=regs.data, scope="local.var")
    array_body = tirx.SeqStmt(
        [
            _marked_array_allocation(regs),
            tirx.BufferStore(regs, _vmi_producer("int32x64"), [0]),
        ]
    )
    plain_body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(plain_slot),
            tirx.BufferStore(plain_slot, tirx.IntImm("int32", 7), [0]),
        ]
    )
    target = _target()
    array_fn = tirx.PrimFunc([], array_body).with_attr("global_symbol", "a_array_fn").with_attr("target", target)
    plain_fn = tirx.PrimFunc([], plain_body).with_attr("global_symbol", "b_plain_fn").with_attr("target", target)
    module = tvm.IRModule(
        {
            tvm.ir.GlobalVar("a_array_fn"): array_fn,
            tvm.ir.GlobalVar("b_plain_fn"): plain_fn,
        }
    )
    source = _source_of(module)
    array_section = _function_section(source, "a_array_fn")
    plain_section = _function_section(source, "b_plain_fn")
    assert "regs_tl_slot_0 = pto.vmi.vcmax()" in array_section
    # The unmarked local.var in the next function reuses the same data Var and
    # must not inherit the register-array category from the first function.
    assert "regs_tl_slot" not in plain_section
    assert "= 7" in plain_section


def _plain_array_buffer(shape=(2,), dtype="int32x64", name="plain_regs"):
    return tirx.decl_buffer(shape, dtype, name=name, scope="local")


@pytest.mark.parametrize(
    ("make_rhs", "kind_name"),
    [
        (lambda: _vmi_producer("int32x64"), "VMI data"),
        (lambda: tirx.call_intrin("int32x64", _Op("tl.vmi.vintlv")), "VMI pair"),
        (lambda: _unknown_vmi_call(), "unknown"),
        (
            lambda: tirx.Add(
                tirx.call_intrin("int32x64", _Op("tl.vmi.vcmax")),
                tirx.call_intrin("int32x64", _Op("tl.vmi.vcmax")),
            ),
            "unknown",
        ),
    ],
)
def test_ordinary_register_array_store_rejects_vmi_values(make_rhs, kind_name):
    array = _plain_array_buffer()
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(array),
            tirx.BufferStore(array, make_rhs(), [0]),
        ]
    )
    _expect_error(
        f"PTO ordinary vector register array store requires an ordinary vector RHS; got {kind_name}",
        lambda: _raw_source(body),
    )


def test_ordinary_register_array_store_rejects_vmi_aliases():
    array = _plain_array_buffer()
    producer = tirx.call_intrin("int32x64", _Op("tl.vmi.vcmax"))
    alias = tirx.Var("vmi_alias", "int32x64")
    bind_body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(array),
            tirx.Bind(alias, producer),
            tirx.BufferStore(array, alias, [0]),
        ]
    )
    _expect_error(
        "PTO ordinary vector register array store requires an ordinary vector RHS; got VMI data",
        lambda: _raw_source(bind_body),
    )

    let_var = tirx.Var("vmi_let", "int32x64")
    let_body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(array),
            tirx.BufferStore(
                array,
                tirx.Let(
                    let_var,
                    tirx.call_intrin("int32x64", _Op("tl.vmi.vcmax")),
                    let_var,
                ),
                [0],
            ),
        ]
    )
    _expect_error(
        "PTO ordinary vector register array store requires an ordinary vector RHS; got VMI data",
        lambda: _raw_source(let_body),
    )


def test_ordinary_register_array_store_keeps_its_ordinary_vector_route():
    array = _plain_array_buffer()
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(array),
            tirx.BufferStore(array, tirx.BufferLoad(array, [0]), [1]),
        ]
    )
    source = _raw_source(body)
    assert "plain_regs_tl_slot_0 = None" in source
    assert "plain_regs_tl_slot_1 = plain_regs_tl_slot_0" in source


def test_plain_local_var_and_local_fragment_stores_keep_their_routes():
    slot = tirx.decl_buffer((1,), "int32", name="plain", scope="local.var")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(slot),
            tirx.BufferStore(slot, tirx.IntImm("int32", 5), [0]),
        ]
    )
    assert "plain = pto.const(5, dtype=pto.si32)" in _raw_source(body)

    frag = tirx.decl_buffer((2,), "int32x64", name="frag", scope="local.fragment")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(frag),
            tirx.BufferStore(frag, tirx.BufferLoad(frag, [0]), [1]),
        ]
    )
    source = _raw_source(body)
    assert "frag_tl_slot_1 = frag_tl_slot_0" in source


# ---------------------------------------------------------------------------
# C. Aliases, Let scoping, Var identity, per-function reset, cycles
# ---------------------------------------------------------------------------
def test_multi_level_alias_chain_keeps_vmi_data():
    artifact, _, fn = _device_fn(_reduce_broadcast("int32", "vcmax"))
    source = _codegen_device_fn(artifact, _alias_chain_source(fn))
    assert "alias_a = r" in source or re.search(r"alias_a = \w+", source)
    assert re.search(r"alias_b = alias_a", source)
    assert _expr(_one(source, "pto.vmi.vbrc(")) == ("pto.vmi.vbrc(alias_b, group=1, size=128)")
    assert source.count("pto.vmi.vcmax(") == 1


def test_direct_nested_producer_call_keeps_vmi_data():
    artifact, _, fn = _device_fn(_reduce_broadcast("int32", "vcmax"))
    source = _codegen_device_fn(artifact, _direct_call_source(fn))
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    assert "pto.vmi.vbrc(pto.vmi.vcmax(" in vbrc_line
    assert vbrc_line.endswith(", group=1, size=128)")
    assert source.count("pto.vmi.vcmax(") == 1


@pytest.mark.parametrize("nested", [False, True])
def test_let_source_is_query_local_and_producer_runs_once(nested):
    artifact, _, fn = _device_fn(_reduce_broadcast("int32", "vcmax"))
    source = _codegen_device_fn(artifact, _let_source(fn, nested=nested))
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    if nested:
        assert "(lambda let_t: (lambda let_u: let_u)(let_t))(pto.vmi.vcmax(" in vbrc_line
    else:
        assert "(lambda let_t: let_t)(pto.vmi.vcmax(" in vbrc_line
    assert vbrc_line.endswith(", group=1, size=128)")
    assert "pto.cast(" not in vbrc_line
    assert source.count("pto.vmi.vcmax(") == 1


def test_same_named_different_vars_are_not_conflated():
    artifact, _, fn = _device_fn(_reduce_broadcast("int32", "vcmax"))
    call = _find_vbrc_call(fn)
    source_var = call.args[0]
    same_name = tirx.Var(source_var.name, source_var.dtype)

    def preorder(node):
        if isinstance(node, tirx.Bind) and isinstance(node.value, tirx.Call) and getattr(node.value.op, "name", "") == "tl.vmi.vbrc":
            second = tirx.call_intrin(
                "int32x128",
                _Op("tl.vmi.vbrc"),
                same_name,
                annotations={"size": 128},
            )
            return tirx.SeqStmt(
                [
                    node,
                    tirx.Bind(same_name, tirx.IntImm("int32", 5)),
                    tirx.Evaluate(second),
                ]
            )
        return None

    source = _codegen_device_fn(artifact, _transform(fn, preorder))
    vbrc_lines = _lines(source, "pto.vmi.vbrc(")
    assert len(vbrc_lines) == 2
    assert re.search(r"pto\.vmi\.vbrc\(\w+, group=1, size=128\)", vbrc_lines[0])
    assert "group" not in vbrc_lines[1]
    # The same-named scalar Var is bridged independently of the VMI data Var,
    # proving the table is keyed by Var identity, not name.  The printer
    # disambiguates the duplicate Python name (r vs r_1).
    assert re.search(
        rf"pto\.vmi\.vbrc\(pto\.cast\({source_var.name}_\d+, pto\.i32\), size=128\)",
        vbrc_lines[1],
    ), vbrc_lines[1]


def _two_function_reset_module(order):
    """One codegen run with the same Var defined in one function and used as a
    true scalar parameter in another.  Function names select the generation
    order (the module map is name-ordered), so both orders are covered without
    relying on insertion order."""
    artifact_a, _, fn_a = _device_fn(_reduce_broadcast("int32", "vcmax"))
    shared = tirx.Var("shared_reset", "int32")

    call_a = _find_vbrc_call(fn_a)
    old_a = call_a.args[0]

    def preorder_a(node):
        if isinstance(node, tirx.Bind) and node.var.same_as(old_a):
            return tirx.Bind(shared, node.value)
        if isinstance(node, tirx.Bind) and isinstance(node.value, tirx.Call):
            value = node.value
            if getattr(value.op, "name", "") == "tl.vmi.vbrc" and value.args[0].same_as(old_a):
                return tirx.Bind(
                    node.var,
                    tirx.Call(
                        value.dtype,
                        value.op,
                        [shared] + list(value.args[1:]),
                        value.annotations,
                        value.span,
                    ),
                )
        return None

    fn_a = fn_a.with_body(tirx.stmt_functor.ir_transform(fn_a.body, preorder_a, None))

    # The SAME Var object is an ordinary scalar parameter here: no Bind, no VMI
    # producer.  A leaked definition table would misclassify it as VMI data.
    scalar_vbrc = tirx.call_intrin("int32x128", _Op("tl.vmi.vbrc"), shared, annotations={"size": 128})
    fn_b = tirx.PrimFunc([shared], tirx.Evaluate(scalar_vbrc))

    if order == "vmi-first":
        vmi_name, scalar_name = "a_vmi_fn", "b_scalar_fn"
    else:
        vmi_name, scalar_name = "b_vmi_fn", "a_scalar_fn"
    target = _target()
    fn_a = fn_a.with_attr("global_symbol", vmi_name).with_attr("target", target)
    fn_b = fn_b.with_attr("global_symbol", scalar_name).with_attr("target", target)
    vmi_entry = (tvm.ir.GlobalVar(vmi_name), fn_a)
    scalar_entry = (tvm.ir.GlobalVar(scalar_name), fn_b)
    entries = [vmi_entry, scalar_entry]
    if order == "scalar-first":
        entries.reverse()
    module = tvm.IRModule(dict(entries))
    with target:
        module = tilelang.transform.LowerIntrin()(module)
        module = tilelang.transform.HoistBroadcastValues()(module)
    return _source_of(module), vmi_name, scalar_name


def _function_section(source, name):
    start = source.index(f"def {name}(")
    end = source.find("def ", start + 1)
    return source[start : end if end != -1 else len(source)]


def test_same_var_across_two_functions_resets_definition_table():
    observed_orders = []
    for order in ("vmi-first", "scalar-first"):
        source, vmi_name, scalar_name = _two_function_reset_module(order)
        section_vmi = _function_section(source, vmi_name)
        section_scalar = _function_section(source, scalar_name)
        assert "pto.vmi.vbrc(shared_reset, group=1, size=128)" in section_vmi
        assert "pto.cast(shared_reset" not in section_vmi
        assert "pto.vmi.vbrc(pto.cast(shared_reset, pto.i32), size=128)" in section_scalar
        assert "pto.vmi.vbrc(shared_reset, group=1" not in section_scalar
        observed_orders.append("vmi-first" if source.index(f"def {vmi_name}(") < source.index(f"def {scalar_name}(") else "scalar-first")
    assert sorted(observed_orders) == ["scalar-first", "vmi-first"]


def test_definition_cycle_is_detected():
    x = tirx.Var("cycle_x", "int32")
    body = tirx.SeqStmt(
        [
            tirx.Bind(x, tirx.Add(x, tirx.IntImm("int32", 1))),
            tirx.Evaluate(_vbrc(x, group=None)),
        ]
    )
    _expect_error("PTO SSA definition cycle detected", lambda: _raw_source(body))


# ---------------------------------------------------------------------------
# D. Unknown-origin propagation and diagnosis
# ---------------------------------------------------------------------------
def test_registered_unknown_vmi_op_through_alias_stays_unknown():
    unknown = tirx.Var("unknown_x", "int32")
    alias = tirx.Var("unknown_alias", "int32")
    body = tirx.SeqStmt(
        [
            tirx.Bind(unknown, tirx.call_intrin("int32", _unknown_vmi_op())),
            tirx.Bind(alias, unknown),
            tirx.Evaluate(_vbrc(alias, group=None)),
        ]
    )
    _expect_error("cannot broadcast unknown (int32, lanes=1)", lambda: _raw_source(body))


def test_unknown_vmi_origin_diagnosed_at_vci():
    x = tirx.Var("x", "int32")
    alias = tirx.Var("alias", "int32")
    body = tirx.SeqStmt(
        [
            tirx.Bind(x, tirx.call_intrin("int32", _Op("tl.vmi.vcmax"))),
            tirx.Bind(alias, x),
            tirx.Evaluate(
                tirx.call_intrin(
                    "int32x128",
                    _Op("tl.vmi.vci"),
                    tirx.Cast("float32", alias),
                    annotations={"size": 128},
                )
            ),
        ]
    )
    _expect_error("tl.vmi.vci base expects a scalar; got unknown", lambda: _raw_source(body))


# ---------------------------------------------------------------------------
# E. Registered producer return-contract matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op_name", VMI_DATA_OPS)
def test_registered_vmi_data_producer_never_enters_scalar_bridge(op_name):
    producer, params = _vmi_data_producer(op_name)
    if op_name in ("vgather", "vgatherb"):
        # The gather printer requires a real access-pointer operand and the
        # function-signature printer cannot emit a synthetic handle parameter.
        # The return contract is asserted through pair_get's source validation,
        # which classifies the gather result before printing it (free variables
        # are only rendered by the TIR printer in the diagnostic): it must be
        # VMI data, not scalar/unknown.
        get = tirx.call_intrin("int32", _Op("tl.vmi.pair_get"), producer, tirx.IntImm("int32", 0))
        _expect_error(
            "tl.vmi.pair_get source must be a VMI pair; got VMI data",
            lambda: _raw_source(tirx.Evaluate(get)),
        )
        return
    source = _raw_source(tirx.Evaluate(_vbrc(producer)), params)
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    assert "pto.cast(" not in vbrc_line
    assert vbrc_line.endswith(", group=1, size=128)")


def test_vload_lane1_is_vmi_data():
    producer = tirx.call_intrin("int32", _Op("tl.vmi.vload"))
    source = _raw_source(tirx.Evaluate(_vbrc(producer)))
    assert "pto.cast(" not in _one(source, "pto.vmi.vbrc(")


@pytest.mark.parametrize("op_name", VMI_MASK_OPS)
def test_vmi_mask_producer_is_not_broadcast_as_data(op_name):
    producer = tirx.call_intrin("boolx128", _Op(f"tl.vmi.{op_name}"))
    _expect_error(
        "tl.vmi.vbrc with group=... requires a VMI vreg; got VMI mask",
        lambda: _raw_source(tirx.Evaluate(_vbrc(producer))),
    )


@pytest.mark.parametrize("op_name", VMI_PAIR_OPS)
def test_vmi_pair_producer_is_not_broadcast_as_data(op_name):
    producer = tirx.call_intrin("int32x128", _Op(f"tl.vmi.{op_name}"))
    _expect_error(
        "tl.vmi.vbrc with group=... requires a VMI vreg; got VMI pair",
        lambda: _raw_source(tirx.Evaluate(_vbrc(producer))),
    )


@pytest.mark.parametrize("op_name", VMI_VOID_OPS)
def test_vmi_void_producer_is_not_broadcast_as_data(op_name):
    producer = tirx.call_intrin("void", _Op(f"tl.vmi.{op_name}"))
    _expect_error(
        "tl.vmi.vbrc with group=... requires a VMI vreg; got void",
        lambda: _raw_source(tirx.Evaluate(_vbrc(producer))),
    )


def test_pair_get_returns_data_from_pair():
    pair = tirx.call_intrin("int32x128", _Op("tl.vmi.vintlv"))
    get = tirx.call_intrin("int32", _Op("tl.vmi.pair_get"), pair, tirx.IntImm("int32", 0))
    source = _raw_source(tirx.Evaluate(_vbrc(get)))
    assert _expr(_one(source, "pto.vmi.vbrc(")) == "pto.vmi.vbrc((pto.vmi.vintlv())[0], group=1, size=128)"


# ---------------------------------------------------------------------------
# F. Scalar-only consumers
# ---------------------------------------------------------------------------
def test_vci_rejects_vmi_data_source():
    _expect_error(
        "tl.vmi.vci base expects a scalar; got VMI data",
        lambda: _kernel_source(_vci_from_vmi()),
    )


def test_vbrc_group_rejects_true_scalar_source():
    _expect_error(
        "tl.vmi.vbrc with group=... requires a VMI vreg; got scalar",
        lambda: _raw_source(tirx.Evaluate(_vbrc(tirx.IntImm("int32", 0), group=1))),
    )


def test_ordinary_scalar_load_after_vmi_store_stays_scalar():
    source = _kernel_source(_scalar_load_after_vmi_store())
    vbrc_line = _one(source, "pto.vmi.vbrc(")
    assert "pto.cast(" in vbrc_line
    assert "group" not in vbrc_line


def test_local_var_store_rejects_vmi_data():
    _expect_error(
        "PTO local.var integer store expects a scalar; got VMI data",
        lambda: _kernel_source(_local_var_vmi_store(value_is_vmi=True)),
    )


def test_local_var_store_keeps_scalar_value():
    source = _kernel_source(_local_var_vmi_store(value_is_vmi=False))
    assert "slot = pto.const(0, dtype=pto.si32)" in source
    assert re.search(r"slot = pto\.cast\(\w+, pto\.si32\)", source)
    assert "pto.cast(slot, pto.i32)" in _one(source, "pto.vmi.vbrc(")


def test_select_branch_rejects_vmi_data():
    data = tirx.Var("data", "int32")
    cond = tirx.Var("cond", "bool")
    body = tirx.SeqStmt(
        [
            tirx.Bind(data, tirx.call_intrin("int32", _Op("tl.vmi.vcmax"))),
            tirx.Evaluate(tirx.Select(cond, data, tirx.IntImm("int32", 0))),
        ]
    )
    _expect_error(
        "PTO select branch expects a scalar; got VMI data",
        lambda: _raw_source(body, (cond,)),
    )


# ---------------------------------------------------------------------------
# G. Deep scalar expressions stay bounded
# ---------------------------------------------------------------------------
def test_deep_scalar_expression_chain_is_classified_linearly():
    x = tirx.Var("x", "int32")
    expr = x
    for i in range(200):
        expr = tirx.Add(expr, tirx.IntImm("int32", i + 1))
    source = _raw_source(tirx.Evaluate(tirx.call_intrin("int32", _Op("tl.vmi.vci"), expr)), (x,))
    # The scalar bridge proves one classification of the 200-deep chain.
    assert "pto.vmi.vinterpret_cast(pto.vmi.vci(pto.cast(" in source
