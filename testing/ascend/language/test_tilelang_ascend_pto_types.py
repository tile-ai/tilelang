"""PTO VMI type and lowering tests.

The PTO tests cover TileLang IR to PTODSL source lowering only. They do not
validate PTOAS/Bisheng compilation, on-device correctness, or performance.
"""

import re
from types import SimpleNamespace

import pytest

import tilelang.language as T
from tilelang.engine.lower import lower
from tvm import tirx
from tvm.tirx import Call

PTO_VMI_OPAQUE_OPS = [
    "vload",
    "vstore",
    "create_mask",
    "vci",
    "vbrc",
    "vintlv",
    "vdintlv",
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
    "vcmp",
    "vcmps",
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
    "vmull",
    "vmula",
    "vdhist",
    "vchist",
    "vgather",
    "vgatherb",
    "vscatter",
]


def _op_name(call_or_op):
    op = getattr(call_or_op, "op", call_or_op)
    return getattr(op, "name", None)


def _indent(line):
    return len(line) - len(line.lstrip())


def _collect_pto_calls(func):
    calls = []

    def visit(node):
        if isinstance(node, Call):
            name = _op_name(node)
            if name is not None and name.startswith("tl.vmi."):
                calls.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


@pytest.mark.parametrize("op_name,combine", [("fmax", T.max), ("fmin", T.min)])
@pytest.mark.pto
def test_pto_float32x2_minmax_codegen(op_name, combine):
    @T.prim_func
    def func(
        A: T.Tensor((2,), "float32"),
        B: T.Tensor((2,), "float32"),
        C: T.Tensor((2,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), "float32")
            b_ub = T.alloc_shared((2,), "float32")
            c_ub = T.alloc_shared((2,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=1):
                for i in T.vectorized(2):
                    c_ub[i] = combine(a_ub[i], b_ub[i])
            T.copy(c_ub, C)

    source = lower(func, target="pto").kernel_source
    assert f"_tl_vectorize_binary_f32x2(pto.{op_name}," in source
    compile(source, "<pto-float32x2-minmax>", "exec")


@pytest.mark.parametrize(
    "scalar_op,unary",
    [
        ("pto.exp", T.exp),
        ("pto.log", T.log),
        ("pto.sqrt", T.sqrt),
    ],
)
@pytest.mark.pto
def test_pto_float32x2_unary_math_codegen(scalar_op, unary):
    @T.prim_func
    def func(A: T.Tensor((2,), "float32"), B: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), "float32")
            b_ub = T.alloc_shared((2,), "float32")
            T.copy(A, a_ub)
            with T.SimtVF(threads=1):
                for i in T.vectorized(2):
                    b_ub[i] = unary(a_ub[i])
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    assert f"_tl_vectorize_unary_f32x2({scalar_op}," in source
    compile(source, "<pto-float32x2-unary-math>", "exec")


@pytest.mark.pto
def test_pto_float32x2_div_codegen():
    @T.prim_func
    def func(
        A: T.Tensor((2,), "float32"),
        B: T.Tensor((2,), "float32"),
        C: T.Tensor((2,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), "float32")
            b_ub = T.alloc_shared((2,), "float32")
            c_ub = T.alloc_shared((2,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=1):
                for i in T.vectorized(2):
                    c_ub[i] = a_ub[i] / b_ub[i]
            T.copy(c_ub, C)

    source = lower(func, target="pto").kernel_source
    assert "from tilelang.contrib.ptodsl.simt import (" in source
    assert "def _tl_vectorize_binary_f32x2" not in source
    assert "_tl_vectorize_binary_f32x2(_tl_scalar_div," in source
    compile(source, "<pto-float32x2-div>", "exec")


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,dst_pto_type",
    [
        ("float32", "float16", "pto.f16x2"),
        ("float32", "bfloat16", "pto.bf16x2"),
        ("float16", "float32", "pto.f32x2"),
        ("bfloat16", "float32", "pto.f32x2"),
    ],
)
@pytest.mark.pto
def test_pto_packed_float_cast_and_local_fragment_codegen(src_dtype, dst_dtype, dst_pto_type):
    @T.prim_func
    def func(A: T.Tensor((2,), src_dtype), B: T.Tensor((2,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), src_dtype)
            b_ub = T.alloc_shared((2,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimtVF(threads=1):
                local = T.alloc_fragment((2,), src_dtype)
                T.copy(a_ub, local)
                T.copy(local, b_ub)
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    assert f', {dst_pto_type}, rounding="r", saturation="nosat")' in source
    assert "pto.alloc_buffer((2,)," in source
    compile(source, "<pto-packed-float-cast>", "exec")


@pytest.mark.pto
def test_pto_type_helpers():
    assert str(T.vmi.vreg(64, T.float32)) == "float32x64"
    assert str(T.vmi.vreg(8, "float16")) == "float16x8"
    assert str(T.vmi.mask(64)) == "boolx64"


@pytest.mark.pto
def test_pto_alloc_local_builds_vector_register_buffer():
    @T.prim_func
    def func():
        with T.Kernel(1) as _, T.SimdVF():
            regs = T.vmi.alloc_local((4,), T.vmi.vreg(64, T.float32))
            for i in T.unroll(4, explicit=True):
                regs[i] = T.vmi.vbrc(T.float32(1), size=64)
            T.evaluate(regs[0])

    allocations = []
    stores = []

    def visit(node):
        if isinstance(node, tirx.SBlock):
            allocations.extend(buffer for buffer in node.alloc_buffers if buffer.scope() == "local")
        elif isinstance(node, tirx.BufferStore) and node.buffer.scope() == "local":
            stores.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert len(allocations) == 1
    assert tuple(int(dim) for dim in allocations[0].shape) == (4,)
    assert str(allocations[0].dtype) == "float32x64"
    assert stores and str(stores[0].value.dtype) == "float32x64"


@pytest.mark.pto
def test_pto_alloc_local_validates_vreg_type_and_shape(monkeypatch):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match="VMI vector type"):
        T.vmi.alloc_local((4,), T.float32)
    with pytest.raises(TypeError, match="compile-time integer"):
        T.vmi.alloc_local((tirx.Var("n", "int32"),), T.vmi.vreg(64, T.float32))
    with pytest.raises(ValueError, match="positive shape dimensions"):
        T.vmi.alloc_local((0,), T.vmi.vreg(64, T.float32))
    with pytest.raises(ValueError, match="at least one shape dimension"):
        T.vmi.alloc_local((), T.vmi.vreg(64, T.float32))


@pytest.mark.pto
def test_pto_namespace_exports_public_ops():
    expected = [
        "alloc_local",
        "alloc_var",
        "vload",
        "vstore",
        "vci",
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
        "vcmp",
        "vcmps",
        "vsel",
        "vselr",
        "vbrc",
        "vcadd",
        "vcmax",
        "vcmin",
        "vcvt",
        "vinterpret_cast",
        "vexpdif",
        "vaxpy",
        "vlrelu",
        "vprelu",
        "vmull",
        "vmula",
        "vdhist",
        "vchist",
        "vgather",
        "vgatherb",
        "vscatter",
        "create_mask",
        "vintlv",
        "vdintlv",
    ]
    for name in expected:
        assert hasattr(T.vmi, name), name
    assert not hasattr(T.vmi, "pair_get")


@pytest.mark.pto
def test_pto_tir_call_dtypes_preserve_vector_and_mask_lanes():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float16")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float16")
            with T.SimdVF():
                mask = T.vmi.create_mask(37, size=64)
                src = T.vmi.vload(a_ub[0], size=64)
                one = T.vmi.vbrc(T.float32(1), size=64)
                out = T.vmi.vadd(src, one, mask)
                reduced = T.vmi.vcadd(out, mask, group=1, reassoc=False)
                cast = T.vmi.vcvt(out, "float16")
                T.vmi.vstore(cast, b_ub[0], mask)
                T.evaluate(reduced)

    dtypes_by_op = {}
    for call in _collect_pto_calls(func):
        dtypes_by_op.setdefault(_op_name(call), set()).add(str(call.dtype))

    assert dtypes_by_op["tl.vmi.create_mask"] == {"boolx64"}
    assert dtypes_by_op["tl.vmi.vload"] == {"float32x64"}
    assert dtypes_by_op["tl.vmi.vbrc"] == {"float32x64"}
    assert dtypes_by_op["tl.vmi.vadd"] == {"float32x64"}
    assert dtypes_by_op["tl.vmi.vcadd"] == {"float32"}
    assert dtypes_by_op["tl.vmi.vcvt"] == {"float16x64"}
    assert dtypes_by_op["tl.vmi.vstore"] == {""}
    vcadd_call = next(call for call in _collect_pto_calls(func) if _op_name(call) == "tl.vmi.vcadd")
    assert dict(vcadd_call.annotations)["reassoc"] == 0


@pytest.mark.pto
def test_pto_integer_vcadd_allows_omitted_reassoc():
    @T.prim_func
    def func(A: T.Buffer((64,), "int32")):
        with T.Kernel(1) as _, T.SimdVF():
            source = T.vmi.vci(T.int32(0), size=64)
            mask = T.vmi.create_mask(64, size=64)
            reduced = T.vmi.vcadd(source, mask)
            T.evaluate(reduced)

    vcadd_call = next(call for call in _collect_pto_calls(func) if _op_name(call) == "tl.vmi.vcadd")
    assert "reassoc" not in dict(vcadd_call.annotations)


@pytest.mark.pto
def test_pto_create_mask_tir_annotations_match_ptodsl_surface():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32")):
        with T.Kernel(1) as _, T.SimdVF():
            mask = T.vmi.create_mask(3, size=32, group=4)
            T.evaluate(mask)

    create_mask_calls = [call for call in _collect_pto_calls(func) if _op_name(call) == "tl.vmi.create_mask"]
    assert len(create_mask_calls) == 1
    call = create_mask_calls[0]
    assert str(call.dtype) == "boolx32"
    assert dict(call.annotations) == {"group": 4, "size": 32}


@pytest.mark.pto
def test_pto_scope_guard_allows_ops_inside_simdvf():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                src = T.vmi.vload(a_ub[0], size=64)
                T.vmi.vstore(src, b_ub[0], mask)

    names = {_op_name(call) for call in _collect_pto_calls(func)}
    assert {"tl.vmi.create_mask", "tl.vmi.vload", "tl.vmi.vstore"}.issubset(names)


@pytest.mark.pto
def test_pto_bufferload_addresses_lower_to_explicit_ptr_offset_args():
    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((128,), "float32")
            b_ub = T.alloc_shared((128,), "float32")
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                src = T.vmi.vload(a_ub[16], size=64)
                T.vmi.vstore(src, b_ub[32], mask)

    pto_calls = _collect_pto_calls(func)
    vload_call = next(call for call in pto_calls if _op_name(call) == "tl.vmi.vload")
    vstore_call = next(call for call in pto_calls if _op_name(call) == "tl.vmi.vstore")

    assert _op_name(vload_call.args[0]) == "tl.access_ptr"
    assert str(vload_call.args[1]) == "16"
    assert _op_name(vstore_call.args[1]) == "tl.access_ptr"
    assert str(vstore_call.args[2]) == "32"


@pytest.mark.pto
def test_pto_fp4_multidim_bufferload_uses_packed_linear_offset():
    @T.prim_func
    def func(
        A: T.Buffer((256,), "bfloat16"),
        B: T.Buffer((2, 256), "float4_e2m1fn"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((256,), "bfloat16")
            b_ub = T.alloc_shared((2, 256), "float4_e2m1fn")
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                fp4 = T.vmi.vcvt(T.vmi.vload(a_ub[0], size=256), "float4_e2m1fn")
                T.vmi.vstore(fp4, b_ub[1, 0], mask)
                loaded = T.vmi.vload(b_ub[1, 0], size=256)
                T.evaluate(loaded)

    pto_calls = _collect_pto_calls(func)
    vload = next(call for call in pto_calls if _op_name(call) == "tl.vmi.vload" and str(call.dtype) == "float4_e2m1fnx256")
    vstore = next(call for call in pto_calls if _op_name(call) == "tl.vmi.vstore")
    assert _op_name(vload.args[0]) == "tl.access_ptr"
    assert str(vload.args[1]) == "128"
    assert _op_name(vstore.args[1]) == "tl.access_ptr"
    assert str(vstore.args[2]) == "128"


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda value, mask: T.vmi.vload(value, size=16, dist_mode="invalid"),
            "does not support dist_mode",
        ),
        (
            lambda value, mask: T.vmi.vload(value, size=16, to_dtype="float32"),
            "to_dtype",
        ),
        (
            lambda value, mask: T.vmi.vload(value, size=16, dist_mode="unpack"),
            "requires to_dtype",
        ),
        (
            lambda value, mask: T.vmi.vload(value, size=16, dist_mode="unpack", to_dtype="float16"),
            "unpack must widen",
        ),
        (
            lambda value, mask: T.vmi.vload(value, size=16, dist_mode="continuous", group=2, stride=1),
            "dist_mode together with group",
        ),
        (
            lambda value, mask: T.vmi.vload(value, size=16, group=2),
            "requires stride",
        ),
        (
            lambda value, mask: T.vmi.vload(value, size=16, block_stride=1),
            "requires block_stride and repeat_stride together",
        ),
        (
            lambda value, mask: T.vmi.vload(value, size=16, stride=1),
            "accepts stride only when group",
        ),
        (
            lambda value, mask: T.vmi.vstore(value, value, mask=mask, group=2, stride=1),
            "group mode does not take a mask",
        ),
        (
            lambda value, mask: T.vmi.vstore(value, value, mask=mask, dist_mode="dintlv"),
            "requires an \\(even, odd\\) pair",
        ),
        (
            lambda value, mask: T.vmi.vstore(T.vmi.VmiPair(value), value, mask=mask),
            "expects a single VMI vector",
        ),
        (
            lambda value, mask: T.vmi.vstore((value, value), value, mask=mask),
            "expects a single VMI vector",
        ),
        (
            lambda value, mask: T.vmi.vstore(value, value, mask=mask, dist_mode="brc"),
            "does not support dist_mode",
        ),
        (
            lambda value, mask: T.vmi.vcvt(value, "float16", mask=mask),
            "does not support masked form",
        ),
        (
            lambda value, mask: T.vmi.vcvt(value),
            "requires to_dtype",
        ),
        (
            lambda value, mask: T.vmi.vcvt(value, "float16", rounding="nearest"),
            "does not support rounding",
        ),
        (
            lambda value, mask: T.vmi.create_mask(8, size=15, group=4),
            "requires size to be divisible by group",
        ),
        (
            lambda value, mask: T.vmi.create_mask(8, size=16, group=True),
            "requires group to be a positive Python integer",
        ),
        (
            lambda value, mask: T.vmi.vbrc(value, size=16, group=2),
            "requires the input lane count to match group",
        ),
        (
            lambda value, mask: T.vmi.vcadd(value, mask),
            "requires an explicit reassoc",
        ),
        (
            lambda value, mask: T.vmi.vcadd(value, mask, reassoc=1),
            "requires reassoc to be the Python boolean",
        ),
        (
            lambda value, mask: T.vmi.create_mask(5, size=16, group=4),
            "active_lanes to be <= the inferred group_size",
        ),
        (
            lambda value, mask: T.vmi.vcmp(value, value, mask, 0),
            "does not support comparison",
        ),
        (
            lambda value, mask: T.vmi.vselr(value, value),
            "requires an integer VMI vector",
        ),
        (
            lambda value, mask: T.vmi.vcmax(value, mask, group=3),
            "source lanes to be divisible by group",
        ),
        (
            lambda value, mask: T.vmi.vmull(value, value, mask),
            "requires identical 32-bit integer vectors",
        ),
    ],
)
@pytest.mark.pto
def test_pto_wrappers_reject_invalid_mode_combinations(monkeypatch, call, message):
    value = SimpleNamespace(dtype="float16x16")
    mask = SimpleNamespace(dtype="boolx16")
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)

    with pytest.raises((TypeError, ValueError), match=message):
        call(value, mask)


@pytest.mark.pto
def test_pto_vinterpret_cast_width_changing_bit_totals(monkeypatch):
    """Same-width keeps lanes; width change requires matching bit totals (ASC vintlv)."""
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(
        T.vmi,
        "_call_vmi",
        lambda op, result_dtype, *args, **kwargs: calls.append((op, str(result_dtype))) or "ok",
    )

    T.vmi.vinterpret_cast(SimpleNamespace(dtype="float32x64"), "uint32")
    assert calls[-1] == ("vinterpret_cast", "uint32x64")

    # ASC vintlv half-split: 128xbf16 → 64xf32
    T.vmi.vinterpret_cast(SimpleNamespace(dtype="bfloat16x128"), "float32")
    assert calls[-1] == ("vinterpret_cast", "float32x64")

    # float16x16 → float32 is legal (256-bit total → float32x8; 8 is a VMI lane count)
    T.vmi.vinterpret_cast(SimpleNamespace(dtype="float16x16"), "float32")
    assert calls[-1] == ("vinterpret_cast", "float32x8")

    with pytest.raises(TypeError, match="bit totals"):
        T.vmi.vinterpret_cast(SimpleNamespace(dtype="float16x15"), "float32")

    # Bit totals match but result lanes are not a formal PTODSL VMI count (16).
    with pytest.raises(ValueError, match="lanes"):
        T.vmi.vinterpret_cast(SimpleNamespace(dtype="int8x128"), "si64")
    with pytest.raises(ValueError, match="lanes"):
        T.vmi.vinterpret_cast(SimpleNamespace(dtype="int8x128"), "int64")


@pytest.mark.pto
def test_pto_wrappers_reject_invalid_operand_contracts(monkeypatch):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    f32 = SimpleNamespace(dtype="float32x16")
    f16 = SimpleNamespace(dtype="float16x16")
    i32 = SimpleNamespace(dtype="int32x16")
    mask16 = SimpleNamespace(dtype="boolx16")
    mask8 = SimpleNamespace(dtype="boolx8")

    with pytest.raises(TypeError, match="identical VMI vector types"):
        T.vmi.vadd(f32, f16, mask16)
    with pytest.raises(TypeError, match="identical VMI vector types"):
        T.vmi.vsel(mask16, f32, f16)
    with pytest.raises(TypeError, match="identical VMI vector types"):
        T.vmi.vintlv(f32, f16, mask16)
    with pytest.raises(TypeError, match="lane counts to match"):
        T.vmi.vscatter(f32, SimpleNamespace(dtype="ptr"), i32, mask8)
    with pytest.raises(TypeError, match="uint16x256 accumulator"):
        T.vmi.vdhist(i32, i32, mask16)


@pytest.mark.pto
def test_pto_gather_rejects_non_ub_buffer():
    with pytest.raises(TypeError, match="requires a UB pointer"):

        @T.prim_func
        def func(A: T.Buffer((64,), "float32")):
            with T.Kernel(1) as _, T.SimdVF():
                offsets = T.vmi.vci(T.int32(0), size=64)
                mask = T.vmi.create_mask(64, size=64)
                T.evaluate(T.vmi.vgather(A[0], offsets, mask))


@pytest.mark.pto
def test_pto_vload_supports_buffer_address_and_pointer_offset(monkeypatch):
    class FakeBuffer:
        pass

    class FakeBufferLoad(FakeBuffer):
        pass

    class FakeBufferRegion(FakeBuffer):
        pass

    class FakePointer:
        pass

    calls = []

    monkeypatch.setattr(T.vmi, "Buffer", FakeBuffer)
    monkeypatch.setattr(T.vmi, "BufferLoad", FakeBufferLoad)
    monkeypatch.setattr(T.vmi, "BufferRegion", FakeBufferRegion)
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(T.vmi, "_vector_result_dtype", lambda *args, **kwargs: "float32x64")
    monkeypatch.setattr(
        T.vmi,
        "_call_vmi",
        lambda op, result_dtype, *args, **kwargs: (
            calls.append((op, result_dtype, tuple(arg for arg in args if arg is not None), kwargs)) or "ok"
        ),
    )

    source = FakeBufferLoad()
    source.buffer = FakeBuffer()
    source.buffer.dtype = "float32"
    source.indices = [3]
    monkeypatch.setattr(
        T.vmi,
        "access_ptr",
        lambda source, access_type, extent=None, offset=0: (
            "access_ptr",
            source,
            access_type,
            extent,
            offset,
        ),
    )
    out = T.vmi.vload(source, size=64)
    assert out == "ok"
    assert calls[-1][0] == "vload"
    assert calls[-1][2] == (("access_ptr", source.buffer, "r", 64, 0), 3)

    region = FakeBufferRegion()
    region.buffer = source.buffer
    for buffer_address in (source, source.buffer, region):
        buffer_address.dtype = "float32"
        with pytest.raises(TypeError, match="express the offset in the buffer index"):
            T.vmi.vload(buffer_address, 0, size=64)
        with pytest.raises(TypeError, match="express the offset in the buffer index"):
            T.vmi.vload(buffer_address, 3, size=64)

    calls.clear()
    ptr = FakePointer()
    ptr.type_annotation = SimpleNamespace(element_type=SimpleNamespace(dtype="float32"))
    out = T.vmi.vload(ptr, 7, size=64)
    assert out == "ok"
    assert calls[-1][0] == "vload"
    assert calls[-1][2] == (ptr, 7)


@pytest.mark.pto
def test_pto_loads_derive_element_dtype_from_real_pointer_annotations(monkeypatch):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        T.vmi,
        "_call_vmi",
        lambda op, result_dtype, *args, **kwargs: SimpleNamespace(dtype=result_dtype),
    )

    source = T.ptr("float16", "shared")
    offsets = SimpleNamespace(dtype="int32x16")
    mask = SimpleNamespace(dtype="boolx16")

    loaded = T.vmi.vload(source, 3, size=16)
    gathered = T.vmi.vgather(source, offsets, mask)
    gathered_bytes = T.vmi.vgatherb(source, offsets, mask)

    assert str(loaded.dtype) == "float16x16"
    assert str(gathered.dtype) == "float16x16"
    assert str(gathered_bytes.dtype) == "float16x16"


@pytest.mark.parametrize(
    "call",
    [
        lambda source, offsets, mask: T.vmi.vload(source, size=16),
        lambda source, offsets, mask: T.vmi.vgather(source, offsets, mask),
        lambda source, offsets, mask: T.vmi.vgatherb(source, offsets, mask),
    ],
)
@pytest.mark.pto
def test_pto_loads_reject_pointer_without_element_dtype(monkeypatch, call):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    source = T.ptr(storage_scope="shared")
    offsets = SimpleNamespace(dtype="int32x16")
    mask = SimpleNamespace(dtype="boolx16")

    with pytest.raises(TypeError, match="known element dtype"):
        call(source, offsets, mask)


@pytest.mark.pto
def test_pto_vstore_supports_buffer_address_and_pointer_offset(monkeypatch):
    class FakeBuffer:
        pass

    class FakeBufferLoad(FakeBuffer):
        pass

    class FakeBufferRegion(FakeBuffer):
        pass

    class FakePointer:
        pass

    vec = SimpleNamespace(dtype="float32x64")
    calls = []

    monkeypatch.setattr(T.vmi, "Buffer", FakeBuffer)
    monkeypatch.setattr(T.vmi, "BufferLoad", FakeBufferLoad)
    monkeypatch.setattr(T.vmi, "BufferRegion", FakeBufferRegion)
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(T.vmi, "_lanes_of", lambda value: 64)
    monkeypatch.setattr(
        T.vmi,
        "_call_vmi",
        lambda op, result_dtype, *args, **kwargs: (
            calls.append((op, result_dtype, tuple(arg for arg in args if arg is not None), kwargs)) or "ok"
        ),
    )

    destination = FakeBufferLoad()
    destination.buffer = FakeBuffer()
    destination.buffer.dtype = "float32"
    destination.indices = [5]
    monkeypatch.setattr(
        T.vmi,
        "access_ptr",
        lambda source, access_type, extent=None, offset=0: (
            "access_ptr",
            source,
            access_type,
            extent,
            offset,
        ),
    )
    out = T.vmi.vstore(vec, destination, mask="pred")
    assert out == "ok"
    assert calls[-1][0] == "vstore"
    assert calls[-1][2] == (vec, ("access_ptr", destination.buffer, "w", 64, 0), 5, "pred")
    assert calls[-1][2][-1] == "pred"

    for buffer_address in (destination, destination.buffer, FakeBufferRegion()):
        with pytest.raises(TypeError, match="express the offset in the buffer index"):
            T.vmi.vstore(vec, buffer_address, 0, mask="pred")
        with pytest.raises(TypeError, match="express the offset in the buffer index"):
            T.vmi.vstore(vec, buffer_address, 3, mask="pred")

    calls.clear()
    ptr = FakePointer()
    out = T.vmi.vstore(vec, ptr, 7, mask="pred")
    assert out == "ok"
    assert calls[-1][0] == "vstore"
    assert calls[-1][2][1:4] == (ptr, 7, "pred")


@pytest.mark.pto
def test_pto_pair_unpacks_and_indexes_once(monkeypatch):
    calls = []

    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        T.vmi.tirx,
        "call_intrin",
        lambda dtype, op, *args, **kwargs: (
            calls.append((dtype, getattr(op, "name", op), args)) or ("call", dtype, getattr(op, "name", op), args)
        ),
    )

    pair = T.vmi.VmiPair(SimpleNamespace(dtype="float32x64"))
    left, right = pair

    assert left[1] == "float32x64"
    assert right[1] == "float32x64"
    assert calls[0][1] == "tl.vmi.pair_get"
    assert calls[0][2][-1] == 0
    assert calls[1][2][-1] == 1
    assert pair[0][1] == "float32x64"
    assert len(pair) == 2


@pytest.mark.pto
def test_pto_vload_dintlv_returns_pair(monkeypatch):
    calls = []

    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(T.vmi, "_vector_result_dtype", lambda *args, **kwargs: "float32x64")
    monkeypatch.setattr(
        T.vmi,
        "_call_vmi",
        lambda op, result_dtype, *args, **kwargs: calls.append((op, result_dtype, args, kwargs)) or SimpleNamespace(dtype="float32x64"),
    )
    monkeypatch.setattr(
        T.vmi,
        "_wrap_pair",
        lambda pair: T.vmi.VmiPair(pair),
    )
    monkeypatch.setattr(
        T.vmi,
        "access_ptr",
        lambda source, access_type, extent=None, offset=0: ("access_ptr", source, access_type, extent, offset),
    )
    monkeypatch.setattr(
        T.vmi.tirx,
        "call_intrin",
        lambda dtype, op, *args, **kwargs: (
            calls.append((dtype, getattr(op, "name", op), args)) or ("call", dtype, getattr(op, "name", op), args)
        ),
    )

    source = SimpleNamespace(
        dtype="ptr",
        type_annotation=SimpleNamespace(element_type=SimpleNamespace(dtype="float32")),
    )
    pair = T.vmi.vload(source, size=64, dist_mode="dintlv")
    assert isinstance(pair, T.vmi.VmiPair)
    assert calls[0][0] == "vload"
    left, right = pair
    assert left[2] == "tl.vmi.pair_get"
    assert right[2] == "tl.vmi.pair_get"
    assert pair[1][2] == "tl.vmi.pair_get"


@pytest.mark.pto
def test_vmi_fp4_rejects_unsupported_paths(monkeypatch):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    fp4 = SimpleNamespace(dtype="float4_e2m1fnx256")

    with pytest.raises(ValueError, match="even FP4 lane count"):
        T.vmi.vload(fp4, size=255)
    with pytest.raises(TypeError, match="does not unpack or convert packed FP4"):
        T.vmi.vload(fp4, size=256, dist_mode="unpack", to_dtype="uint8")
    with pytest.raises(TypeError, match='does not support packed FP4 with dist_mode="dintlv"'):
        T.vmi.vload(fp4, size=256, dist_mode="dintlv")
    with pytest.raises(TypeError, match="does not support packed FP4 vectors"):
        T.vmi.vadd(fp4, fp4, SimpleNamespace(dtype="boolx256"))
    with pytest.raises(TypeError, match="does not support packed FP4 vectors"):
        T.vmi.vsel(SimpleNamespace(dtype="boolx256"), fp4, fp4)


@pytest.mark.pto
def test_vmi_fp4_vcvt_validates_pto_specific_options(monkeypatch):
    calls = []
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        T.vmi,
        "_call_vmi",
        lambda op, result_dtype, *args, **kwargs: calls.append((op, result_dtype, args, kwargs)) or "ok",
    )
    bf16 = SimpleNamespace(dtype="bfloat16x256")

    assert T.vmi.vcvt(bf16, "float4_e2m1fn", rounding="C") == "ok"
    assert calls[-1][3]["rounding"] == "C"
    with pytest.raises(ValueError, match="expected one of A, C, F, R, Z"):
        T.vmi.vcvt(bf16, "float4_e2m1fn", rounding="H")
    with pytest.raises(ValueError, match="does not support saturate"):
        T.vmi.vcvt(bf16, "float4_e2m1fn", saturate="SAT")


@pytest.mark.pto
def test_vmi_gather_rejects_packed_fp4_source(monkeypatch):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    fp4_ptr = SimpleNamespace(dtype="ptr", type_annotation=SimpleNamespace(element_type=SimpleNamespace(dtype="float4_e2m1fn")))
    offsets = SimpleNamespace(dtype="int32x64")
    mask = SimpleNamespace(dtype="boolx64")

    with pytest.raises(TypeError, match="does not support packed FP4 source buffers"):
        T.vmi.vgather(fp4_ptr, offsets, mask)
    with pytest.raises(TypeError, match="does not support packed FP4 source buffers"):
        T.vmi.vgatherb(fp4_ptr, offsets, mask)


@pytest.mark.pto
def test_vmi_fp4_vstore_requires_physical_mask_lanes(monkeypatch):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(T.vmi, "_call_vmi", lambda *args, **kwargs: "ok")
    fp4 = SimpleNamespace(dtype="float4_e2m1fnx256")
    ptr = SimpleNamespace(dtype="ptr")

    with pytest.raises(TypeError, match=r"physical FP4x2 mask with half as many lanes.*expected 128, got 256"):
        T.vmi.vstore(fp4, ptr, mask=SimpleNamespace(dtype="boolx256"))
    assert T.vmi.vstore(fp4, ptr, mask=SimpleNamespace(dtype="boolx128")) == "ok"


@pytest.mark.pto
def test_vmi_vstore_dintlv_rejects_mismatched_pair_types(monkeypatch):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match="requires identical VMI vector types"):
        T.vmi.vstore(
            (SimpleNamespace(dtype="float16x64"), SimpleNamespace(dtype="float16x32")),
            SimpleNamespace(dtype="ptr"),
            dist_mode="dintlv",
        )


@pytest.mark.pto
def test_vmi_fp4_vload_keeps_logical_size_and_packs_offset(monkeypatch):
    calls = []
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        T.vmi,
        "_call_vmi",
        lambda op, result_dtype, *args, **kwargs: calls.append((op, result_dtype, args, kwargs)) or "ok",
    )
    fp4_ptr = SimpleNamespace(dtype="ptr", type_annotation=SimpleNamespace(element_type=SimpleNamespace(dtype="float4_e2m1fn")))

    assert T.vmi.vload(fp4_ptr, offset=256, size=256) == "ok"
    assert calls[-1] == (
        "vload",
        "float4_e2m1fnx256",
        (fp4_ptr, 128),
        {
            "size": 256,
            "to_dtype": None,
            "stride": None,
            "block_stride": None,
            "repeat_stride": None,
            "dist_mode": None,
            "group": None,
            "loc": None,
            "ip": None,
        },
    )


@pytest.mark.pto
def test_vmi_fp4_rejects_non_contiguous_modes(monkeypatch):
    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    fp4_ptr = SimpleNamespace(dtype="ptr", type_annotation=SimpleNamespace(element_type=SimpleNamespace(dtype="float4_e2m1fn")))
    with pytest.raises(TypeError, match="only supports contiguous packed FP4 loads"):
        T.vmi.vload(fp4_ptr, size=256, group=2, stride=128)
    with pytest.raises(TypeError, match="only supports contiguous packed FP4 loads"):
        T.vmi.vload(fp4_ptr, size=256, dist_mode="brc")
    with pytest.raises(TypeError, match="only supports contiguous packed FP4 stores"):
        T.vmi.vstore(SimpleNamespace(dtype="float4_e2m1fnx256"), fp4_ptr, group=2, stride=128)


@pytest.mark.pto
def test_pto_pair_return_path_can_be_reused_for_vstore(monkeypatch):
    pair = T.vmi.VmiPair(SimpleNamespace(dtype="float32x64"))
    calls = []

    monkeypatch.setattr(T.vmi, "require_vmi_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        T.vmi.tirx,
        "call_intrin",
        lambda dtype, op, *args, **kwargs: ("call", dtype, getattr(op, "name", op), args),
    )
    monkeypatch.setattr(
        T.vmi, "_call_vmi", lambda op, result_dtype, *args, **kwargs: calls.append((op, result_dtype, args, kwargs)) or "ok"
    )
    monkeypatch.setattr(T.vmi, "_lanes_of", lambda value: 64)
    monkeypatch.setattr(
        T.vmi,
        "access_ptr",
        lambda source, access_type, extent=None, offset=0: ("access_ptr", source, access_type, extent, offset),
    )

    out = T.vmi.vstore(pair, SimpleNamespace(dtype="ptr"), mask=SimpleNamespace(dtype="pred"), dist_mode="dintlv")
    assert out == "ok"
    assert calls[-1][0] == "vstore"
    assert calls[-1][2][0][2] == "tl.vmi.pair_get"
    assert calls[-1][2][1][2] == "tl.vmi.pair_get"


@pytest.mark.pto
def test_pto_codegen_emits_static_local_register_lists():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            with T.SimdVF():
                regs = T.vmi.alloc_local((4,), T.vmi.vreg(64, T.float32))
                mask = T.vmi.create_mask(64, size=64)
                for i in T.unroll(2, explicit=True):
                    for j in T.unroll(2, explicit=True):
                        regs[i * 2 + j] = T.vmi.vload(a_ub[0], size=64)
                T.vmi.vstore(T.vmi.vadd(regs[0], regs[3], mask), b_ub[0], mask)

    source = lower(func, target="pto").kernel_source
    assert "= [None] * 4" in source
    assert "pto.static_range(" not in source
    for index in range(4):
        assert f"regs[{index}] = pto.vmi.vload" in source
    assert re.search(r"pto\.vmi\.vadd\([^\n]*\[0\], [^\n]*\[3\]", source)


@pytest.mark.pto
def test_pto_codegen_emits_range_for_non_explicit_unroll():
    @T.prim_func
    def func(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((256,), "float32")
            b_ub = T.alloc_shared((256,), "float32")
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                for i in T.unroll(4, explicit=False):
                    value = T.vmi.vload(a_ub[i * 64], size=64)
                    T.vmi.vstore(value, b_ub[i * 64], mask)

    lower(func, target="pto")


@pytest.mark.pto
def test_pto_codegen_rejects_non_explicit_unroll_local_register_index():
    @T.prim_func
    def func():
        with T.Kernel(1) as _, T.SimdVF():
            regs = T.vmi.alloc_local((4,), T.vmi.vreg(64, T.float32))
            for i in T.unroll(4, explicit=False):
                regs[i] = T.vmi.vbrc(T.float32(1), size=64)

    with pytest.raises(Exception, match=r"T\.unroll\(\.\.\., explicit=False\)"):
        lower(func, target="pto")


@pytest.mark.pto
def test_pto_codegen_rejects_runtime_local_register_index():
    @T.prim_func
    def func():
        with T.Kernel(1) as _, T.SimdVF():
            regs = T.vmi.alloc_local((4,), T.vmi.vreg(64, T.float32))
            for i in T.serial(4):
                regs[i] = T.vmi.vbrc(T.float32(1), size=64)

    with pytest.raises(Exception, match="requires a compile-time constant index"):
        lower(func, target="pto")


@pytest.mark.pto
def test_pto_codegen_emits_vector_calls_and_pair_indexing():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.vmi.create_mask(32, size=64, group=2)
                src = T.vmi.vload(a_ub[0], size=64)
                lo, hi = T.vmi.vintlv(src, src, mask)
                out = T.vmi.vadd(lo, hi, mask)
                T.vmi.vstore(out, b_ub[0], mask)
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    assert "pto.vmi.create_mask(" in source
    assert "group=2" in source
    assert "group_size=" not in source
    assert "size=64" in source
    assert "pto.vmi.vload(" in source
    assert "pto.vmi.vintlv(" in source
    assert "pto.vmi.vadd(" in source
    assert "pto.vmi.vcvt(" not in source or "to_dtype=pto.f16" in source
    assert "pto.vmi.vload(scalar.load(" not in source
    assert "pto.vmi.vstore(out, scalar.load(" not in source
    assert "[0]" in source
    assert "[1]" in source


@pytest.mark.pto
def test_pto_codegen_preserves_vmi_merge_mode():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.vmi.create_mask(37, size=64)
                x = T.vmi.vload(a_ub[0], size=64)
                y = T.vmi.vbrc(T.float32(1), size=64)
                merged = T.vmi.vdiv(x, y, mask, pmode="merge")
                T.vmi.vstore(merged, b_ub[0], mask)
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    assert "pto.vmi.vdiv(" in source
    assert 'pmode="merge"' in source


@pytest.mark.pto
def test_pto_codegen_keeps_dintlv_vstore_pair_grouped():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                src = T.vmi.vload(a_ub[0], size=64)
                interleaved = T.vmi.vintlv(src, src, mask)
                T.vmi.vstore(interleaved, b_ub[0], mask, dist_mode="dintlv")
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    vstore_line = next(line for line in source.splitlines() if "pto.vmi.vstore(" in line and 'dist_mode="dintlv"' in line)
    assert re.search(r"pto\.vmi\.vstore\(\(\(.+\)\[0\], \(.+\)\[1\]\), ", vstore_line)
    assert not re.search(r"pto\.vmi\.vstore\(\(.+\)\[0\], \(.+\)\[1\], ", vstore_line)


@pytest.mark.pto
def test_pto_codegen_covers_every_public_vector_op():
    @T.prim_func
    def func(
        A: T.Buffer((64,), "float32"),
        B: T.Buffer((64,), "float32"),
        I: T.Buffer((64,), "int32"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            i_ub = T.alloc_shared((64,), "int32")
            hist_acc_ub = T.alloc_shared((256,), "uint16")
            hist_src_ub = T.alloc_shared((256,), "uint8")
            with T.SimdVF():
                mask = T.vmi.create_mask(37, size=64)
                hist_mask = T.vmi.create_mask(256, size=256)
                x = T.vmi.vload(a_ub[0], size=64)
                y = T.vmi.vbrc(T.float32(1), size=64)
                i = T.vmi.vload(i_ub[0], size=64)
                idx = T.vmi.vci(T.int32(0), size=64)
                hist_acc = T.vmi.vload(hist_acc_ub[0], size=256)
                hist_src = T.vmi.vload(hist_src_ub[0], size=256)

                add = T.vmi.vadd(x, y, mask)
                sub = T.vmi.vsub(x, y, mask)
                mul = T.vmi.vmul(x, y, mask)
                div = T.vmi.vdiv(x, y, mask)
                vmax = T.vmi.vmax(x, y, mask)
                vmin = T.vmi.vmin(x, y, mask)
                vand = T.vmi.vand(i, idx, mask)
                vor = T.vmi.vor(i, idx, mask)
                vxor = T.vmi.vxor(i, idx, mask)
                vshl = T.vmi.vshl(i, idx, mask)
                vshr = T.vmi.vshr(i, idx, mask)

                vabs = T.vmi.vabs(x, mask)
                vneg = T.vmi.vneg(x, mask)
                vrelu = T.vmi.vrelu(x, mask)
                vexp = T.vmi.vexp(x, mask)
                vln = T.vmi.vln(x, mask)
                vsqrt = T.vmi.vsqrt(x, mask)
                vnot = T.vmi.vnot(i, mask)

                vadds = T.vmi.vadds(x, T.float32(1), mask)
                vmuls = T.vmi.vmuls(x, T.float32(2), mask)
                vmaxs = T.vmi.vmaxs(x, T.float32(3), mask)
                vmins = T.vmi.vmins(x, T.float32(4), mask)
                vshls = T.vmi.vshls(i, T.int32(1), mask)
                vshrs = T.vmi.vshrs(i, T.int32(1), mask)

                cmp = T.vmi.vcmp(x, y, mask, "gt")
                cmps = T.vmi.vcmps(x, T.float32(0), mask, "ge")
                sel = T.vmi.vsel(mask, x, y)
                selr = T.vmi.vselr(i, idx)

                vcadd = T.vmi.vcadd(x, mask, group=1, reassoc=False)
                vcmax = T.vmi.vcmax(x, mask, group=1)
                vcmin = T.vmi.vcmin(x, mask, group=1)
                vcvt = T.vmi.vcvt(x, "float16")
                cast = T.vmi.vinterpret_cast(i, "float32")
                expdif = T.vmi.vexpdif(x, y, mask)
                axpy = T.vmi.vaxpy(x, y, T.float32(0.5), mask)
                lrelu = T.vmi.vlrelu(x, T.float32(0.125), mask)
                prelu = T.vmi.vprelu(x, y, mask)

                mul_lo, mul_hi = T.vmi.vmull(i, idx, mask)
                mula = T.vmi.vmula(x, x, y, mask)
                dhist = T.vmi.vdhist(hist_acc, hist_src, hist_mask)
                chist = T.vmi.vchist(hist_acc, hist_src, hist_mask)
                gather = T.vmi.vgather(a_ub[0], idx, mask)
                gatherb = T.vmi.vgatherb(a_ub[0], idx, mask)
                intlv_lo, intlv_hi = T.vmi.vintlv(x, y, mask)
                dintlv_lo, dintlv_hi = T.vmi.vdintlv(x, y, mask)

                T.vmi.vstore(sel, b_ub[0], mask)
                T.vmi.vscatter(add, b_ub[0], idx, mask)
                T.evaluate(sub)
                T.evaluate(mul)
                T.evaluate(div)
                T.evaluate(vmax)
                T.evaluate(vmin)
                T.evaluate(vand)
                T.evaluate(vor)
                T.evaluate(vxor)
                T.evaluate(vshl)
                T.evaluate(vshr)
                T.evaluate(vabs)
                T.evaluate(vneg)
                T.evaluate(vrelu)
                T.evaluate(vexp)
                T.evaluate(vln)
                T.evaluate(vsqrt)
                T.evaluate(vnot)
                T.evaluate(vadds)
                T.evaluate(vmuls)
                T.evaluate(vmaxs)
                T.evaluate(vmins)
                T.evaluate(vshls)
                T.evaluate(vshrs)
                T.evaluate(cmp)
                T.evaluate(cmps)
                T.evaluate(selr)
                T.evaluate(vcadd)
                T.evaluate(vcmax)
                T.evaluate(vcmin)
                T.evaluate(vcvt)
                T.evaluate(cast)
                T.evaluate(expdif)
                T.evaluate(axpy)
                T.evaluate(lrelu)
                T.evaluate(prelu)
                T.evaluate(mul_lo)
                T.evaluate(mul_hi)
                T.evaluate(mula)
                T.evaluate(dhist)
                T.evaluate(chist)
                T.evaluate(gather)
                T.evaluate(gatherb)
                T.evaluate(intlv_lo)
                T.evaluate(intlv_hi)
                T.evaluate(dintlv_lo)
                T.evaluate(dintlv_hi)

    source = lower(func, target="pto").kernel_source
    missing = sorted(op_name for op_name in PTO_VMI_OPAQUE_OPS if f"pto.vmi.{op_name}(" not in source)
    assert not missing, f"missing PTO source generation coverage for {missing}"
    assert "pto.vmi.vmull(" in source
    assert "pto.vmi.vintlv(" in source
    assert "pto.vmi.vdintlv(" in source
    assert "pto.vmi.vcadd(" in source and "reassoc=False" in source
    assert "pto.vmi.vcvt(" not in source or "to_dtype=pto.f16" in source
    assert "pto.vmi.vinterpret_cast(" not in source or "to_dtype=pto.f32" in source


@pytest.mark.pto
def test_pto_codegen_wraps_literal_scalar_sources_by_dtype():
    @T.prim_func
    def func():
        with T.Kernel(1) as _, T.SimdVF():
            brc_f32 = T.vmi.vbrc(T.float32(1), size=64)
            brc_f16 = T.vmi.vbrc(T.float16(1), size=64)
            brc_bf16 = T.vmi.vbrc(T.bfloat16(1), size=64)
            brc_f8 = T.vmi.vbrc(T.float8_e4m3fn(1), size=64)
            brc_i64 = T.vmi.vbrc(T.int64(7), size=64)
            brc_i32 = T.vmi.vbrc(T.int32(7), size=64)
            brc_i16 = T.vmi.vbrc(T.int16(7), size=64)
            brc_i8 = T.vmi.vbrc(T.int8(7), size=64)
            idx_i64 = T.vmi.vci(T.int64(0), size=64)
            idx_i32 = T.vmi.vci(T.int32(0), size=64)
            idx_i16 = T.vmi.vci(T.int16(0), size=64)
            idx_i8 = T.vmi.vci(T.int8(0), size=64)
            T.evaluate(brc_f32)
            T.evaluate(brc_f16)
            T.evaluate(brc_bf16)
            T.evaluate(brc_f8)
            T.evaluate(brc_i64)
            T.evaluate(brc_i32)
            T.evaluate(brc_i16)
            T.evaluate(brc_i8)
            T.evaluate(idx_i64)
            T.evaluate(idx_i32)
            T.evaluate(idx_i16)
            T.evaluate(idx_i8)

    source = lower(func, target="pto").kernel_source
    assert "pto.vmi.vbrc(pto.f32(" in source
    assert "pto.vmi.vbrc(pto.f16(" in source
    assert "pto.vmi.vbrc(pto.bf16(" in source
    assert "pto.vmi.vbrc(pto.f8e4m3(" in source
    assert "pto.vmi.vbrc(pto.si64(7), size=64)" in source
    assert "pto.vmi.vbrc(pto.si32(7), size=64)" in source
    assert "pto.vmi.vbrc(pto.si16(7), size=64)" in source
    assert "pto.vmi.vbrc(pto.si8(7), size=64)" in source
    assert "pto.vmi.vci(pto.si64(0), size=64)" in source
    assert "pto.vmi.vci(pto.si32(0), size=64)" in source
    assert "pto.vmi.vci(pto.si16(0), size=64)" in source
    assert "pto.vmi.vci(pto.si8(0), size=64)" in source


@pytest.mark.pto
def test_pto_codegen_vdup_types_scalar_sources():
    """vdup must make PTOAS scalar/result element types explicit."""

    @T.prim_func
    def func(x: T.int32, f: T.float32):
        with T.Kernel(1) as _, T.SimdVF():
            mask = T.vmi.create_mask(64, size=64)
            # Integer literals lose signedness when printed bare; the target
            # element type must therefore be materialized in the PTO source.
            signed = T.simd.vdup(T.int32(1), "int32", mask)
            unsigned = T.simd.vdup(T.int32(1), "uint32", mask)
            matching = T.simd.vdup(x, "int32", mask)
            dynamic = T.simd.vdup(x, "uint32", mask)
            float_to_int = T.simd.vdup(T.float32(1.5), "int32", mask)
            dynamic_float_to_int = T.simd.vdup(f, "int32", mask)
            T.evaluate(signed)
            T.evaluate(unsigned)
            T.evaluate(matching)
            T.evaluate(dynamic)
            T.evaluate(float_to_int)
            T.evaluate(dynamic_float_to_int)

    source = lower(func, target="pto").kernel_source
    assert "pto.vdup(pto.si32(1)," in source
    assert "pto.vdup(pto.ui32(1)," in source
    assert "pto.vdup(x, mask)" in source
    assert "pto.vdup(scalar.cast(x, pto.ui32)," in source
    assert 'pto.vcvt(pto.vdup(pto.f32(float.fromhex(\'0x1.8p+0\')), mask), pto.si32, mask, rnd="Z", sat="SAT")' in source
    assert 'pto.vcvt(pto.vdup(f, mask), pto.si32, mask, rnd="Z", sat="SAT")' in source


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype", "signed_dtype"),
    [
        ("float32", "int8", "pto.si8"),
        ("float32", "int16", "pto.si16"),
        ("float32", "int32", "pto.si32"),
        ("float32", "int64", "pto.si64"),
    ],
)
@pytest.mark.pto
def test_pto_codegen_uses_signed_integers_for_vcvt(source_dtype, target_dtype, signed_dtype):
    @T.prim_func
    def func(A: T.Buffer((256,), source_dtype)):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((256,), source_dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                source = T.vmi.vload(a_ub[0], size=256)
                converted = T.vmi.vcvt(source, target_dtype)
                T.evaluate(converted)

    source = lower(func, target="pto").kernel_source
    assert "pto.vbitcast(" not in source
    assert f"to_dtype={signed_dtype}" in source
    assert f"to_dtype=pto.i{target_dtype.removeprefix('int')}" not in source


@pytest.mark.pto
def test_vmi_pto_codegen_uses_physical_fp4_storage_units():
    @T.prim_func
    def func(
        A: T.Buffer((256,), "bfloat16"),
        B: T.Buffer((512,), "float4_e2m1fn"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((256,), "bfloat16")
            b_ub = T.alloc_shared((512,), "float4_e2m1fn")
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                bf16 = T.vmi.vload(a_ub[0], size=256)
                fp4 = T.vmi.vcvt(bf16, "float4_e2m1fn")
                T.vmi.vstore(fp4, b_ub[256], mask)
                loaded = T.vmi.vload(b_ub[256], size=256)
                T.evaluate(loaded)

    source = lower(func, target="pto").kernel_source
    vstore_line = next(line for line in source.splitlines() if "pto.vmi.vstore(fp4," in line)
    vload_line = next(line for line in source.splitlines() if "pto.vmi.vload(" in line and "f4e2m1x2" in line)
    assert "mask = pto.vmi.create_mask(128, size=128)" in source
    assert ", 128, " in vstore_line
    assert ", 128, " in vload_line
    assert "size=(256 // 2)" in vload_line


@pytest.mark.pto
def test_vmi_pto_codegen_preserves_physical_fp4_mask_group():
    @T.prim_func
    def func(
        A: T.Buffer((256,), "bfloat16"),
        B: T.Buffer((256,), "float4_e2m1fn"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((256,), "bfloat16")
            b_ub = T.alloc_shared((256,), "float4_e2m1fn")
            with T.SimdVF():
                physical_mask = T.vmi.create_mask(4, size=128, group=4)
                fp4 = T.vmi.vcvt(T.vmi.vload(a_ub[0], size=256), "float4_e2m1fn")
                T.vmi.vstore(fp4, b_ub[0], physical_mask)

    source = lower(func, target="pto").kernel_source
    assert "physical_mask = pto.vmi.create_mask(4, group=4, size=128)" in source


@pytest.mark.pto
def test_pto_codegen_rejects_non_pto_ascend_backend():
    # Use a boolx256 mask so Ascend SimdVF type checks (#353) pass and we still
    # hit the VMI-on-AscendC rejection (boolx64 masks fail earlier on predicates).
    @T.prim_func
    def func():
        with T.Kernel(1), T.SimdVF():
            mask = T.vmi.create_mask(256, size=256)
            T.evaluate(mask)

    with pytest.raises(Exception, match=r"Ascend CCE codegen does not support tl\.vmi\.create_mask"):
        lower(func, target="ascend")


@pytest.mark.pto
def test_simdvf_pto_codegen_still_emits_existing_simd_source():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                x = T.simd.vld(a_ub[0])
                y = T.simd.vadd(x, x)
                T.simd.vsts(b_ub[0], y)
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    compile(source, "<pto-source>", "exec")
    vecscope_lines = [line for line in source.splitlines() if "with pto.vecscope():" in line]
    assert len(vecscope_lines) == 1
    vecscope_indent = _indent(vecscope_lines[0])
    vector_lines = [line for line in source.splitlines() if any(op in line for op in ("pto.vlds(", "pto.vadd(", "pto.vsts("))]
    assert vector_lines
    assert all(_indent(line) > vecscope_indent for line in vector_lines)
    assert "pto.vlds(" in source
    assert "pto.vadd(" in source
    assert "pto.vsts(" in source


@pytest.mark.pto
def test_empty_simdvf_pto_codegen_emits_valid_python():
    @T.prim_func
    def func():
        with T.Kernel(1) as _, T.SimdVF():
            pass

    source = lower(func, target="pto").kernel_source
    compile(source, "<pto-empty-simdvf>", "exec")

    lines = source.splitlines()
    vecscope_lines = [(index, line) for index, line in enumerate(lines) if "with pto.vecscope():" in line]
    assert len(vecscope_lines) == 1
    vecscope_index, vecscope_line = vecscope_lines[0]
    body_lines = [line for line in lines[vecscope_index + 1 :] if line.strip()]
    assert body_lines
    assert body_lines[0].strip() == "pass"
    assert _indent(body_lines[0]) > _indent(vecscope_line)


@pytest.mark.parametrize("op_name", PTO_VMI_OPAQUE_OPS)
@pytest.mark.pto
def test_pto_builtin_effects_are_opaque(op_name):
    effect = T.vmi.tirx.op.Op.get(f"tl.vmi.{op_name}").get_attr("TCallEffectKind")
    assert effect == T.vmi.tirx.CallEffectKind.Opaque


@pytest.mark.pto
def test_pto_pair_get_is_pure():
    effect = T.vmi.tirx.op.Op.get("tl.vmi.pair_get").get_attr("TCallEffectKind")
    assert effect == T.vmi.tirx.CallEffectKind.Pure


@pytest.mark.parametrize(
    "call",
    [
        lambda: T.vmi.vreg(0, T.float32),
        lambda: T.vmi.mask(0),
        lambda: T.vmi.vreg(64, T.float32x2),
    ],
)
@pytest.mark.pto
def test_pto_type_helpers_reject_invalid_input(call):
    with pytest.raises((TypeError, ValueError)):
        call()
