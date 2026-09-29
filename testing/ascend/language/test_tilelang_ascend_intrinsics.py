"""Ascend SIMD API contracts and instruction emission."""

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.ascend.language import simd as ascend_simd
from tilelang.engine.lower import lower
from tvm import tirx
from tvm.script.ir_builder import IRBuilder
from tvm.tirx import Call
from testing.ascend._ir import calls


def _op_name(call_or_op):
    op = getattr(call_or_op, "op", call_or_op)
    return getattr(op, "name", None)


def test_ascend_simd_pair_supports_heterogeneous_dtypes():
    carrier = tirx.Var("pair", "int32x64")
    pair = ascend_simd.SimdPair(carrier, ("boolx256", "int32x64"))

    assert tuple(map(str, pair.dtype)) == ("boolx256", "int32x64")

    carry, result = pair
    assert str(carry.dtype) == "boolx256"
    assert str(result.dtype) == "int32x64"
    assert _op_name(carry) == "tl.simd.pair_get"
    assert _op_name(result) == "tl.simd.pair_get"
    assert carry.args[0].same_as(carrier)
    assert result.args[0].same_as(carrier)
    assert int(carry.args[1]) == 0
    assert int(result.args[1]) == 1

    homogeneous = ascend_simd.SimdPair(carrier)
    assert tuple(map(str, homogeneous.dtype)) == ("int32x64", "int32x64")


def test_ascend_simd_pair_validates_dtypes_and_index():
    carrier = tirx.Var("pair", "int32x64")

    with pytest.raises(TypeError, match="tuple or list"):
        ascend_simd.SimdPair(carrier, "int32x64")
    with pytest.raises(ValueError, match="exactly two"):
        ascend_simd.SimdPair(carrier, ("int32x64",))

    pair = ascend_simd.SimdPair(carrier, ("boolx256", "int32x64"))
    with pytest.raises(TypeError, match="integer 0 or 1"):
        ascend_simd.pair_get(pair, tirx.Var("index", "int32"))
    with pytest.raises(IndexError, match="0 or 1"):
        ascend_simd.pair_get(pair, 2)


def test_ascend_simd_vexpdif_rejects_widening_form():
    src0 = tirx.Var("src0", "float16x128")
    src1 = tirx.Var("src1", "float16x128")
    mask = tirx.Var("mask", "boolx256")

    with pytest.raises(TypeError, match="requires matching float32 vectors"):
        ascend_simd.vexpdif(src0, src1, mask)


def test_ascend_simd_vexpdif_codegen():
    @T.prim_func
    def func(
        A: T.Tensor((64,), "float32"),
        B: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                src0 = T.simd.vld(a_ub[0])
                src1 = T.simd.vld(b_ub[0])
                result = T.simd.vexpdif(src0, src1, mask)
                T.simd.vsts(c_ub[0], result, mask)
            T.copy(c_ub, C)

    source = lower(func, target="ascend").kernel_source
    assert "simd_inst::vexpdif(" in source
    assert "simd_inst::vexpdif<" not in source


def test_manual_sync_intrinsic_codegen():
    @T.prim_func
    def kernel(A: T.Tensor((1,), "float32"), flag: T.int32):
        with T.Kernel(1):
            T.ascend_pipe_barrier("PIPE_ALL")
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            T.ascend_sync_inter_arrive("PIPE_FIX", 3)
            T.ascend_sync_inter_wait("PIPE_MTE3", flag)
            T.ascend_threadfence()
            T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 8)
            T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 8)
            A[0] = T.float32(1)

    source = lower(kernel, target="ascend").kernel_source
    for instruction in (
        "asc_sync();",
        "asc_threadfence();",
        "asc_sync_notify(PIPE_S, PIPE_MTE3, static_cast<event_t>(0));",
        "asc_sync_wait(PIPE_S, PIPE_MTE3, static_cast<event_t>(0));",
        "asc_sync_inter_arrive(PIPE_FIX, 3);",
        "asc_sync_inter_wait(PIPE_MTE3, flag);",
        "asc_sync_inter_arrive(PIPE_MTE3, 8);",
        "asc_sync_inter_wait(PIPE_MTE3, 8);",
    ):
        assert instruction in source


def test_ascend_simd_mem_bar_pto_codegen():
    @T.prim_func
    def func():
        with T.Kernel(1) as _, T.SimdVF():
            T.simd.mem_bar("VST_VLD")

    source = lower(func, target="pto").kernel_source
    assert 'pto.mem_bar("VST_VLD")' in source


@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_ascend_simd_vaddc_codegen(dtype):
    @T.prim_func
    def func(
        A: T.Tensor((64,), dtype),
        B: T.Tensor((64,), dtype),
        C: T.Tensor((64,), dtype),
        Carry: T.Tensor((8,), "uint32"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), dtype)
            b_ub = T.alloc_shared((64,), dtype)
            c_ub = T.alloc_shared((64,), dtype)
            carry_ub = T.alloc_shared((8,), "uint32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                src0 = T.simd.vld(a_ub[0])
                src1 = T.simd.vld(b_ub[0])
                carry, result = T.simd.vaddc(src0, src1)
                T.simd.vsts(c_ub[0], result, dist="NORM_B32")
                T.simd.pst(carry_ub[0], carry)
            T.copy(c_ub, C)
            T.copy(carry_ub, Carry)

    pair_get_dtypes = []

    def collect_pair_get(node):
        if isinstance(node, Call) and _op_name(node) == "tl.simd.pair_get":
            pair_get_dtypes.append(str(node.dtype))

    tirx.stmt_functor.post_order_visit(func.body, collect_pair_get)
    assert sorted(pair_get_dtypes) == ["boolx256", f"{dtype}x64"]

    source = lower(func, target="ascend").kernel_source
    assert source.count("simd_inst::vaddc(") == 1
    assert ".v0" in source
    assert ".v1" in source


def test_ascend_simd_vdiv_precision_override():
    @T.prim_func
    def func(
        A: T.Tensor((64,), "float32"),
        B: T.Tensor((64,), "float32"),
        C: T.Tensor((192,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((192,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                T.simd.vsts(c_ub[0], T.simd.vdiv(a, b, mask, precision="exact"), mask)
                T.simd.vsts(c_ub[64], T.simd.vdiv(a, b, mask), mask)
                T.simd.vsts(c_ub[128], T.simd.vdiv(a, b, mask, precision="ftz_true"), mask)
            T.copy(c_ub, C)

    config_key = tilelang.PassConfigKey.TL_ENABLE_FAST_MATH.value
    with tilelang.transform.PassContext(config={config_key: False}):
        precise_default_source = lower(func, target="ascend").kernel_source
    with tilelang.transform.PassContext(config={config_key: True}):
        fast_default_source = lower(func, target="ascend").kernel_source

    assert precise_default_source.count("simd_inst::vdiv_0ulp_ftz_true(") == 2
    assert precise_default_source.count("simd_inst::vdiv(") == 1
    assert fast_default_source.count("simd_inst::vdiv_0ulp_ftz_true(") == 1
    assert fast_default_source.count("simd_inst::vdiv(") == 2

    with tilelang.transform.PassContext(config={config_key: False}):
        pto_precise_default_source = lower(func, target="pto").kernel_source
    with tilelang.transform.PassContext(config={config_key: True}):
        pto_fast_default_source = lower(func, target="pto").kernel_source

    assert pto_precise_default_source.count("tl.vdiv_precise_f32(") == 2
    assert pto_precise_default_source.count("pto.vdiv(") == 1
    assert pto_fast_default_source.count("tl.vdiv_precise_f32(") == 1
    assert pto_fast_default_source.count("pto.vdiv(") == 2


def test_ascend_simd_sfu_precision_merging():
    """ftz_false in MODE_MERGING selects the precision wrappers."""

    @T.prim_func
    def func(
        A: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.simd.pset(32, "PAT_VL8")
                full = T.simd.pset(32)
                src = T.simd.vld(a_ub[0])
                dst = T.simd.alloc_local((1,), "float32")
                dst[0] = src
                dst[0] = T.simd.vexp(src, mask, mode="MODE_MERGING", precision="ftz_false")
                dst[0] = T.simd.vln(src, mask, mode="MODE_MERGING", precision="ftz_false")
                dst[0] = T.simd.vsqrt(src, mask, mode="MODE_MERGING", precision="ftz_false")
                T.simd.vsts(c_ub[0], dst[0], full)
            T.copy(c_ub, C)

    source = lower(func, target="ascend").kernel_source
    assert "simd_inst::vexp_1ulp_ftz_false(" in source
    assert "simd_inst::vln_1ulp_ftz_false(" in source
    assert "simd_inst::vsqrt_0ulp_ftz_false(" in source
    assert "::vexp(" not in source
    assert "::vln(" not in source
    assert "::vsqrt(" not in source

    # PTO lowers merging to a zeroing SFU call wrapped in pto.vsel; the
    # ftz_false precision tier selects the subnormal-preserving helpers.
    pto_source = lower(func, target="pto").kernel_source
    assert "pto.vsel(" in pto_source
    for wrapper in ("tl.vexp_1ulp_ftz_false", "tl.vln_1ulp_ftz_false", "tl.vsqrt_0ulp_ftz_false"):
        assert f"{wrapper}(" in pto_source
    assert "MODE_MERGING" not in pto_source


def test_pto_simd_sfu_precision_source():
    @T.prim_func
    def func(
        A: T.Buffer((64,), "float32"),
        C: T.Buffer((320,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((320,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                src = T.simd.vld(a_ub[0])
                T.simd.vsts(c_ub[0], ascend_simd.vexp(src, full, precision="ftz_false"), full)
                T.simd.vsts(c_ub[64], ascend_simd.vexp(src, full, precision="ftz_true"), full)
                T.simd.vsts(c_ub[128], ascend_simd.vexp(src, full), full)
                T.simd.vsts(c_ub[192], ascend_simd.vln(src, full, precision="ftz_false"), full)
                T.simd.vsts(c_ub[256], ascend_simd.vsqrt(src, full, precision="ftz_false"), full)
            T.copy(c_ub, C)

    config_key = tilelang.PassConfigKey.TL_ENABLE_FAST_MATH.value
    for fast_math in (False, True):
        with tilelang.transform.PassContext(config={config_key: fast_math}):
            source = lower(func, target="pto").kernel_source
        assert source.count("tl.vexp_1ulp_ftz_false(") == 1
        assert source.count("pto.vexp(") == 2
        assert source.count("tl.vln_1ulp_ftz_false(") == 1
        assert source.count("tl.vsqrt_0ulp_ftz_false(") == 1


def test_ascend_simd_vsstb_threads_pointer_state():
    """POST_UPDATE returns a handle that updates one mutable pointer."""

    @T.prim_func
    def func(A: T.Tensor((256,), "bfloat16"), B: T.Tensor((256,), "bfloat16")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((256,), "bfloat16")
            b_ub = T.alloc_shared((256,), "bfloat16")
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.simd.pset(16)
                src = T.simd.vld(a_ub[0])
                T.simd.vsts(b_ub[0], src, mask, dist="ONEPT_B32")
                dst_ptr = T.simd.make_ubuf_ptr(b_ub[0], "bfloat16")
                dst_ptr = T.simd.vsstb(src, dst_ptr, T.int32((3 << 16) | 1), mask, update=True)
                dst_ptr = T.simd.vsstb(src, dst_ptr, T.int32((3 << 16) | 1), mask, update=True)
                T.simd.mem_bar("VST_VLD")
            T.copy(b_ub, B)

    pointer_stores = []

    def visit(node):
        if isinstance(node, tirx.BufferStore) and node.buffer.scope() == "local.var" and node.buffer.dtype == "handle":
            pointer_stores.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert len(pointer_stores) == 3
    pointer = pointer_stores[0].buffer
    for store in pointer_stores:
        assert store.buffer.same_as(pointer)

    for store in pointer_stores[1:]:
        update = store.value
        assert isinstance(update, Call)
        assert _op_name(update) == "tl.simd.vsstb"
        assert update.dtype == "handle"
        assert update.args[1].dtype == "handle"
        assert str(update.args[4]) == '"POST_UPDATE"'

    source = lower(func, target="ascend").kernel_source
    assert source.count("simd_inst::vsstb(") == 2
    assert source.count("POST_UPDATE") == 2

    pto_source = lower(func, target="pto").kernel_source
    assert "dst_ptr = None" in pto_source
    assert pto_source.count("pto.vsstb(") == 2
    assert "dst_ptr, 3, 1, " in pto_source
    assert 'dist="1PT_B32"' in pto_source
    assert "ONEPT_B32" not in pto_source
    assert "_tl_coerce_i64(pto.addptr(" not in pto_source
    assert "_tl_coerce_i64(pto.vsstb(" not in pto_source
    assert 'pto.mem_bar("VST_VLD")' in pto_source


@pytest.mark.parametrize("dtype,bits", [("bfloat16", 16), ("float32", 32), ("uint8", 8), ("float8_e4m3", 8)])
def test_vld2_loads_once_and_exposes_both_vectors(dtype, bits):
    lanes = 2048 // bits
    distribution = f"DINTLV_B{bits}"
    store_distribution = f"NORM_B{bits}"

    @T.prim_func
    def kernel(A: T.Tensor((lanes * 2,), dtype), B: T.Tensor((lanes * 2,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((lanes * 2,), dtype)
            dst = T.alloc_shared((lanes * 2,), dtype)
            T.copy(A, src)
            with T.SimdVF():
                mask = T.simd.pset(bits)
                first, second = T.simd.vld2(src[0], dist=distribution)
                T.simd.vsts(dst[0], first, mask, dist=store_distribution)
                T.simd.vsts(dst[lanes], second, mask, dist=store_distribution)
            T.copy(dst, B)

    source = lower(kernel, target="ascend").kernel_source
    assert source.count("simd_inst::vld_x2<") == 1
    assert distribution in source
    assert ".v0" in source and ".v1" in source


def test_vld_postupdate_rejects_invalid_pointer_and_increment():
    buf = tirx.decl_buffer((128,), "uint16", scope="shared")
    with pytest.raises(ValueError, match="mutable pointer"):
        ascend_simd.vld(buf[0], post_inc=128)
    untyped = tirx.decl_buffer((1,), "handle", scope="local.var")[0]
    with pytest.raises(ValueError, match="declared with make_ubuf_ptr"):
        ascend_simd.vld(untyped, post_inc=128)
    with IRBuilder(), T.sblock("root"):
        ptr = ascend_simd.make_ubuf_ptr(buf[0], "uint16")[0]
        with pytest.raises(ValueError, match="must match dtype"):
            ascend_simd.vld(ptr, "BRC_B8", post_inc=1)
        with pytest.raises(TypeError, match="int32 element increment"):
            ascend_simd.vld(ptr, post_inc=True)
        with pytest.raises(ValueError, match="fit int32"):
            ascend_simd.vld(ptr, post_inc=1 << 32)
        wide_ptr = ascend_simd.make_ubuf_ptr(buf[0], "uint64")[0]
        with pytest.raises(ValueError, match="8/16/32-bit pointer"):
            ascend_simd.vld(wide_ptr, post_inc=1)
        with pytest.raises(ValueError, match="source element dtype must match"):
            ascend_simd.vsstb(tirx.Var("src", "float32x64"), ptr, 1, tirx.Var("mask", "boolx256"), update=True)


@pytest.mark.parametrize("dist", ["US_B32", "DS_B32", "E2B", "E2B_B8", "UNPK4_B16", "UNKNOWN_B16", None])
def test_vld_postupdate_rejects_unknown_distribution(dist):
    with IRBuilder(), T.sblock("root"):
        buf = T.alloc_shared((128,), "uint16")
        ptr = ascend_simd.make_ubuf_ptr(buf[0], "uint16")[0]
        with pytest.raises(ValueError, match="Unsupported vld post-update distribution"):
            ascend_simd.vld(ptr, dist, post_inc=1)


def test_vld2_rejects_a_single_vector_distribution():
    src = tirx.decl_buffer((256,), "bfloat16", scope="shared.dyn")
    with pytest.raises(ValueError, match="DINTLV"):
        T.simd.vld2(src[0], dist="NORM_B16")


def test_ascend_simd_vpack_u16_to_u8_codegen():
    """Compile-only check that S.vpack(uint16) lowers to simd_inst::vpack<uint8_t>."""

    @T.prim_func
    def func(A: T.Tensor((128,), "uint16"), B: T.Tensor((256,), "uint8")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((128,), "uint16")
            b_ub = T.alloc_shared((256,), "uint8")
            T.copy(A, a_ub)
            with T.SimdVF():
                m16 = T.simd.pset(16)
                a = T.simd.vld(a_ub[0], dist="NORM_B16")
                packed = T.simd.vpack(a, 0)
                T.simd.vsts(b_ub[0], packed, m16, dist="NORM_B8")
            T.copy(b_ub, B)

    source = lower(func, target="ascend").kernel_source
    assert "simd_inst::vpack<" in source
    assert "uint8_t" in source


def test_ascend_simd_histv2_codegen():
    @T.prim_func
    def func(A: T.Tensor((256,), "uint8"), B: T.Tensor((256,), "uint16")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((256,), "uint8")
            b_ub = T.alloc_shared((256,), "uint16")
            T.copy(A, a_ub)
            with T.SimdVF():
                m8 = T.simd.pset(8)
                m16 = T.simd.pset(16)
                src = T.simd.vld(a_ub[0], dist="NORM")
                frequency = T.simd.alloc_var("uint16")
                cumulative = T.simd.alloc_var("uint16")
                frequency = T.simd.vdup(T.uint16(0), "uint16", m16)
                cumulative = T.simd.vdup(T.uint16(0), "uint16", m16)
                T.simd.dhistv2(frequency, src, m8, bin=0)
                T.simd.chistv2(cumulative, src, m8, bin=1)
                T.simd.vsts(b_ub[0], frequency, m16, dist="NORM_B16")
                T.simd.vsts(b_ub[128], cumulative, m16, dist="NORM_B16")
            T.copy(b_ub, B)

    source = lower(func, target="ascend").kernel_source
    assert "simd_inst::dhistv2" in source and "Bin_N0" in source
    assert "simd_inst::chistv2" in source and "Bin_N1" in source


@pytest.mark.parametrize(
    "value,error", [(-1, ValueError), (2, ValueError), ("Bin_N0", TypeError), (None, TypeError), (0.0, TypeError), (False, TypeError)]
)
@pytest.mark.parametrize("operation", [T.simd.dhistv2, T.simd.chistv2])
def test_histogram_rejects_invalid_bin(operation, value, error):
    src = tirx.Var("src", "uint8x256")
    dst = tirx.decl_buffer((1,), "uint16x128", name="dst", scope="local.var")[0]
    mask = tirx.Var("mask", "boolx256")
    with pytest.raises(error, match="0 or 1"):
        operation(dst, src, mask, bin=value)


def test_ascend_simd_vld_e2b_b16_extent():
    """E2B_B16 loads use access_ptr extent=8 (8 dense values → 8×16-lane blocks)."""

    @T.prim_func
    def func(A: T.Tensor((128,), "bfloat16"), B: T.Tensor((128,), "bfloat16")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((128,), "bfloat16")
            b_ub = T.alloc_shared((128,), "bfloat16")
            T.copy(A, a_ub)
            with T.SimdVF():
                m16 = T.simd.pset(16)
                x = T.simd.vld(a_ub[0], dist="E2B_B16")
                T.simd.vsts(b_ub[0], x, m16, dist="NORM_B16")
            T.copy(b_ub, B)

    (load,) = calls(func, "tl.simd.vld")
    assert int(load.args[0].args[1]) == 8
    source = lower(func, target="ascend").kernel_source
    # Latest ascend lowers E2B_B16 through vlds with a zero hardware offset.
    assert "simd_inst::vlds_brc_elem2datablock<" in source


def test_ascend_simd_vsts_extent_override():
    """Optional vsts(..., extent=N) overrides the default access_ptr footprint."""

    @T.prim_func
    def func(A: T.Tensor((128,), "bfloat16"), B: T.Tensor((128,), "bfloat16")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((128,), "bfloat16")
            b_ub = T.alloc_shared((128,), "bfloat16")
            T.copy(A, a_ub)
            with T.SimdVF():
                m8 = T.simd.pset(8)
                x = T.simd.vld(a_ub[0], dist="NORM_B16")
                # Dense PAT_VL8 recip pack: write only 8 elements, not full 128.
                T.simd.vsts(b_ub[0], x, m8, dist="NORM_B16", extent=8)
            T.copy(b_ub, B)

    (store,) = calls(func, "tl.simd.vsts")
    assert int(store.args[0].args[1]) == 8
    assert int(store.args[0].args[2]) == 2
