import re

import numpy as np
import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.ascend.language import simd as ascend_simd
from tilelang.engine.lower import lower
from tvm import tirx
from tvm.tirx import Call
from tvm.tirx.stmt import Bind


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
        A: T.Buffer((64,), "float32"),
        B: T.Buffer((64,), "float32"),
        C: T.Buffer((64,), "float32"),
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


def test_ascend_pipe_barrier():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_pipe_barrier("PIPE_ALL")
            T.ascend_pipe_barrier("PIPE_V")
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "asc_sync();" in source
    assert "asc_sync_pipe(PIPE_V);" not in source


def test_ascend_set_wait_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "asc_sync_notify(PIPE_S, PIPE_MTE3, static_cast<event_t>(0));" in source
    assert "asc_sync_wait(PIPE_S, PIPE_MTE3, static_cast<event_t>(0));" in source


def test_ascend_sync_inter_arrive_wait():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32"), flag_id: T.int32):
        with T.Kernel(1) as _:
            T.ascend_sync_inter_arrive("PIPE_FIX", 3)
            T.ascend_sync_inter_wait("PIPE_MTE3", flag_id)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "#include <c_api/asc_simd.h>" not in source
    assert "asc_sync_inter_arrive(PIPE_FIX, 3);" in source
    assert "asc_sync_inter_wait(PIPE_MTE3, flag_id);" in source


def test_ascend_threadfence():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_threadfence()
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "asc_threadfence();" in source


def test_ascend_cross_core_set_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 0x8)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "asc_sync_inter_arrive(PIPE_MTE3, 8)" in source


def test_ascend_cross_core_wait_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 0x8)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "asc_sync_inter_wait(PIPE_MTE3, 8)" in source


def test_ascend_simd_3510_intrinsics_codegen():
    @T.prim_func
    def func(
        C: T.Buffer((128,), "float32"),
        O: T.Buffer((64,), "int32"),
        H: T.Buffer((128,), "float16"),
        S: T.Buffer((128,), "int16"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((128,), "float32")
            c_ub = T.alloc_shared((128,), "float32")
            i_ub = T.alloc_shared((64,), "int32")
            o_ub = T.alloc_shared((64,), "int32")
            h_ub = T.alloc_shared((128,), "float16")
            s_ub = T.alloc_shared((128,), "int16")

            with T.SimdVF():
                full = T.simd.pset(32)
                half = T.simd.pset(16)
                p8 = T.simd.pge(8, "PAT_VL128")
                p16 = T.simd.pge(16, "PAT_VL64")
                p32 = T.simd.pge(32, "PAT_VL16")
                pred = T.simd.pand(p32, full, full)
                pred = T.simd.por(pred, p8, full)
                pred = T.simd.pxor(pred, p16, full)
                pred = T.simd.pnot(pred, full)
                pred = T.simd.psel(pred, full, p32)

                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(a_ub[64])
                a = T.simd.vadd(a, b, pred)
                a = T.simd.vsub(a, T.simd.vdup(T.float32(1), "float32", pred), pred)
                a = T.simd.vmul(a, T.simd.vdup(T.float32(2), "float32", pred), pred)
                a = T.simd.vdiv(a, T.simd.vdup(T.float32(2), "float32", pred), pred)
                a = T.simd.vmax(a, b, pred)
                a = T.simd.vmin(a, b, pred)
                a = T.simd.vexp(a, pred)
                a = T.simd.vln(a, pred)
                a = T.simd.vabs(a, pred)
                a = T.simd.vsqrt(a, pred)
                a = T.simd.vneg(a, pred)
                a = T.simd.vrelu(a, pred)
                a = T.simd.vmaxs(a, T.float32(1), pred)
                a = T.simd.vmins(a, T.float32(3), pred)
                a = T.simd.vmuls(a, T.float32(2), pred)
                a = T.simd.vadds(a, T.float32(1), pred)
                a = T.simd.vsel(
                    a,
                    b,
                    T.simd.pand(
                        T.simd.vcmp(a, b, pred, "ge"),
                        T.simd.vcmps(a, T.float32(0), pred, "gt"),
                        pred,
                    ),
                )
                b = T.simd.vexpdif(a, b, pred)
                b = T.simd.vadd(b, T.simd.vdupv(T.simd.vcmax(a, pred), pred), pred)
                b = T.simd.vadd(b, T.simd.vcpadd(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcadd(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcmin(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcgadd(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcgmax(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcgmin(a, pred), pred)
                a, b = T.simd.vintlv(a, b)
                a, b = T.simd.vdintlv(a, b)
                a = T.simd.vadd(a, b, pred)
                a = T.simd.vadd(
                    a,
                    T.simd.vgatherb(a_ub[0], T.simd.vci(T.uint32(0), "uint32"), pred),
                    pred,
                )
                a = T.simd.vadd(
                    a,
                    T.simd.vgather2(a_ub[0], T.simd.vci(T.uint32(0), "uint32"), pred),
                    pred,
                )
                a = T.simd.vadd(
                    a,
                    T.simd.vcvt(
                        T.simd.vld(h_ub[0], dist="UNPK_B16"),
                        "float32",
                        half,
                        part=0,
                    ),
                    pred,
                )
                b = T.simd.vsqz(a, pred)
                T.simd.vsts(c_ub[0], a, pred)
                T.simd.vsts(c_ub[64], b, pred)
                b = T.simd.vcvt(a, "float16", pred)
                T.simd.vsts(h_ub[0], b, half, dist="PK_B32")
                b = T.simd.vpack(a)
                T.simd.vsts(h_ub[64], b, half, dist="PK_B32")
                T.simd.vsstb(a, c_ub[0], T.int32(1), pred)
                T.simd.vscatter(a, c_ub[0], T.simd.vci(T.uint32(0), "uint32"), pred)
                acc = T.simd.alloc_local(1, "float32")
                acc[0] = a
                T.simd.vaxpy(acc[0], a, T.float32(2), pred)
                a = T.simd.vadd(a, acc[0], pred)
                T.simd.mem_bar("VST_VLD")

                a = T.simd.vld(i_ub[0])
                b = T.simd.vdup(T.int32(1), "int32", pred)
                a = T.simd.vand(a, T.simd.vci(T.int32(0), "int32"), pred)
                a = T.simd.vor(a, b, pred)
                a = T.simd.vxor(a, b, pred)
                a = T.simd.vnot(a, pred)
                a = T.simd.vshl(a, b, pred)
                a = T.simd.vshr(a, b, pred)
                a = T.simd.vselr(a, T.simd.vci(T.uint32(0), "uint32"))
                a = T.simd.vshls(a, T.int32(1), pred)
                a = T.simd.vshrs(a, T.int32(1), pred)
                T.simd.vsts(o_ub[0], a, pred)
                b = T.simd.vpack(a)
                T.simd.vsts(s_ub[0], b, half, dist="NORM_B16")
                a = T.simd.vcadd(b, half)
                T.simd.vsts(o_ub[0], a, pred)

            T.copy(c_ub, C)
            T.copy(o_ub, O)
            T.copy(h_ub, H)
            T.copy(s_ub, S)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "simd_inst::vcpadd(" in source
    assert "simd_inst::vxor(" in source
    assert "simd_inst::vnot(" in source


@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_ascend_simd_vaddc_codegen(dtype):
    @T.prim_func
    def func(
        A: T.Buffer((64,), dtype),
        B: T.Buffer((64,), dtype),
        C: T.Buffer((64,), dtype),
        Carry: T.Buffer((8,), "uint32"),
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
    print(source)
    assert source.count("simd_inst::vaddc(") == 1
    assert ".v0" in source
    assert ".v1" in source


def test_ascend_simd_vdiv_precision_override():
    @T.prim_func
    def func(
        A: T.Buffer((64,), "float32"),
        B: T.Buffer((64,), "float32"),
        C: T.Buffer((192,), "float32"),
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


def test_ascend_simd_sfu_precision_merging():
    """ftz_false in MODE_MERGING selects the precision wrappers."""

    @T.prim_func
    def func(
        A: T.Buffer((64,), "float32"),
        C: T.Buffer((64,), "float32"),
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


def test_ascend_simd_vsstb_threads_pointer_state():
    """POST_UPDATE returns a handle that updates one mutable pointer."""

    @T.prim_func
    def func(A: T.Buffer((256,), "bfloat16"), B: T.Buffer((256,), "bfloat16")):
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
    print(source)
    assert "__ubuf__ void* dst_ptr" in source
    assert source.count("simd_inst::vsstb(") == 2
    assert source.count("POST_UPDATE") == 2


def test_ascend_simd_vld2_dintlv_b16_codegen():
    """Compile-only check that S.vld2(DINTLV_B16) lowers to simd_inst::vld_x2."""

    @T.prim_func
    def func(A: T.Buffer((256,), "bfloat16"), B: T.Buffer((256,), "bfloat16")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((256,), "bfloat16")
            b_ub = T.alloc_shared((256,), "bfloat16")
            T.copy(A, a_ub)
            with T.SimdVF():
                m16 = T.simd.pset(16)
                x0, x1 = T.simd.vld2(a_ub[0], dist="DINTLV_B16")
                T.simd.vsts(b_ub[0], x0, m16, dist="NORM_B16")
                T.simd.vsts(b_ub[128], x1, m16, dist="NORM_B16")
            T.copy(b_ub, B)

    source = lower(func, target="ascend").kernel_source
    print(source)
    assert "simd_inst::vld_x2<" in source
    assert "DINTLV_B16" in source


def test_ascend_simd_vld2_dintlv_b32_codegen():
    """Compile-only check that S.vld2(DINTLV_B32) supports float32."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((128,), "float32")
            b_ub = T.alloc_shared((128,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                m32 = T.simd.pset(32)
                x0, x1 = T.simd.vld2(a_ub[0], dist="DINTLV_B32")
                T.simd.vsts(b_ub[0], x0, m32, dist="NORM_B32")
                T.simd.vsts(b_ub[64], x1, m32, dist="NORM_B32")
            T.copy(b_ub, B)

    source = lower(func, target="ascend").kernel_source
    print(source)
    assert "simd_inst::vld_x2<" in source
    assert "DINTLV_B32" in source


def test_ascend_simd_vld2_dintlv_b8_codegen():
    """Compile-only check that S.vld2(DINTLV_B8) lowers for uint8 / float8."""

    @T.prim_func
    def func_u8(A: T.Buffer((512,), "uint8"), B: T.Buffer((512,), "uint8")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((512,), "uint8")
            b_ub = T.alloc_shared((512,), "uint8")
            T.copy(A, a_ub)
            with T.SimdVF():
                m8 = T.simd.pset(8)
                x0, x1 = T.simd.vld2(a_ub[0], dist="DINTLV_B8")
                T.simd.vsts(b_ub[0], x0, m8, dist="NORM_B8")
                T.simd.vsts(b_ub[256], x1, m8, dist="NORM_B8")
            T.copy(b_ub, B)

    source = lower(func_u8, target="ascend").kernel_source
    print(source)
    assert "simd_inst::vld_x2<" in source
    assert "DINTLV_B8" in source

    @T.prim_func
    def func_fp8(A: T.Buffer((512,), "float8_e4m3"), B: T.Buffer((512,), "float8_e4m3")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((512,), "float8_e4m3")
            b_ub = T.alloc_shared((512,), "float8_e4m3")
            T.copy(A, a_ub)
            with T.SimdVF():
                m8 = T.simd.pset(8)
                x0, x1 = T.simd.vld2(a_ub[0], dist="DINTLV_B8")
                T.simd.vsts(b_ub[0], x0, m8, dist="NORM_B8")
                T.simd.vsts(b_ub[256], x1, m8, dist="NORM_B8")
            T.copy(b_ub, B)

    source_fp8 = lower(func_fp8, target="ascend").kernel_source
    print(source_fp8)
    assert "simd_inst::vld_x2<" in source_fp8
    assert "DINTLV_B8" in source_fp8


def test_ascend_simd_vld2_binds_once_at_site():
    """vld2 must Bind once at the call site; unpack uses .v0/.v1 on that bind.

    Kernel correctness depends on a single memory load, not a CSE/hoist of an
    opaque load into two independent vld_x2 calls.
    """

    @T.prim_func
    def func(A: T.Buffer((512,), "uint8"), B: T.Buffer((512,), "uint8")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((512,), "uint8")
            b_ub = T.alloc_shared((512,), "uint8")
            T.copy(A, a_ub)
            with T.SimdVF():
                m8 = T.simd.pset(8)
                x0, x1 = T.simd.vld2(a_ub[0], dist="DINTLV_B8")
                T.simd.vsts(b_ub[0], x0, m8, dist="NORM_B8")
                T.simd.vsts(b_ub[256], x1, m8, dist="NORM_B8")
            T.copy(b_ub, B)

    vld2_binds = []
    inline_vld2_pair_gets = []

    def visit(node):
        if isinstance(node, Bind) and isinstance(node.value, Call) and _op_name(node.value) == "tl.simd.vld2":
            vld2_binds.append(node)
        if isinstance(node, Call) and _op_name(node) == "tl.simd.pair_get":
            pair_arg = node.args[0]
            if isinstance(pair_arg, Call) and _op_name(pair_arg) == "tl.simd.vld2":
                inline_vld2_pair_gets.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert len(vld2_binds) == 1, f"expected one frontend Bind of vld2, got {len(vld2_binds)}"
    assert not inline_vld2_pair_gets, "pair_get must reference the bound var, not an inline vld2"

    source = lower(func, target="ascend").kernel_source
    print(source)
    assert source.count("simd_inst::vld_x2<") == 1
    assert ".v0" in source and ".v1" in source


def test_ascend_simd_dintlv_part_lane_roundtrip():
    """Numpy model of CCE dintlv→vintlv→PART0..3→store-vintlv must be identity.

    Catches lane-reconstruction bugs that wide bf16 tolerances can mask.
    """

    def vintlv(a: np.ndarray, b: np.ndarray):
        n = a.shape[0]
        half = n // 2
        out0 = np.empty(n, dtype=a.dtype)
        out1 = np.empty(n, dtype=a.dtype)
        out0[0::2] = a[:half]
        out0[1::2] = b[:half]
        out1[0::2] = a[half:]
        out1[1::2] = b[half:]
        return out0, out1

    def part(v: np.ndarray, p: int):
        return v[p::4].copy()

    src = np.arange(512, dtype=np.uint8)
    d0, d1 = src[0::2].copy(), src[1::2].copy()
    d0, d1 = vintlv(d0, d1)

    y00, y01, y02, y03 = (part(d0, p) for p in range(4))
    y10, y11, y12, y13 = (part(d1, p) for p in range(4))

    # Store-side vintlv restore (matches cast_back_asc.dequant_fp8_512).
    y00, y02 = vintlv(y00, y02)
    y01, y03 = vintlv(y01, y03)
    y10, y12 = vintlv(y10, y12)
    y11, y13 = vintlv(y11, y13)
    y00, y01 = vintlv(y00, y01)
    y02, y03 = vintlv(y02, y03)
    y10, y11 = vintlv(y10, y11)
    y12, y13 = vintlv(y12, y13)

    restored = np.concatenate([y00, y01, y02, y03, y10, y11, y12, y13])
    np.testing.assert_array_equal(restored, src)


def test_ascend_simd_vld2_rejects_unsupported_dist():
    """Unsupported vld2 dists must fail at API time."""
    raised = None

    try:

        @T.prim_func
        def func(A: T.Buffer((256,), "bfloat16"), B: T.Buffer((128,), "bfloat16")):
            with T.Kernel(1) as _:
                a_ub = T.alloc_shared((256,), "bfloat16")
                b_ub = T.alloc_shared((128,), "bfloat16")
                with T.SimdVF():
                    m16 = T.simd.pset(16)
                    x0, x1 = T.simd.vld2(a_ub[0], dist="NORM_B16")
                    T.simd.vsts(b_ub[0], x0, m16, dist="NORM_B16")

    except ValueError as e:
        raised = e

    assert raised is not None and "DINTLV" in str(raised), raised


def test_ascend_simd_vpack_u16_to_u8_codegen():
    """Compile-only check that S.vpack(uint16) lowers to simd_inst::vpack<uint8_t>."""

    @T.prim_func
    def func(A: T.Buffer((128,), "uint16"), B: T.Buffer((256,), "uint8")):
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
    print(source)
    assert "simd_inst::vpack<" in source
    assert "uint8_t" in source


def test_ascend_simd_histv2_codegen():
    @T.prim_func
    def func(A: T.Buffer((256,), "uint8"), B: T.Buffer((256,), "uint16")):
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
    print(source)
    assert "simd_inst::dhistv2" in source and "Bin_N0" in source
    assert "simd_inst::chistv2" in source and "Bin_N1" in source


@pytest.mark.parametrize("invalid_bin", [-1, 2, 7])
def test_ascend_simd_histv2_rejects_out_of_range_bins(invalid_bin):
    with pytest.raises(ValueError, match="0 or 1"):
        ascend_simd._histogram_bin(invalid_bin)


@pytest.mark.parametrize("invalid_bin", ["Bin_N0", "Bin_N1", None, 0.0, False])
def test_ascend_simd_histv2_rejects_non_integer_bins(invalid_bin):
    with pytest.raises(TypeError, match="integer 0 or 1"):
        ascend_simd._histogram_bin(invalid_bin)


def test_ascend_simd_vld_e2b_b16_extent():
    """E2B_B16 loads use access_ptr extent=8 (8 dense values → 8×16-lane blocks)."""

    @T.prim_func
    def func(A: T.Buffer((128,), "bfloat16"), B: T.Buffer((128,), "bfloat16")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((128,), "bfloat16")
            b_ub = T.alloc_shared((128,), "bfloat16")
            T.copy(A, a_ub)
            with T.SimdVF():
                m16 = T.simd.pset(16)
                x = T.simd.vld(a_ub[0], dist="E2B_B16")
                T.simd.vsts(b_ub[0], x, m16, dist="NORM_B16")
            T.copy(b_ub, B)

    tir = str(func)
    assert "access_ptr(a_ub[0], 8, 1)" in tir, tir
    assert "E2B_B16" in tir
    source = lower(func, target="ascend").kernel_source
    print(source)
    # Latest ascend lowers E2B_B16 through vlds with a zero hardware offset.
    assert "simd_inst::vlds_brc_elem2datablock<" in source


def test_ascend_simd_vsts_extent_override():
    """Optional vsts(..., extent=N) overrides the default access_ptr footprint."""

    @T.prim_func
    def func(A: T.Buffer((128,), "bfloat16"), B: T.Buffer((128,), "bfloat16")):
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

    tir = str(func)
    # Write mask=2; extent must be the override (8), not default lanes (128).
    assert "access_ptr(b_ub[0], 8, 2)" in tir, tir
    assert "access_ptr(b_ub[0], 128, 2)" not in tir
    source = lower(func, target="ascend").kernel_source
    print(source)
    assert "simd_inst::vsts" in source


def test_copy_pad_value_gm_to_ub():
    # 30 float32 = 120B per row (not 32B-aligned); dst UB is over-allocated to
    # 32 cols so each row can be right-padded up to 128B.
    @T.prim_func
    def func(A: T.Buffer((4, 30), "float32"), B: T.Buffer((4, 32), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((4, 32), "float32")
            T.copy(A[:, :], a_ub[:, :30], pad_value=0.0)
            T.copy(a_ub[:, :], B[:, :])

    source = lower(func, target="ascend").kernel_source
    print(source)
    pad_idx = source.find("asc_set_copy_pad_val")
    copy_idx = source.find("asc_copy_gm2ub_align")
    assert pad_idx != -1 and copy_idx != -1
    # The pad-register write must precede the padded copy.
    assert pad_idx < copy_idx
    # Padded copy: right_padding=2, constant padding enabled, dst_stride=128B.
    assert re.search(
        r"asc_copy_gm2ub_align\([^;]*,\s*4,\s*120,\s*0,\s*2,\s*1,\s*"
        r"static_cast<asc_load_l2_cache_mode>\(0\),\s*120,\s*128\);",
        source,
    ), source


if __name__ == "__main__":
    tilelang.testing.main()
