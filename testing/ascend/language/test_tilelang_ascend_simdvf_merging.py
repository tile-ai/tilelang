"""Predicated SIMD arithmetic preserves inactive lanes and precision choices."""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.engine.lower import lower


BINARY_OPS = ("vadd", "vsub", "vmul", "vabsdif", "vmax", "vmin", "vand", "vor", "vxor", "vshl", "vshr")
SCALAR_OPS = ("vadds", "vmaxs", "vmins", "vmuls", "vshls", "vshrs")
UNARY_OPS = ("vabs", "vneg", "vrelu", "vnot", "vcadd", "vcmax", "vcmin")
SFU_OPS = ("vexp", "vln", "vsqrt")
INTEGER_OPS = ("vand", "vor", "vxor", "vshl", "vshr", "vnot", "vshls", "vshrs")
CASES = [(op, "int32" if op in INTEGER_OPS else "float32") for op in BINARY_OPS + SCALAR_OPS + UNARY_OPS + SFU_OPS]
CASES += [(op, "float32") for op in ("vdiv", "vdup", "vdupv", "vaxpy", "vmula", "vmadd")]
CASES += [(op, "bfloat16") for op in ("vdup", "vdupv", "vnot", "vor", "vxor", "vmuls")]
CASES += [("vxor", "float16"), ("vxor", "float32"), ("vcadd", "int16"), ("vcadd", "uint16")]


def merging_kernel(op_name, dtype, mode="MODE_MERGING", precision=None, pos="POS_LOWEST", scalar_dtype=None):
    bits = tvm.DataType(dtype).bits
    lanes = 2048 // bits
    out_dtype = {"int16": "int32", "uint16": "uint32"}.get(dtype, dtype) if op_name == "vcadd" else dtype
    out_bits = tvm.DataType(out_dtype).bits
    out_lanes = 2048 // out_bits
    pred_dtype = f"uint{bits}"
    scalar_dtype = scalar_dtype or dtype
    op = getattr(T.simd, op_name)

    @T.macro
    def update(dst, a, b, mask):
        if op_name in BINARY_OPS:
            dst[0] = op(a, b, mask, mode=mode)
        elif op_name in SCALAR_OPS:
            dst[0] = op(a, T.cast(1, dtype), mask, mode=mode)
        elif op_name in UNARY_OPS:
            dst[0] = op(a, mask, mode=mode)
        elif op_name in SFU_OPS:
            dst[0] = op(a, mask, mode=mode, precision=precision)
        elif op_name == "vdiv":
            dst[0] = op(a, b, mask, mode=mode, precision=precision)
        elif op_name == "vdup":
            dst[0] = op(T.cast(1, scalar_dtype), dtype, mask, mode=mode)
        elif op_name == "vdupv":
            dst[0] = op(a, mask, pos=pos, mode=mode)
        elif op_name == "vaxpy":
            op(dst[0], a, T.cast(1, dtype), mask, mode=mode)
        else:
            op(dst[0], a, b, mask, mode=mode)

    @T.prim_func
    def kernel(
        A: T.Tensor((lanes,), dtype),
        B: T.Tensor((lanes,), dtype),
        Old: T.Tensor((out_lanes,), out_dtype),
        Mask: T.Tensor((lanes,), pred_dtype),
        Out: T.Tensor((out_lanes,), out_dtype),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((lanes,), dtype)
            b_ub = T.alloc_shared((lanes,), dtype)
            old_ub = T.alloc_shared((out_lanes,), out_dtype)
            mask_ub = T.alloc_shared((lanes,), pred_dtype)
            out_ub = T.alloc_shared((out_lanes,), out_dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(Old, old_ub)
            T.copy(Mask, mask_ub)
            with T.SimdVF():
                full = T.simd.pset(out_bits)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                mask = T.simd.vcmps(T.simd.vld(mask_ub[0]), T.cast(0, pred_dtype), op="gt")
                dst = T.simd.alloc_local((1,), out_dtype)
                dst[0] = T.simd.vld(old_ub[0])
                update(dst, a, b, mask)
                T.simd.vsts(out_ub[0], dst[0], full)
            T.copy(out_ub, Out)

    return kernel


@pytest.mark.parametrize(
    "op_name,precision,wrapper",
    [
        ("vdiv", None, "vdiv_0ulp_ftz_true"),
        ("vdiv", "exact", "vdiv_0ulp_ftz_true"),
        ("vexp", "ftz_false", "vexp_1ulp_ftz_false"),
        ("vln", "ftz_false", "vln_1ulp_ftz_false"),
        ("vsqrt", "ftz_false", "vsqrt_0ulp_ftz_false"),
    ],
)
def test_merging_precision_wrappers(op_name, precision, wrapper):
    source = lower(merging_kernel(op_name, "float32", precision=precision), target="ascend").kernel_source
    assert f"simd_inst::{wrapper}(*" in source


@pytest.mark.pto
@pytest.mark.parametrize(
    "op_name,precision,wrapper",
    [
        ("vdiv", None, "tl.vdiv_precise_f32"),
        ("vdiv", "exact", "tl.vdiv_precise_f32"),
        ("vdiv", "ftz_true", "pto.vdiv"),
        ("vexp", "ftz_false", "tl.vexp_1ulp_ftz_false"),
        ("vln", "ftz_false", "tl.vln_1ulp_ftz_false"),
        ("vsqrt", "ftz_false", "tl.vsqrt_0ulp_ftz_false"),
    ],
)
def test_merging_precision_wrappers_pto(op_name, precision, wrapper):
    source = lower(merging_kernel(op_name, "float32", precision=precision), target="pto").kernel_source
    assert "dst_tl_slot_0 = pto.vsel(" in source
    assert f"{wrapper}(" in source
    assert "MODE_MERGING" not in source


def reference(op_name, a, b, old, mask, pos):
    a32, b32 = a.float(), b.float()
    if op_name in ("vand", "vor", "vxor", "vnot"):
        int_dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32}[a.element_size()]
        x, y = a.view(int_dtype), b.view(int_dtype)
        bits = {"vand": lambda: x & y, "vor": lambda: x | y, "vxor": lambda: x ^ y, "vnot": lambda: ~x}[op_name]()
        active = bits.view(a.dtype)
    elif op_name in ("vcadd", "vcmax", "vcmin"):
        expected = old.clone()
        values = a32[mask]
        if op_name == "vcadd":
            expected[0] = values.sum().to(old.dtype)
        else:
            value = (
                (values.max() if op_name == "vcmax" else values.min())
                if values.numel()
                else torch.tensor(-float("inf") if op_name == "vcmax" else float("inf"))
            )
            expected[0] = value.to(old.dtype)
            index = torch.nonzero(mask & (a32 == value)).flatten()[0].item() if values.numel() else 0
            expected.view(torch.int32 if old.element_size() == 4 else torch.int16)[1] = index
        return expected
    else:
        active = {
            "vadd": lambda: a32 + b32,
            "vsub": lambda: a32 - b32,
            "vmul": lambda: a32 * b32,
            "vabsdif": lambda: (a32 - b32).abs(),
            "vmax": lambda: torch.maximum(a32, b32),
            "vmin": lambda: torch.minimum(a32, b32),
            "vshl": lambda: a.to(torch.int64) << b.to(torch.int64),
            "vshr": lambda: a.to(torch.int64) >> b.to(torch.int64),
            "vadds": lambda: a32 + 1,
            "vmuls": lambda: a32,
            "vmaxs": lambda: a32.clamp(min=1),
            "vmins": lambda: a32.clamp(max=1),
            "vshls": lambda: a.to(torch.int64) << 1,
            "vshrs": lambda: a.to(torch.int64) >> 1,
            "vabs": lambda: a32.abs(),
            "vneg": lambda: -a32,
            "vrelu": lambda: a32.clamp(min=0),
            "vexp": lambda: a32.exp(),
            "vln": lambda: a32.log(),
            "vsqrt": lambda: a32.sqrt(),
            "vdiv": lambda: a32 / b32,
            "vdup": lambda: torch.ones_like(a32),
            # POS selects a fixed source lane; mask controls destination writes.
            "vdupv": lambda: torch.full_like(a32, a32[-1 if pos == "POS_HIGHEST" else 0].item()),
            "vaxpy": lambda: old.float() + a32,
            "vmula": lambda: old.float() + a32 * b32,
            "vmadd": lambda: old.float() * a32 + b32,
        }[op_name]().to(old.dtype)
    expected = old.clone()
    expected[mask] = active[mask]
    return expected


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("op_name,dtype", CASES + [("vdupv", "float16")])
def test_merging_runtime(target, op_name, dtype):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    lanes = 2048 // tvm.DataType(dtype).bits
    td = getattr(torch, dtype)
    out_dtype = {"int16": "int32", "uint16": "uint32"}.get(dtype, dtype) if op_name == "vcadd" else dtype
    out_td = getattr(torch, out_dtype)
    a = (torch.arange(lanes) + 1 if op_name == "vdupv" else torch.arange(lanes) % 8 + 1).to(td)
    b = (torch.arange(lanes) % 3 + 1).to(td)
    old = (torch.arange(2048 // tvm.DataType(out_dtype).bits) % 17 + 7).to(out_td)
    pos = "POS_HIGHEST" if op_name == "vdupv" else "POS_LOWEST"
    kernel = tilelang.compile(merging_kernel(op_name, dtype, precision="ftz_true" if op_name == "vdiv" else None, pos=pos), target=target)
    device_a, device_b, device_old = a.to("npu"), b.to("npu"), old.to("npu")
    for mask in (torch.zeros(lanes, dtype=torch.bool), torch.ones(lanes, dtype=torch.bool), torch.arange(lanes) % 3 == 1):
        expected = reference(op_name, a, b, old, mask, pos)
        actual = torch.empty_like(device_old)
        pred_dtype = getattr(torch, f"uint{tvm.DataType(dtype).bits}")
        kernel(device_a, device_b, device_old, mask.to(pred_dtype).to("npu"), actual)
        actual = actual.cpu()
        if op_name in SFU_OPS or op_name == "vdiv":
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert torch.equal(actual[~mask], old[~mask])
        else:
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (op_name, dtype, actual, expected)


def _reduction_kernel(op_name, dtype="float32"):
    bits = tvm.DataType(dtype).bits
    lanes = 2048 // bits
    pred_dtype = f"uint{bits}"
    op = getattr(T.simd, op_name)

    @T.macro
    def update(dst, src, mask):
        dst[0] = op(src, mask, mode="MODE_MERGING")

    @T.prim_func
    def kernel(
        A: T.Tensor((lanes,), dtype),
        Old: T.Tensor((lanes,), dtype),
        Mask: T.Tensor((lanes,), pred_dtype),
        Out: T.Tensor((lanes,), dtype),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((lanes,), dtype)
            old_ub = T.alloc_shared((lanes,), dtype)
            mask_ub = T.alloc_shared((lanes,), pred_dtype)
            out_ub = T.alloc_shared((lanes,), dtype)
            T.copy(A, a_ub)
            T.copy(Old, old_ub)
            T.copy(Mask, mask_ub)
            with T.SimdVF():
                full = T.simd.pset(bits)
                src = T.simd.vld(a_ub[0])
                mask = T.simd.vcmps(T.simd.vld(mask_ub[0]), T.cast(0, pred_dtype), op="gt")
                dst = T.simd.alloc_local((1,), dtype)
                dst[0] = T.simd.vld(old_ub[0])
                update(dst, src, mask)
                T.simd.vsts(out_ub[0], dst[0], full)
            T.copy(out_ub, Out)

    return kernel


@pytest.mark.parametrize("op_name", ["vcpadd", "vcgadd", "vcgmax", "vcgmin"])
def test_reduction_merging_codegen(op_name):
    source = lower(_reduction_kernel(op_name), target="ascend").kernel_source
    assert f"::{op_name}(*" in source
    assert "MODE_MERGING" in source


@pytest.mark.pto
@pytest.mark.parametrize("op_name", ["vcpadd", "vcgadd", "vcgmax", "vcgmin"])
def test_reduction_merging_codegen_pto(op_name):
    source = lower(_reduction_kernel(op_name), target="pto").kernel_source
    assert f"pto.{op_name}(" in source
    assert "dst_tl_slot_0 = pto.vsel(" in source
    assert ", dst_tl_slot_0, " in source
    assert "MODE_MERGING" not in source
    if op_name == "vcpadd":
        assert 'pto.pset_b32("PAT_VL32")' in source
    else:
        assert 'pto.pset_b32("PAT_VL8")' in source


def _vcvt_kernel(src_dtype, dst_dtype, mode="MODE_MERGING"):
    src_bits = tvm.DataType(src_dtype).bits
    dst_bits = tvm.DataType(dst_dtype).bits
    src_lanes = 2048 // src_bits
    dst_lanes = 2048 // dst_bits
    pred_dtype = f"uint{src_bits}"

    @T.prim_func
    def kernel(
        A: T.Tensor((src_lanes,), src_dtype),
        Old: T.Tensor((dst_lanes,), dst_dtype),
        Mask: T.Tensor((src_lanes,), pred_dtype),
        Out: T.Tensor((dst_lanes,), dst_dtype),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((src_lanes,), src_dtype)
            old_ub = T.alloc_shared((dst_lanes,), dst_dtype)
            mask_ub = T.alloc_shared((src_lanes,), pred_dtype)
            out_ub = T.alloc_shared((dst_lanes,), dst_dtype)
            T.copy(A, a_ub)
            T.copy(Old, old_ub)
            T.copy(Mask, mask_ub)
            with T.SimdVF():
                full = T.simd.pset(dst_bits)
                src = T.simd.vld(a_ub[0])
                mask = T.simd.vcmps(T.simd.vld(mask_ub[0]), T.cast(0, pred_dtype), op="gt")
                dst = T.simd.alloc_local((1,), dst_dtype)
                dst[0] = T.simd.vld(old_ub[0])
                dst[0] = T.simd.vcvt(src, dst_dtype, mask, mode=mode)
                T.simd.vsts(out_ub[0], dst[0], full)
            T.copy(out_ub, Out)

    return kernel


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        ("int32", "float32"),
        ("float32", "int32"),
        ("float32", "float16"),
        ("float16", "float32"),
    ],
)
def test_vcvt_merging_codegen(src_dtype, dst_dtype):
    source = lower(_vcvt_kernel(src_dtype, dst_dtype), target="ascend").kernel_source
    assert "::vcvt(*" in source
    assert "MODE_MERGING" in source


@pytest.mark.pto
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        ("int32", "float32"),
        ("float32", "int32"),
        ("float32", "float16"),
        ("float16", "float32"),
    ],
)
def test_vcvt_merging_codegen_pto(src_dtype, dst_dtype):
    source = lower(_vcvt_kernel(src_dtype, dst_dtype), target="pto").kernel_source
    assert "pto.vcvt(" in source
    assert "dst_tl_slot_0 = pto.vsel(" in source
    assert ", dst_tl_slot_0, " in source
    dst_bits = tvm.DataType(dst_dtype).bits
    assert f'pto.mask_type("b{dst_bits}")' in source
    assert "MODE_MERGING" not in source


@pytest.mark.pto
@pytest.mark.parametrize("op_name", ["vcgadd", "vcgmax", "vcgmin"])
def test_pto_group_reduction_merging_runtime(op_name):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    lanes = 64
    a = (torch.arange(lanes) % 8 + 1).to(torch.float32)
    old = (torch.arange(lanes) % 17 + 7).to(torch.float32)
    kernel = tilelang.compile(_reduction_kernel(op_name), target="pto")
    for mask in (
        torch.zeros(lanes, dtype=torch.bool),
        torch.ones(lanes, dtype=torch.bool),
        torch.arange(lanes) % 3 == 1,
    ):
        expected = old.clone()
        for j in range(8):
            group = a[j * 8 : (j + 1) * 8][mask[j * 8 : (j + 1) * 8]]
            if group.numel():
                value = {"vcgadd": group.sum, "vcgmax": group.max, "vcgmin": group.min}[op_name]()
            else:
                value = torch.tensor({"vcgadd": 0.0, "vcgmax": -float("inf"), "vcgmin": float("inf")}[op_name])
            expected[j] = value
        actual = torch.empty_like(old, device="npu")
        kernel(a.to("npu"), old.to("npu"), mask.to(torch.uint32).to("npu"), actual)
        torch.npu.synchronize()
        assert torch.equal(actual.cpu(), expected), (op_name, int(mask.sum()))


@pytest.mark.pto
def test_pto_vcpadd_merging_runtime():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    lanes = 64
    a = (torch.arange(lanes) % 8 + 1).to(torch.float32)
    old = (torch.arange(lanes) % 17 + 7).to(torch.float32)
    kernel = tilelang.compile(_reduction_kernel("vcpadd"), target="pto")
    for mask in (
        torch.zeros(lanes, dtype=torch.bool),
        torch.ones(lanes, dtype=torch.bool),
        torch.arange(lanes) % 3 == 1,
    ):
        expected = old.clone()
        for i in range(lanes // 2):
            pair = a[2 * i : 2 * i + 2][mask[2 * i : 2 * i + 2]]
            expected[i] = pair.sum()
        actual = torch.empty_like(old, device="npu")
        kernel(a.to("npu"), old.to("npu"), mask.to(torch.uint32).to("npu"), actual)
        torch.npu.synchronize()
        assert torch.equal(actual.cpu(), expected), int(mask.sum())


@pytest.mark.pto
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        ("int32", "float32"),
        ("float32", "int32"),
        ("float32", "float16"),
        ("float16", "float32"),
    ],
)
def test_pto_vcvt_merging_runtime(src_dtype, dst_dtype):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    src_bits = tvm.DataType(src_dtype).bits
    dst_bits = tvm.DataType(dst_dtype).bits
    src_lanes = 2048 // src_bits
    dst_lanes = 2048 // dst_bits
    pred_dtype = getattr(torch, f"uint{src_bits}")
    if src_dtype == "float32":
        a = torch.arange(src_lanes, dtype=torch.float32) * 0.5 - 8
    elif src_dtype == "float16":
        a = (torch.arange(src_lanes, dtype=torch.float32) - 16).to(torch.float16)
    else:
        a = torch.arange(src_lanes, dtype=torch.int32) - 16
    old = (torch.arange(dst_lanes, dtype=torch.float32) % 17 + 7).to(getattr(torch, dst_dtype))
    kernels = {
        mode: tilelang.compile(_vcvt_kernel(src_dtype, dst_dtype, mode=mode), target="pto") for mode in ("MODE_ZEROING", "MODE_MERGING")
    }
    for mask in (
        torch.zeros(src_lanes, dtype=torch.bool),
        torch.ones(src_lanes, dtype=torch.bool),
        torch.arange(src_lanes) % 3 == 1,
    ):
        outs = {}
        for mode, kernel in kernels.items():
            actual = torch.empty(dst_lanes, dtype=getattr(torch, dst_dtype), device="npu")
            kernel(a.to("npu"), old.to("npu"), mask.to(pred_dtype).to("npu"), actual)
            torch.npu.synchronize()
            outs[mode] = actual.cpu()
        if src_bits == dst_bits:
            sel = mask
        elif dst_bits < src_bits:
            sel = torch.zeros(dst_lanes, dtype=torch.bool)
            sel[0::2] = mask
        else:
            sel = mask[0::2]
        expected = torch.where(sel, outs["MODE_ZEROING"], old)
        assert torch.equal(outs["MODE_MERGING"], expected), (src_dtype, dst_dtype, int(mask.sum()))


def _vcvt_low_precision_kernel(src_dtype, dst_dtype, mode):
    """bf16->FP4 / f32->FP8 conversions, masked at destination b8 granularity."""
    src_bits = tvm.DataType(src_dtype).bits
    src_lanes = 2048 // src_bits
    dst_bits = tvm.DataType(dst_dtype).bits
    dst_lanes = 2048 // dst_bits
    mask_lanes = 2048 // 8

    @T.prim_func
    def kernel(
        A: T.Tensor((src_lanes,), src_dtype),
        Old: T.Tensor((dst_lanes,), dst_dtype),
        Mask: T.Tensor((mask_lanes,), "uint8"),
        Out: T.Tensor((dst_lanes,), dst_dtype),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((src_lanes,), src_dtype)
            old_ub = T.alloc_shared((dst_lanes,), dst_dtype)
            mask_ub = T.alloc_shared((mask_lanes,), "uint8")
            out_ub = T.alloc_shared((dst_lanes,), dst_dtype)
            T.copy(A, a_ub)
            T.copy(Old, old_ub)
            T.copy(Mask, mask_ub)
            with T.SimdVF():
                full = T.simd.pset(8)
                src = T.simd.vld(a_ub[0])
                mask = T.simd.vcmps(T.simd.vld(mask_ub[0]), T.cast(0, "uint8"), op="gt")
                dst = T.simd.alloc_local((1,), dst_dtype)
                dst[0] = T.simd.vld(old_ub[0])
                dst[0] = T.simd.vcvt(src, dst_dtype, mask, mode=mode)
                T.simd.vsts(out_ub[0], dst[0], full)
            T.copy(out_ub, Out)

    return kernel


@pytest.mark.pto
@pytest.mark.parametrize(
    "src_dtype,dst_dtype,torch_dst",
    [
        ("float32", "float8_e4m3", torch.float8_e4m3fn),
        ("bfloat16", "float4_e2m1fn", torch.int8),
    ],
)
def test_pto_vcvt_low_precision_merging_runtime(src_dtype, dst_dtype, torch_dst):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    src_bits = tvm.DataType(src_dtype).bits
    src_lanes = 2048 // src_bits
    mask_lanes = 2048 // 8
    if src_dtype == "float32":
        a = torch.randn(src_lanes, dtype=torch.float32) * 2
    else:
        a = (torch.arange(src_lanes, dtype=torch.float32) % 16 - 8).to(torch.bfloat16)
    old = (torch.arange(mask_lanes, dtype=torch.int32) % 123 - 61).to(torch.int8)
    kernels = {
        mode: tilelang.compile(_vcvt_low_precision_kernel(src_dtype, dst_dtype, mode=mode), target="pto")
        for mode in ("MODE_ZEROING", "MODE_MERGING")
    }
    for mask in (
        torch.zeros(mask_lanes, dtype=torch.bool),
        torch.ones(mask_lanes, dtype=torch.bool),
        torch.arange(mask_lanes) % 3 == 1,
    ):
        outs = {}
        for mode, kernel in kernels.items():
            actual = torch.empty(mask_lanes, dtype=torch.int8, device="npu")
            if torch_dst != torch.int8:
                actual = actual.view(torch_dst)
            kernel(a.to("npu"), old.view(torch_dst).to("npu"), mask.to(torch.uint8).to("npu"), actual)
            torch.npu.synchronize()
            outs[mode] = actual.cpu().view(torch.int8)
        expected = torch.where(mask, outs["MODE_ZEROING"], old)
        assert torch.equal(outs["MODE_MERGING"], expected), (src_dtype, dst_dtype, int(mask.sum()))
