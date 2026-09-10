"""Native merging codegen, dtype fallbacks, and inactive-lane preservation."""

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


@pytest.mark.parametrize("op_name,dtype", CASES)
def test_merging_codegen(op_name, dtype):
    source = lower(merging_kernel(op_name, dtype, precision="ftz_true" if op_name == "vdiv" else None), target="ascend").kernel_source
    if op_name == "vdup" and dtype == "bfloat16":
        assert "simd_inst::vdup(*" in source
    else:
        native_name = "vdup" if op_name == "vdupv" else op_name
        assert f"::{native_name}(*" in source
        assert f"simd_inst::{native_name}(*" not in source
        assert f"simd_inst::{op_name}(*" not in source


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


@pytest.mark.parametrize("op_name,dtype", [("vadd", "int64"), ("vdupv", "float8_e4m3fn"), ("vneg", "bfloat16")])
def test_merging_keeps_unvalidated_types_on_wrappers(op_name, dtype):
    # Codegen only: these cases ensure this change does not switch additional
    # overloads to CCE. They do not assert that the old wrappers support them.
    source = lower(merging_kernel(op_name, dtype), target="ascend").kernel_source
    assert f"simd_inst::{op_name}(*" in source


def test_merging_vdup_fallback_uses_destination_dtype():
    source = lower(merging_kernel("vdup", "bfloat16", scalar_dtype="float32"), target="ascend").kernel_source
    assert "simd_inst::vdup(*" in source


@pytest.mark.parametrize("op_name", ["vadd", "vdup", "vdupv", "vcmax", "vaxpy"])
def test_zeroing_keeps_capi_wrappers(op_name):
    source = lower(merging_kernel(op_name, "float32", mode="MODE_ZEROING"), target="ascend").kernel_source
    wrapper = "vdup<float>" if op_name == "vdup" else op_name
    assert f"simd_inst::{wrapper}(" in source


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


@pytest.mark.parametrize("op_name,dtype", CASES + [("vdupv", "float16")])
def test_merging_runtime(op_name, dtype):
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
    kernel = tilelang.compile(merging_kernel(op_name, dtype, precision="ftz_true" if op_name == "vdiv" else None, pos=pos), target="ascend")
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
