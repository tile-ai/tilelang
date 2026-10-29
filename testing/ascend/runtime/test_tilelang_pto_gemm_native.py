"""PTO L1 GEMM owns mutex slots without consuming surrounding event flags."""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


def _gemm_with_live_event(k):
    @T.prim_func
    def kernel(A: T.Tensor((64, k), "float16"), B: T.Tensor((64, k), "float16"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((64, k), "float16")
            b = T.alloc_l1((64, k), "float16")
            c = T.alloc_l0c((64, 64), "float32")
            T.copy(A, a)
            T.copy(B, b)
            # This legal hardware event belongs to the caller. GEMM uses the
            # separate mutex pool and must leave the event available to wait.
            T.ascend_set_flag("M_MTE1", 0)
            for stage in T.serial(2):
                T.gemm(a, b, c, transpose_B=True, clear_accum=T.if_then_else(stage == 0, 2, 0))
            T.ascend_wait_flag("M_MTE1", 0)
            T.copy(c, C)

    return kernel


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("k", [64, 512, 1536], ids=["single-stage", "two-stage", "looped-stages"])
def test_l1_gemm_preserves_caller_event(target, k):
    compiled = tilelang.compile(_gemm_with_live_event(k), target=target, out_idx=-1)
    # Exactly representable products make synchronization corruption visible
    # without a numerical tolerance depending on the matrix dimensions.
    generator = torch.Generator().manual_seed(13)
    a = torch.randint(-2, 3, (64, k), generator=generator).to(torch.float16)
    b = torch.randint(-2, 3, (64, k), generator=generator).to(torch.float16)
    actual = compiled(a.npu(), b.npu()).cpu()
    expected = 2 * (a.float() @ b.float().T)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _precise_division(active_lanes, mode):
    @T.prim_func
    def kernel(
        A: T.Tensor((64,), "float32"),
        B: T.Tensor((64,), "float32"),
        Old: T.Tensor((64,), "float32"),
        Out: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "float32")
            b = T.alloc_shared((64,), "float32")
            old = T.alloc_shared((64,), "float32")
            out = T.alloc_shared((64,), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(Old, old)
            with T.SimdVF():
                full = T.simd.pset(32)
                mask = T.simd.update_mask(active_lanes, 32)
                result = T.simd.alloc_local((1,), "float32")
                result[0] = T.simd.vld(old[0])
                result[0] = T.simd.vdiv(T.simd.vld(a[0]), T.simd.vld(b[0]), mask, mode=mode, precision="exact")
                T.simd.vsts(out[0], result[0], full)
            T.copy(out, Out)

    return kernel


def _division_inputs():
    a = torch.linspace(-127.0, 129.0, 64)
    b = torch.linspace(0.13, 3.7, 64)
    # Signed zeros, infinities, NaNs, exponent boundaries and subnormals.
    a_bits = [
        0,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x3F800000,
        0xBF800000,
        0x7F7FFFFF,
        0x00800000,
        0x80800000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x3F800001,
        0xBF800001,
    ]
    b_bits = [
        0x3F800000,
        0x3F800000,
        0x7F800000,
        0x3F800000,
        0x3F800000,
        0,
        0x80000000,
        0x00800000,
        0x40000000,
        0x40000000,
        0x3F800000,
        0x3F800000,
        0x00800000,
        0x00800000,
        0x3F7FFFFF,
        0x3F7FFFFF,
    ]
    a[:16] = torch.tensor(a_bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    b[:16] = torch.tensor(b_bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    return a, b


def test_precise_division_uses_native_pto_precision():
    source = tilelang.lower(_precise_division(37, "MODE_ZEROING"), target="pto", enable_device_compile=False).kernel_source
    assert 'precision="high_precision"' in source
    assert "vdiv_precise_f32" not in source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("active_lanes,mode", [(64, "MODE_ZEROING"), (37, "MODE_ZEROING"), (37, "MODE_MERGING")])
def test_precise_division_matches_ascend_semantics(active_lanes, mode):
    func = _precise_division(active_lanes, mode)
    a, b = _division_inputs()
    old = torch.arange(64, dtype=torch.float32) + 100
    args = (a.npu(), b.npu(), old.npu())
    reference = tilelang.compile(func, target="ascend", out_idx=-1)(*args).cpu()
    actual = tilelang.compile(func, target="pto", out_idx=-1)(*args).cpu()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
    comparable = ~torch.isnan(reference)
    torch.testing.assert_close(actual.view(torch.int32)[comparable], reference.view(torch.int32)[comparable], rtol=0, atol=0)
    if active_lanes < 64:
        inactive = old[active_lanes:] if mode == "MODE_MERGING" else torch.zeros(64 - active_lanes)
        torch.testing.assert_close(actual[active_lanes:], inactive, rtol=0, atol=0)
