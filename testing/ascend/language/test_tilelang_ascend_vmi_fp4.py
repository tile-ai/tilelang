"""On-device coverage for VMI BF16-to-packed-FP4 conversion."""

from __future__ import annotations

import shutil

import pytest
import torch

import tilelang
import tilelang.ascend.language as T


LANES = 256
ACTIVE_PHYSICAL_LANES = 37
FP4_CODES = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _vmi_bf16_to_fp4():
    @T.prim_func
    def main(A: T.Buffer((LANES,), "bfloat16"), B: T.Buffer((LANES,), "float4_e2m1fn")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((LANES,), "bfloat16")
            b_ub = T.alloc_shared((LANES,), "float4_e2m1fn")
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.vmi.create_mask(LANES // 2, size=LANES // 2)
                bf16 = T.vmi.vload(a_ub[0], size=LANES)
                fp4 = T.vmi.vcvt(bf16, "float4_e2m1fn")
                T.vmi.vstore(fp4, b_ub[0], mask)
            T.copy(b_ub, B)

    return main


def _vmi_bf16_to_fp4_partial():
    @T.prim_func
    def main(
        A: T.Buffer((LANES,), "bfloat16"),
        B: T.Buffer((LANES,), "float4_e2m1fn"),
        C: T.Buffer((LANES,), "float4_e2m1fn"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((LANES,), "bfloat16")
            c_ub = T.alloc_shared((LANES,), "float4_e2m1fn")
            T.copy(A, a_ub)
            T.copy(B, c_ub)
            with T.SimdVF():
                # FP4 stores consume physical f4e2m1x2 predicate lanes.
                mask = T.vmi.create_mask(ACTIVE_PHYSICAL_LANES, size=LANES // 2)
                bf16 = T.vmi.vload(a_ub[0], size=LANES)
                fp4 = T.vmi.vcvt(bf16, "float4_e2m1fn")
                T.vmi.vstore(fp4, c_ub[0], mask)
            T.copy(c_ub, C)

    return main


def _fp4_test_data():
    repeats = (LANES + FP4_CODES.numel() - 1) // FP4_CODES.numel()
    codes = FP4_CODES.repeat(repeats)[:LANES]
    values = FP4_VALUES.repeat(repeats)[:LANES].to(torch.bfloat16)
    return values, codes[0::2] | (codes[1::2] << 4)


@pytest.mark.pto
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "npu") or not torch.npu.is_available() or shutil.which("ptoas") is None,
    reason="NPU FP4 storage is unavailable",
)
def test_vmi_bf16_to_fp4_e2e():
    values, expected_bytes = _fp4_test_data()
    kernel = tilelang.compile(_vmi_bf16_to_fp4(), target="pto", out_idx=-1)

    result = kernel(values.npu())
    torch.npu.synchronize()

    assert result.dtype == torch.float4_e2m1fn_x2
    assert result.shape == (LANES // 2,)
    torch.testing.assert_close(result.view(torch.uint8).cpu(), expected_bytes)


@pytest.mark.pto
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "npu") or not torch.npu.is_available() or shutil.which("ptoas") is None,
    reason="NPU FP4 storage is unavailable",
)
def test_vmi_bf16_to_fp4_physical_partial_mask_e2e():
    values, packed_values = _fp4_test_data()
    initial_bytes = torch.full((LANES // 2,), 0xFF, dtype=torch.uint8, device="npu")
    initial = initial_bytes.view(torch.float4_e2m1fn_x2)
    kernel = tilelang.compile(_vmi_bf16_to_fp4_partial(), target="pto", out_idx=-1)

    result = kernel(values.npu(), initial)
    torch.npu.synchronize()

    expected_bytes = initial_bytes.cpu()
    expected_bytes[:ACTIVE_PHYSICAL_LANES] = packed_values[:ACTIVE_PHYSICAL_LANES]
    torch.testing.assert_close(result.view(torch.uint8).cpu(), expected_bytes)
