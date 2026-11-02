"""SWZL GEMM loads must match the actual shared layout (issue #43)."""

import re

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_linear_layout,
    make_quarter_bank_swizzled_layout,
)
from tilelang.ppu.intrinsics.macro.mma_macro_generator import PPUTensorCoreIntrinEmitter


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "bfloat16"])
@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize("stages", [0, 2])
@pytest.mark.parametrize(
    "row_bytes,layout_fn,mode",
    [
        (128, make_full_bank_swizzled_layout, 0),
        (128, make_half_bank_swizzled_layout, 1),
        (128, make_quarter_bank_swizzled_layout, None),
        (128, make_linear_layout, None),
        (64, make_half_bank_swizzled_layout, 1),
        (64, make_quarter_bank_swizzled_layout, None),
        (64, make_linear_layout, None),
    ],
)
def test_gemm_swzl_layout(dtype, operand, stages, row_bytes, layout_fn, mode):
    k = row_bytes * 8 // tilelang.tvm.DataType(dtype).bits

    @T.prim_func
    def main(
        A: T.Tensor((64, k), dtype),
        B: T.Tensor((64, k), dtype),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, k), dtype)
            B_shared = T.alloc_shared((64, k), dtype)
            C_local = T.alloc_fragment((64, 64), "float32")
            if operand == "A":
                T.annotate_layout({A_shared: layout_fn(A_shared)})
            else:
                T.annotate_layout({B_shared: layout_fn(B_shared)})
            T.clear(C_local)
            for _ in T.Pipelined(2, num_stages=stages):
                T.copy(A, A_shared)
                T.copy(B, B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C)

    torch.manual_seed(20261026)
    a = torch.randn(64, k, device="cuda").to(getattr(torch, dtype))
    b = torch.randn(64, k, device="cuda").to(getattr(torch, dtype))
    expected = 2 * (a.float() @ b.float().T)
    for disabled in (False, True):
        kernel = tilelang.compile(
            main,
            out_idx=[2],
            pass_configs={"tl.disable_aiu_lower": True, "tl.disable_ldmat_swzl": disabled},
        )
        source = kernel.get_kernel_source()
        loads = [line for line in source.splitlines() if "tix_ldmatrix_swzl" in line]
        selected = [line for line in loads if f"{operand}_shared" in line]
        other = "B" if operand == "A" else "A"
        assert bool(selected) == (mode is not None and not disabled)
        if selected:
            assert all(re.search(rf", {mode}, (?:\(bool\)[01]|true|false)\);", line) for line in selected), selected
        # A fallback for one operand must not disable a compatible other one.
        assert any(f"{other}_shared" in line for line in loads) == (not disabled)
        torch.testing.assert_close(kernel(a, b), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "shape,extent,has_layout,expected",
    [
        ((16, 64), 64, True, (1, True)),
        ((16, 64), 64, False, (0, False)),
        ((8, 64), 64, True, (0, False)),
        ((16, 64), 8, True, (0, False)),
    ],
)
def test_emitter_swzl_requires_layout_and_x4_shape(shape, extent, has_layout, expected):
    buffer = tilelang.tvm.tirx.decl_buffer(shape, "bfloat16", scope="shared")
    region = tilelang.tvm.tirx.BufferRegion(buffer, [tilelang.tvm.ir.Range(0, shape[0]), tilelang.tvm.ir.Range(0, extent)])
    layouts = {buffer: make_half_bank_swizzled_layout(buffer)} if has_layout else None
    emitter = PPUTensorCoreIntrinEmitter(ppu_arch=15, layout_map=layouts, warp_row_tiles=16, warp_col_tiles=16)
    assert emitter._ldmat_swzl_mode(region) == expected


if __name__ == "__main__":
    tilelang.testing.main()
