"""
Test the layout-driven UB(ND)->UB(NZ) T.copy frontend.

Does: UB ND (32, 128) -> UB NZ (33, 128) via scatter, with optional cast.
Then writes the physical NZ result back to GM for verification.
"""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_nz_layout


def nd2nz_scatter_test(sd: torch.dtype, dd: torch.dtype, rows: int, cols: int, inside_vf=False):
    """Scatter a tile from ND layout to NZ layout in UB.

    ND layout:  (rows, cols) dense
    NZ layout:  (rows + 1, cols) with padding row to avoid bank conflict
    """

    @T.prim_func
    def main(
        input_gm: T.Tensor((rows, cols), sd),
        output_gm: T.Tensor((rows + 1, cols), dd),
    ):
        with T.Kernel(1):
            src_ub = T.alloc_shared((rows, cols), sd)  # ND layout
            dst_ub = T.alloc_shared((rows + 1, cols), dd)  # NZ layout
            T.annotate_layout({dst_ub: make_ascend_compact_nz_layout(dst_ub)})
            T.copy(input_gm, src_ub)
            if inside_vf:
                with T.SimdVF():
                    T.copy(src_ub, dst_ub[0:rows, 0:cols])
            else:
                T.copy(src_ub, dst_ub[0:rows, 0:cols])
            T.copy(dst_ub, output_gm)

    return main


ROWS = 32
COLS = 128


def _manual_compact_nz_dual_copy():
    @T.prim_func
    def main():
        with T.Kernel(1):
            x_l1 = T.alloc_l1((64, COLS), "float32")
            x_nz = T.alloc_shared((33, COLS), "float32")
            T.annotate_layout(
                {
                    x_l1: make_ascend_nz_layout(x_l1),
                    x_nz: make_ascend_compact_nz_layout(x_nz),
                }
            )
            with T.Cube():
                pass
            with T.Vector():
                T.dual_copy(x_nz[:32, :], x_l1[:, :])

    return main


def _manual_mixed_dual_copy():
    @T.prim_func
    def main(
        cube_a_gm: T.Tensor((16, 16), "bfloat16"),
        input_gm: T.Tensor((128, COLS), "float32"),
        output_gm: T.Tensor((128, COLS), "float32"),
    ):
        with T.Kernel(1):
            cube_a_l1 = T.alloc_l1((16, 16), "bfloat16")
            x_ub = T.alloc_shared((64, COLS), "float32")
            with T.Cube():
                T.copy(cube_a_gm, cube_a_l1)
            with T.Vector():
                T.dual_copy(input_gm, x_ub)
                T.dual_copy(x_ub, output_gm)

    return main


def test_manual_pipeline_rewrites_compact_nz_dual_copy():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}):
        source = tilelang.lower(_manual_compact_nz_dual_copy(), target="ascend").kernel_source
    assert "__global__ __mix__(1, 2)" in source
    assert source.count("asc_get_sub_block_id()") == 1
    assert source.count("asc_copy_ub2l1") == 1


def test_manual_mixed_pipeline_shares_sub_block_id():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}):
        source = tilelang.lower(_manual_mixed_dual_copy(), target="ascend").kernel_source
    assert "__global__ __mix__(1, 2)" in source
    assert source.count("asc_get_sub_block_id()") == 1
    assert "asc_copy_gm2ub_align" in source
    assert "asc_copy_ub2gm_align" in source


@pytest.mark.parametrize(
    "sd,dd,inside_vf",
    [
        (torch.float16, torch.float16, False),
        (torch.bfloat16, torch.bfloat16, False),
        (torch.bfloat16, torch.float32, False),
        (torch.float32, torch.float32, False),
        (torch.float32, torch.bfloat16, False),
        pytest.param(torch.bfloat16, torch.float32, True, id="inside-simd-vf"),
    ],
)
def test_nd2nz_scatter(sd, dd, inside_vf):
    # Prepare input: a recognizable pattern
    input_gm = torch.arange(ROWS * COLS, dtype=sd, device="npu").reshape(ROWS, COLS) + 1.0

    program = nd2nz_scatter_test(sd, dd, ROWS, COLS, inside_vf)
    kernel = tilelang.compile(
        program,
        target="ascend",
        out_idx=-1,
    )

    output_gm = kernel(input_gm)
    torch.npu.synchronize()

    frac_len = 32 // output_gm.element_size()
    input_cast = input_gm.to(dtype=dd)
    output_back = output_gm.view(COLS // frac_len, ROWS + 1, frac_len)[:, :-1, :].permute(1, 0, 2).reshape(ROWS, COLS)
    assert torch.equal(input_cast, output_back), f"sd={sd} dd={dd}"
