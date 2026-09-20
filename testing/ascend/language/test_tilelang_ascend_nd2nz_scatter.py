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


def nd2nz_scatter_test(sd: torch.dtype, dd: torch.dtype, rows: int, cols: int):
    """Scatter a tile from ND layout to NZ layout in UB.

    ND layout:  (rows, cols) dense
    NZ layout:  (rows + 1, cols) with padding row to avoid bank conflict
    """

    @T.prim_func
    def main(
        input_gm: T.Buffer((rows, cols), sd),
        output_gm: T.Buffer((rows + 1, cols), dd),
    ):
        with T.Kernel(1):
            src_ub = T.alloc_shared((rows, cols), sd)  # ND layout
            dst_ub = T.alloc_shared((rows + 1, cols), dd)  # NZ layout
            T.annotate_layout({dst_ub: make_ascend_compact_nz_layout(dst_ub)})
            T.copy(input_gm, src_ub)
            T.copy(src_ub, dst_ub[0:rows, 0:cols])
            T.copy(dst_ub, output_gm)

    return main


def _invalid_unpadded_nd2nz_copy(rows: int, cols: int):
    @T.prim_func
    def main(input_gm: T.Buffer((rows, cols), "float32")):
        with T.Kernel(1):
            src_ub = T.alloc_shared((rows, cols), "float32")
            dst_ub = T.alloc_shared((rows, cols), "float32")
            T.annotate_layout({dst_ub: make_ascend_compact_nz_layout(dst_ub)})
            T.copy(input_gm, src_ub)
            T.copy(src_ub, dst_ub)

    return main


def _staged_nd2nz_copy():
    stages = 2
    rows = 16
    cols = 16

    @T.prim_func
    def main(
        input_gm: T.Buffer((stages, rows, cols), "float16"),
        output_gm: T.Buffer((stages, rows + 1, cols), "float16"),
    ):
        with T.Kernel(1):
            src_ub = T.alloc_shared((stages, rows, cols), "float16")
            dst_ub = T.alloc_shared((stages, rows + 1, cols), "float16")
            T.annotate_layout({dst_ub: make_ascend_compact_nz_layout(dst_ub)})
            T.copy(input_gm, src_ub)
            for stage in T.serial(stages):
                T.copy(src_ub[stage, :rows, :cols], dst_ub[stage, :rows, :cols])
            T.copy(dst_ub, output_gm)

    return main


def _simtvf_nd2nz_copy():
    @T.prim_func
    def main(input_gm: T.Buffer((ROWS, COLS), "float32"), output_gm: T.Buffer((ROWS + 1, COLS), "float32")):
        with T.Kernel(1):
            src_ub = T.alloc_shared((ROWS, COLS), "float32")
            dst_ub = T.alloc_shared((ROWS + 1, COLS), "float32")
            T.annotate_layout({dst_ub: make_ascend_compact_nz_layout(dst_ub)})
            T.copy(input_gm, src_ub)
            with T.SimtVF(threads=128):
                T.copy(src_ub, dst_ub[:ROWS, :COLS])
            T.copy(dst_ub, output_gm)

    return main


def _simdvf_nd2nz_copy():
    @T.prim_func
    def main(
        input_gm: T.Buffer((ROWS, COLS), "float32"),
        output_gm: T.Buffer((ROWS + 1, COLS), "float32"),
    ):
        with T.Kernel(1):
            src_ub = T.alloc_shared((ROWS, COLS), "float32")
            dst_ub = T.alloc_shared((ROWS + 1, COLS), "float32")
            T.annotate_layout({dst_ub: make_ascend_compact_nz_layout(dst_ub)})
            T.copy(input_gm, src_ub)
            with T.SimdVF():
                T.copy(src_ub, dst_ub[:ROWS, :COLS])
            T.copy(dst_ub, output_gm)

    return main


def _reshaped_source_nd2nz_copy():
    @T.prim_func
    def main(input_gm: T.Buffer((ROWS, COLS), "bfloat16")):
        with T.Kernel(1):
            src_base = T.alloc_shared((ROWS, COLS * 2), "bfloat16")
            src_view = T.reshape(src_base, (ROWS * 2, COLS))
            dst = T.alloc_shared((ROWS + 1, COLS), "float32")
            T.annotate_layout({dst: make_ascend_compact_nz_layout(dst)})
            T.copy(input_gm, src_view[:ROWS, :])
            T.copy(src_view[:ROWS, :], dst[:ROWS, :])

    return main


def _strided_source_nd2nz_copy():
    @T.prim_func
    def main():
        with T.Kernel(1):
            src_base = T.alloc_shared((ROWS, COLS), "float32")
            src_view = T.view(src_base, strides=(COLS * 2, 1))
            dst = T.alloc_shared((ROWS + 1, COLS), "float32")
            T.annotate_layout({dst: make_ascend_compact_nz_layout(dst)})
            T.copy(src_view, dst[:ROWS, :])

    return main


def _leading_strided_source_nd2nz_copy(leading_stride: int):
    @T.prim_func
    def main():
        with T.Kernel(1):
            src_base = T.alloc_shared((2, ROWS, COLS), "float32")
            src_view = T.view(src_base, strides=(leading_stride, COLS, 1))
            dst = T.alloc_shared((ROWS + 1, COLS), "float32")
            T.annotate_layout({dst: make_ascend_compact_nz_layout(dst)})
            T.copy(src_view[1, :, :], dst[:ROWS, :])

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
        cube_a_gm: T.Buffer((16, 16), "bfloat16"),
        cube_b_gm: T.Buffer((16, 16), "bfloat16"),
        input_gm: T.Buffer((128, COLS), "float32"),
        output_gm: T.Buffer((128, COLS), "float32"),
    ):
        with T.Kernel(1):
            cube_a_l1 = T.alloc_l1((16, 16), "bfloat16")
            cube_b_l1 = T.alloc_l1((16, 16), "bfloat16")
            cube_accum = T.alloc_l0c((16, 16), "float32")
            x_ub = T.alloc_shared((64, COLS), "float32")
            with T.Cube():
                T.copy(cube_a_gm, cube_a_l1)
                T.copy(cube_b_gm, cube_b_l1)
                T.gemm(cube_a_l1, cube_b_l1, cube_accum, transpose_B=True, clear_accum=True)
            with T.Vector():
                T.dual_copy(input_gm, x_ub)
                T.dual_copy(x_ub, output_gm)

    return main


def test_manual_pipeline_rewrites_compact_nz_dual_copy():
    kernel = tilelang.compile(
        _manual_compact_nz_dual_copy(),
        out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    source = kernel.get_kernel_source()
    assert "__global__ __mix__(1, 2)" in source
    assert source.count("asc_get_sub_block_id()") == 1
    assert source.count("asc_copy_ub2l1") == 1


@pytest.mark.parametrize(
    "target,kernel_marker,sid_marker,load_marker,store_marker",
    [
        (
            "ascend",
            "__global__ __mix__(1, 2)",
            "asc_get_sub_block_id()",
            "asc_copy_gm2ub_align",
            "asc_copy_ub2gm_align",
        ),
        (
            "pto",
            'with pto.section("cube")',
            "pto.get_subblock_idx()",
            "pto.mte_gm_ub",
            "pto.mte_ub_gm",
        ),
    ],
    ids=["ascend", "pto"],
)
def test_manual_mixed_pipeline_reuses_one_sid_for_dual_copy(target, kernel_marker, sid_marker, load_marker, store_marker):
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}):
        source = tilelang.lower(_manual_mixed_dual_copy(), target=target).kernel_source

    assert kernel_marker in source
    if target == "pto":
        assert 'with pto.section("vector")' in source
    assert source.count(sid_marker) == 1
    assert source.count(load_marker) == 1
    assert source.count(store_marker) == 1


@pytest.mark.parametrize(
    "sd,dd",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_nd2nz_scatter(sd, dd):
    # Prepare input: a recognizable pattern
    input_gm = torch.arange(ROWS * COLS, dtype=sd, device="npu").reshape(ROWS, COLS) + 1.0

    program = nd2nz_scatter_test(sd, dd, ROWS, COLS)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
    )

    output_gm = kernel(input_gm)
    torch.npu.synchronize()

    frac_len = 32 // output_gm.element_size()
    input_cast = input_gm.to(dtype=dd)
    output_back = output_gm.view(COLS // frac_len, ROWS + 1, frac_len)[:, :-1, :].permute(1, 0, 2).reshape(ROWS, COLS)
    assert torch.equal(input_cast, output_back), f"sd={sd} dd={dd}"


def test_nd2nz_scatter_tcopy_codegen():
    # tilelang.lower expects the caller to hold the target scope; the
    # vectorize planner consults Target.current(). Same below.
    with tvm.target.Target("ascend"):
        source = tilelang.lower(
            nd2nz_scatter_test(torch.float32, torch.bfloat16, ROWS, COLS),
            target="ascend",
        ).kernel_source

    assert "__global__ __vector__ void main_kernel" in source
    assert "ascend_nd2nz_scatter<32, 128, float, bfloat16_t>" in source
    assert "asc_copy_ub2l1" not in source


def test_nd2nz_scatter_tcopy_requires_padding_row():
    with tvm.target.Target("ascend"), pytest.raises(ValueError, match="reserve exactly one padding row"):
        tilelang.lower(_invalid_unpadded_nd2nz_copy(ROWS, COLS), target="ascend")


def test_staged_nd2nz_scatter_uses_loop_bounds():
    with tvm.target.Target("ascend"):
        source = tilelang.lower(_staged_nd2nz_copy(), target="ascend").kernel_source
    assert "ascend_nd2nz_scatter<16, 16, half, half>" in source


def test_reshaped_source_tcopy_codegen():
    with tvm.target.Target("ascend"):
        source = tilelang.lower(_reshaped_source_nd2nz_copy(), target="ascend").kernel_source
    assert "ascend_nd2nz_scatter<32, 128, bfloat16_t, float>" in source


def test_nd2nz_scatter_rejects_strided_source():
    with tvm.target.Target("ascend"), pytest.raises(ValueError, match="compact trailing source matrix"):
        tilelang.lower(_strided_source_nd2nz_copy(), target="ascend")


def test_nd2nz_scatter_accepts_aligned_leading_stride():
    with tvm.target.Target("ascend"):
        source = tilelang.lower(_leading_strided_source_nd2nz_copy(ROWS * COLS), target="ascend").kernel_source
    assert "ascend_nd2nz_scatter<32, 128, float, float>" in source


def test_nd2nz_scatter_rejects_unaligned_source_address():
    with tvm.target.Target("ascend"), pytest.raises(ValueError, match="32-byte-aligned source address"):
        tilelang.lower(_leading_strided_source_nd2nz_copy(ROWS * COLS - 1), target="ascend")


def test_simtvf_copy_is_not_rewritten_to_simdvf_scatter():
    with tvm.target.Target("ascend"):
        source = tilelang.lower(_simtvf_nd2nz_copy(), target="ascend").kernel_source
    assert "__simt_vf__" in source
    assert "ascend_nd2nz_scatter" not in source


def test_simdvf_copy_uses_nd2nz_callee():
    kernel = tilelang.compile(_simdvf_nd2nz_copy(), out_idx=-1)
    source = kernel.get_kernel_source()
    assert "__simd_vf__ inline void main_kernel_simd_vf_0" in source
    assert "ascend_nd2nz_scatter_callee<32, 128, float, float>" in source


if __name__ == "__main__":
    tilelang.testing.main()
