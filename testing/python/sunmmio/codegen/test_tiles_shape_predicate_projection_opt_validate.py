import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_aligned_row_major, make_zz_layout

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")

STRICT_OPT_ARGS = ("--verify-each", "--suvm-to-llvm-pipeline")


@target("Sunmmio")
def tiles_shape_rank1_side_buffer_predicate_kernel(
    num_tokens=16,
    rows=64,
    cols=64,
    valid_m0=50,
    valid_n0=50,
    valid_m1=60,
    valid_n1=60,
    accum_dtype=T.float32,
):
    assert valid_m0 <= rows
    assert valid_m1 <= rows
    assert valid_n0 <= cols
    assert valid_n1 <= cols
    assert rows % 32 == 0
    assert cols % 32 == 0
    assert num_tokens % 16 == 0

    token_policy = T.placement.mesh_as_line(0)
    out_shape = (num_tokens, rows, cols)
    out_layout = make_zz_layout(out_shape, [1, 2], (32, 32))
    a_layout = make_zz_layout((rows, cols), [0, 1], (32, 32))
    b_layout = make_aligned_row_major((rows,), accum_dtype, align_bytes=64)

    @T.prim_func
    def main(
        Out: T.MeshTensor(out_shape, token_policy, accum_dtype, layout=out_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((rows, cols), accum_dtype)
            B_shared = T.alloc_shared((rows,), accum_dtype)

            T.annotate_layout({A_shared: a_layout, B_shared: b_layout})

            T.fill(A_shared, T.float32(4.0))
            T.fill(B_shared, T.float32(2.0))

            for i, j in T.Tiles([valid_m0, valid_n0]):
                A_shared[i, j] = A_shared[i, j] / B_shared[i]

            for i, j in T.Tiles([valid_m1, valid_n1]):
                A_shared[i, j] = A_shared[i, j] / B_shared[j]

            T.copy(A_shared, Out[0, 0:rows, 0:cols])

    return main


def test_tiles_shape_predicate_is_projected_to_access_axes(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    mlir_filename = "tiles_shape_predicate_projection_suvm.mlir"
    log_subdir = "tiles_shape_predicate_projection"
    validate_sunmmio_codegen_with_npuir_opt(
        tiles_shape_rank1_side_buffer_predicate_kernel(),
        tmp_path,
        mlir_filename=mlir_filename,
        opt_args=STRICT_OPT_ARGS,
        log_ir=True,
        log_dir=tmp_path,
        log_subdir=log_subdir,
    )

    tir_src = (tmp_path / log_subdir / "tiles_shape_predicate_projection_suvm.tir.log").read_text(encoding="utf-8")

    assert "A_shared.vload([i0 * 32 + ki, i1 * 32 + kj], predicate=i0 * 32 + ki < 50 and i1 * 32 + kj < 50)" in tir_src
    assert "B_shared.vload([i0 * 32 + ki], predicate=i0 * 32 + ki < 50)" in tir_src
    assert "B_shared.vload([i0 * 32 + ki], predicate=i0 * 32 + ki < 50 and i1 * 32 + kj < 50)" not in tir_src

    assert "B_shared.vload([i1 * 32 + kj], predicate=i1 * 32 + kj < 60)" in tir_src


if __name__ == "__main__":
    tilelang.testing.main()
