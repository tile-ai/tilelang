import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_row_major, make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    find_async_op_lines,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


@target("Sunmmio")
def layout_transform_roundtrip_kernel(
    m=128,
    n=128,
    dtype=T.bfloat16,
):
    shard_policy = T.placement.replicated()
    dram_layout = make_zz_layout((m, n), axes=[0, 1], block_shape=(32, 32))
    rsram_layout = make_row_major((m, n))

    @T.prim_func
    def main(
        A: T.MeshTensor((m, n), shard_policy, dtype, layout=dram_layout),  # type: ignore
        B: T.MeshTensor((m, n), shard_policy, dtype, layout=dram_layout),  # type: ignore
    ):
        with T.Kernel() as _cid:
            A_rsram = T.alloc_shared((m, n), dtype, scope="shared.rsram")
            T.annotate_layout({A_rsram: rsram_layout})

            T.copy(A, A_rsram)
            T.copy(A_rsram, B)

    return main


def test_layout_transform_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        layout_transform_roundtrip_kernel(),
        tmp_path,
        mlir_filename="layout_transform_roundtrip_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.transform_layout_async",
            "suvm.sync",
        ),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    assert_source_contains(src, ("suvm.get_partitioned_tile_view",))
    assert "!suvm.token" not in src
    transform_lines = find_async_op_lines(src, "suvm.transform_layout_async")
    assert len(transform_lines) >= 2
    assert all("#suvm.unit<odma1>" in line for line in transform_lines)
    assert "sunmmio.fake" not in src


if __name__ == "__main__":
    tilelang.testing.main()
