import os
import re

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_aligned_row_major

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")

RAW_MLIR_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def dynamic_rank1_tail_mask_kernel(vector_size=512, dtype=T.float32):
    out_shape = (vector_size,)
    out_layout = make_aligned_row_major(out_shape, dtype, align_bytes=1024)
    lengths_shape = (128,)
    lengths_layout = make_aligned_row_major(lengths_shape, T.int32, align_bytes=1024)
    vector_layout = make_aligned_row_major(out_shape, dtype, align_bytes=1024)

    @T.prim_func
    def main(
        out: T.MeshTensor(out_shape, T.placement.replicated(), dtype, layout=out_layout),  # type: ignore
        lengths: T.MeshTensor(lengths_shape, T.placement.replicated(), T.int32, layout=lengths_layout),  # type: ignore
    ):
        with T.Kernel():
            out_shared = T.alloc_shared(out_shape, dtype)
            lengths_shared = T.alloc_shared(lengths_shape, T.int32)
            valid = T.alloc_var(T.int32, init=0)
            T.annotate_layout(
                {
                    out_shared: vector_layout,
                    lengths_shared: lengths_layout,
                }
            )

            T.copy(lengths, lengths_shared)
            valid = T.max(T.min(lengths_shared[0], vector_size), 0)
            T.clear(out_shared)
            for i in T.Tiles([valid], parallel=True):
                out_shared[i] = T.Cast(dtype, 1)

            T.copy(out_shared, out[0:vector_size])

    return main


def test_dynamic_rank1_tail_uses_axis_mask(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        dynamic_rank1_tail_mask_kernel(dtype=T.bfloat16),
        tmp_path,
        mlir_filename="dynamic_rank1_tail_mask_suvm.mlir",
        expected_tokens=("suvm.tile.range", "suvm.tile.cmpi"),
        opt_args=RAW_MLIR_OPT_ARGS,
    )
    # TileAxisMask expands to tile.range + cmpi in the SUVM builder.
    assert "suvm.tile.rect_mask" not in src
    range_match = re.search(r"suvm\.tile\.range : !suvm\.tile<(?P<extent>\d+)xi16>", src)
    assert range_match is not None
    extent = range_match["extent"]
    assert re.search(
        rf"suvm\.tile\.cmpi\s+ult, .* : !suvm\.tile<{extent}xi16>, i16 -> !suvm\.tile<{extent}xi1>",
        src,
    )
    assert not re.search(rf"suvm\.tile\.cmpi\s+slt, .* : !suvm\.tile<{extent}xi16>", src)


if __name__ == "__main__":
    tilelang.testing.main()
