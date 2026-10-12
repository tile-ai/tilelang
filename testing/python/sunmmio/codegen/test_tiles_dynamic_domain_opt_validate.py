import os
import re

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
def dynamic_rank2_domain_kernel(h=4, matrix_size=32, dtype=T.float32):
    out_shape = (16, matrix_size, matrix_size)
    token_policy = T.placement.mesh_as_line(0)
    out_layout = make_zz_layout(out_shape, [1, 2], (32, 32))
    lengths_shape = (128,)
    lengths_layout = make_aligned_row_major(lengths_shape, T.int32, align_bytes=1024)
    matrix_layout = make_zz_layout((matrix_size, matrix_size), [0, 1], (32, 32))

    @T.prim_func
    def main(
        out: T.MeshTensor(out_shape, token_policy, dtype, layout=out_layout),  # type: ignore
        lengths: T.MeshTensor(lengths_shape, T.placement.replicated(), T.int32, layout=lengths_layout),  # type: ignore
    ):
        with T.Kernel():
            src = T.alloc_shared((matrix_size, matrix_size), dtype)
            dst = T.alloc_shared((matrix_size, matrix_size), dtype)
            lengths_shared = T.alloc_shared(lengths_shape, T.int32)
            valid = T.alloc_var(T.int32, init=0)
            T.annotate_layout(
                {
                    src: matrix_layout,
                    dst: matrix_layout,
                    lengths_shared: lengths_layout,
                }
            )

            T.copy(lengths, lengths_shared)
            valid = T.max(T.min(lengths_shared[0], matrix_size), 0)
            T.fill(src, 1)
            T.fill(dst, 0)
            for i, j in T.Tiles([h, valid], parallel=True):
                dst[i, j] = src[i, j]

            T.copy(dst, out[0, 0:matrix_size, 0:matrix_size])

    return main


def test_dynamic_rank2_domain_is_materialized_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        dynamic_rank2_domain_kernel(),
        tmp_path,
        mlir_filename="dynamic_rank2_domain_suvm.mlir",
        opt_args=STRICT_OPT_ARGS,
    )
    valid_match = re.search(r"(?P<valid>%\d+) = arith\.maxsi", src)
    assert valid_match is not None
    domain_casts = re.findall(rf"(?P<domain>%\d+) = arith\.index_cast {valid_match['valid']} : i32 to index", src)
    assert len(domain_casts) == 1
    assert len(re.findall(rf"{re.escape(domain_casts[0])}(?!\d)", src)) >= 3


if __name__ == "__main__":
    tilelang.testing.main()
