import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_aligned_row_major, make_zz_layout

from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")

LOOSE_OPT_ARGS = ("--verify-each",)
SUNMMIO_A4E_NCORES = 16


@target("Sunmmio")
def hybrid_tail_predicate_mask_kernel(
    num_tokens=4096,
    hc=4,
    mix_pad=32,
    comb_pad=32,
    sqrsum_align_bytes=1024,
    accum_dtype=T.float32,
):
    """MHC-like sinkhorn subgraph for hybrid tail predicate mask lowering.

    The important pattern is a small logical `(hc, hc)` tile-domain over a
    physical `(hc, 32)` RSRAM buffer, interleaved with `T.reduce_sum/max`.
    Before the canonical-mask fix, the hybrid scope lowered predicates such as
    `j * 32 + kj < 4` to `tile<1x32xi1>` and then emitted an illegal bool
    `tile.broadcast` to `tile<4x32xi1>`.
    """

    local_token_capacity = num_tokens // SUNMMIO_A4E_NCORES
    assert num_tokens % SUNMMIO_A4E_NCORES == 0
    assert local_token_capacity * 4 == sqrsum_align_bytes
    assert hc * 2 + hc * hc <= mix_pad
    assert hc <= comb_pad
    assert mix_pad % 32 == 0
    assert comb_pad % 32 == 0

    token_policy = T.placement.mesh_as_line(0)
    replicated_policy = T.placement.replicated()

    gemm_mul_shape = (num_tokens, mix_pad)
    sqrsum_shape = (num_tokens,)
    scale_shape = (1, mix_pad)
    base_shape = (1, mix_pad)
    comb_shape = (num_tokens, hc, comb_pad)

    gemm_mul_layout = make_zz_layout(gemm_mul_shape, [0, 1], (32, 32))
    sqrsum_layout = make_aligned_row_major(sqrsum_shape, accum_dtype, align_bytes=sqrsum_align_bytes)
    scale_layout = make_zz_layout(scale_shape, [0, 1], (32, 32))
    base_layout = make_zz_layout(base_shape, [0, 1], (32, 32))
    comb_layout = make_zz_layout(comb_shape, [1, 2], (32, 32))

    row_layout = make_zz_layout((1, mix_pad), [0, 1], (32, 32))
    vector_layout = make_aligned_row_major((mix_pad,), accum_dtype, align_bytes=64)
    sqrsum_stage_layout = make_aligned_row_major(
        (local_token_capacity,),
        accum_dtype,
        align_bytes=sqrsum_align_bytes,
    )
    cm_layout = make_zz_layout((hc, mix_pad), [0, 1], (32, 32))

    @T.prim_func
    def main(
        GemmOutMul: T.MeshTensor(gemm_mul_shape, token_policy, accum_dtype, layout=gemm_mul_layout),  # type: ignore
        GemmOutSqrsum: T.MeshTensor(sqrsum_shape, token_policy, accum_dtype, layout=sqrsum_layout),  # type: ignore
        HcScale: T.MeshTensor(scale_shape, replicated_policy, accum_dtype, layout=scale_layout),  # type: ignore
        HcBase: T.MeshTensor(base_shape, replicated_policy, accum_dtype, layout=base_layout),  # type: ignore
        CombMix: T.MeshTensor(comb_shape, token_policy, accum_dtype, layout=comb_layout),  # type: ignore
    ):
        with T.Kernel():
            scale_shared = T.alloc_shared((1, mix_pad), accum_dtype)
            base_shared = T.alloc_shared((1, mix_pad), accum_dtype)
            gemm_mul_shared = T.alloc_shared((1, mix_pad), accum_dtype)
            sqrsum_stage_shared = T.alloc_shared((local_token_capacity,), accum_dtype)
            sqrsum_norm_shared = T.alloc_shared((mix_pad,), accum_dtype)
            rms_shared = T.alloc_shared((mix_pad,), accum_dtype)
            mixes_shared = T.alloc_shared((mix_pad,), accum_dtype)
            cm = T.alloc_shared((hc, mix_pad), accum_dtype)
            row_max = T.alloc_shared((hc,), accum_dtype)
            row_sum = T.alloc_shared((hc,), accum_dtype)
            col_sum = T.alloc_shared((mix_pad,), accum_dtype)

            T.annotate_layout(
                {
                    scale_shared: row_layout,
                    base_shared: row_layout,
                    gemm_mul_shared: row_layout,
                    sqrsum_stage_shared: sqrsum_stage_layout,
                    sqrsum_norm_shared: vector_layout,
                    rms_shared: vector_layout,
                    mixes_shared: vector_layout,
                    cm: cm_layout,
                }
            )

            T.copy(HcScale[0:1, 0:mix_pad], scale_shared)
            T.copy(HcBase[0:1, 0:mix_pad], base_shared)
            T.copy(GemmOutSqrsum[0:local_token_capacity], sqrsum_stage_shared)

            scale2 = T.alloc_var(accum_dtype)
            sqrsum_value = T.alloc_var(accum_dtype)
            mix_value = T.alloc_var(accum_dtype)
            base_value = T.alloc_var(accum_dtype)
            scale2 = scale_shared[0, 2]

            for token in T.serial(1):
                T.copy(GemmOutMul[token : token + 1, 0:mix_pad], gemm_mul_shared)

                sqrsum_value = sqrsum_stage_shared[token]
                for m in T.Tiles([mix_pad]):
                    sqrsum_norm_shared[m] = sqrsum_value / T.float32(512.0) + T.float32(1.0e-6)
                for m in T.Tiles([mix_pad]):
                    rms_shared[m] = T.rsqrt(sqrsum_norm_shared[m])
                for m in T.Tiles([mix_pad]):
                    mixes_shared[m] = gemm_mul_shared[0, m] * rms_shared[m]

                T.fill(cm, 0)
                for i in T.serial(hc):
                    for j in T.serial(hc):
                        mix_value = mixes_shared[2 * hc + i * hc + j]
                        base_value = base_shared[0, 2 * hc + i * hc + j]
                        cm[i, j] = mix_value * scale2 + base_value

                T.reduce_max(cm[0:hc, 0:hc], row_max[0:hc], dim=1, clear=True)
                for i, j in T.Tiles([hc, hc]):
                    cm[i, j] = T.exp(cm[i, j] - row_max[i])

                T.reduce_sum(cm[0:hc, 0:hc], row_sum[0:hc], dim=1, clear=True)
                for i, j in T.Tiles([hc, hc]):
                    cm[i, j] = cm[i, j] / (row_sum[i] + T.float32(1.0e-6))

                T.reduce_sum(cm[0:hc, 0:hc], col_sum[0:hc], dim=0, clear=True)
                for i, j in T.Tiles([hc, hc]):
                    cm[i, j] = cm[i, j] / (col_sum[j] + T.float32(1.0e-6))

                T.copy(cm, CombMix[token, 0:hc, 0:comb_pad])

    return main


def test_hybrid_tail_predicate_mask_lowers_without_bool_broadcast(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        hybrid_tail_predicate_mask_kernel(),
        tmp_path,
        mlir_filename="hybrid_tail_predicate_mask_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "suvm.tile.cmpi", "suvm.tile.select"),
        opt_args=LOOSE_OPT_ARGS,
    )

    assert_source_contains(
        src,
        (
            "suvm.tile.reduce",
            "suvm.tile.cmpi",
            "suvm.tile.select",
            "!suvm.tile_view<4x32xf32>",
            "suvm.tile.extract_slice",
            "suvm.tile.insert_slice",
        ),
    )
    assert "!suvm.tile<1x32xi1> -> !suvm.tile<4x32xi1>" not in src


if __name__ == "__main__":
    tilelang.testing.main()
