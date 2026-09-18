import json
import os

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    validate_sunmmio_codegen_with_npuir_opt,
    assert_source_contains,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


@target("Sunmmio")
def softmax_dynamic(block_M=256, block_N=128, in_dtype=T.float32, out_dtype=T.float32):
    """JIT kernel factory for dynamic-(M, N) row-wise softmax."""
    M = T.dynamic("m")
    N = T.dynamic("n")
    mesh_cols = 4

    zz_layout = make_zz_layout((M, N), [0, 1], (32, 32))
    placement = T.placement.full_shard(0, 1)

    accum_dtype = T.float32
    scale = 1.44269504

    @T.prim_func
    def softmax(
        X: T.MeshTensor((M, N), placement, in_dtype, layout=zz_layout),
        Y: T.MeshTensor((M, N), placement, out_dtype, layout=zz_layout),
    ):
        with T.Kernel():
            sharded_M, sharded_N = X.local_shape

            X_shared = T.alloc_shared((block_M, block_N), in_dtype)
            Y_shared = T.alloc_shared((block_M, block_N), out_dtype)
            exp_x = T.alloc_shared((block_M, block_N), accum_dtype)
            tile_max = T.alloc_shared((block_M,), accum_dtype)
            tile_sum = T.alloc_shared((block_M,), accum_dtype)
            local_lse = T.alloc_shared((block_M,), accum_dtype)
            lse_dist = T.alloc_shared((mesh_cols, block_M), accum_dtype)
            lse_max = T.alloc_shared((block_M,), accum_dtype)
            global_lse = T.alloc_shared((block_M,), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                T.fill(local_lse, -T.infinity(accum_dtype))
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(
                        X[
                            bx * block_M : (bx + 1) * block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        X_shared,
                    )
                    T.reduce_max(X_shared, tile_max, dim=1, clear=True)
                    for i, j in T.Tiles([block_M, block_N]):
                        exp_x[i, j] = T.exp2(X_shared[i, j] * scale - tile_max[i] * scale)
                    T.reduce_sum(exp_x, tile_sum, dim=1, clear=True)
                    for i in T.Tiles([block_M]):
                        local_lse[i] = tile_max[i] * scale + T.log2(T.exp2(local_lse[i] - tile_max[i] * scale) + tile_sum[i])

                T.comm.all_gather(local_lse, lse_dist, direction="h")
                T.reduce_max(lse_dist, lse_max, dim=0, clear=True)
                for i, j in T.Tiles([mesh_cols, block_M]):
                    lse_dist[i, j] = T.exp2(lse_dist[i, j] - lse_max[j])
                T.reduce_sum(lse_dist, global_lse, dim=0, clear=True)
                for i in T.Tiles([block_M]):
                    global_lse[i] = lse_max[i] + T.log2(global_lse[i])

                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(
                        X[
                            bx * block_M : (bx + 1) * block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                        X_shared,
                    )
                    for i, j in T.Tiles([block_M, block_N]):
                        Y_shared[i, j] = T.exp2(X_shared[i, j] * scale - global_lse[i])
                    T.copy(
                        Y_shared,
                        Y[
                            bx * block_M : (bx + 1) * block_M,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )

    return softmax


def test_simple_global_copy_gemm_codegen_validates_with_npuir_opt(tmp_path, monkeypatch):
    coverage_path = tmp_path / "softmax_dynamic_coverage.json"
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", str(coverage_path))
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")

    src = validate_sunmmio_codegen_with_npuir_opt(
        softmax_dynamic(),
        tmp_path,
        mlir_filename="softmax_dynamic_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.tile.reduce",
            "suvm.mcast_tok",
            "suvm.sync",
        ),
    )
    assert_source_contains(
        src,
        ("suvm.tile.reduce", "suvm.tile.exp", "suvm.tile.ln", "suvm.mcast_tok"),
    )
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    main = coverage["main"]
    tiles = coverage["tiles"]
    assert main["missing_node_types"] == []
    assert main["missing_call_ops"] == []
    assert tiles["missing_node_types"] == []
    assert tiles["missing_call_ops"] == []

    assert "tir.For" in main["expected_node_types"]
    assert "tir.For" in main["visited_node_types"]
    assert "tir.For" in tiles["expected_node_types"]
    assert "tir.For" in tiles["visited_node_types"]
    assert "tl.vector_core_in_tile_reduce" not in main["expected_call_ops"]
    assert "tl.vector_core_in_tile_reduce" in tiles["expected_call_ops"]
    assert "tl.vector_core_in_tile_reduce" in tiles["visited_call_ops"]


if __name__ == "__main__":
    tilelang.testing.main()
