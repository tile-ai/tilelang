import os

import tilelang.language as T
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import compile_test, target
from testing.python.sunmmio.common.formal_verify import *


@target("Sunmmio")
def kernel_overall(M, N, K, block_M, block_N, block_K, dtype="bfloat16", accum_dtype="float32"):
    shard_policy = T.placement.full_shard(0, 1)

    A_shape = (M, K)
    B_shape = (K, N)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, [0, 1], (32, 32))
    B_layout = make_zz_layout(B_shape, [0, 1], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(A_shape, shard_policy, dtype, layout=A_layout),
        B: T.MeshTensor(B_shape, shard_policy, dtype, layout=B_layout),
        Bias: T.MeshTensor(C_shape, shard_policy, accum_dtype, layout=C_layout),
        C: T.MeshTensor(C_shape, shard_policy, accum_dtype, layout=C_layout),
    ):
        # Initialize Kernel Context
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            # [wanghz18] Automatic SRAM Scope Inference
            # We declare generic 'shared' scope, expecting InferSramScope pass to
            # refine them to 'shared.asram', 'shared.wsram', 'shared.rsram'
            A_shared = T.alloc_shared((block_M, block_K), dtype=dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype=dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            Bias_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_N, block_N)):
                for by in T.serial(T.ceildiv(sharded_M, block_M)):
                    T.clear(C_shared)  # Avoid Fill op unsupported scope error

                    # [wanghz18] GEMM Lowering to mma_sunmmio intrinsic
                    for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=2):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_shared)

                    # Load Bias
                    T.copy(Bias[by * block_M, bx * block_N], Bias_shared)

                    # [weizzh] Tiles Loop for Element-wise operation
                    # This loop should be legalized and vectorized by LegalizeTilesLoop/TilesLoop passes
                    for i, j in T.Tiles(C_shared, parallel=True):
                        C_shared[i, j] = C_shared[i, j] + Bias_shared[i, j]

                    # [xiaoyao-NKU] Inter-core Communication (Broadcast)
                    C_remote = T.alloc_shared((block_M, block_N), accum_dtype)
                    T.comm.broadcast(C_shared, C_remote, (0, 0), direction="h")

                    # Store result
                    T.copy(C_remote, C[by * block_M, bx * block_N])

    return main


def test_overall(is_log=False):
    func = kernel_overall(256, 256, 128, 64, 64, 32)
    script_device_mode = [
        "with T.launch_thread",
        "T.odma_unit(",
        "T.sunmmio_sync(",
        "T.dma_copy(",
        "T.mma_sunmmio(",
        "T.barrier_init(",
        "T.barrier_arrive_and_wait(",
        "T.broadcast_(",
    ]

    script_lower_tile_op = [
        "T.dma_copy(",
        "T.mma_sunmmio(",
        "T.broadcast_(",
    ]

    script_InjectSunmmioSync = [
        "with T.launch_thread",
        "T.odma_unit(",
        "T.sunmmio_sync(",
        "T.dma_copy(",
        "T.mma_sunmmio(",
        "T.barrier_init(",
        "T.barrier_arrive_and_wait(",
        "T.broadcast_(",
    ]

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_InjectSunmmioSync,
        },
        "DeviceMod": {
            "script_expected": script_device_mode,
        },
    }
    test_config = get_or_add_default_verify(func, test_config)
    if not is_log:
        compile_test(func, out_idx=[2], target="Sunmmio", test_config=test_config)
    else:
        compile_test(
            func,
            out_idx=[2],
            target="Sunmmio",
            log_pass_output=True,
            log_dir=os.path.join(os.path.dirname(__file__), "_debug", "overall"),
            remove_header=True,
        )


if __name__ == "__main__":
    test_overall()
    # test_overall(is_log=True)
