import os

import tilelang.language as T
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import compile_test, target
from testing.python.sunmmio.common.formal_verify import *


@target("Sunmmio")
def summa_matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    """
    SUMMA (Scalable Universal Matrix Multiplication Algorithm)
    for a 4x4 mesh.

    Each core accumulates its local C tiles over all global K panels. For each
    panel, A is broadcast along the current core row and B along its column.
    """
    shard_policy = T.placement.full_shard(0, 1)

    A_shape = (M, K)
    B_shape = (K, N)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, [0, 1], (32, 32))
    B_layout = make_zz_layout(B_shape, [0, 1], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def kernel(
        A: T.MeshTensor(A_shape, shard_policy, dtype, layout=A_layout),
        B: T.MeshTensor(B_shape, shard_policy, dtype, layout=B_layout),
        C: T.MeshTensor(C_shape, shard_policy, accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, _ = A.local_shape
            _, sharded_N = B.local_shape
            core_row = _cid // T.mesh_ncols()
            core_col = _cid % T.mesh_ncols()

            # Multicast lands in RSRAM before A is copied into MMA's ASRAM input.
            A_broadcast = T.alloc_shared((block_M, block_K), dtype, scope="shared.rsram")
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_local)
                    K_steps = T.ceildiv(K, block_K)

                    for k_tile in range(K_steps):
                        a_src_col = k_tile % T.mesh_ncols()
                        b_src_row = k_tile % T.mesh_nrows()
                        a_local_k = (k_tile // T.mesh_ncols()) * block_K
                        b_local_k = (k_tile // T.mesh_nrows()) * block_K

                        T.comm.broadcast(
                            A[
                                bx * block_M : bx * block_M + block_M,
                                a_local_k : a_local_k + block_K,
                            ],
                            A_broadcast,
                            (core_row, a_src_col),
                            direction="h",
                        )
                        T.copy(A_broadcast, A_shared)
                        T.comm.broadcast(
                            B[
                                b_local_k : b_local_k + block_K,
                                by * block_N : by * block_N + block_N,
                            ],
                            B_shared,
                            (b_src_row, core_col),
                            direction="v",
                        )
                        T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C[bx * block_M, by * block_N])

    return kernel


def test_summa(is_log=False):
    func = summa_matmul(128, 128, 128, 32, 32, 32)

    script_device_mode = [
        "with T.launch_thread",
        "T.odma_unit(",
        "T.sunmmio_sync(",
        "T.dma_copy(",
        "T.broadcast_(",
        "T.mma_sunmmio(",
        "T.barrier_init(",
    ]

    script_lower_tile_op = [
        'A = T.match_buffer(A_handle, (32, 32), "float16", strides=(32, 1))',
        'B = T.match_buffer(B_handle, (32, 32), "float16", strides=(32, 1))',
        "C = T.match_buffer(C_handle, (32, 32), strides=(32, 1))",
        'bx = T.launch_thread("blockIdx.x", 16)',
        "for bx_1, by in T.grid(1, 1):",
        "for k_tile in range(4):",
        "T.dma_copy(T.region(A[bx_1 * 32, 0], 1, 32, 32), T.region(A_rsram_stage[0, 0], 2, 32, 32), 0)",
        "bx // 4 * 4 + k_tile",
        "k_tile * 4 + bx % 4",
        "T.mma_sunmmio(",
        "T.dma_copy(T.region(C_local[0, 0], 1, 32, 32), T.region(C[bx_1 * 32, by * 32], 2, 32, 32), 0)",
    ]

    script_InjectSunmmioSync = [
        "with T.launch_thread",
        "T.odma_unit(",
        "T.sunmmio_sync(",
        "T.dma_copy(",
        "T.broadcast_(",
        "T.mma_sunmmio(",
        "T.barrier_init(",
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
            log_dir=os.path.join(os.path.dirname(__file__), "_debug", "summa"),
            remove_header=True,
        )


if __name__ == "__main__":
    test_summa()
    # test_summa(is_log=True)
