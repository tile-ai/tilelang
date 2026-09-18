import os
import tilelang.language as T
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import compile_test, target
from testing.python.sunmmio.common.formal_verify import *


@target("Sunmmio")
def kernel_flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=1,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, heads, dim]
    # Different precisions will cause different number of allocates. The default allocate is allocated according to uint8, so when the data type is bfloat16, the number of allocates will be doubled.
    dtype = T.bfloat16
    # accum_dtype = T.float32
    accum_dtype = T.bfloat16
    shard_policy = T.placement.full_shard(0, 2)

    Q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    K_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    V_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, shard_policy, dtype, layout=Q_layout),
        K: T.MeshTensor(kv_shape, shard_policy, dtype, layout=K_layout),
        V: T.MeshTensor(kv_shape, shard_policy, dtype, layout=V_layout),
        Output: T.MeshTensor(q_shape, shard_policy, dtype, layout=O_layout),
    ):
        with T.Kernel() as _cid:
            sharded_batch = Q.local_shape[0]
            sharded_heads = Q.local_shape[2]

            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], accum_dtype, scope="shared.asram")
            acc_o = T.alloc_shared([block_M, dim], accum_dtype, scope="shared.rsram")
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            q_tiles = T.ceildiv(seq_len, block_M)
            for bz in T.serial(sharded_batch):
                for by in T.serial(sharded_heads):
                    for bx in T.serial(q_tiles):
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = (
                            T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                            if is_causal
                            else T.ceildiv(seq_len, block_N)
                        )

                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                            if is_causal:
                                for i, j in T.Tiles([block_M, block_N]):
                                    acc_s[i, j] = T.if_then_else(
                                        bx * block_M + i >= k * block_N + j,
                                        0,
                                        -T.infinity(acc_s.dtype),
                                    )
                                # T.fill(acc_s, T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)))
                            else:
                                for i, j in T.Tiles([block_M, block_N]):
                                    acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)

                            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                            for i in T.Tiles([block_M]):
                                scores_max_prev[i] = scores_max[i]
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Tiles([block_M]):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                            for i in T.Tiles([block_M]):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)

                            for i in T.Tiles([block_M]):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            T.copy(acc_s, acc_s_cast)

                            for i, j in T.Tiles([block_M, dim]):
                                acc_o[i, j] *= scores_scale[i]

                            T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o)

                        for i, j in T.Tiles([block_M, dim]):
                            acc_o[i, j] /= logsum[i]
                        T.copy(acc_o, O_shared)
                        T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def test_flashattn(is_log=False):
    func = kernel_flashattn(8, 32, 4096, 128, False, block_M=128, block_N=128, num_stages=1, threads=1)

    script_device_mode = [
        "with T.launch_thread",
        "T.odma_unit(",
        "T.sunmmio_sync(",
        "T.dma_copy(",
        "T.sunmmio_layout_transform(",
        "T.mma_sunmmio(",
    ]

    script_lower_tile_op = [
        """
        Q = T.match_buffer(Q_handle, (2, 4096, 8, 128), "bfloat16", strides=(4194304, 1024, 128, 1))
        K = T.match_buffer(K_handle, (2, 4096, 8, 128), "bfloat16", strides=(4194304, 1024, 128, 1))
        V = T.match_buffer(V_handle, (2, 4096, 8, 128), "bfloat16", strides=(4194304, 1024, 128, 1))
        Output = T.match_buffer(Output_handle, (2, 4096, 8, 128), "bfloat16", strides=(4194304, 1024, 128, 1))
        """,
        """
            bx = T.launch_thread("blockIdx.x", 16)
            with T.block("tilelang_root"):
        """,
        "for bz, by, bx_1 in T.grid(2, 8, 32):",
        "T.dma_copy(T.region(Q[bz, bx_1 * 128, by, 0], 1, 1, 128, 1, 128), T.region(Q_layout_stage[0, 0, 0, 0], 2, 1, 128, 1, 128), 0)",
        "T.sunmmio_layout_transform(T.region(Q_layout_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_rsram_stage[0, 0, 0, 0], 2, 1, 128, 1, 128))",
        "T.dma_copy(T.region(Q_rsram_stage[0, 0, 0, 0], 1, 1, 128, 1, 128), T.region(Q_shared[0, 0], 2, 128, 128), 0)",
    ]

    script_inject_sunmmio_sync = [
        "with T.launch_thread",
        "T.odma_unit(",
        "T.sunmmio_sync(",
        "T.dma_copy(",
        "T.sunmmio_layout_transform(",
        "T.mma_sunmmio(",
    ]

    test_config = {
        "LowerTileOp": {
            "script_expected": script_lower_tile_op,
        },
        "InjectSunmmioSync": {
            "script_expected": script_inject_sunmmio_sync,
        },
        "DeviceMod": {
            "script_expected": script_device_mode,
            "show_generated_script": True,
        },
    }

    test_config = get_or_add_default_verify(func, test_config)
    if not is_log:
        compile_test(func, target="Sunmmio", test_config=test_config)
    else:
        compile_test(
            func,
            out_idx=[2],
            target="Sunmmio",
            log_pass_output=True,
            log_dir=os.path.join(os.path.dirname(__file__), "_debug", "flashattn"),
            remove_header=True,
        )


if __name__ == "__main__":
    test_flashattn()
    # test_flashattn(is_log=True)
