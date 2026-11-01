import tilelang
from tilelang import language as T

from tile_kernels.config import get_num_vec_cores


@tilelang.jit
def get_normalize_weight_kernel_asc(num_topk: int):
    """Normalize top-k routing weights on Ascend NPU."""
    num_cores = get_num_vec_cores()
    num_tokens = T.dynamic("num_tokens")

    tile = 128
    num_stages = 4
    threads = 128

    @T.prim_func
    def normalize_weight_kernel(
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        denominator: T.Tensor[(num_tokens,), T.float32],
        normalized_weights: T.Tensor[(num_tokens, num_topk), T.float32],
    ):
        with T.Kernel(num_cores) as core_id:
            for tile_id in T.Persistent([T.ceildiv(num_tokens, tile)], num_cores, core_id, group_size=1, num_stages=num_stages):
                row = tile_id * tile
                valid_rows = T.min(tile, num_tokens - row)
                weights_ub = T.alloc_shared((tile, num_topk), T.float32)
                norm_ub = T.alloc_shared((tile, num_topk), T.float32)
                denom_ub = T.alloc_shared((tile,), T.float32)
                T.annotate_buffer_versions({weights_ub: num_stages, norm_ub: num_stages, denom_ub: num_stages})

                T.copy(topk_weights[row : row + valid_rows, :], weights_ub[:valid_rows, :])

                with T.SimtVF(threads=threads):
                    weights_frag = T.alloc_fragment((tile, num_topk), T.float32)
                    denom_frag = T.alloc_fragment((tile,), T.float32)
                    norm_frag = T.alloc_fragment((tile, num_topk), T.float32)

                    T.copy(weights_ub, weights_frag)
                    T.reduce_sum(weights_frag, denom_frag, dim=1)
                    for i in T.Parallel(tile):
                        denom_frag[i] += 1e-20
                    for i, j in T.Parallel(tile, num_topk):
                        norm_frag[i, j] = weights_frag[i, j] / denom_frag[i]
                    T.copy(denom_frag, denom_ub)
                    T.copy(norm_frag, norm_ub)

                T.copy(denom_ub[:valid_rows], denominator[row : row + valid_rows])
                T.copy(norm_ub[:valid_rows, :], normalized_weights[row : row + valid_rows, :])

    return normalize_weight_kernel
