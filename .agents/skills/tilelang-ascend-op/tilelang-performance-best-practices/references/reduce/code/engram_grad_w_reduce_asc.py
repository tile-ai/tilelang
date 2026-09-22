import tilelang
from tilelang import language as T
from tilelang.language import simd as S

from tile_kernels.config import get_num_vec_cores

_VEC = 64


def _choose_num_batches(num_rows: int, max_rows_per_batch: int) -> int:
    if num_rows == 1:
        return 1
    min_batches = (num_rows + max_rows_per_batch - 1) // max_rows_per_batch
    if min_batches == 1:
        for candidate in range(2, min(4, num_rows) + 1):
            if num_rows % candidate == 0:
                return candidate
        return num_rows
    for candidate in range(min_batches, num_rows + 1):
        if num_rows % candidate == 0:
            return candidate
    return num_rows


@tilelang.jit
def get_engram_grad_w_reduce_kernel_asc(hidden_size: int, num_persistent_blocks: int, hc_mult: int = 4):
    assert num_persistent_blocks > 0

    if hidden_size <= 3072:
        blk_d = 256
    elif hidden_size == 6144 and num_persistent_blocks > 74:
        blk_d = 256
    else:
        blk_d = 512
    assert hidden_size % blk_d == 0

    max_rows_per_batch = 74 if hidden_size > 3072 and blk_d == 256 else 37
    num_row_batches = _choose_num_batches(num_persistent_blocks, max_rows_per_batch)
    rows_per_batch = num_persistent_blocks // num_row_batches
    use_explicit_unroll = rows_per_batch <= 14
    use_two_accumulators = hidden_size <= 2560 and rows_per_batch >= 32
    buffer_versions = 2 if num_row_batches > 1 else 1
    row_pipeline_stages = 2 if num_row_batches > 1 else 0
    num_tiles = hidden_size // blk_d
    num_chunks = blk_d // _VEC
    num_cores = get_num_vec_cores()
    persistent_stages = 3

    @T.prim_func
    def engram_grad_w_reduce_kernel_asc(
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult, hidden_size), T.float],
        weight_hidden: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_embed: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        grad_weight_hidden: T.Tensor[(hc_mult, hidden_size), T.float],
        grad_weight_embed: T.Tensor[(hc_mult, hidden_size), T.float],
    ):
        with T.Kernel(num_cores) as core_id:
            wh_ub = T.alloc_shared((blk_d,), T.bfloat16)
            we_ub = T.alloc_shared((blk_d,), T.bfloat16)
            grad_wh_ub = T.alloc_shared((blk_d,), T.float)
            grad_we_ub = T.alloc_shared((blk_d,), T.float)
            grad_w_batch_ub = T.alloc_shared((rows_per_batch, blk_d), T.float)
            grad_w_acc_ub = T.alloc_shared((blk_d,), T.float)
            T.annotate_buffer_versions({
                wh_ub: persistent_stages,
                we_ub: persistent_stages,
                grad_wh_ub: persistent_stages,
                grad_we_ub: persistent_stages,
                grad_w_batch_ub: buffer_versions,
            })

            for pid_h, pid_b in T.Persistent([hc_mult, num_tiles], num_cores, core_id, group_size=1, num_stages=persistent_stages):
                col_start = pid_b * blk_d
                col_end = col_start + blk_d
                T.copy(weight_hidden[pid_h, col_start:col_end], wh_ub)
                T.copy(weight_embed[pid_h, col_start:col_end], we_ub)
                T.copy(grad_weight_hidden[pid_h, col_start:col_end], grad_wh_ub)
                T.copy(grad_weight_embed[pid_h, col_start:col_end], grad_we_ub)

                with T.SimdVF():
                    for c in range(num_chunks):
                        col = c * _VEC
                        S.vsts(grad_w_acc_ub[col], S.vdup(0.0, T.float32))

                for batch in T.Pipelined(num_row_batches, num_stages=row_pipeline_stages):
                    row_start = batch * rows_per_batch
                    T.copy(grad_w_partial[row_start : row_start + rows_per_batch, pid_h, col_start:col_end], grad_w_batch_ub, l2_cache_ctrl="NOTALLOC_KEEP")
                    with T.SimdVF():
                        acc0 = S.alloc_var(T.float32)
                        acc1 = S.alloc_var(T.float32)
                        for c in range(num_chunks):
                            col = c * _VEC
                            acc0 = S.vld(grad_w_acc_ub[col])
                            if use_explicit_unroll:
                                for row in T.unroll(rows_per_batch, explicit=True):
                                    acc0 = S.vadd(acc0, S.vld(grad_w_batch_ub[row, col]))
                            elif use_two_accumulators:
                                acc1 = S.vdup(0.0, T.float32)
                                for pair in T.serial(rows_per_batch // 2):
                                    row0 = pair * 2
                                    acc0 = S.vadd(acc0, S.vld(grad_w_batch_ub[row0, col]))
                                    acc1 = S.vadd(acc1, S.vld(grad_w_batch_ub[row0 + 1, col]))
                                if rows_per_batch % 2:
                                    acc0 = S.vadd(acc0, S.vld(grad_w_batch_ub[rows_per_batch - 1, col]))
                                acc0 = S.vadd(acc0, acc1)
                            else:
                                for row in T.serial(rows_per_batch):
                                    acc0 = S.vadd(acc0, S.vld(grad_w_batch_ub[row, col]))
                            S.vsts(grad_w_acc_ub[col], acc0)

                with T.SimdVF():
                    for c in range(num_chunks):
                        col = c * _VEC
                        total = S.vld(grad_w_acc_ub[col])
                        wh = S.vcvt(S.vld(wh_ub[col], dist="UNPK_B16"), T.float32, part=0)
                        we = S.vcvt(S.vld(we_ub[col], dist="UNPK_B16"), T.float32, part=0)
                        S.vsts(grad_wh_ub[col], S.vadd(S.vld(grad_wh_ub[col]), S.vmul(total, we)))
                        S.vsts(grad_we_ub[col], S.vadd(S.vld(grad_we_ub[col]), S.vmul(total, wh)))

                T.copy(grad_wh_ub, grad_weight_hidden[pid_h, col_start:col_end])
                T.copy(grad_we_ub, grad_weight_embed[pid_h, col_start:col_end])

    return engram_grad_w_reduce_kernel_asc
