import argparse
import itertools
from typing import Callable

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_fwd(
    batch: int,
    heads: int,
    dim_qk: int,
    dim_vo: int,
    softmax_scale: float,
    mask_fn: Callable[[int, int, int, int, int, int, T.ptr], bool],
    block_mask_fn: Callable[[int, int, int, int, int, int, int, int, T.ptr], bool],
    block_qo: int = 64,
    block_kv: int = 64,
    num_stages: int = 2,
    thread_num: int = 128,
):
    scale = softmax_scale * 1.44269504  # log2(e)
    dtype = T.bfloat16
    accum_dtype = T.float32

    seq_len_qo = T.dynamic('seq_len_qo')
    seq_len_kv = T.dynamic('seq_len_kv')

    @T.prim_func
    def flash_fwd(
        q_global: T.Tensor([batch, seq_len_qo, heads, dim_qk], dtype),
        k_global: T.Tensor([batch, seq_len_kv, heads, dim_qk], dtype),
        v_global: T.Tensor([batch, seq_len_kv, heads, dim_vo], dtype),
        o_global: T.Tensor([batch, seq_len_qo, heads, dim_vo], dtype),
        lse_global: T.Tensor([batch, heads, seq_len_qo], accum_dtype),
        mask_custom_data: T.ptr,
        block_mask_custom_data: T.ptr,
    ):
        with T.Kernel(T.ceildiv(seq_len_qo, block_qo), heads, batch, threads=thread_num) as (
            bx,
            by,
            bz,
        ):
            q_shared = T.alloc_shared([block_qo, dim_qk], dtype)
            k_shared = T.alloc_shared([block_kv, dim_qk], dtype)
            v_shared = T.alloc_shared([block_kv, dim_vo], dtype)
            acc_s = T.alloc_fragment([block_qo, block_kv], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_qo, block_kv], dtype)
            acc_o = T.alloc_fragment([block_qo, dim_vo], accum_dtype)
            scores_max = T.alloc_fragment([block_qo], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_qo], accum_dtype)
            scores_scale = T.alloc_fragment([block_qo], accum_dtype)
            scores_sum = T.alloc_fragment([block_qo], accum_dtype)
            scaled_sum_exp = T.alloc_fragment([block_qo], accum_dtype)

            T.copy(q_global[bz, bx * block_qo : (bx + 1) * block_qo, by, :], q_shared)
            T.fill(acc_o, 0)
            T.fill(scaled_sum_exp, 0)
            T.fill(scores_max, -2.**100)

            for k in T.Pipelined(T.ceildiv(seq_len_kv, block_kv), num_stages=num_stages):
                if block_mask_fn(
                    bz,
                    by,
                    bx * block_qo,
                    T.min(seq_len_qo, (bx + 1) * block_qo),
                    k * block_kv,
                    T.min(seq_len_kv, (k + 1) * block_kv),
                    seq_len_qo,
                    seq_len_kv,
                    block_mask_custom_data,
                ):
                    T.copy(k_global[bz, k * block_kv : (k + 1) * block_kv, by, :], k_shared)
                    for i, j in T.Parallel(block_qo, block_kv):
                        acc_s[i, j] = T.if_then_else(
                            mask_fn(
                                bz,
                                by,
                                bx * block_qo + i,
                                k * block_kv + j,
                                seq_len_qo,
                                seq_len_kv,
                                mask_custom_data,
                            )
                            and k * block_kv + j < seq_len_kv,
                            0,
                            -2.**100,
                        )
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(v_global[bz, k * block_kv : (k + 1) * block_kv, by, :], v_shared)
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_qo):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_qo):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_qo, dim_vo):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_qo, block_kv):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_qo):
                        scaled_sum_exp[i] = scaled_sum_exp[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Parallel(block_qo, dim_vo):
                acc_o[i, j] /= scaled_sum_exp[i]
            T.copy(acc_o, o_global[bz, bx * block_qo : (bx + 1) * block_qo, by, :])

            for i in T.Parallel(block_qo):
                scaled_sum_exp[i] = T.log2(scaled_sum_exp[i]) + scores_max[i] * scale
            T.copy(scaled_sum_exp, lse_global[bz, by, bx * block_qo : (bx + 1) * block_qo])

    return flash_fwd


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_preprocess(batch, heads, dim):
    dtype = T.bfloat16
    accum_dtype = T.float32
    seq_len = T.dynamic('seq_len')
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        o_global: T.Tensor(shape, dtype),
        grad_o_global: T.Tensor(shape, dtype),
        delta_global: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(
                    o_global[
                        bz,
                        by * blk : (by + 1) * blk,
                        bx,
                        k * blk : (k + 1) * blk,
                    ],
                    o,
                )
                T.copy(
                    grad_o_global[
                        bz,
                        by * blk : (by + 1) * blk,
                        bx,
                        k * blk : (k + 1) * blk,
                    ],
                    do,
                )
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, delta_global[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_bwd(
    batch: int,
    heads: int,
    dim_qk: int,
    dim_vo: int,
    softmax_scale: float,
    mask_fn: Callable[[int, int, int, int, int, int], bool],
    block_mask_fn: Callable[[int, int, int, int, int, int, int, int], bool],
    block_qo: int = 32,
    block_kv: int = 64,
    num_stages: int = 2,
    thread_num: int = 128,
):
    scale = softmax_scale * 1.44269504  # log2(e)
    dtype = T.bfloat16
    accum_dtype = T.float32

    seq_len_qo = T.dynamic('seq_len_qo')
    seq_len_kv = T.dynamic('seq_len_kv')

    @T.prim_func
    def flash_bwd(
        q_global: T.Tensor([batch, seq_len_qo, heads, dim_qk], dtype),
        k_global: T.Tensor([batch, seq_len_kv, heads, dim_qk], dtype),
        v_global: T.Tensor([batch, seq_len_kv, heads, dim_vo], dtype),
        grad_o_global: T.Tensor([batch, seq_len_qo, heads, dim_vo], dtype),
        lse_global: T.Tensor([batch, heads, seq_len_qo], accum_dtype),
        delta_global: T.Tensor([batch, heads, seq_len_qo], accum_dtype),
        grad_q_global: T.Tensor(
            [batch, heads, T.ceildiv(seq_len_qo, block_qo) * block_qo, dim_qk], accum_dtype
        ),
        grad_k_global: T.Tensor([batch, seq_len_kv, heads, dim_qk], dtype),
        grad_v_global: T.Tensor([batch, seq_len_kv, heads, dim_vo], dtype),
        mask_custom_data: T.ptr,
        block_mask_custom_data: T.ptr,
    ):
        with T.Kernel(heads, T.ceildiv(seq_len_kv, block_kv), batch, threads=thread_num) as (
            bx,
            by,
            bz,
        ):
            T.disable_warp_group_reg_alloc()
            K_shared = T.alloc_shared([block_kv, dim_qk], dtype)
            dsT_shared = T.alloc_shared([block_kv, block_qo], dtype)
            q = T.alloc_shared([block_qo, dim_qk], dtype)
            V_shared = T.alloc_shared([block_kv, dim_vo], dtype)
            qkT = T.alloc_fragment([block_kv, block_qo], accum_dtype)
            dsT = T.alloc_fragment([block_kv, block_qo], accum_dtype)
            qkT_cast = T.alloc_shared([block_kv, block_qo], dtype)
            lse_shared = T.alloc_shared([block_qo], accum_dtype)
            delta = T.alloc_shared([block_qo], accum_dtype)
            do = T.alloc_shared([block_qo, dim_vo], dtype)
            dv = T.alloc_fragment([block_kv, dim_vo], accum_dtype)
            dk = T.alloc_fragment([block_kv, dim_qk], accum_dtype)
            dq = T.alloc_fragment([block_qo, dim_qk], accum_dtype)
            dv_shared = T.alloc_shared([block_kv, dim_vo], dtype)
            dk_shared = T.alloc_shared([block_kv, dim_qk], dtype)
            dq_shared = T.alloc_shared([block_qo, dim_qk], accum_dtype)

            T.copy(k_global[bz, by * block_kv : (by + 1) * block_kv, bx, :], K_shared)
            T.copy(v_global[bz, by * block_kv : (by + 1) * block_kv, bx, :], V_shared)
            T.clear(dv)
            T.clear(dk)
            for k in T.Pipelined(0, T.ceildiv(seq_len_qo, block_qo), num_stages=num_stages):
                if block_mask_fn(
                    bz,
                    bx,
                    k * block_qo,
                    T.min(seq_len_qo, (k + 1) * block_qo),
                    by * block_kv,
                    T.min(seq_len_kv, (by + 1) * block_kv),
                    seq_len_qo,
                    seq_len_kv,
                    block_mask_custom_data,
                ):
                    T.copy(q_global[bz, k * block_qo : (k + 1) * block_qo, bx, :], q)
                    T.clear(qkT)
                    T.gemm(K_shared, q, qkT, transpose_B=True)
                    T.copy(grad_o_global[bz, k * block_qo : (k + 1) * block_qo, bx, :], do)
                    T.clear(dsT)
                    T.gemm(V_shared, do, dsT, transpose_B=True)

                    T.copy(lse_global[bz, bx, k * block_qo : (k + 1) * block_qo], lse_shared)
                    for i, j in T.Parallel(block_kv, block_qo):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                    # We don't need to handle OOB positions,
                    # since OOB values won't affect other positions here.
                    for i, j in T.Parallel(block_kv, block_qo):
                        qkT[i, j] = T.if_then_else(
                            mask_fn(
                                bz,
                                bx,
                                k * block_qo + j,
                                by * block_kv + i,
                                seq_len_qo,
                                seq_len_kv,
                                mask_custom_data,
                            ),
                            qkT[i, j],
                            0,
                        )
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv)

                    T.copy(delta_global[bz, bx, k * block_qo : (k + 1) * block_qo], delta)

                    for i, j in T.Parallel(block_kv, block_qo):
                        dsT_shared[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * softmax_scale
                    T.gemm(dsT_shared, q, dk)

                    T.clear(dq)
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                    T.copy(dq, dq_shared)
                    T.atomic_add(
                        grad_q_global[bz, bx, k * block_qo : (k + 1) * block_qo, :],
                        dq_shared,
                    )
            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, grad_v_global[bz, by * block_kv : (by + 1) * block_kv, bx, :])
            T.copy(dk_shared, grad_k_global[bz, by * block_kv : (by + 1) * block_kv, bx, :])

    return flash_bwd


class _FlexAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float,
        mask_fn: Callable[[int, int], bool],
        block_mask_fn: Callable[[int, int, int, int], bool],
        mask_custom_data: torch.Tensor,
        block_mask_custom_data: torch.Tensor,
    ):
        batch, seq_len_qo, head, dim_qk = q.shape
        _, seq_len_kv, _, dim_vo = v.shape
        o = torch.empty((batch, seq_len_qo, head, dim_vo), dtype=q.dtype, device=q.device)
        lse = torch.empty((batch, head, seq_len_qo), dtype=torch.float32, device=q.device)
        mod = flashattn_fwd(
            batch,
            head,
            dim_qk,
            dim_vo,
            softmax_scale,
            mask_fn,
            block_mask_fn,
        )
        mod(
            q,
            k,
            v,
            o,
            lse,
            mask_custom_data,
            block_mask_custom_data,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            o,
            lse,
            mask_custom_data,
            block_mask_custom_data,
        )
        ctx.softmax_scale = softmax_scale
        ctx.mask_fn = mask_fn
        ctx.block_mask_fn = block_mask_fn
        return o, lse

    @staticmethod
    def backward(ctx: torch.autograd.Function, do: torch.Tensor, dlse: None):
        (
            q,
            k,
            v,
            o,
            lse,
            mask_custom_data,
            block_mask_custom_data,
        ) = ctx.saved_tensors
        batch, seq_len_qo, head, dim_qk = q.shape
        _, seq_len_kv, _, dim_vo = v.shape

        do, q, k, v, o = [x.contiguous() for x in (do, q, k, v, o)]

        delta = torch.empty((batch, head, seq_len_qo), dtype=torch.float32, device=q.device)
        flashattn_bwd_preprocess(batch, head, dim_vo)(o, do, delta)

        block_qo = 32
        mod = flashattn_bwd(
            batch,
            head,
            dim_qk,
            dim_vo,
            ctx.softmax_scale,
            ctx.mask_fn,
            ctx.block_mask_fn,
            block_qo=block_qo,
        )
        dq = torch.zeros(
            (batch, head, (seq_len_qo + block_qo - 1) // block_qo * block_qo, dim_qk),
            device=q.device,
            dtype=torch.float32,
        )
        dk = torch.zeros_like(k)
        dv = torch.empty_like(v)
        mod(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq,
            dk,
            dv,
            mask_custom_data,
            block_mask_custom_data,
        )
        dq = dq[:, :, :seq_len_qo, :].transpose(1, 2).bfloat16()  # TODO: tilelang cast
        return dq, dk, dv, *([None] * 5)


flex_attn = _FlexAttn.apply


def ref_program(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    bool_mask: torch.Tensor,
):
    score = torch.einsum('bshd,bthd->bhst', q, k) * softmax_scale
    score = score.masked_fill(~bool_mask, float('-inf'))
    p = torch.softmax(score, dim=-1)
    lse = torch.logsumexp(score, dim=-1) * 1.44269504
    o = torch.einsum('bhst,bthd->bshd', p, v)
    return o, lse


def main(
    BATCH: int,
    H: int,
    N_CTX_QO: int,
    D_HEAD_QK: int,
    N_CTX_KV: int,
    D_HEAD_VO: int,
    mode: str,
):
    if mode == 'causal':
        bool_mask = (
            (N_CTX_QO - torch.arange(N_CTX_QO, device='cuda'))[:, None]
            <= (N_CTX_KV - torch.arange(N_CTX_KV, device='cuda'))[None, :]
        ).expand(BATCH, H, N_CTX_QO, N_CTX_KV)

        @T.macro
        def mask_fn(
            b,
            h,
            q_idx,
            k_idx,
            seq_len_qo,
            seq_len_kv,
            mask_custom_data,
        ):
            return (seq_len_qo - q_idx) <= (seq_len_kv - k_idx)

        @T.macro
        def block_mask_fn(
            b,
            h,
            q_start,
            q_end,
            k_start,
            k_end,
            seq_len_qo,
            seq_len_kv,
            block_mask_custom_data,
        ):
            return (seq_len_qo - q_end - 1) <= (seq_len_kv - k_start)

        mask_custom_data = None
        block_mask_custom_data = None

    elif mode == 'full':
        bool_mask = torch.ones((BATCH, H, N_CTX_QO, N_CTX_KV), dtype=torch.bool, device='cuda')

        @T.macro
        def mask_fn(
            b,
            h,
            q_idx,
            k_idx,
            seq_len_qo,
            seq_len_kv,
            mask_custom_data,
        ):
            return True

        @T.macro
        def block_mask_fn(
            b,
            h,
            q_start,
            q_end,
            k_start,
            k_end,
            seq_len_qo,
            seq_len_kv,
            block_mask_custom_data,
        ):
            return True

        mask_custom_data = None
        block_mask_custom_data = None
    elif mode == 'random':
        bool_mask = torch.rand((BATCH, H, N_CTX_QO, N_CTX_KV), device='cuda') < 0.5

        @T.macro
        def mask_fn(
            b,
            h,
            q_idx,
            k_idx,
            seq_len_qo,
            seq_len_kv,
            mask_custom_data,
        ):
            mask = T.make_tensor(
                mask_custom_data,
                shape=(BATCH, H, seq_len_qo, seq_len_kv),
                dtype=T.bool,
            )
            return mask[b, h, q_idx, k_idx]

        @T.macro
        def block_mask_fn(
            b,
            h,
            q_start,
            q_end,
            k_start,
            k_end,
            seq_len_qo,
            seq_len_kv,
            block_mask_custom_data,
        ):
            block_mask = T.make_tensor(
                block_mask_custom_data,
                shape=(BATCH, H, T.ceildiv(seq_len_qo, 128), T.ceildiv(seq_len_kv, 128)),
                dtype=T.bool,
            )
            return T.if_then_else(
                q_start // 128 == (q_end - 1) // 128 and k_start // 128 == (k_end - 1) // 128,
                block_mask[b, h, q_start // 128, k_start // 128],
                True,
            )

        mask_custom_data = bool_mask
        block_mask_custom_data = (
            bool_mask.unflatten(-2, (-1, 128)).unflatten(-1, (-1, 128)).any(dim=(-1, -3))
        )

    Q = (
        torch.empty(BATCH, N_CTX_QO, H, D_HEAD_QK, dtype=torch.bfloat16, device='cuda')
        .normal_()
        .requires_grad_()
    )
    K = (
        torch.empty(BATCH, N_CTX_KV, H, D_HEAD_QK, dtype=torch.bfloat16, device='cuda')
        .normal_()
        .requires_grad_()
    )
    V = (
        torch.empty(BATCH, N_CTX_KV, H, D_HEAD_VO, dtype=torch.bfloat16, device='cuda')
        .normal_()
        .requires_grad_()
    )
    dO = (
        torch.empty(BATCH, N_CTX_QO, H, D_HEAD_VO, dtype=torch.bfloat16, device='cuda')
        .normal_()
        .requires_grad_()
    )
    O, lse = flex_attn(
        Q,
        K,
        V,
        D_HEAD_QK**-0.5,
        mask_fn,
        block_mask_fn,
        mask_custom_data,
        block_mask_custom_data,
    )
    O.backward(dO)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None

    Q_ref = Q.float().detach().requires_grad_()
    K_ref = K.float().detach().requires_grad_()
    V_ref = V.float().detach().requires_grad_()
    O_ref, lse_ref = ref_program(Q_ref, K_ref, V_ref, D_HEAD_QK**-0.5, bool_mask)
    O_ref.backward(dO)
    dQ_ref, Q_ref.grad = Q_ref.grad.clone(), None
    dK_ref, K_ref.grad = K_ref.grad.clone(), None
    dV_ref, V_ref.grad = V_ref.grad.clone(), None

    torch.testing.assert_close(lse, lse_ref, rtol=3e-6, atol=1e-5)
    torch.testing.assert_close(O.float(), O_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dV.float(), dV_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dK.float(), dK_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dQ.float(), dQ_ref, rtol=1e-2, atol=1e-2)
    print('All checks passed.✅')

    def run():
        Q.grad = K.grad = V.grad = None
        flex_attn(
            Q,
            K,
            V,
            D_HEAD_QK**-0.5,
            mask_fn,
            block_mask_fn,
            mask_custom_data,
            block_mask_custom_data,
        )[0].backward(dO)

    with torch.profiler.profile() as prof:
        for _ in range(5):
            run()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--h', type=int, default=16, help='Number of heads')
    parser.add_argument('--n_ctx_qo', type=int, default=4096, help='Context size')
    parser.add_argument('--d_head_qk', type=int, default=192, help='Head dimension')
    parser.add_argument('--n_ctx_kv', type=int, default=8192, help='Context size')
    parser.add_argument('--d_head_vo', type=int, default=128, help='Head dimension')
    parser.add_argument('--mask_mode', type=str, default='random', help='Causal flag')
    args = parser.parse_args()
    main(
        args.batch,
        args.h,
        args.n_ctx_qo,
        args.d_head_qk,
        args.n_ctx_kv,
        args.d_head_vo,
        args.mask_mode,
    )
