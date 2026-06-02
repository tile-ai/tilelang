"""TileLang UMA 2CTA FlashAttention kernel matching TileScale fa4_uma.cu.

This is the second CUDA-parity target.  The TileLang DSL owns the launch,
persistent cluster tile loop, memory/barrier topology, and role dispatch.  The
initial role bodies are lowered through UMA-specific builtins in copy_sm100.h so
we can first match the reference structure, then split the helpers into smaller
generic primitives once correctness and performance are stable.
"""

import argparse
from typing import Optional

import torch
import tilelang
import tilelang.layout
import tilelang.language as T
from tilelang.carver.arch import driver


PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    "tl.disable_thread_storage_sync": True,
    "tl.outline_warp_spec_branches": True,
}


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG, target="cuda -arch=sm_100")
def attention_kernel_uma(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    num_kv_heads: Optional[int] = None,
    is_causal: bool = False,
):
    if dim != 128:
        raise ValueError("attention_kernel_uma currently supports head_dim=128 only")
    if is_causal:
        raise ValueError("attention_kernel_uma currently implements non-causal attention only")
    if num_kv_heads is None:
        num_kv_heads = heads
    if heads % num_kv_heads != 0:
        raise ValueError(f"heads={heads} must be divisible by num_kv_heads={num_kv_heads}")

    block_m = 256
    block_m_cta = 128
    block_n = 128
    page_rows = 32
    q_stages = 2
    kv_stages = 3
    b_per_cta = 64
    tile_cols = 64
    threads = 512
    q_rows_per_cluster = q_stages * block_m
    q_tiles = T.ceildiv(seq_len, q_rows_per_cluster)
    total_tiles = q_tiles * heads * batch
    sm_num = driver.get_num_sms()
    grid = T.max(2, (T.min(total_tiles * 2, sm_num) // 2) * 2)
    total_clusters = grid // 2
    loop_extent = T.ceildiv(seq_len, block_n)
    scale_log2 = (1.0 / dim) ** 0.5 * 1.44269504089

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, num_kv_heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    q_stage_elems = block_m_cta * dim
    kv_stage_elems = b_per_cta * dim

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(grid, threads=threads, cluster_dims=2) as block_id:
            # Match fa4_uma.cu launch bounds while relying on runtime
            # setmaxnreg.inc/dec for role-specific register donation.
            T.annotate_min_blocks_per_sm(1)
            T.use_2cta_tmem()

            Q_shared = T.alloc_shared([q_stages, block_m_cta, dim], dtype)
            O_shared = T.alloc_shared([q_stages, block_m_cta, dim], dtype)
            K_shared = T.alloc_shared([kv_stages, b_per_cta, dim], dtype)
            V_shared = T.alloc_shared([kv_stages, b_per_cta, dim], dtype)
            # fa4_uma overlaps sum_smem with the front of rs_smem after the
            # per-kb rescale loop to stay under the SM100 shared-memory cap.
            rs_shared = T.alloc_shared([2, q_stages, block_m_cta], accum_dtype)

            Base_tmem = T.alloc_tmem([block_m_cta, 512], accum_dtype)
            S0_tmem = T.alloc_tmem([block_m_cta, block_n], accum_dtype, alias=Base_tmem, col_offset=0)
            P0_tmem = T.alloc_tmem([block_m_cta, block_n], dtype, alias=Base_tmem, col_offset=64)
            S1_tmem = T.alloc_tmem([block_m_cta, block_n], accum_dtype, alias=Base_tmem, col_offset=128)
            P1_tmem = T.alloc_tmem([block_m_cta, block_n], dtype, alias=Base_tmem, col_offset=192)
            O0_tmem = T.alloc_tmem([block_m_cta, dim], accum_dtype, alias=Base_tmem, col_offset=256)
            O1_tmem = T.alloc_tmem([block_m_cta, dim], accum_dtype, alias=Base_tmem, col_offset=384)

            base_layout = T.Layout([block_m_cta, 512], lambda i, j: [i, j])
            score_layout = T.Layout([block_m_cta, block_n], lambda i, j: [i, j])
            output_layout = T.Layout([block_m_cta, dim], lambda i, j: [i, j])
            T.annotate_layout({
                Q_shared: tilelang.layout.make_full_bank_swizzled_layout(Q_shared),
                O_shared: tilelang.layout.make_full_bank_swizzled_layout(O_shared),
                K_shared: tilelang.layout.make_full_bank_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_full_bank_swizzled_layout(V_shared),
                Base_tmem: base_layout,
                S0_tmem: score_layout,
                P0_tmem: score_layout,
                S1_tmem: score_layout,
                P1_tmem: score_layout,
                O0_tmem: output_layout,
                O1_tmem: output_layout,
            })

            mb_q = T.alloc_cluster_barrier([2] * q_stages)
            mb_k = T.alloc_cluster_barrier([2] * kv_stages)
            mb_s = T.alloc_cluster_barrier([1] * q_stages)
            mb_p = T.alloc_cluster_barrier([256] * q_stages)
            mb_p2 = T.alloc_cluster_barrier([256] * q_stages)
            mb_v = T.alloc_cluster_barrier([2] * kv_stages)
            mb_k_rel = T.alloc_cluster_barrier([1] * kv_stages)
            mb_v_rel = T.alloc_cluster_barrier([1] * kv_stages)
            mb_pv = T.alloc_cluster_barrier([1] * kv_stages)
            mb_corr = T.alloc_cluster_barrier([256] * q_stages)
            mb_epi = T.alloc_barrier([block_m_cta] * q_stages)
            mb_q_rel = T.alloc_cluster_barrier(1)
            mb_o_rel = T.alloc_barrier(1)
            mb_o_tmem_rel = T.alloc_cluster_barrier(256)

            tid = T.alloc_var(T.int32, T.get_thread_binding())
            warp = T.alloc_var(T.int32, tid // 32)
            warp_group = T.alloc_var(T.int32, warp // 4)
            cta_rank = T.alloc_var(T.int32, T.block_rank_in_cluster())
            cluster_id = T.alloc_var(T.int32, T.cluster_id_x())
            T.assume(cta_rank < 2)

            for tile_iter in T.serial(T.ceildiv(total_tiles, total_clusters)):
                tile_id = tile_iter * total_clusters + cluster_id
                if tile_id < total_tiles:
                    tile_phase = tile_iter & 1
                    tile_k_base = tile_iter * loop_extent
                    tile_mb = T.truncmod(tile_id, q_tiles)
                    tile_head = T.truncmod(T.truncdiv(tile_id, q_tiles), heads)
                    tile_batch = T.truncdiv(tile_id, q_tiles * heads)
                    tile_ms = tile_mb * q_rows_per_cluster + cta_rank * block_m_cta
                    tile_kv_head = T.truncdiv(tile_head * num_kv_heads, heads)
                    tile_q_col_base = tile_head * dim
                    tile_kv_col_base = tile_kv_head * dim
                    tile_q_row_base = tile_batch * seq_len + tile_ms
                    tile_kv_row_base = tile_batch * seq_len
                    tile_v_col_base = tile_kv_col_base + cta_rank * b_per_cta
                    if warp_group == 0:
                        T.set_max_nreg(168, 1)
                        T.tcgen05_fa4_uma_softmax_warp_2cta(
                            S0_tmem[0, 0],
                            P0_tmem[0, 0],
                            T.access_ptr(rs_shared, "w"),
                            T.access_ptr(rs_shared, "w"),
                            mb_s[0],
                            mb_p[0],
                            mb_p2[0],
                            loop_extent,
                            seq_len,
                            tile_k_base,
                            scale_log2,
                        )
                    elif warp_group == 1:
                        T.set_max_nreg(168, 1)
                        T.tcgen05_fa4_uma_softmax_warp_2cta(
                            S1_tmem[0, 0],
                            P1_tmem[0, 0],
                            T.access_ptr(rs_shared, "w"),
                            T.access_ptr(rs_shared, "w"),
                            mb_s[1],
                            mb_p[1],
                            mb_p2[1],
                            loop_extent,
                            seq_len,
                            tile_k_base,
                            scale_log2,
                        )
                    elif warp_group == 2:
                        T.set_max_nreg(96, 0)
                        T.tcgen05_fa4_uma_correction_warp_2cta(
                            T.access_ptr(O_shared, "w"),
                            T.access_ptr(rs_shared, "r"),
                            T.access_ptr(rs_shared, "r"),
                            mb_corr[0],
                            mb_pv[0],
                            mb_epi[0],
                            mb_o_rel,
                            mb_o_tmem_rel,
                            O0_tmem[0, 0],
                            loop_extent,
                            tile_k_base,
                            tile_phase,
                        )
                    elif warp == 12:
                        T.set_max_nreg(80, 0)
                        T.tcgen05_fa4_uma_mma_warp_2cta(
                            cta_rank,
                            T.access_ptr(Q_shared, "r"),
                            T.access_ptr(K_shared, "r"),
                            T.access_ptr(V_shared, "r"),
                            mb_q[0],
                            mb_k[0],
                            mb_s[0],
                            mb_p[0],
                            mb_p2[0],
                            mb_v[0],
                            mb_pv[0],
                            mb_corr[0],
                            mb_k_rel[0],
                            mb_v_rel[0],
                            mb_q_rel,
                            mb_o_tmem_rel,
                            S0_tmem[0, 0],
                            loop_extent,
                            tile_k_base,
                            tile_phase,
                        )
                    elif warp == 13:
                        T.set_max_nreg(80, 0)
                        q_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(Q, "r"),
                            heads * dim, batch * seq_len,
                            2, heads * dim * 2,
                            tile_cols, block_m_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        k_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(K, "r"),
                            num_kv_heads * dim, batch * seq_len,
                            2, num_kv_heads * dim * 2,
                            tile_cols, b_per_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        v_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(V, "r"),
                            num_kv_heads * dim, batch * seq_len,
                            2, num_kv_heads * dim * 2,
                            tile_cols, b_per_cta,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        T.tcgen05_fa4_uma_producer_warp_2cta(
                            q_desc,
                            k_desc,
                            v_desc,
                            T.access_ptr(Q_shared, "w"),
                            T.access_ptr(K_shared, "w"),
                            T.access_ptr(V_shared, "w"),
                            mb_q[0],
                            mb_k[0],
                            mb_v[0],
                            mb_k_rel[0],
                            mb_v_rel[0],
                            mb_q_rel,
                            loop_extent,
                            tile_k_base,
                            tile_q_row_base,
                            tile_kv_row_base,
                            tile_q_col_base,
                            tile_kv_col_base,
                            tile_v_col_base,
                            tile_phase,
                        )
                    elif warp == 14:
                        T.set_max_nreg(80, 0)
                        output_desc = T.create_tma_descriptor(
                            9, 2, T.access_ptr(Output, "w"),
                            heads * dim, batch * seq_len,
                            2, heads * dim * 2,
                            tile_cols, page_rows,
                            1, 1,
                            0, 3, 2, 0,
                        )
                        T.tcgen05_fa4_uma_epilogue_warp_2cta(
                            T.access_ptr(O_shared, "r"),
                            output_desc,
                            mb_epi[0],
                            mb_o_rel,
                            tile_ms,
                            tile_batch,
                            tile_q_col_base,
                            tile_phase,
                            seq_len,
                        )
                    else:
                        T.set_max_nreg(80, 0)
                        T.evaluate(0)

            if warp == 12:
                T.deallocate_tmem(Base_tmem)

    return main


def reference_attention(Q, K, V):
    Q_f = Q.to(torch.float32)
    K_f = K.to(torch.float32)
    V_f = V.to(torch.float32)
    if Q_f.shape[2] != K_f.shape[2]:
        groups = Q_f.shape[2] // K_f.shape[2]
        K_f = K_f.repeat_interleave(groups, dim=2)
        V_f = V_f.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bshd,bthd->bhst", Q_f, K_f) * (1.0 / Q.shape[-1] ** 0.5)
    return torch.einsum("bhst,bthd->bshd", scores.softmax(dim=-1), V_f).to(Q.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=None)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--skip_ref", action="store_true")
    ap.add_argument("--print_source", action="store_true")
    ap.add_argument("--compile_only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    kv_heads = args.kv_heads or args.heads
    fn = attention_kernel_uma(
        args.batch,
        args.heads,
        args.seq,
        args.dim,
        num_kv_heads=kv_heads,
    )
    if args.print_source:
        print(fn.get_kernel_source())
    if args.compile_only:
        return

    Q = torch.randn(args.batch, args.seq, args.heads, args.dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(args.batch, args.seq, kv_heads, args.dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(args.batch, args.seq, kv_heads, args.dim, dtype=torch.bfloat16, device="cuda")

    O = fn(Q, K, V)
    if args.skip_ref:
        torch.cuda.synchronize()
        print(f"shape={tuple(O.shape)}  reference=skipped")
    else:
        O_ref = reference_attention(Q, K, V)
        err_abs = (O.to(torch.float32) - O_ref.to(torch.float32)).abs()
        print(f"shape={tuple(O.shape)}  max_abs={err_abs.max().item():.4f}  mean_abs={err_abs.mean().item():.4f}")

    if args.bench:
        from tilelang.profiler import do_bench

        for _ in range(3):
            _ = fn(Q, K, V)
        torch.cuda.synchronize()
        lat = do_bench(lambda: fn(Q, K, V), warmup=25, rep=100)
        flops = 2.0 * 2.0 * args.batch * args.heads * args.seq * args.seq * args.dim
        print(f"latency={lat:.3f} ms  perf={flops / lat * 1e-9:.2f} TFLOPS")


if __name__ == "__main__":
    main()
