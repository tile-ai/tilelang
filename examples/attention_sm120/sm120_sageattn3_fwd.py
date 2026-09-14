"""SM120 SageAttention3-style NVFP4 attention forward in TileLang (v1: P goes through shared memory).

Algorithm (thu-ml/SageAttention `sageattention3_blackwell`, paper arXiv:2505.11594): Q, K, V are
NVFP4 (e2m1 + per-16-element e4m3 scales; K smoothed by its sequence mean, Q by its 128-row block
mean with the compensation term delta_s = q_mean @ K^T), S = FP4MM(Q, K) + delta_s, online softmax
with the two-level P quantization (P2 = P * 448 * 6 folded into exp2, per-16-key e4m3 scale
= e4m3_rn(max16(P2) / 6), P-hat = e2m1_rn(P2 / scale)), O += FP4MM(P-hat, V^T); O /= rowsum(P2).

Data contract (see sageattn3_quant.py): K rows are permuted inside every 32-key group
(PERM32) so that, under the m16n8k64 accumulator layout, every lane holds 8 consecutive original
keys and the C-fragment -> A-fragment relabeling is the identity; V^T and P therefore stay in the
original key order. Inputs: packed fp4 (2 elem/byte, low nibble first) and row-major uint32 scale
words (4 e4m3 bytes = 4 consecutive 16-groups, LSB first).

kernel-only usage: python sm120_sageattn3_fwd.py --seq 4096 --verify --bench
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

import tilelang
import tilelang.language as T
from tilelang.layout import Fragment
from tilelang.profiler import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402

LOG2E = 1.4426950408889634
LOG2_P1 = math.log2(1.0 / (448.0 * 6.0))  # two-level: P2 = P * 2688 (softmax_fused.h)
LOG2_FP4 = math.log2(1.0 / 6.0)  # per-16 scale = max16(P2) / 6

# One fp32 -> one e2m1 nibble (low 4 bits of the result), x divided by the block scale first,
# via the same cvt.rn.satfinite.e2m1x2.f32 the original uses (utils.h packed_float_to_e2m1).
CVT_E2M1_SRC = r"""
__device__ __forceinline__ unsigned int tl_cvt_e2m1_rn_div(float x, float s) {
  unsigned int out;
  asm volatile(
      "{\n"
      ".reg .b8 b;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b, %1, %2;\n"
      "cvt.u32.u8 %0, b;\n"
      "}"
      : "=r"(out)
      : "f"(0.0f), "f"(x / s));
  return out & 0xFu;
}
"""


def _p_group_layout(block_m: int, n_groups: int, warp_rows: int = 32) -> Fragment:
    """Fragment layout of G[row, key_group, key_in_group] (P2 in original key order).

    Thread mapping = the m16n8k64 e2m1 A-operand fragment (FullRow warps of ``warp_rows`` rows):
    for key = 16*group + k, lane 4*(row%8) + (key%32)//8 holds keys 8q..8q+7 (+32) of rows row%8
    and row%8+8. Index = position inside the per-lane A fragment (k64 atom outermost, then the
    16-row atom, then row-half, k-half, key%8), matching make_mma_load_layout's repeat order.
    Written with per-variable floor-divs so the layout checker can prove bijectivity.
    """

    def fwd_thread(i, m, k):
        return (i // warp_rows) * 32 + (i % 8) * 4 + (m % 2) * 2 + k // 8

    def fwd_index(i, m, k):
        return (m // 4) * (warp_rows * 2) + ((i % warp_rows) // 16) * 32 + ((i % 16) // 8) * 8 + ((m % 4) // 2) * 16 + k % 8

    return Fragment((block_m, n_groups, 16), forward_thread_fn=fwd_thread, forward_index_fn=fwd_index)


def _p_word_layout(block_m: int, n_groups: int, warp_rows: int = 32) -> Fragment:
    """Fragment layout of Pw[row, key_group, k8]: the uint32 holding keys 8*k8..8*k8+7 (= G index // 8)."""

    def fwd_thread(i, m, k8):
        return (i // warp_rows) * 32 + (i % 8) * 4 + (m % 2) * 2 + k8

    def fwd_index(i, m, k8):
        return (m // 4) * (warp_rows // 4) + ((i % warp_rows) // 16) * 4 + ((i % 16) // 8) + ((m % 4) // 2) * 2

    return Fragment((block_m, n_groups, 2), forward_thread_fn=fwd_thread, forward_index_fn=fwd_index)


DEFAULT_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    # The automatic producer/consumer warp specialization currently drops the fragment
    # reduce / exp2 / pack stages of this kernel (pass bug under investigation); the simple
    # form is used until that is fixed.
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def build_sm120_sageattn3_fwd(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int = 128,
    block_M: int = 128,
    block_N: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    out_dtype=T.bfloat16,
):
    assert dim == 128, "v1 supports head_dim = 128"
    assert block_M == 128 and block_N == 128, "v1 tile contract: 128x128"
    n_warps = threads // 32
    warp_rows = block_M // n_warps  # FullRow policy: every warp owns warp_rows full rows
    assert warp_rows in (16, 32), "FullRow warp partition must give 16 or 32 rows per warp (threads 256 or 128)"
    assert seq_len % block_N == 0, "seq_len must be a multiple of 128 (pad like the original)"

    fp4 = T.float4_e2m1fn
    accum = T.float32
    sf_words_qk = dim // 64
    sf_words_pv = block_N // 64
    n_groups = block_N // 16
    softmax_scale = dim**-0.5
    sl2 = softmax_scale * LOG2E
    n_kv_blocks = seq_len // block_N
    g_layout = _p_group_layout(block_M, n_groups, warp_rows)
    pw_layout = _p_word_layout(block_M, n_groups, warp_rows)

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_len, dim), fp4),
        K: T.Tensor((batch, heads, seq_len, dim), fp4),
        VT: T.Tensor((batch, heads, dim, seq_len), fp4),
        SFQ: T.Tensor((batch, heads, seq_len, sf_words_qk), T.uint32),
        SFK: T.Tensor((batch, heads, seq_len, sf_words_qk), T.uint32),
        SFV: T.Tensor((batch, heads, dim, seq_len // 64), T.uint32),
        DS: T.Tensor((batch, heads, seq_len // block_M, seq_len), accum),
        O: T.Tensor((batch, heads, seq_len, dim), out_dtype),
    ):
        with T.Kernel(seq_len // block_M, heads, batch, threads=threads) as (bx, by, bz):
            T.import_source(CVT_E2M1_SRC)
            Q_sh = T.alloc_shared((block_M, dim), fp4)
            K_sh = T.alloc_shared((block_N, dim), fp4)
            V_sh = T.alloc_shared((dim, block_N), fp4)
            P_sh = T.alloc_shared((block_M, block_N), fp4)
            P_u32 = T.view(P_sh, (block_M, block_N // 8), dtype=T.uint32)
            SFQ_sh = T.alloc_shared((block_M, sf_words_qk), T.uint32)
            SFK_sh = T.alloc_shared((block_N, sf_words_qk), T.uint32)
            SFV_sh = T.alloc_shared((dim, sf_words_pv), T.uint32)
            SFP_sh = T.alloc_shared((block_M, sf_words_pv), T.uint32)
            SFP_u8 = T.view(SFP_sh, (block_M, n_groups), dtype=T.uint8)
            DS_sh = T.alloc_shared((block_N,), accum)
            O_sh = T.alloc_shared((block_M, dim), out_dtype)

            acc_s = T.alloc_fragment((block_M, block_N), accum)
            G = T.alloc_fragment((block_M, n_groups, 16), accum)
            smax = T.alloc_fragment((block_M, n_groups), accum)
            absmax = T.alloc_fragment((block_M, n_groups), accum)
            rowsum8 = T.alloc_fragment((block_M, n_groups), accum)
            Pw = T.alloc_fragment((block_M, n_groups, 2), T.uint32)
            acc_o = T.alloc_fragment((block_M, dim), accum)
            m_i = T.alloc_fragment((block_M,), accum)
            m_prev = T.alloc_fragment((block_M,), accum)
            ms = T.alloc_fragment((block_M,), accum)
            rescale = T.alloc_fragment((block_M,), accum)
            l_tile = T.alloc_fragment((block_M,), accum)
            l_i = T.alloc_fragment((block_M,), accum)
            T.annotate_layout({G: g_layout, Pw: pw_layout})

            T.copy(Q[bz, by, bx * block_M, 0], Q_sh)
            for r, w in T.Parallel(block_M, sf_words_qk):
                SFQ_sh[r, w] = SFQ[bz, by, bx * block_M + r, w]
            T.fill(acc_o, 0)
            T.fill(l_i, 0)
            for i in T.Parallel(block_M):  # explicit (T.fill with -inf was dropped by lowering)
                m_i[i] = -T.infinity(accum)

            for kt in T.Pipelined(n_kv_blocks, num_stages=num_stages):
                T.copy(K[bz, by, kt * block_N, 0], K_sh)
                T.copy(VT[bz, by, 0, kt * block_N], V_sh)
                for r, w in T.Parallel(block_N, sf_words_qk):
                    SFK_sh[r, w] = SFK[bz, by, kt * block_N + r, w]
                for r, w in T.Parallel(dim, sf_words_pv):
                    SFV_sh[r, w] = SFV[bz, by, r, kt * sf_words_pv + w]
                T.copy(DS[bz, by, bx, kt * block_N : (kt + 1) * block_N], DS_sh)

                # S = delta_s (broadcast over rows) + FP4MM(Q, K^T). S columns are in the
                # permuted-K order, delta_s is indexed by original key: o(j) = 32*(j//32) + PERM32[j%32].
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = DS_sh[(j // 32) * 32 + ((j % 32) // 8) * 2 + ((j % 8) // 2) * 8 + j % 2]
                T.mma_gemm_blockscaled(
                    Q_sh,
                    K_sh,
                    acc_s,
                    SFQ_sh,
                    SFK_sh,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=False,
                    k_start=0,
                    sf_a_granularity_k=16,
                    sf_b_granularity_k=16,
                    sf_layout="rowmajor",
                    scale_dtype="ue4m3",
                )

                # Regroup S columns (permuted-K order) into original-key 16-groups: with the K
                # permutation this is a pure in-lane register renaming (no shuffles).
                for i, kk, hi, q1, q0, n, jj in T.Parallel(block_M, 2, 2, 2, 2, 4, 2):
                    G[i, 4 * kk + 2 * hi + q1, 8 * q0 + 2 * n + jj] = acc_s[i, 64 * kk + 32 * hi + 8 * n + 4 * q1 + 2 * q0 + jj]

                # Online softmax with the fused per-16 max (paper's "reuse shuffle").
                T.reduce_max(G, smax, dim=2, clear=True)
                T.copy(m_i, m_prev)
                T.reduce_max(smax, m_i, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    ms[i] = m_i[i] * sl2 + LOG2_P1
                    rescale[i] = T.exp2((m_prev[i] - m_i[i]) * sl2)  # same expression order as the original
                for i, m, k in T.Parallel(block_M, n_groups, 16):
                    G[i, m, k] = T.exp2(G[i, m, k] * sl2 - ms[i])
                for i, m in T.Parallel(block_M, n_groups):
                    absmax[i, m] = T.exp2(smax[i, m] * sl2 - ms[i] + LOG2_FP4)
                T.reduce_sum(G, rowsum8, dim=2, clear=True)
                T.reduce_sum(rowsum8, l_tile, dim=1, clear=True)

                # P quantization: e4m3 scale bytes (RN satfinite) and e2m1 nibbles packed 8 per
                # uint32 (one lane owns 8 consecutive keys -> whole-word smem writes).
                for i, m in T.Parallel(block_M, n_groups):
                    SFP_u8[i, m] = T.reinterpret(T.cast(absmax[i, m], T.float8_e4m3fn), T.uint8)
                for i, m, k8 in T.Parallel(block_M, n_groups, 2):
                    Pw[i, m, k8] = 0
                    for t in T.serial(8):
                        Pw[i, m, k8] = Pw[i, m, k8] | T.shift_left(
                            T.call_extern("uint32", "tl_cvt_e2m1_rn_div", G[i, m, 8 * k8 + t], absmax[i, m]),
                            T.cast(4 * t, T.uint32),
                        )
                for i, m, k8 in T.Parallel(block_M, n_groups, 2):
                    P_u32[i, 2 * m + k8] = Pw[i, m, k8]

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= rescale[i]
                T.mma_gemm_blockscaled(
                    P_sh,
                    V_sh,
                    acc_o,
                    SFP_sh,
                    SFV_sh,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=False,
                    k_start=0,
                    sf_a_granularity_k=16,
                    sf_b_granularity_k=16,
                    sf_layout="rowmajor",
                    scale_dtype="ue4m3",
                )
                for i in T.Parallel(block_M):
                    l_i[i] = l_i[i] * rescale[i] + l_tile[i]

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] = acc_o[i, j] / l_i[i]
            T.copy(acc_o, O_sh)
            T.copy(O_sh, O[bz, by, bx * block_M, 0])

    return main


def sm120_sageattn3_fwd(*args, pass_configs: dict | None = None, **kwargs):
    """JIT-compiled kernel; ``pass_configs`` extends DEFAULT_PASS_CONFIGS (experiments/autotune)."""
    cfg = dict(DEFAULT_PASS_CONFIGS)
    if pass_configs:
        cfg.update(pass_configs)
    return tilelang.jit(out_idx=[7], pass_configs=cfg)(build_sm120_sageattn3_fwd)(*args, **kwargs)


# --------------------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------------------
def prepare_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """bf16 [B,H,N,D] -> (canonical dict, TileLang kernel args tuple, delta_s)."""
    q_sm, k_sm, v_p, _, delta_s = sq.preprocess_qkv(q, k, v)
    canon = sq.quantize_canonical(q_sm, k_sm, v_p)
    t = sq.export_for_tilelang(canon)
    args = (
        t["q"].view(torch.int8),
        t["k"].view(torch.int8),
        t["vt"].view(torch.int8),
        t["sfq"],
        t["sfk"],
        t["sfv"],
        delta_s,
    )
    return canon, args, delta_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--dump-source", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enable-ws", action="store_true", help="re-enable auto warp specialization (currently broken)")
    args = ap.parse_args()
    pass_cfg = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False} if args.enable_ws else {}
    torch.manual_seed(args.seed)
    b, h, n, d = args.batch, args.heads, args.seq, args.dim
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    canon, kargs, delta_s = prepare_inputs(q, k, v)
    kernel = sm120_sageattn3_fwd(b, h, n, d, num_stages=args.num_stages, threads=args.threads, pass_configs=pass_cfg)
    if args.dump_source:
        print(kernel.get_kernel_source())
    o = kernel(*kargs)
    torch.cuda.synchronize()
    if args.verify:
        o_ref = sq.reference_attention(q, k, v, d**-0.5)
        o_gold = sq.golden_attention(canon, delta_s, d**-0.5)
        print(
            f"TL vs golden: align_ratio={sq.alignment_ratio(o.float(), o_gold, o_ref):.4f} max|diff|={float((o.float() - o_gold).abs().max()):.3e}"
        )
        print(f"TL vs fp64 ref: cos={sq.cos_sim(o.float(), o_ref):.6f} L1={sq.rel_l1(o.float(), o_ref):.4f}")
        try:
            import fp4attn_cuda

            s = sq.export_for_sage(canon)
            o_sage, _ = fp4attn_cuda.fwd(s["q"], s["k"], s["v"], s["sfq"], s["sfk"], s["sfv"], delta_s, n, None, d**-0.5, False, True, True)
            print(
                f"TL vs original kernel: align_ratio={sq.alignment_ratio(o.float(), o_sage.float(), o_ref):.4f} max|diff|={float((o.float() - o_sage.float()).abs().max()):.3e}  (original vs ref cos={sq.cos_sim(o_sage.float(), o_ref):.6f})"
            )
        except ImportError:
            print("original kernel not installed; skipped TL-vs-original")
    if args.bench:
        ms = do_bench(lambda: kernel(*kargs), warmup=25, rep=100, backend="cudagraph", return_mode="median")
        tops = 4.0 * b * h * n * n * d / (ms * 1e-3) / 1e12
        print(f"kernel-only: {ms:.4f} ms  {tops:.1f} TOPS  (B={b} H={h} N={n} D={d})")


if __name__ == "__main__":
    main()
