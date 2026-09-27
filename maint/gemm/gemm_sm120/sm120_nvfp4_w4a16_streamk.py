"""SM120 W4A16 NVFP4 GEMM: persistent stream-K kernel and per-GPU tuned dispatch (maintenance path).

Extends ``examples/gemm_sm120/sm120_nvfp4_w4a16_gemm.py`` (same prepared weight layout, register dequantization,
fragment and tile kernels) with:

* ``nvfp4_w4a16_streamk_kernel``: ``G = num_SMs * c`` CTAs walk contiguous ranges of the (tile, k-tile) iteration
  space (stream-K, no wave quantization), with a hand-written cp.async stage loader (one barrier per k tile);
* ``get_config``: tables tuned per GPU (keyed by SM count; today only a 188-SM RTX PRO 6000 Blackwell Max-Q and the
  linear-layer shapes of a 27B-class model) and, for every other GPU or shape, a heuristic that uses stream-K above
  96 rows;
* ``nvfp4_w4a16_linear``: the same interface as the example's, dispatching FRAG, FragConfig, TileConfig and
  StreamKConfig.

``benchmark_sm120_nvfp4_w4a16_gemm.py`` in this directory times this path.

Limits: the stream-K loader uses 32-bit byte offsets (x and the packed weight must each be below 4 GiB, checked on the
host); the split-K tickets are per device, so launches on one device must not run concurrently on two streams.
"""

import math
from typing import NamedTuple

import torch

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

from examples.gemm_sm120 import sm120_nvfp4_w4a16_gemm as base
from examples.gemm_sm120.sm120_nvfp4_w4a16_gemm import (  # noqa: F401 - re-exported for callers of this module
    FRAG,
    MAX_M,
    FragConfig,
    NVFP4W4A16Weight,
    TileConfig,
    is_supported_device,
    prepare_nvfp4_weight,
)

# ---------------------------------------------------------------------------------------------------------------
# Stream-K kernel (block_K = 64): a grid of G CTAs, G a run-time value. The (tile, k-tile) iteration space (tile-major;
# tile t = (m tile t % m_tiles, n tile t // m_tiles), so the m tiles of one weight column are adjacent and a weight
# tile is read from DRAM once) is cut into G contiguous ranges [g*TT/G, (g+1)*TT/G). A CTA walks its range tile by
# tile; a tile it covers completely is written straight to Y, a partial segment goes to its own FP32 slot
# P[2g + last] (last = the segment ends the CTA's range; a CTA has at most one first and one last partial segment).
# The last CTA to arrive at a split tile (atomic ticket) sums the slots of CTAs g0..g1 in that fixed order.
# The k loop is a hand-written cp.async pipeline: a C++ stage loader (generated per config) writes the A tile (same
# 128-byte XOR swizzle as make_mma_swizzle_layout), packed weights and scales to raw shared addresses, so TileLang sees
# only shared reads in the loop and inserts no barriers of its own: one commit group per stage (T.ptx_commit_group),
# constant wait counts (T.ptx_wait_group), one __syncthreads per k tile. T.Pipelined cannot express this loop: the k
# range of each tile is a run-time value.


def _streamk_loader_src(block_M, nblk, K, threads, block_K=64):
    rowch = block_K // 8  # 16-byte chunks per A row
    kch = block_K // 32  # 32-wide k chunks (one 512-byte weight block + 64 scale bytes each)
    a_n, w_n, s_n = block_M * rowch, nblk * kch * 32, nblk * kch * 4  # 16-byte copies per stage
    assert block_K in (32, 64) and s_n <= threads
    # per-thread copy loops; a trailing partial round (fewer copies than threads) is guarded
    a_it, a_g = -(-a_n // threads), ("" if a_n % threads == 0 else f"if (e < {a_n}u) ")
    w_it, w_g = -(-w_n // threads), ("" if w_n % threads == 0 else f"if (e < {w_n}u) ")
    # make_mma_swizzle_layout for bf16 rows of 128 B (block_K 64): chunk ^ (row % 8); of 64 B: chunk ^ (row % 8) / 2
    swz = "(ch ^ (row & 7))" if block_K == 64 else "(ch ^ ((row & 7) >> 1))"
    return (
        base._DEQUANT_SRC
        + f"""
__device__ __forceinline__ unsigned nvfp4_w4a16_saddr(const void *p) {{ return static_cast<unsigned>(__cvta_generic_to_shared(p)); }}
// no "memory" clobber (as TileLang's cp_async_gs): the copy is asynchronous; reads of a stage are ordered after its
// wait + __syncthreads, and nvcc stays free to interleave the next stage's issue with this stage's LDS/LDSM/HMMA.
__device__ __forceinline__ void nvfp4_w4a16_cp16(unsigned s, const void *g) {{
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(s), "l"(g));
}}
// zero-filling variant (src-size n = 0 or 16): padding rows >= M read nothing
__device__ __forceinline__ void nvfp4_w4a16_cp16z(unsigned s, const void *g, unsigned n) {{
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" :: "r"(s), "l"(g), "r"(n));
}}
// one pipeline stage (k tile of {block_K}): A rows m0.., 32x32 weight blocks bx*{nblk}.. and their scales.
// 32-bit byte offsets (the host checks that x and the packed weights are below 4 GiB).
__device__ __forceinline__ void nvfp4_w4a16_load_stage(unsigned sA, unsigned sW, unsigned sS, int st, const void *xp,
                                                       const void *wp, const void *sp, int M, int m0, int bx, int kc0,
                                                       int tid) {{
  const char *x = static_cast<const char *>(xp), *w = static_cast<const char *>(wp), *sg = static_cast<const char *>(sp);
  const unsigned ust = st, ukc = kc0, ut = tid, ubx = bx;
#pragma unroll
  for (unsigned v = 0; v < {a_it}u; ++v) {{
    const unsigned e = ut + v * {threads}u, row = e / {rowch}u, ch = e % {rowch}u;
    const bool ok = m0 + (int)row < M;
    const unsigned grow = ok ? (unsigned)(m0 + (int)row) : (unsigned)(M - 1);
    {a_g}nvfp4_w4a16_cp16z(sA + ((ust * {block_M}u + row) * {block_K * 2}u + ({swz} << 4)),
                           x + (grow * {K * 2}u + ukc * 64u + ch * 16u), ok ? 16u : 0u);
  }}
#pragma unroll
  for (unsigned v = 0; v < {w_it}u; ++v) {{
    const unsigned e = ut + v * {threads}u, r = e / {kch * 32}u, c = (e >> 5) % {kch}u, l = e & 31u;
    {w_g}nvfp4_w4a16_cp16(sW + (((ust * {nblk}u + r) * {kch}u + c) * 512u + l * 16u),
                          w + (((ubx * {nblk}u + r) * {K // 32}u + ukc + c) * 512u + l * 16u));
  }}
  if (ut < {s_n}u) {{
    const unsigned r = ut / {kch * 4}u, c = (ut >> 2) % {kch}u, l = ut & 3u;
    nvfp4_w4a16_cp16(sS + (((ust * {nblk}u + r) * {kch}u + c) * 64u + l * 16u),
                     sg + (((ubx * {nblk}u + r) * {K // 32}u + ukc + c) * 64u + l * 16u));
  }}
}}
"""
    )


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nvfp4_w4a16_streamk_kernel(
    N: int, K: int, warps_m: int, WM: int, warps_n: int, nb: int, num_stages: int, min_blocks: int = 1, late_issue: bool = False
):
    """late_issue: issue the next stage's cp.async after the first k16 step's MMAs instead of right after the barrier,
    so the tensor pipe restarts without waiting behind the copy-issue instructions."""
    block_K = 64
    assert WM % 16 == 0 and nb in (1, 2) and num_stages >= 2
    block_M = warps_m * WM
    WN = 32 * nb
    block_N = warps_n * WN
    nblk = block_N // 32
    threads = 32 * warps_m * warps_n
    kchunks = block_K // 32
    n_tiles = N // block_N
    KT = K // block_K
    assert N % block_N == 0 and K % block_K == 0, (N, K, block_N, block_K)
    mt = WM // 16
    M = T.dynamic("M")
    PS = T.dynamic("PS")
    bf16, f32, i32, i64 = T.bfloat16, T.float32, T.int32, T.int64
    src = _streamk_loader_src(block_M, nblk, K, threads, block_K)

    emitter = TensorCoreIntrinEmitter(
        a_dtype=bf16,
        b_dtype=bf16,
        accum_dtype=f32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=warps_m,
        block_col_warps=warps_n,
        warp_row_tiles=WM,
        warp_col_tiles=WN,
        chunk=block_K,
    )

    @T.macro
    def _issue(kt, x, Wq, Sq, sb, stg, M, m0, bx, ks, ke, tid):
        """cp.async issue of k tile ks + kt + num_stages - 1 into stage stg[1] (if any) + its commit group."""
        if kt + num_stages - 1 < ke - ks:
            T.call_extern(
                "handle",
                "nvfp4_w4a16_load_stage",
                sb[0],
                sb[1],
                sb[2],
                stg[1],
                T.access_ptr(x, "r"),
                T.access_ptr(Wq, "r"),
                T.access_ptr(Sq, "r"),
                M,
                m0,
                bx,
                (ks + kt + num_stages - 1) * kchunks,
                tid,
            )
        T.ptx_commit_group()

    @T.prim_func
    def main(
        x: T.Tensor((M, K), bf16),
        Y: T.Tensor((M, N), bf16),
        Wq: T.Tensor((N // 32, K // 32, 128), i32),
        Sq: T.Tensor((N // 32, K // 32, 16), i32),
        alpha: T.Tensor((N,), f32),
        P: T.Tensor((PS, block_M, block_N), f32),
        counters: T.Tensor((base._NUM_COUNTERS,), i32),
        G: T.int32,  # run-time grid size
    ):
        with T.Kernel(G, threads=threads, prelude=src) as g:
            # Without this, every floor division by G gets sign fix-ups and the kernel ran 1.1-1.5x slower (measured).
            T.assume(G > 0)
            if min_blocks > 1:
                T.annotate_min_blocks_per_sm(min_blocks)
            A_s = T.alloc_shared((num_stages, block_M, block_K), bf16)
            Wq_s = T.alloc_shared((num_stages, nblk, kchunks, 128), i32)
            S_s = T.alloc_shared((num_stages, nblk, kchunks, 16), i32)
            ticket_s = T.alloc_shared((1,), i32)
            sb = T.alloc_local((3,), T.uint32)
            stg = T.alloc_local((2,), i32)
            ts = T.alloc_local((4,), i32)
            q_l = T.alloc_local((nb * 4,), i32)
            s_l = T.alloc_local((nb * 2,), i32)
            B_l = T.alloc_local((nb * 32,), bf16)
            A_l = T.alloc_local((mt * 8,), bf16)
            C_l = T.alloc_local((mt * nb * 2 * 8,), f32)
            T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})

            tid = T.get_thread_binding()
            warp = tid // 32
            lane = tid % 32
            wm = warp % warps_m  # matches the emitter's warp_m = (tid // 32) % warps_m
            wn = warp // warps_m
            sb[0] = T.call_extern(T.uint32, "nvfp4_w4a16_saddr", T.access_ptr(A_s, "w"))
            sb[1] = T.call_extern(T.uint32, "nvfp4_w4a16_saddr", T.access_ptr(Wq_s, "w"))
            sb[2] = T.call_extern(T.uint32, "nvfp4_w4a16_saddr", T.access_ptr(S_s, "w"))
            # per-CTA / per-tile scalars live in registers (bound Python expressions would be re-inlined at every use,
            # e.g. a run-time division inside the k loop)
            cs = T.alloc_local((4,), i32)
            cs[0] = T.ceildiv(M, block_M)  # m tiles
            cs[1] = cs[0] * (n_tiles * KT)  # TT: iterations of the whole problem
            cs[2] = T.cast((T.cast(g, i64) * cs[1]) // G, i32)  # it0
            cs[3] = T.cast((T.cast(g + 1, i64) * cs[1]) // G, i32)  # it1
            m_tiles = cs[0]
            TT = cs[1]
            it0 = cs[2]
            it1 = cs[3]

            for t in T.serial(it0 // KT, (it1 - 1) // KT + 1):
                ts[0] = (t % m_tiles) * block_M  # m0
                ts[1] = t // m_tiles  # bx (n tile)
                ts[2] = T.max(it0, t * KT) - t * KT  # ks
                ts[3] = T.min(it1, (t + 1) * KT) - t * KT  # ke
                m0 = ts[0]
                bx = ts[1]
                ks = ts[2]
                ke = ts[3]
                T.clear(C_l)
                for p in T.unroll(num_stages - 1):
                    if p < ke - ks:
                        T.call_extern(
                            "handle",
                            "nvfp4_w4a16_load_stage",
                            sb[0],
                            sb[1],
                            sb[2],
                            p,
                            T.access_ptr(x, "r"),
                            T.access_ptr(Wq, "r"),
                            T.access_ptr(Sq, "r"),
                            M,
                            m0,
                            bx,
                            (ks + p) * kchunks,
                            tid,
                        )
                    T.ptx_commit_group()
                stg[0] = 0  # stage consumed this k tile
                stg[1] = num_stages - 1  # stage refilled this k tile
                for kt in T.serial(ke - ks):
                    T.ptx_wait_group(num_stages - 2)
                    T.sync_threads()
                    if not late_issue:
                        _issue(kt, x, Wq, Sq, sb, stg, M, m0, bx, ks, ke, tid)
                    st = stg[0]
                    for c in T.unroll(kchunks):
                        for b in T.unroll(nb):
                            for v in T.vectorized(4):
                                q_l[b * 4 + v] = Wq_s[st, wn * nb + b, c, lane * 4 + v]
                            for v in T.vectorized(2):
                                s_l[b * 2 + v] = S_s[st, wn * nb + b, c, (lane // 4) * 2 + v]
                            T.call_extern(
                                "handle",
                                "nvfp4_w4a16_dequant_chunk",
                                T.access_ptr(q_l[b * 4], "r", extent=4),
                                T.access_ptr(s_l[b * 2], "r", extent=2),
                                T.access_ptr(B_l[b * 32], "w", extent=32),
                            )
                        for s in T.unroll(2):
                            emitter.ldmatrix_a(A_l, A_s[st, 0:block_M, 0:block_K], c * 2 + s)
                            for i in T.unroll(mt):
                                for b in T.unroll(nb):
                                    for j in T.unroll(2):
                                        base._mma_pair(A_l, i * 8, B_l, b * 32 + (s * 2 + j) * 8, C_l, ((i * nb + b) * 2 + j) * 8)
                            if late_issue and c == 0 and s == 0:
                                _issue(kt, x, Wq, Sq, sb, stg, M, m0, bx, ks, ke, tid)
                    stg[0] = T.if_then_else(stg[0] == num_stages - 1, 0, stg[0] + 1)
                    stg[1] = T.if_then_else(stg[1] == num_stages - 1, 0, stg[1] + 1)
                T.ptx_wait_group(0)
                T.sync_threads()

                if ks == 0 and ke == KT:
                    for i in T.unroll(mt):
                        for b in T.unroll(nb):
                            for j in T.unroll(2):
                                for r2 in T.unroll(4):
                                    prow = wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                                    gcol = bx * block_N + wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                                    if m0 + prow < M:
                                        for v in T.vectorized(2):
                                            Y[m0 + prow, gcol + v] = T.cast(
                                                C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16
                                            )
                else:
                    slot = 2 * g + T.if_then_else(it1 <= (t + 1) * KT, 1, 0)
                    for i in T.unroll(mt):
                        for b in T.unroll(nb):
                            for j in T.unroll(2):
                                for r2 in T.unroll(4):
                                    prow = wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                                    pcol = wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                                    for v in T.vectorized(2):
                                        P[slot, prow, pcol + v] = (
                                            C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[bx * block_N + pcol + v]
                                        )
                    T.call_extern("handle", "__threadfence")
                    T.sync_threads()
                    if tid == 0:
                        ticket_s[0] = T.atomic_add(counters[t], 1, return_prev=True)
                    T.sync_threads()
                    g0 = T.cast(((T.cast(t * KT + 1, i64) * G) - 1) // TT, i32)
                    g1 = T.cast(((T.cast((t + 1) * KT, i64) * G) - 1) // TT, i32)
                    if ticket_s[0] == g1 - g0:
                        T.call_extern("handle", "__threadfence")
                        T.clear(C_l)
                        for q in T.serial(g1 - g0 + 1):
                            sl = 2 * (g0 + q) + T.if_then_else(T.cast((T.cast(g0 + q + 1, i64) * TT) // G, i32) <= (t + 1) * KT, 1, 0)
                            for i in T.unroll(mt):
                                for b in T.unroll(nb):
                                    for j in T.unroll(2):
                                        for r2 in T.unroll(4):
                                            prow = wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                                            pcol = wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                                            for v in T.vectorized(2):
                                                C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] = (
                                                    C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] + P[sl, prow, pcol + v]
                                                )
                        for i in T.unroll(mt):
                            for b in T.unroll(nb):
                                for j in T.unroll(2):
                                    for r2 in T.unroll(4):
                                        prow = wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                                        gcol = bx * block_N + wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                                        if m0 + prow < M:
                                            for v in T.vectorized(2):
                                                Y[m0 + prow, gcol + v] = T.cast(C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v], bf16)
                        if tid == 0:
                            counters[t] = 0
                    T.sync_threads()

    return main


class StreamKConfig(NamedTuple):
    """nvfp4_w4a16_streamk_kernel (block_K 64) over num_SMs * ctas_per_sm CTAs."""

    warps_m: int
    WM: int
    warps_n: int
    nb: int
    num_stages: int
    ctas_per_sm: int = 1
    min_blocks: int = 1
    late_issue: bool = False


# Tables tuned per GPU, keyed by SM count and then by weight shape: {num_sms: {(N, K): [(M_max, config), ...]}}, entries
# ascending; the first entry with M <= M_max wins. Any other GPU or shape uses _heuristic_config.
# 188 SMs: RTX PRO 6000 Blackwell Max-Q (sm_120), the linear-layer shapes of a 27B-class model.
_TUNED_CONFIGS = {
    188: {
        (16384, 5120): [
            (1, base.FRAG),
            (8, base.TileConfig(1, 16, 4, 1, 64, 4, 4)),
            (16, base.TileConfig(1, 16, 4, 1, 64, 2, 4)),
            (32, base.TileConfig(1, 32, 8, 1, 64, 2, 4)),
            (49, base.FRAG),
            (56, base.TileConfig(1, 64, 4, 1, 64, 4, 3)),
            (64, base.FRAG),
            (80, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (96, StreamKConfig(2, 48, 4, 1, 3, 1, 1, False)),
            (112, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (128, StreamKConfig(1, 64, 8, 1, 3, 1, 1, False)),
            (160, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (192, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (224, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (240, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (256, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (288, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (320, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (384, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (448, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (512, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (576, StreamKConfig(1, 64, 4, 1, 3, 2, 2, True)),
            (640, base.TileConfig(2, 64, 4, 2, 64, 1, 3)),
            (896, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (1024, base.TileConfig(2, 64, 4, 2, 64, 1, 3)),
        ],
        (14336, 5120): [
            (1, base.FRAG),
            (8, base.TileConfig(1, 16, 4, 1, 64, 4, 4)),
            (16, base.TileConfig(1, 16, 4, 1, 64, 2, 4)),
            (32, base.TileConfig(1, 32, 8, 1, 64, 2, 4)),
            (49, base.FRAG),
            (56, base.TileConfig(1, 64, 4, 1, 64, 5, 3)),
            (64, base.FRAG),
            (80, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (96, StreamKConfig(2, 48, 4, 1, 3, 1, 1, True)),
            (112, base.FRAG),
            (128, StreamKConfig(1, 64, 4, 1, 3, 2, 2, True)),
            (160, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (176, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (192, base.TileConfig(1, 64, 8, 1, 64, 1, 3)),
            (224, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (240, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (256, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (288, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (320, base.TileConfig(1, 64, 4, 1, 64, 1, 3)),
            (384, base.TileConfig(2, 64, 4, 2, 64, 1, 3)),
            (448, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (512, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (576, StreamKConfig(1, 64, 4, 1, 3, 2, 2, True)),
            (640, base.TileConfig(1, 64, 4, 1, 64, 1, 3)),
            (768, base.TileConfig(2, 64, 4, 2, 64, 1, 3)),
            (1024, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
        ],
        (5120, 6144): [
            (8, base.FRAG),
            (16, base.TileConfig(1, 16, 4, 1, 64, 4, 4)),
            (32, base.TileConfig(1, 32, 4, 1, 64, 4, 4)),
            (49, base.FRAG),
            (56, base.TileConfig(1, 64, 4, 1, 64, 4, 3)),
            (80, base.FRAG),
            (96, StreamKConfig(2, 48, 4, 1, 3, 1, 1, False)),
            (112, base.TileConfig(1, 64, 4, 1, 64, 2, 3)),
            (128, base.TileConfig(1, 64, 4, 1, 64, 4, 3)),
            (160, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (192, base.TileConfig(1, 64, 4, 1, 64, 3, 3)),
            (240, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (256, base.TileConfig(1, 64, 8, 1, 64, 2, 3)),
            (288, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (320, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (384, base.TileConfig(2, 64, 4, 2, 64, 3, 3)),
            (448, base.TileConfig(1, 64, 4, 1, 64, 2, 3)),
            (512, base.TileConfig(2, 64, 4, 2, 64, 2, 3)),
            (576, base.TileConfig(1, 64, 4, 1, 64, 1, 3)),
            (768, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (896, base.TileConfig(1, 64, 4, 1, 64, 1, 3)),
            (1024, base.TileConfig(2, 64, 4, 2, 64, 1, 3)),
        ],
        (34816, 5120): [
            (16, base.TileConfig(1, 16, 8, 1, 64, 1, 6)),
            (32, base.TileConfig(1, 32, 8, 1, 64, 1, 4)),
            (48, base.FRAG),
            (56, base.TileConfig(1, 64, 4, 1, 64, 2, 3)),
            (64, base.FRAG),
            (80, StreamKConfig(1, 80, 4, 1, 3, 2, 2, True)),
            (96, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (112, StreamKConfig(1, 112, 4, 1, 2, 2, 2, True)),
            (128, base.TileConfig(1, 64, 4, 1, 64, 1, 3)),
            (160, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (192, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (224, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (240, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (256, base.TileConfig(1, 64, 4, 1, 64, 1, 3)),
            (288, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (320, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (384, StreamKConfig(1, 128, 4, 1, 2, 2, 2, False)),
            (448, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (512, base.TileConfig(2, 64, 4, 2, 64, 1, 3)),
            (576, base.TileConfig(1, 64, 4, 1, 64, 1, 3)),
            (640, base.TileConfig(2, 64, 4, 2, 64, 1, 3)),
            (896, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (1024, base.TileConfig(2, 64, 4, 2, 64, 1, 3)),
        ],
        (5120, 17408): [
            (8, base.FRAG),
            (16, base.TileConfig(1, 16, 4, 1, 64, 4, 4)),
            (32, base.TileConfig(1, 32, 4, 1, 64, 4, 4)),
            (64, base.FRAG),
            (80, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (96, StreamKConfig(2, 48, 4, 1, 3, 1, 1, True)),
            (112, base.FRAG),
            (128, StreamKConfig(1, 64, 8, 1, 3, 1, 1, False)),
            (160, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (176, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (192, StreamKConfig(1, 64, 8, 1, 3, 1, 1, False)),
            (224, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (240, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (256, base.TileConfig(2, 64, 4, 2, 64, 4, 3)),
            (288, StreamKConfig(1, 96, 4, 1, 2, 2, 2, False)),
            (320, StreamKConfig(1, 80, 4, 1, 3, 2, 2, False)),
            (384, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (448, StreamKConfig(1, 112, 4, 1, 2, 2, 2, False)),
            (512, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
            (576, base.TileConfig(1, 64, 4, 1, 64, 1, 3)),
            (1024, StreamKConfig(2, 64, 4, 2, 3, 1, 1, True)),
        ],
    },
}


def _heuristic_config(N, K, M, num_sms):
    """Untuned shapes.

    * M <= 96: fragment-kernel passes (weight-stream bound; one weight read per 64 rows).
    * Small problems (fewer than 12 k tiles per CTA of a two-CTAs-per-SM 64 x 128 stream-K grid): the example's tile
      heuristic; stream-K ranges that short cost more in partial tiles than they save.
    * 96 < M <= 256 (or N not a multiple of 256): stream-K with 64 x 128 tiles, two CTAs per SM.
    * M > 256: 128 x 256 tiles; plain tiling (no split) when the tiles already fill the SMs evenly (>= 75% of the last
      wave), else stream-K, which avoids both wave quantization and a full FP32 split-K partial pass.

    Fitted on eight shapes on one 188-SM GPU; per-GPU tables (_TUNED_CONFIGS) are the way to do better."""
    if M <= 96:
        return FRAG
    iterations = -(-M // 64) * (N // 128) * (K // 64)  # (tile, k tile) pairs of a 64 x 128 stream-K grid
    if iterations < 12 * 2 * num_sms:
        return base.get_config(N, K, M, num_sms)
    if M <= 256 or N % 256:
        return StreamKConfig(1, 64, 4, 1, 3, ctas_per_sm=2, min_blocks=2, late_issue=True)
    tiles = -(-M // 128) * (N // 256)
    if tiles / (num_sms * math.ceil(tiles / num_sms)) >= 0.75:
        return TileConfig(2, 64, 4, 2, 64, 1, 3)
    return StreamKConfig(2, 64, 4, 2, 3, late_issue=True)


def get_config(N, K, M, num_sms=None):
    """The configuration nvfp4_w4a16_linear uses for one (N, K) weight and 1 <= M <= MAX_M rows: FRAG, a FragConfig,
    a TileConfig or a StreamKConfig. The tuned table applies only on a GPU with the SM count it was tuned on."""
    if num_sms is None:
        num_sms = base._num_sms(torch.cuda.current_device())
    for m_max, cfg in _TUNED_CONFIGS.get(num_sms, {}).get((N, K), ()):
        if m_max >= M:
            return cfg
    return _heuristic_config(N, K, M, num_sms)


def smem_bytes(cfg):
    """Shared memory per CTA of a TileConfig or StreamKConfig."""
    if isinstance(cfg, TileConfig):
        return base.smem_bytes(cfg)
    block_M, nblk = cfg.warps_m * cfg.WM, cfg.warps_n * cfg.nb
    return cfg.num_stages * (block_M * 64 * 2 + nblk * 2 * 576) + 4


def _run_streamk(x, y, w, cfg):
    M, K = x.shape
    ws = base._workspace(x.device)
    if M * K * 2 >= 2**32 or w.N * K // 2 >= 2**32:
        raise ValueError("the stream-K loader uses 32-bit byte offsets: x and the packed weight must be below 4 GiB")
    block_M, block_N = cfg.warps_m * cfg.WM, cfg.warps_n * 32 * cfg.nb
    tiles = -(-M // block_M) * (w.N // block_N)
    base._check_counters(tiles, f"M={M}, {cfg}")
    G = min(base._num_sms(x.device.index) * cfg.ctas_per_sm, tiles * (K // 64))
    kernel = base._kernel(
        ("streamk", w.N, w.K) + tuple(cfg),
        nvfp4_w4a16_streamk_kernel,
        w.N,
        w.K,
        cfg.warps_m,
        cfg.WM,
        cfg.warps_n,
        cfg.nb,
        cfg.num_stages,
        cfg.min_blocks,
        bool(cfg.late_issue),
    )
    if tiles == G:  # every CTA covers exactly one whole tile: P is never touched
        P = ws.dummy_partials((1, block_M, block_N))
    else:
        P = torch.empty(2 * G, block_M, block_N, dtype=torch.float32, device=x.device)
    kernel(x, y, w.Wq, w.Sq, w.alpha, P, ws.counters, G)


def launch(x, y, w, cfg):
    """Run one chunk (M <= MAX_M rows) with a FRAG, FragConfig, TileConfig or StreamKConfig config."""
    if isinstance(cfg, StreamKConfig):
        _run_streamk(x, y, w, cfg)
    else:
        base.launch(x, y, w, cfg)


def nvfp4_w4a16_linear(x, weight, config=None, out=None):
    """y = x @ W^T in BF16, as the example's nvfp4_w4a16_linear, with this module's get_config and the stream-K
    kernel. config: None (get_config() per chunk), FRAG, a FragConfig, a TileConfig or a StreamKConfig."""
    x, out = base.check_inputs(x, weight, out)
    num_sms = base._num_sms(x.device.index)
    for r0 in range(0, x.shape[0], MAX_M):
        xs, ys = x[r0 : r0 + MAX_M], out[r0 : r0 + MAX_M]
        launch(xs, ys, weight, config if config is not None else get_config(weight.N, weight.K, xs.shape[0], num_sms))
    return out
