"""SM120 W4A16 NVFP4 GEMM for decode-sized batches (BF16 activations x NVFP4 weights).

Computes ``y = x @ W^T`` for ``x`` of shape ``(M, K)`` in BF16 and a weight ``W`` of shape ``(N, K)`` stored
as NVFP4 the way quantized checkpoints ship it:

* ``packed``: ``int8/uint8 (N, K / 2)``, two FP4 E2M1 codes per byte, low nibble = even ``k``;
* ``scale``: ``float8_e4m3fn`` (or raw ``uint8`` UE4M3 bytes) ``(N, K / 16)``, one block scale per 16 ``k``;
* ``alpha``: a float or an ``(N,)`` tensor, the per-tensor (or per-column) multiplier, i.e.
  ``1 / weight_global_scale`` for checkpoints that store a global scale,

so that ``y[m, n] = alpha[n] * sum_k x[m, k] * fp4(packed[n, k]) * scale[n, k // 16]``, accumulated in FP32 and
rounded once to BF16. The weights are dequantized into BF16 *register* fragments and multiplied with BF16
``mma.sync.m16n8k16``; activations stay BF16 (no activation quantization, unlike the W4A4 block-scaled path in
``sm120_nvfp4_blockscaled_gemm.py``). The kernels target the small-M regime of LLM decoding (M = 1 .. 1024
tokens), where the GEMM is bound by the weight stream (small M) or by BF16 tensor-core throughput (larger M).

Fragment-ordered weight layout (``prepare_nvfp4_weight``, run once at load time)
-------------------------------------------------------------------------------
The weight is re-laid out once into 32 (n) x 32 (k) blocks of 512 bytes, ordered like the ``mma.m16n8k16`` B
fragments of one warp: lane ``l`` owns 16 consecutive bytes, the 32 weights it feeds to the tensor core for two
n16 tiles x two k16 steps. Word ``w = 2 * st + j`` (k16 step ``st``, n16 tile ``j``) holds element ``e = 2r + h`` at
``n = 16j + l // 4 + 8 (r // 2)``, ``k = 16 st + 2 (l % 4) + h + 8 (r % 2)``. One ``ld.shared.v4`` per lane per block
therefore delivers a complete B fragment; no BF16 weight tile is ever written to shared memory.

Inside each 32-bit word the eight nibbles are bit-scattered (``_SLOT_POS``) so that three of the four ``bf16x2``
fragment registers come out of one shift and one AND with ``0x81C081C0`` and the fourth out of three shifts and
three LOP3s. Each result is the BF16 pair ``fp4 * 2^-126`` (E2M1 codes land in the BF16 subnormal range).

The UE4M3 block scales are re-encoded once as one byte ``b`` with ``bf16(0x7000 | b << 4) == scale * 2^119``, which is
exact for every E4M3 value including subnormals, and stored without replication (64 bytes per 32 x 32 block, the
same 12.5% overhead as the checkpoint). Per lane one PRMT plus one IMAD turn a scale word into ``bf16x2`` scale pairs,
and one ``mul.rn.bf16x2`` per fragment register gives ``fp4 * scale * 2^-7`` exactly. ``alpha * 2^7`` is applied
once in the FP32 epilogue. Zero scales zero their codes and negative scales flip the code signs at preparation time.
The dequantization costs about 2.1 instructions per weight. The prepared copy is exactly as large as the checkpoint
tensors (``N * K / 2 + N * K / 16`` bytes).

Kernels and dispatch (``nvfp4_w4a16_linear``)
---------------------------------------------
This example keeps to two non-persistent kernels:

* ``nvfp4_w4a16_frag_kernel``: one CTA row covers all ``M <= 64`` rows; grid ``(N / block_N, split_k)``. Used for
  small ``M`` and, up to 96 rows, as 64-row passes.
* ``nvfp4_w4a16_tile_kernel``: grid ``(m tiles, n tiles, split_k)`` with tile heights in 16-row steps, a
  ``T.Pipelined`` cp.async pipeline, and deterministic split-K.

Split tiles are reduced deterministically: FP32 partials, and the last CTA to arrive at a tile (atomic ticket) sums
them in fixed order. Tickets reset themselves, so every launch can be captured in a CUDA graph. The tickets live in
a per-device workspace, so launches on one device must not run concurrently on two streams.

``get_config`` picks a kernel and tile per (N, K, M) with a shape-independent heuristic. A persistent stream-K kernel
and tables tuned per GPU, which are faster above 96 rows, live in
``maint/gemm/gemm_sm120/sm120_nvfp4_w4a16_streamk.py`` and reuse this module's weight layout and kernels.

Scope: the kernels use ``mma.sync.m16n8k16`` (BF16), ``cp.async`` and ``mul.rn.bf16x2`` (sm_90 or newer), and at most
99 KB of shared memory per CTA. They are tested and tuned on SM120 (compute capability 12.0) only, which is why the
example and its tests require it.

Run from the repository root:

    python examples/gemm_sm120/sm120_nvfp4_w4a16_gemm.py --m 16 --n 5120 --k 6144 --verify
"""

import argparse
import functools
import math
import sys
from pathlib import Path
from typing import NamedTuple

import torch

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

FRAG_MAX_M = 64  # rows covered by one pass of the fragment kernel
MAX_M = 1024  # rows per launch; nvfp4_w4a16_linear splits larger batches into MAX_M-row chunks
_ALPHA_FOLD = 128.0  # the kernels accumulate fp4 * scale * 2^-7
_NUM_COUNTERS = 1 << 14  # split-K ticket counters per device (tiles of one split launch)

# ---------------------------------------------------------------------------------------------------------------
# Register dequantization of one lane's chunk: 4 bit-scattered weight words + 2 scale words -> 16 bf16x2 B registers.
#   weight word w = 2*st + j: k16 step st (0, 1), n16 tile j (0, 1) of the lane's 32x32 block.
#   out[4*w + r]: r=0 (n_a, k0..k0+1)  r=1 (n_a, k0+8..k0+9)  r=2 (n_b, k0..k0+1)  r=3 (n_b, k0+8..k0+9)
#       with n_a = 16j + lane/4, n_b = n_a + 8, k0 = 16*st + 2*(lane%4)  (mma.m16n8k16 B layout, two n8 tiles)
#   scale word st, byte 2*x + j = scale code of (n = 16j + 8x + lane/4, k16 step st); p_j = {code(j, a), code(j, b)}.
_DEQUANT_SRC = r"""
__device__ __forceinline__ unsigned nvfp4_w4a16_mul_lo(unsigned a, unsigned p) {
  unsigned d;
  asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 b;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 b, {l, l};\n\t"
      "mul.rn.bf16x2 %0, %1, b;\n\t}" : "=r"(d) : "r"(a), "r"(p));
  return d;
}
__device__ __forceinline__ unsigned nvfp4_w4a16_mul_hi(unsigned a, unsigned p) {
  unsigned d;
  asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 b;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 b, {h, h};\n\t"
      "mul.rn.bf16x2 %0, %1, b;\n\t}" : "=r"(d) : "r"(a), "r"(p));
  return d;
}
template <typename TQ, typename TS, typename TO>
__device__ __forceinline__ void nvfp4_w4a16_dequant_chunk(TQ *qp, TS *sp, TO *op) {
  const unsigned *q = reinterpret_cast<const unsigned *>(qp);
  const unsigned *s = reinterpret_cast<const unsigned *>(sp);
  unsigned *out = reinterpret_cast<unsigned *>(op);
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const unsigned sw = s[st];
    // bf16x2 {scale(n_a), scale(n_b)} * 2^119 for n16 tile j = 0 (bytes 0, 2) and j = 1 (bytes 1, 3)
    const unsigned p0 = __byte_perm(sw, 0u, 0x4240u) * 16u + 0x70007000u;
    const unsigned p1 = __byte_perm(sw, 0u, 0x4341u) * 16u + 0x70007000u;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const unsigned x = q[2 * st + j];
      const unsigned p = j ? p1 : p0;
      const unsigned ra = x & 0x81C081C0u;
      const unsigned rb = (x << 3) & 0x81C081C0u;
      const unsigned rc = (x << 6) & 0x81C081C0u;
      const unsigned rd = ((x << 1) & 0x80008000u) | ((x >> 3) & 0x01800180u) | ((x >> 7) & 0x00400040u);
      out[8 * st + 4 * j + 0] = nvfp4_w4a16_mul_lo(ra, p);
      out[8 * st + 4 * j + 1] = nvfp4_w4a16_mul_lo(rb, p);
      out[8 * st + 4 * j + 2] = nvfp4_w4a16_mul_hi(rc, p);
      out[8 * st + 4 * j + 3] = nvfp4_w4a16_mul_hi(rd, p);
    }
  }
}
"""

# Bit positions (sign, e1, e0, m) of the nibble feeding fragment register r (slot a/b/c/d above), half h (0 = low
# bf16 = even k, 1 = high bf16 = odd k). Slots a, b, c: target positions shifted by 0, 3, 6; slot d: the rest.
_SLOT_POS = {
    0: ((15, 8, 7, 6), (31, 24, 23, 22)),
    1: ((12, 5, 4, 3), (28, 21, 20, 19)),
    2: ((9, 2, 1, 0), (25, 18, 17, 16)),
    3: ((14, 11, 10, 13), (30, 27, 26, 29)),
}


@T.macro
def _mma_pair(A_l, a_off, B_l, b_off, C_l, c_off):
    """Two m16n8k16 BF16 MMAs: one m16 x n16 tile from one A fragment and two n8 halves of a B fragment."""
    T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data, a_off, B_l.data, b_off, C_l.data, c_off, T.bool(False))
    T.ptx_mma(
        "float32",
        "m16n8k16",
        "row",
        "col",
        "bf16",
        "bf16",
        "fp32",
        A_l.data,
        a_off,
        B_l.data,
        b_off + 4,
        C_l.data,
        c_off + 4,
        T.bool(False),
    )


# ---------------------------------------------------------------------------------------------------------------
# Fragment kernel: grid (N / block_N, split_k); block_N = 32 * n_warps; one warp owns 32 columns for all M rows.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nvfp4_w4a16_frag_kernel(N: int, K: int, block_M: int, n_warps: int, block_K: int, split_k: int, num_stages: int):
    """Streams packed blocks, scales and the A tile into shared memory with cp.async inside ``T.Pipelined``. Each lane
    does one 128-bit shared load of its 16 weight bytes and one 64-bit load of its 8 scale bytes (quad broadcast),
    dequantizes in registers and issues ``mma.sync`` with that B fragment. A (all ``M <= block_M`` rows) is read with
    ldmatrix from a swizzled tile; rows ``>= M`` re-read row ``M - 1`` and are never stored. ``T.ptx_mma`` with a
    hand-built B fragment is used instead of ``T.gemm(A_shared, B_fragment)`` because the fragment then comes out of
    packed-word bit operations (a per-element ``T.Parallel`` fill costs about 10 instructions per weight)."""
    assert block_M in (16, 32, 48, 64) and n_warps in (1, 2, 4, 8) and block_K % 32 == 0
    block_N = 32 * n_warps
    threads = 32 * n_warps
    kchunks = block_K // 32  # 32-wide k chunks per k tile (one 16-byte fragment load per lane each)
    n_tiles = N // block_N
    k_tiles = K // block_K
    assert N % block_N == 0 and K % block_K == 0 and k_tiles % split_k == 0, (N, K, block_N, block_K, split_k)
    kt_per_split = k_tiles // split_k
    warp_rows = block_M // 16
    M = T.dynamic("M")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

    emitter = TensorCoreIntrinEmitter(
        a_dtype=bf16,
        b_dtype=bf16,
        accum_dtype=f32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=1,
        block_col_warps=n_warps,
        warp_row_tiles=block_M,
        warp_col_tiles=32,
        chunk=block_K,
    )
    P_shape = (split_k, M, N) if split_k > 1 else (1, 1, 1)

    def _partial_sum(P, i, col):
        """Fixed-order (0, 1, ..., split_k - 1) FP32 sum of the split-K partials: deterministic."""
        acc = P[0, i, col]
        for sk in range(1, split_k):
            acc = acc + P[sk, i, col]
        return acc

    @T.prim_func
    def main(
        x: T.Tensor((M, K), bf16),
        Y: T.Tensor((M, N), bf16),
        Wq: T.Tensor((N // 32, K // 32, 128), i32),
        Sq: T.Tensor((N // 32, K // 32, 16), i32),
        alpha: T.Tensor((N,), f32),
        P: T.Tensor(P_shape, f32),
        counters: T.Tensor((_NUM_COUNTERS,), i32),
    ):
        with T.Kernel(n_tiles, split_k, threads=threads, prelude=_DEQUANT_SRC) as (bx, bk):
            A_s = T.alloc_shared((block_M, block_K), bf16)
            Wq_s = T.alloc_shared((n_warps, kchunks, 128), i32)
            S_s = T.alloc_shared((n_warps, kchunks, 16), i32)
            ticket_s = T.alloc_shared((1,), i32)
            q_l = T.alloc_local((4,), i32)
            s_l = T.alloc_local((2,), i32)
            B_l = T.alloc_local((32,), bf16)  # 2 k16 steps x 2 n16 tiles x 8 values
            A_l = T.alloc_local((warp_rows * 8,), bf16)  # one k16 step of A for all m16 tiles
            C_l = T.alloc_local((warp_rows * 2 * 8,), f32)  # warp_rows m16 x 2 n16 accumulators
            T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})

            tid = T.get_thread_binding()
            warp = tid // 32
            lane = tid % 32
            T.clear(C_l)

            for kt in T.Pipelined(kt_per_split, num_stages=num_stages):
                kc0 = (bk * kt_per_split + kt) * kchunks
                # Rows >= M re-read row M - 1 (clamped: no address past the end of x is formed); those accumulator rows
                # are never stored. A pure global->shared store keeps this an async (cp.async) stage.
                for i, kk in T.Parallel(block_M, block_K):
                    A_s[i, kk] = x[T.min(i, M - 1), kc0 * 32 + kk]
                T.copy(Wq[bx * n_warps, kc0, 0], Wq_s)
                T.copy(Sq[bx * n_warps, kc0, 0], S_s)
                for c in T.unroll(kchunks):
                    for v in T.vectorized(4):
                        q_l[v] = Wq_s[warp, c, lane * 4 + v]
                    for v in T.vectorized(2):
                        s_l[v] = S_s[warp, c, (lane // 4) * 2 + v]
                    T.call_extern(
                        "handle", "nvfp4_w4a16_dequant_chunk", T.access_ptr(q_l, "r"), T.access_ptr(s_l, "r"), T.access_ptr(B_l, "w")
                    )
                    for s in T.unroll(2):
                        emitter.ldmatrix_a(A_l, A_s, c * 2 + s)
                        for i in T.unroll(warp_rows):
                            for j in T.unroll(2):
                                _mma_pair(A_l, i * 8, B_l, (s * 2 + j) * 8, C_l, (i * 2 + j) * 8)

            # C_l[(i*2+j)*8 + r] sits at row 16i + 8*((r%4)//2) + lane//4, col 16j + 8*(r//4) + 2*(lane%4) + r%2
            for i in T.unroll(warp_rows):
                for j in T.unroll(2):
                    for r2 in T.unroll(4):
                        grow = i * 16 + 8 * (r2 % 2) + lane // 4
                        gcol = bx * block_N + warp * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                        if grow < M:
                            if split_k == 1:
                                for v in T.vectorized(2):
                                    Y[grow, gcol + v] = T.cast(C_l[(i * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                            else:
                                for v in T.vectorized(2):
                                    P[bk, grow, gcol + v] = C_l[(i * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]

            if split_k > 1:
                # threadfence reduction: the last CTA of this N tile reduces the partials in fixed order.
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tid == 0:
                    ticket_s[0] = T.atomic_add(counters[bx], 1, return_prev=True)
                T.sync_threads()
                if ticket_s[0] == split_k - 1:
                    T.call_extern("handle", "__threadfence")
                    for i, j in T.Parallel(block_M, block_N):
                        if i < M:
                            Y[i, bx * block_N + j] = T.cast(_partial_sum(P, i, bx * block_N + j), bf16)
                    if tid == 0:
                        counters[bx] = 0

    return main


# ---------------------------------------------------------------------------------------------------------------
# Tile kernel: the CTA covers block_M = warps_m * WM rows x block_N = warps_n * 32 * nb columns; warp (wm, wn) owns
# WM rows x 32 * nb columns. Per 32-wide k chunk each warp dequantizes its nb 32x32 weight blocks once into register
# B fragments and reuses them for its WM / 16 m16 tiles.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nvfp4_w4a16_tile_kernel(
    N: int, K: int, warps_m: int, WM: int, warps_n: int, nb: int, block_K: int, split_k: int, num_stages: int, min_blocks: int = 1
):
    """Grid (m tiles, n tiles, split_k); ``T.Pipelined`` cp.async pipeline; deterministic split-K. Padding rows
    ``>= M`` wrap to row ``(m0 + i) % M``: always a valid row, never stored, and spread over distinct rows (clamping
    them all to row ``M - 1`` makes every CTA hit the same L2 lines)."""
    assert WM % 16 == 0 and nb in (1, 2) and block_K % 32 == 0
    block_M = warps_m * WM
    WN = 32 * nb
    block_N = warps_n * WN
    nblk = block_N // 32  # 32-column weight blocks per CTA
    threads = 32 * warps_m * warps_n
    kchunks = block_K // 32
    n_tiles = N // block_N
    k_tiles = K // block_K
    assert N % block_N == 0 and K % block_K == 0 and k_tiles % split_k == 0, (N, K, block_N, block_K, split_k)
    kt_per_split = k_tiles // split_k
    mt = WM // 16  # m16 tiles per warp
    M = T.dynamic("M")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

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
    P_shape = (split_k, M, N) if split_k > 1 else (1, 1, 1)

    def _partial_sum(P, i, col):
        acc = P[0, i, col]
        for sk in range(1, split_k):
            acc = acc + P[sk, i, col]
        return acc

    @T.prim_func
    def main(
        x: T.Tensor((M, K), bf16),
        Y: T.Tensor((M, N), bf16),
        Wq: T.Tensor((N // 32, K // 32, 128), i32),
        Sq: T.Tensor((N // 32, K // 32, 16), i32),
        alpha: T.Tensor((N,), f32),
        P: T.Tensor(P_shape, f32),
        counters: T.Tensor((_NUM_COUNTERS,), i32),
    ):
        with T.Kernel(T.ceildiv(M, block_M), n_tiles, split_k, threads=threads, prelude=_DEQUANT_SRC) as (bm, bx, bk):
            if min_blocks > 1:
                T.annotate_min_blocks_per_sm(min_blocks)
            A_s = T.alloc_shared((block_M, block_K), bf16)
            Wq_s = T.alloc_shared((nblk, kchunks, 128), i32)
            S_s = T.alloc_shared((nblk, kchunks, 16), i32)
            ticket_s = T.alloc_shared((1,), i32)
            q_l = T.alloc_local((nb * 4,), i32)
            s_l = T.alloc_local((nb * 2,), i32)
            B_l = T.alloc_local((nb * 32,), bf16)  # per block: 2 k16 steps x 2 n16 tiles x 8 values
            A_l = T.alloc_local((mt * 8,), bf16)  # one k16 step of A for the warp's m16 tiles
            C_l = T.alloc_local((mt * nb * 2 * 8,), f32)  # mt m16 x nb blocks x 2 n16 accumulators
            T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})

            tid = T.get_thread_binding()
            warp = tid // 32
            lane = tid % 32
            wm = warp % warps_m  # matches the emitter's warp binding (m first)
            wn = warp // warps_m
            m0 = bm * block_M
            T.clear(C_l)

            for kt in T.Pipelined(kt_per_split, num_stages=num_stages):
                kc0 = (bk * kt_per_split + kt) * kchunks
                # (a predicated zero-fill load would not be pipelined as cp.async by T.Pipelined)
                for i, kk in T.Parallel(block_M, block_K):
                    A_s[i, kk] = x[(m0 + i) % M, kc0 * 32 + kk]
                T.copy(Wq[bx * nblk, kc0, 0], Wq_s)
                T.copy(Sq[bx * nblk, kc0, 0], S_s)
                for c in T.unroll(kchunks):
                    for b in T.unroll(nb):
                        for v in T.vectorized(4):
                            q_l[b * 4 + v] = Wq_s[wn * nb + b, c, lane * 4 + v]
                        for v in T.vectorized(2):
                            s_l[b * 2 + v] = S_s[wn * nb + b, c, (lane // 4) * 2 + v]
                        T.call_extern(
                            "handle",
                            "nvfp4_w4a16_dequant_chunk",
                            T.access_ptr(q_l[b * 4], "r", extent=4),
                            T.access_ptr(s_l[b * 2], "r", extent=2),
                            T.access_ptr(B_l[b * 32], "w", extent=32),
                        )
                    for s in T.unroll(2):
                        emitter.ldmatrix_a(A_l, A_s, c * 2 + s)
                        for i in T.unroll(mt):
                            for b in T.unroll(nb):
                                for j in T.unroll(2):
                                    _mma_pair(A_l, i * 8, B_l, b * 32 + (s * 2 + j) * 8, C_l, ((i * nb + b) * 2 + j) * 8)

            # C_l[((i*nb+b)*2+j)*8 + r] sits at row 16i + 8*((r%4)//2) + lane//4,
            #                                   col 32b + 16j + 8*(r//4) + 2*(lane%4) + r%2   (warp-relative)
            for i in T.unroll(mt):
                for b in T.unroll(nb):
                    for j in T.unroll(2):
                        for r2 in T.unroll(4):
                            grow = m0 + wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                            gcol = bx * block_N + wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                            if grow < M:
                                if split_k == 1:
                                    for v in T.vectorized(2):
                                        Y[grow, gcol + v] = T.cast(C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                                else:
                                    for v in T.vectorized(2):
                                        P[bk, grow, gcol + v] = C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]

            if split_k > 1:
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tid == 0:
                    ticket_s[0] = T.atomic_add(counters[bm * n_tiles + bx], 1, return_prev=True)
                T.sync_threads()
                if ticket_s[0] == split_k - 1:
                    T.call_extern("handle", "__threadfence")
                    for i, j in T.Parallel(block_M, block_N):
                        if m0 + i < M:
                            Y[m0 + i, bx * block_N + j] = T.cast(_partial_sum(P, m0 + i, bx * block_N + j), bf16)
                    if tid == 0:
                        counters[bm * n_tiles + bx] = 0

    return main


# ---------------------------------------------------------------------------------------------------------------
# Weight preparation (host side, once per weight).
def _to_int32(w64):
    """int64 holding uint32 bit patterns -> int32 with the same bits."""
    return torch.where(w64 >= 2**31, w64 - 2**32, w64).to(torch.int32)


def _fragment_maps(device):
    """Index maps of one 32x32 block.

    w_idx (1024,): nibble stream order [lane][word][element e] -> n_in * 32 + k_in  (e = 2r + h, see _DEQUANT_SRC)
    s_idx (64,):   scale byte order [g][st][x][j]               -> n_in * 2 + kb"""
    lane = torch.arange(32).view(32, 1, 1)
    w = torch.arange(4).view(1, 4, 1)
    e = torch.arange(8).view(1, 1, 8)
    g, t = lane // 4, lane % 4
    st, j = w // 2, w % 2
    r, h = e // 2, e % 2
    n_in = 16 * j + g + 8 * (r // 2)
    k_in = 16 * st + 2 * t + h + 8 * (r % 2)
    w_idx = (n_in * 32 + k_in).reshape(-1)
    b = torch.arange(64)
    gb, sb, xb, jb = b // 8, (b // 4) % 2, (b // 2) % 2, b % 2
    s_idx = (16 * jb + 8 * xb + gb) * 2 + sb
    return w_idx.to(device), s_idx.to(device)


def _encode_scales(scale):
    """E4M3 scales (N, K/16) -> (byte codes b with bf16(0x7000 | b << 4) == |scale| * 2^119, zero mask, negative mask)."""
    sc = scale.float()
    if not bool(torch.isfinite(sc).all()):
        raise ValueError("NaN block scales are not supported")
    zero = sc == 0
    neg = torch.signbit(sc) & ~zero
    bits = (sc.abs() * (2.0**119)).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    nz = ~zero
    # Holds for every finite E4M3 value (the dtype check in prepare_nvfp4_weight guarantees E4M3 input).
    if not (bool(((bits >> 12)[nz] == 7).all()) and bool(((bits & 0xF)[nz] == 0).all())):
        raise ValueError("block scales are not exactly representable in the fragment scale encoding")
    code = ((bits >> 4) & 0xFF).masked_fill(zero, 0).to(torch.uint8)
    return code, zero, neg


class NVFP4W4A16Weight(NamedTuple):
    """An NVFP4 weight in the fragment-ordered layout of this module (see the module docstring).

    Wq: int32 (N/32, K/32, 128), one 512-byte block per 32 (n) x 32 (k) weights, in mma B-fragment order.
    Sq: int32 (N/32, K/32, 16), the 64 re-encoded scale bytes of each block.
    alpha: fp32 (N,), the output multiplier times 2^7."""

    N: int
    K: int
    Wq: torch.Tensor
    Sq: torch.Tensor
    alpha: torch.Tensor

    @property
    def device(self):
        return self.Wq.device


def prepare_nvfp4_weight(packed, scale, alpha=1.0):
    """Re-lay out an NVFP4 checkpoint weight for ``nvfp4_w4a16_linear`` (once, at load time).

    packed: int8/uint8 (N, K/2), two E2M1 codes per byte, low nibble = even k.
    scale: float8_e4m3fn, or uint8 UE4M3 bytes, (N, K/16), one block scale per 16 consecutive k.
    alpha: float, one-element tensor or (N,) tensor multiplying each output column (1 / global_scale).
    Requires N % 128 == 0 and K % 256 == 0. NaN scales are rejected. On a CUDA device this also creates the
    device's launch workspace, so that a first call inside a CUDA-graph capture allocates nothing persistent."""
    if not isinstance(packed, torch.Tensor) or packed.dtype not in (torch.int8, torch.uint8) or packed.ndim != 2:
        raise TypeError(f"packed must be a 2-D int8/uint8 tensor, got {getattr(packed, 'dtype', type(packed))}")
    if not isinstance(scale, torch.Tensor) or scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise TypeError(f"scale must be a float8_e4m3fn or uint8 (UE4M3 bytes) tensor, got {getattr(scale, 'dtype', type(scale))}")
    N, K = packed.shape[0], packed.shape[1] * 2
    if N % 128 or K % 256:
        raise ValueError(f"need N % 128 == 0 and K % 256 == 0, got N={N}, K={K}")
    if tuple(scale.shape) != (N, K // 16):
        raise ValueError(f"scale must have shape {(N, K // 16)}, got {tuple(scale.shape)}")
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e4m3fn)
    dev = packed.device
    w_idx, s_idx = _fragment_maps(dev)
    scode, zero, neg = _encode_scales(scale.to(dev))

    Wq = torch.empty(N // 32, K // 32, 128, dtype=torch.int32, device=dev)
    rows = max(32, ((1 << 22) // K) // 32 * 32)  # bounded temporaries (~4M codes per slab)
    u8 = packed.view(torch.uint8)
    for n0 in range(0, N, rows):
        n1 = min(N, n0 + rows)
        u = u8[n0:n1]
        nib = torch.stack(((u & 0xF), (u >> 4)), dim=-1).reshape(n1 - n0, K)  # nib[n, k] = e2m1 code
        # exact scale folding: zero-scale blocks -> code 0, negative scales -> flipped code sign
        nib = nib ^ (neg[n0:n1].repeat_interleave(16, 1).to(torch.uint8) << 3)
        nib = nib.masked_fill(zero[n0:n1].repeat_interleave(16, 1), 0)
        nb = (n1 - n0) // 32
        blk = nib.view(nb, 32, K // 32, 32).permute(0, 2, 1, 3).reshape(nb, K // 32, 1024)
        stream = blk[:, :, w_idx].view(nb, K // 32, 32, 4, 8).to(torch.int64)  # [lane][word][e]
        word = torch.zeros(nb, K // 32, 32, 4, dtype=torch.int64, device=dev)
        for e in range(8):
            ps, p1, p0, pm = _SLOT_POS[e // 2][e % 2]
            c = stream[..., e]
            word |= ((c >> 3) & 1) << ps
            word |= ((c >> 2) & 1) << p1
            word |= ((c >> 1) & 1) << p0
            word |= (c & 1) << pm
        Wq[n0 // 32 : n1 // 32] = _to_int32(word).view(nb, K // 32, 128)
    sblk = scode.view(N // 32, 32, K // 32, 2).permute(0, 2, 1, 3).reshape(N // 32, K // 32, 64)
    Sq = sblk[:, :, s_idx].contiguous().view(torch.int32).contiguous()  # (N/32, K/32, 16)
    alpha = torch.as_tensor(alpha, dtype=torch.float32, device=dev).reshape(-1)
    if alpha.numel() not in (1, N):
        raise ValueError(f"alpha must be a scalar or have N={N} elements, got {alpha.numel()}")
    alpha = (alpha.expand(N) * _ALPHA_FOLD).contiguous()
    if dev.type == "cuda":
        _workspace(dev)
    return NVFP4W4A16Weight(N, K, Wq.contiguous(), Sq, alpha)


# ---------------------------------------------------------------------------------------------------------------
# Configurations. A config is FRAG, a FragConfig or a TileConfig.
class FragConfig(NamedTuple):
    """nvfp4_w4a16_frag_kernel: one tile of block_M >= M rows x 32 * n_warps columns, split-K over grid.y."""

    block_M: int
    n_warps: int
    block_K: int
    split_k: int
    num_stages: int


class TileConfig(NamedTuple):
    """nvfp4_w4a16_tile_kernel: block_M = warps_m * WM rows x block_N = warps_n * 32 * nb columns."""

    warps_m: int
    WM: int
    warps_n: int
    nb: int
    block_K: int
    split_k: int
    num_stages: int
    min_blocks: int = 1


# FRAG: the fragment kernel, run in FRAG_MAX_M-row passes, each with the FragConfig that frag_config() picks for
# its row count. A FragConfig instead fixes the tile for every pass (passes of FragConfig.block_M rows).
FRAG = "frag"


@functools.lru_cache(maxsize=None)
def _num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _choose_split(n_tiles, k_tiles, num_sms):
    """Smallest divisor of k_tiles giving >= num_sms CTAs, else the largest one keeping <= 2 * num_sms (else 1)."""
    divs = [d for d in range(1, k_tiles + 1) if k_tiles % d == 0]
    ok = [d for d in divs if n_tiles * d <= 2 * num_sms]
    if not ok:
        return 1
    enough = [d for d in ok if n_tiles * d >= num_sms]
    return enough[0] if enough else ok[-1]


def _choose_split_balanced(n_tiles, k_tiles, num_sms, max_split=8):
    """Tensor-bound sizes: the divisor of k_tiles that best balances CTAs over the SMs.

    Every CTA does the same work, so the busiest SM runs ceil(C / SMs) of them: efficiency (C/SMs) / ceil(C/SMs).
    Ties go to the smaller split (less FP32 partial traffic)."""
    best, best_eff = 1, -1.0
    for d in range(1, min(max_split, k_tiles) + 1):
        c = n_tiles * d
        if k_tiles % d or (d > 1 and c > 4 * num_sms):
            continue
        eff = (c / num_sms) / math.ceil(c / num_sms)
        if eff > best_eff + 1e-9:
            best, best_eff = d, eff
    return best


def frag_config(N, K, M, num_sms):
    """FragConfig for M <= 64 rows.

    M <= 32 is weight-stream bound: at least one wave of co-resident CTAs, 256 columns per CTA on wide layers and 3 k
    tiles in flight. M > 32 is tensor bound (the MMA work is block_M rows whatever M is): 128 columns per CTA and a
    split-K that balances CTAs over the SMs."""
    block_M = 16 if M <= 16 else (32 if M <= 32 else (48 if M <= 48 else 64))
    block_K = 64
    W = N // 32
    if block_M <= 32:
        n_warps = 8 if W >= 256 else 4
        while W % n_warps:
            n_warps //= 2
        split_k = _choose_split(W // n_warps, K // block_K, num_sms)
        num_stages = 3 if (block_M == 32 and n_warps == 8) else 4
    else:
        n_warps = 4
        while W % n_warps:
            n_warps //= 2
        split_k = _choose_split_balanced(W // n_warps, K // block_K, num_sms)
        num_stages = 3
    return FragConfig(block_M, n_warps, block_K, split_k, num_stages)


def get_config(N, K, M, num_sms=None):
    """The configuration nvfp4_w4a16_linear uses for one (N, K) weight and 1 <= M <= MAX_M rows.

    * M <= 96: FRAG, fragment-kernel passes (weight-stream bound; one weight read per 64 rows).
    * M > 96: the tile kernel with 64 x 128 tiles at two CTAs per SM and the split-K that best balances the CTAs
      over the SMs; above 512 rows, 128 x 256 tiles without split when they fill the SMs evenly (N % 256 == 0 and
      the last wave at least 75% full).

    The stream-K kernel and per-GPU tuned tables in maint/gemm/gemm_sm120/sm120_nvfp4_w4a16_streamk.py are faster
    above 96 rows on most shapes."""
    if M <= 96:
        return FRAG
    if num_sms is None:
        num_sms = _num_sms(torch.cuda.current_device())
    big_tiles = -(-M // 128) * (N // 256)
    if M > 512 and N % 256 == 0 and big_tiles / (num_sms * math.ceil(big_tiles / num_sms)) >= 0.75:
        return TileConfig(2, 64, 4, 2, 64, 1, 3)
    tiles = -(-M // 64) * (N // 128)
    return TileConfig(1, 64, 4, 1, 64, _choose_split_balanced(tiles, K // 64, 2 * num_sms), 3, min_blocks=2)


def smem_bytes(cfg):
    """Shared memory per CTA of a TileConfig."""
    block_M, nblk = cfg.warps_m * cfg.WM, cfg.warps_n * cfg.nb
    return cfg.num_stages * (block_M * cfg.block_K * 2 + nblk * (cfg.block_K // 32) * 576) + 4


# ---------------------------------------------------------------------------------------------------------------
# Launch.
_KERNELS = {}
_WORKSPACES = {}
_DUMMY_FLOATS = 128 * 256  # largest (block_M x block_N) partial tile any kernel of this example or maint/ binds


class _Workspace:
    """Per-device launch state: self-resetting split-K ticket counters and a dummy FP32 partials buffer (bound when a
    launch does not split, never touched)."""

    def __init__(self, device):
        self.counters = torch.zeros(_NUM_COUNTERS, dtype=torch.int32, device=device)
        self.dummy = torch.zeros(_DUMMY_FLOATS, dtype=torch.float32, device=device)

    def dummy_partials(self, shape):
        numel = math.prod(shape)
        if numel > _DUMMY_FLOATS:
            raise ValueError(f"dummy partials {shape} exceed the workspace")
        return self.dummy[:numel].view(shape)


def _workspace(device):
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    ws = _WORKSPACES.get(device)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            # torch.zeros would come from the graph's private pool and only record the memset; the tickets must be
            # zero and persistent before the first replay.
            raise RuntimeError(
                f"no NVFP4 W4A16 workspace on {device}: call prepare_nvfp4_weight or nvfp4_w4a16_linear once "
                "on this device before capturing a CUDA graph"
            )
        ws = _WORKSPACES[device] = _Workspace(device)
    return ws


def _kernel(key, factory, *args):
    # Kept on purpose next to tilelang.jit's own cache: a dict lookup skips the per-call cache-key construction of
    # the JIT factory, which is measurable next to a 10-30 us decode GEMM when it is not replayed from a CUDA graph.
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = factory(*args)
    return kernel


def _check_counters(n, what):
    if n > _NUM_COUNTERS:
        raise ValueError(f"{what}: {n} split tiles exceed the {_NUM_COUNTERS} ticket counters")


def _run_frag(x, y, w, cfg):
    M = x.shape[0]
    if cfg.block_M < M:
        raise ValueError(f"FragConfig block_M={cfg.block_M} is smaller than M={M}")
    if M > 16 and M % 16:
        # The fragment kernel maps padding rows to row M - 1; with many of them every CTA hits the same L2 lines.
        # Run the tile kernel with the same tile instead: its padding rows wrap to distinct rows.
        _run_tile(x, y, w, TileConfig(1, cfg.block_M, cfg.n_warps, 1, cfg.block_K, cfg.split_k, cfg.num_stages))
        return
    ws = _workspace(x.device)
    _check_counters(w.N // (32 * cfg.n_warps), f"{cfg}")
    kernel = _kernel(("frag", w.N, w.K) + tuple(cfg), nvfp4_w4a16_frag_kernel, w.N, w.K, *cfg)
    if cfg.split_k > 1:
        P = torch.empty(cfg.split_k, M, w.N, dtype=torch.float32, device=x.device)
    else:
        P = ws.dummy_partials((1, 1, 1))
    kernel(x, y, w.Wq, w.Sq, w.alpha, P, ws.counters)


def _run_tile(x, y, w, cfg):
    M = x.shape[0]
    ws = _workspace(x.device)
    block_M, block_N = cfg.warps_m * cfg.WM, cfg.warps_n * 32 * cfg.nb
    _check_counters(-(-M // block_M) * (w.N // block_N), f"M={M}, {cfg}")
    kernel = _kernel(("tile", w.N, w.K) + tuple(cfg), nvfp4_w4a16_tile_kernel, w.N, w.K, *cfg)
    if cfg.split_k > 1:
        P = torch.empty(cfg.split_k, M, w.N, dtype=torch.float32, device=x.device)
    else:
        P = ws.dummy_partials((1, 1, 1))
    kernel(x, y, w.Wq, w.Sq, w.alpha, P, ws.counters)


def launch(x, y, w, cfg):
    """Run one chunk (M <= MAX_M rows) with a FRAG, FragConfig or TileConfig config."""
    M = x.shape[0]
    if isinstance(cfg, FragConfig):
        for r0 in range(0, M, cfg.block_M):
            _run_frag(x[r0 : r0 + cfg.block_M], y[r0 : r0 + cfg.block_M], w, cfg)
    elif cfg == FRAG:
        num_sms = _num_sms(x.device.index)
        for r0 in range(0, M, FRAG_MAX_M):
            xs, ys = x[r0 : r0 + FRAG_MAX_M], y[r0 : r0 + FRAG_MAX_M]
            _run_frag(xs, ys, w, frag_config(w.N, w.K, xs.shape[0], num_sms))
    elif isinstance(cfg, TileConfig):
        _run_tile(x, y, w, cfg)
    else:
        raise TypeError(f"unknown config {cfg!r}: expected FRAG, a FragConfig or a TileConfig")


def check_inputs(x, weight, out=None):
    """Validate the arguments of nvfp4_w4a16_linear; returns (x, out) with x contiguous and 16-byte aligned and out
    allocated if it was None."""
    if not isinstance(weight, NVFP4W4A16Weight):
        raise TypeError(f"weight must come from prepare_nvfp4_weight, got {type(weight).__name__}")
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != weight.K:
        raise ValueError(
            f"x must be a bfloat16 (M, {weight.K}) tensor, got {getattr(x, 'dtype', type(x))} {tuple(getattr(x, 'shape', ()))}"
        )
    if x.device != weight.device:
        raise ValueError(f"x is on {x.device} but the weight is on {weight.device}")
    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()  # the kernels load x in 16-byte vectors
    M = x.shape[0]
    if out is None:
        out = torch.empty(M, weight.N, dtype=torch.bfloat16, device=x.device)
    elif (
        out.dtype != torch.bfloat16
        or tuple(out.shape) != (M, weight.N)
        or not out.is_contiguous()
        or out.device != x.device
        or out.data_ptr() % 16
    ):
        raise ValueError(f"out must be a contiguous, 16-byte aligned bfloat16 ({M}, {weight.N}) tensor on {x.device}")
    return x, out


def nvfp4_w4a16_linear(x, weight, config=None, out=None):
    """y = x @ W^T in BF16 for x (M, K) BF16 and a prepared NVFP4 weight (see prepare_nvfp4_weight).

    config: None (get_config() per chunk), FRAG, a FragConfig or a TileConfig; it applies to every chunk.
    Deterministic and CUDA-graph capturable (per-call partials are allocated with torch.empty, split-K tickets live in
    a per-device workspace created by prepare_nvfp4_weight and reset themselves). Batches above MAX_M rows run as
    MAX_M-row chunks. Each (N, K, config) compiles on first use."""
    x, out = check_inputs(x, weight, out)
    for r0 in range(0, x.shape[0], MAX_M):
        xs, ys = x[r0 : r0 + MAX_M], out[r0 : r0 + MAX_M]
        cfg = config if config is not None else get_config(weight.N, weight.K, xs.shape[0], _num_sms(x.device.index))
        launch(xs, ys, weight, cfg)
    return out


def is_supported_device(device=None):
    """True on the GPUs this example is tested and tuned on (compute capability 12.0)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) == (12, 0)


# ---------------------------------------------------------------------------------------------------------------
# Command line example.
def _nvfp4_helpers():
    """The NVFP4 reference helpers in examples/dequantize_gemm/quantize.

    The repository root is put on sys.path on purpose, so that ``python examples/gemm_sm120/...`` works as well as
    importing this module from the repository root (tests, maint/ benchmarks)."""
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from examples.dequantize_gemm.quantize.nvfp4 import (
        decode_packed_fp4_e2m1,
        decode_ue4m3_scale_bytes,
        quantize_bf16_to_nvfp4_blockscaled,
    )

    return quantize_bf16_to_nvfp4_blockscaled, decode_packed_fp4_e2m1, decode_ue4m3_scale_bytes


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--global-scale", type=float, default=448.0, help="NVFP4 global weight scale (alpha = 1 / global scale)")
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="cupti")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    from tilelang.profiler import do_bench

    args = parse_args(argv)
    if not is_supported_device():
        raise RuntimeError("this example is tested on SM120 (compute capability 12.0) GPUs only")
    quantize, decode_fp4, decode_scales = _nvfp4_helpers()
    torch.manual_seed(args.seed)
    M, N, K = args.m, args.n, args.k

    # An NVFP4 "checkpoint" weight: quantize a random BF16 weight scaled by the global scale.
    w_bf16 = (torch.randn(N, K, device="cuda") * 0.02 * args.global_scale).to(torch.bfloat16)
    packed, _, scale_bytes = quantize(w_bf16, return_scale_bytes=True)
    alpha = 1.0 / args.global_scale
    weight = prepare_nvfp4_weight(packed, scale_bytes, alpha)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    y = nvfp4_w4a16_linear(x, weight)
    print(f"Shape: M={M}, N={N}, K={K}, config={get_config(N, K, M)}")
    if args.verify:
        w_ref = decode_fp4(packed) * decode_scales(scale_bytes).repeat_interleave(16, dim=1) * alpha
        ref = x.float() @ w_ref.T
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        assert rel < 2.5e-3, f"relative error {rel:.3e}"
        assert torch.equal(y, nvfp4_w4a16_linear(x, weight)), "not deterministic"
        print(f"TileLang correctness: passed (relative error {rel:.2e})")

    latency_ms = do_bench(lambda: nvfp4_w4a16_linear(x, weight), backend=args.backend)
    ref_ms = do_bench(lambda: x @ w_bf16.T, backend=args.backend)
    tflops = 2 * M * N * K / latency_ms / 1e9
    weight_gbs = (N * K // 2 + N * K // 16) / latency_ms / 1e6
    print(f"TileLang W4A16 latency: {latency_ms * 1e3:.2f} us, {tflops:.2f} TFLOPS, {weight_gbs:.0f} GB/s (weights)")
    print(f"torch BF16 matmul latency: {ref_ms * 1e3:.2f} us")


if __name__ == "__main__":
    main()
