"""Fused lightning-indexer scoring and top-k selection for DeepSeek sparse attention.

The indexer scores key ``j`` for query ``i`` as ``sum_h w[i, h] * relu(q[i, h] . k[j])``
and every query keeps the ``topk`` best keys of its own key range. One SM90 kernel
does both: a block scores a tile of keys for two queries at a time and keeps their
running top-k in shared memory, so the [Sq, Sk] score matrix is never written out.
"""

import torch
import tilelang
import tilelang.language as T

_THREADS = 256  # two warp groups, one per query of a block
_PASS_CONFIG = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    # Barriers are placed by hand.
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    # Buffer liveness does not model TMA writes that are still in flight.
    tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
}


def _coarse_bucket(score):
    """Top byte of the float16 pattern of ``score``, remapped to grow with it."""
    bits = T.reinterpret(T.cast(score, T.float16), T.uint16)
    return T.cast(T.if_then_else(score < 0, ~bits & 0xFFFF, bits | 0x8000) >> 8, T.int32)


def _ordered_bits(score):
    bits = T.reinterpret(score, T.uint32)
    return T.if_then_else(score < 0, ~bits, bits | T.uint32(0x80000000))


def _digit(score, idx, byte):
    """Byte ``byte`` of the sort key: four score bytes, then four bytes of the inverted id."""
    bits = T.if_then_else(byte < 4, _ordered_bits(score), ~T.cast(idx, T.uint32))
    return T.cast((bits >> (24 - (byte % 4) * 8)) & 0xFF, T.int32)


@T.macro
def _suffix_counts(hist, pivot, need):
    """Turn bucket counts into suffix counts and find the bucket holding rank ``need``.

    On return hist[b] counts the entries in buckets >= b, pivot[0] is the bucket
    with hist[b] >= need > hist[b + 1] and pivot[1] is hist[b + 1]. Warp 0 scans
    the 256 buckets, eight per lane. The two barriers publish the counts to it
    and its result to everyone. At least ``need`` entries must have been counted.
    """
    tx = T.get_thread_binding()
    total = T.alloc_var(T.int32)
    above = T.alloc_var(T.int32)
    count = T.alloc_var(T.int32)
    T.sync_threads()
    if tx < 32:
        total = 0
        for j in T.unroll(8):
            total += hist[tx * 8 + j]
        above = total
        for s in T.unroll(5):
            other = T.shfl_down(above, 1 << s)
            if tx + (1 << s) < 32:
                above += other
        above -= total
        for j in T.unroll(8):
            b = tx * 8 + 7 - j
            count = above + hist[b]
            if count >= need and above < need:
                pivot[0] = b
                pivot[1] = above
            hist[b] = count
            above = count
    T.sync_threads()


@T.macro
def _select_topk(val, idx, val_out, idx_out, hist, pivot, n_tied, tied, tau, cnt, bq):
    """Keep the best ``topk`` of the cnt[bq] > topk candidates of query ``bq``.

    Candidates are ordered by score, then by smaller key id. A coarse radix pass
    settles every bucket above the one holding the k-th score; the entries of
    that bucket are refined a byte at a time, four score bytes and then four id
    bytes.

    All threads call it after a barrier. On return tau[bq] is the k-th score
    and cnt[bq] = topk.
    """
    tx = T.get_thread_binding()
    topk = val_out.shape[0]
    total = T.alloc_var(T.int32)
    need = T.alloc_var(T.int32)
    cut = T.alloc_var(T.int32)
    bucket = T.alloc_var(T.int32)
    slot = T.alloc_var(T.int32)
    n = T.alloc_var(T.int32)
    total = cnt[bq]
    need = topk
    for b in T.Parallel(257):
        hist[b] = 0
    if tx == 0:
        n_tied[0] = 0
    T.sync_threads()
    for i in T.serial(T.ceildiv(total, _THREADS)):
        c = i * _THREADS + tx
        if c < total:
            T.atomic_add(hist[_coarse_bucket(val[bq, c])], 1)
    _suffix_counts(hist, pivot, need)
    cut = pivot[0]
    need -= pivot[1]
    for i in T.serial(T.ceildiv(total, _THREADS)):
        c = i * _THREADS + tx
        if c < total:
            bucket = _coarse_bucket(val[bq, c])
            if bucket > cut:
                # hist[b + 1] counts the entries above bucket b: its first output slot.
                slot = T.atomic_add(hist[bucket + 1], 1, return_prev=True)
                val_out[slot] = val[bq, c]
                idx_out[slot] = idx[bq, c]
            elif bucket == cut:
                slot = T.atomic_add(n_tied[0], 1, return_prev=True)
                tied[0, slot] = c
    for byte in T.serial(8):
        if need > 0:
            src = byte % 2
            base = topk - need
            T.sync_threads()
            n = n_tied[src]
            if n == need:
                # The whole bucket is kept.
                for i in T.serial(T.ceildiv(n, _THREADS)):
                    t = i * _THREADS + tx
                    if t < n:
                        c = tied[src, t]
                        val_out[base + t] = val[bq, c]
                        idx_out[base + t] = idx[bq, c]
                need = 0
            else:
                for b in T.Parallel(257):
                    hist[b] = 0
                if tx == 0:
                    n_tied[src ^ 1] = 0
                T.sync_threads()
                for i in T.serial(T.ceildiv(n, _THREADS)):
                    t = i * _THREADS + tx
                    if t < n:
                        c = tied[src, t]
                        T.atomic_add(hist[_digit(val[bq, c], idx[bq, c], byte)], 1)
                _suffix_counts(hist, pivot, need)
                cut = pivot[0]
                need -= pivot[1]
                for i in T.serial(T.ceildiv(n, _THREADS)):
                    t = i * _THREADS + tx
                    if t < n:
                        c = tied[src, t]
                        bucket = _digit(val[bq, c], idx[bq, c], byte)
                        # Ids are unique: after the last byte the pivot bucket holds one entry.
                        if bucket > cut or (byte == 7 and bucket == cut):
                            slot = base + T.atomic_add(hist[bucket + 1], 1, return_prev=True)
                            val_out[slot] = val[bq, c]
                            idx_out[slot] = idx[bq, c]
                        elif bucket == cut:
                            slot = T.atomic_add(n_tied[src ^ 1], 1, return_prev=True)
                            tied[src ^ 1, slot] = c
    T.sync_threads()
    for i in T.Parallel(topk):
        val[bq, i] = val_out[i]
        idx[bq, i] = idx_out[i]
    kept = T.alloc_fragment((topk,), T.float32)
    kth = T.alloc_fragment((1,), T.float32)
    T.copy(val_out, kept)
    T.reduce_min(kept, kth, dim=0, clear=True)
    if tx == 0:
        tau[bq] = kth[0]
        cnt[bq] = topk
    T.sync_threads()


@tilelang.jit(pass_configs=_PASS_CONFIG)
def _indexer_topk_kernel(heads, dim, topk):
    Sq = T.dynamic("Sq")
    Sk = T.dynamic("Sk")
    block_Q = _THREADS // 128
    # 64-key tiles let two blocks share an SM up to topk = 512.
    block_N = 64 if topk <= 512 else 128
    capacity = 2 * topk

    @T.prim_func
    def dsa_indexer_topk(
        IndexQ: T.Tensor((Sq, heads, dim), T.bfloat16),
        Weights: T.Tensor((Sq, heads), T.bfloat16),
        IndexK: T.Tensor((Sk, dim), T.bfloat16),
        KeyRange: T.Tensor((Sq, 2), T.int32),
        TopkIndices: T.Tensor((Sq, topk), T.int32),
        TopkLength: T.Tensor((Sq,), T.int32),
    ):
        with T.Kernel(T.ceildiv(Sq, block_Q), threads=_THREADS) as bx:
            tx = T.get_thread_binding()
            q0 = bx * block_Q

            # Candidates of each query live in [0, cnt); tau is the k-th score once topk are known.
            val = T.alloc_shared((block_Q, capacity), T.float32)
            idx = T.alloc_shared((block_Q, capacity), T.int32)
            cnt = T.alloc_shared((block_Q,), T.int32)
            tau = T.alloc_shared((block_Q,), T.float32)
            val_out = T.alloc_shared((topk,), T.float32)
            idx_out = T.alloc_shared((topk,), T.int32)
            hist = T.alloc_shared((257,), T.int32)
            pivot = T.alloc_shared((2,), T.int32)
            n_tied = T.alloc_shared((2,), T.int32)
            tied = T.alloc_shared((2, capacity), T.int32)
            q_s = T.alloc_shared((block_Q * heads, dim), T.bfloat16)
            k_s = T.alloc_shared((2, block_N, dim), T.bfloat16)
            loaded = T.alloc_barrier([1, 1])

            acc = T.alloc_fragment((block_N, block_Q * heads), T.float32)
            score = T.alloc_fragment((block_N, block_Q), T.float32)
            w = T.alloc_fragment((block_Q, heads), T.float32)
            lo = T.alloc_fragment((block_Q,), T.int32)
            hi = T.alloc_fragment((block_Q,), T.int32)
            threshold = T.alloc_fragment((block_Q,), T.float32)
            first_key = T.alloc_var(T.int32)
            end_key = T.alloc_var(T.int32)

            # The block scans the union of its queries' ranges; each query masks to its own.
            # With an odd Sq the last block repeats the last query.
            first_key = Sk
            end_key = 0
            for bq in T.serial(block_Q):
                qi = T.min(q0 + bq, Sq - 1)
                first_key = T.min(first_key, T.max(KeyRange[qi, 0], 0))
                end_key = T.max(end_key, T.min(KeyRange[qi, 1], Sk))
            for bq in T.Parallel(block_Q):
                qi = T.min(q0 + bq, Sq - 1)
                lo[bq] = T.max(KeyRange[qi, 0], 0)
                hi[bq] = T.min(KeyRange[qi, 1], Sk)
            for bq, h in T.Parallel(block_Q, heads):
                w[bq, h] = Weights[T.min(q0 + bq, Sq - 1), h]
            for i, d in T.Parallel(block_Q * heads, dim):
                q_s[i, d] = IndexQ[T.min(q0 + i // heads, Sq - 1), i % heads, d]
            T.fill(cnt, 0)
            T.fill(tau, -T.infinity(T.float32))

            n_tiles = T.ceildiv(end_key - first_key, block_N)
            if n_tiles > 0:
                T.tma_copy(IndexK[first_key : first_key + block_N, :], k_s[0, :, :], barrier=loaded[0], annotations={"emit_arrive": 1})
            T.sync_threads()
            for pair in T.serial(T.ceildiv(n_tiles, 2)):
                for bq in T.serial(block_Q):
                    if cnt[bq] + 2 * block_N > capacity:
                        _select_topk(val, idx, val_out, idx_out, hist, pivot, n_tied, tied, tau, cnt, bq)
                for bq in T.Parallel(block_Q):
                    threshold[bq] = tau[bq]
                # Every thread has read cnt above before anyone appends again.
                T.sync_threads()
                for stage in T.unroll(2):
                    tile = 2 * pair + stage
                    if tile < n_tiles:
                        # The next tile loads into the other buffer while this one is scored.
                        if tile + 1 < n_tiles:
                            next_key = first_key + (tile + 1) * block_N
                            T.tma_copy(
                                IndexK[next_key : next_key + block_N, :],
                                k_s[1 - stage, :, :],
                                barrier=loaded[1 - stage],
                                annotations={"emit_arrive": 1},
                            )
                        T.mbarrier_wait_parity(loaded[stage], pair % 2)
                        T.gemm(k_s[stage, :, :], q_s, acc, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                        acc_h = T.reshape(acc, (block_N, block_Q, heads))
                        for i, bq, h in T.Parallel(block_N, block_Q, heads):
                            acc_h[i, bq, h] = T.max(acc_h[i, bq, h], 0) * w[bq, h]
                        T.reduce_sum(acc_h, score, dim=-1, clear=True)
                        # Keys arrive in increasing order, so one that only ties tau has a larger
                        # id than every candidate already kept and can be dropped.
                        for i, bq in T.Parallel(block_N, block_Q):
                            key = first_key + tile * block_N + i
                            if key >= lo[bq] and key < hi[bq] and score[i, bq] > threshold[bq]:
                                slot = T.alloc_var(T.int32)
                                slot = T.atomic_add(cnt[bq], 1, return_prev=True)
                                val[bq, slot] = score[i, bq]
                                idx[bq, slot] = key
                        # Publishes the appends and releases this tile's buffer for the tile after next.
                        T.sync_threads()

            for bq in T.serial(block_Q):
                if cnt[bq] > topk:
                    _select_topk(val, idx, val_out, idx_out, hist, pivot, n_tied, tied, tau, cnt, bq)
            for bq, i in T.Parallel(block_Q, topk):
                if q0 + bq < Sq:
                    TopkIndices[q0 + bq, i] = T.if_then_else(i < cnt[bq], idx[bq, i], -1)
            if tx < block_Q and q0 + tx < Sq:
                TopkLength[q0 + tx] = cnt[tx]

    return dsa_indexer_topk


def causal_key_ranges(positions, cu_seqlens=None):
    """Key range [document start, position + 1) of queries sitting at ``positions``.

    ``cu_seqlens`` holds the document boundaries of the packed key sequence,
    from 0 to Sk; without it the keys form one document.
    """
    positions = positions.long()
    if cu_seqlens is None:
        start = torch.zeros_like(positions)
    else:
        cu_seqlens = cu_seqlens.to(positions.device).long()
        start = cu_seqlens[torch.bucketize(positions, cu_seqlens[1:-1], right=True)]
    return torch.stack([start, positions + 1], dim=1).int()


def indexer_topk(q, weights, k, key_ranges, topk):
    """Select the ``topk`` best keys of every query by indexer score.

    ``q`` [Sq, H, D], ``weights`` [Sq, H] and ``k`` [Sk, D] are contiguous
    bfloat16 CUDA tensors, with H 32 or 64 and D 64 or 128. ``key_ranges``
    [Sq, 2] int32 gives each query a half-open range [lo, hi) of keys, clipped
    to [0, Sk). Queries are processed in pairs that scan the union of their
    ranges, so neighbors should have similar ranges.

    Returns ``indices`` [Sq, topk] int32 and ``lengths`` [Sq] int32. Row i holds
    the ids of its min(topk, hi - lo) best keys in no particular order, then -1.
    Scores are accumulated in fp32 and must stay finite: a key whose score
    overflows to -inf or becomes NaN is never selected. Equal scores go to the
    smaller key id.
    """
    sq, heads, dim = q.shape
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("This example requires an SM90 GPU")
    # A candidate list holds topk entries plus two key tiles of at least 64 keys,
    # and two lists of 2 * 2048 entries fill shared memory. Other H and D are not tested.
    if topk not in (128, 256, 512, 1024, 2048) or heads not in (32, 64) or dim not in (64, 128):
        raise ValueError("topk must be a power of two in [128, 2048], H 32 or 64 and D 64 or 128")
    # The kernel call checks shapes, dtypes and strides, but not alignment.
    if any(x.data_ptr() % 16 for x in (q, weights, k)):
        raise ValueError("q, weights and k must be 16-byte aligned")
    if sq == 0 or k.shape[0] == 0:
        return q.new_full((sq, topk), -1, dtype=torch.int32), q.new_zeros((sq,), dtype=torch.int32)
    indices = torch.empty((sq, topk), dtype=torch.int32, device=q.device)
    lengths = torch.empty((sq,), dtype=torch.int32, device=q.device)
    _indexer_topk_kernel(heads, dim, topk)(q, weights, k, key_ranges, indices, lengths)
    return indices, lengths


def reference_scores(q, weights, k):
    """Dense fp32 scores [Sq, Sk] of the bfloat16 inputs."""
    logits = torch.relu(torch.einsum("qhd,kd->qhk", q.float(), k.float()))
    return torch.einsum("qhk,qh->qk", logits, weights.float())


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len, heads, dim, topk = 4096, 64, 128, 2048
    q = torch.randn(seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, heads, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq_len, dim, device="cuda", dtype=torch.bfloat16)
    ranges = causal_key_ranges(torch.arange(seq_len, device="cuda"))
    indices, lengths = indexer_topk(q, weights, k, ranges, topk)

    # The last query sees every key.
    expected = reference_scores(q[-1:], weights[-1:], k).topk(topk).indices
    overlap = torch.isin(indices[-1], expected).float().mean().item()
    print(f"lengths {lengths[0].item()}..{lengths[-1].item()}, last row overlap with torch.topk: {overlap:.4f}")
