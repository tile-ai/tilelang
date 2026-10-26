"""Streaming exact, unsorted Top-K in TileLang (no custom CUDA or inline PTX).

Inspired by https://github.com/deepseek-ai/DeepSelect.
"""

import math

import torch
import tilelang
import tilelang.language as T

if __package__:
    from .deep_select_config import select_config
else:
    from deep_select_config import select_config


def _ordered_key(value, dtype):
    if dtype == "bfloat16":
        bits = T.cast(T.reinterpret(value, T.uint16), T.uint32)
        return T.if_then_else((bits & 0x8000) != 0, ~bits & 0xFFFF, bits | 0x8000)
    bits = T.reinterpret(value, T.uint32)
    return T.if_then_else((bits & T.uint32(0x80000000)) != 0, ~bits, bits | T.uint32(0x80000000))


def _key_value(key, dtype):
    if dtype == "bfloat16":
        bits = T.cast(T.if_then_else((key & 0x8000) != 0, key & 0x7FFF, ~key & 0xFFFF), "uint16")
        return T.reinterpret(bits, "bfloat16")
    bits = T.if_then_else((key & T.uint32(0x80000000)) != 0, key & T.uint32(0x7FFFFFFF), ~key)
    return T.reinterpret(bits, "float32")


@T.macro
def _radix_threshold(keys, histogram, scratch, cutoff, k, bits, msb_ready=False):
    """MSB-first 8-bit radix selection with shared increment-one histograms.

    Collective over the whole CTA. ``keys`` is a per-thread uint32 vector;
    zero denotes padding and at least K nonzero keys must exist across the CTA.
    ``histogram`` has at least 257 shared counters; ``scratch`` is shared
    uint32[2, warps] with at least two warps. ``cutoff`` is local uint32[1].

    The entry barrier allows histogram storage to alias the shared candidate
    keys, once they have been loaded into registers. The exit barrier protects
    the broadcast scratch before reuse. Candidate indices need not be live yet.
    BF16 refines two bytes; FP32 may stop at a whole-bucket lower boundary.
    Bucket 256 is a write-only sink for keys outside the current prefix.
    Padding may enter bucket zero: at least K positive keys exist, so it
    cannot change the Kth largest key or the positive output cutoff.
    ``msb_ready`` reuses a complete first-byte histogram built by the caller;
    the entry barrier publishes it. Subsequent digits still clear and rebuild.
    """
    tx = T.get_thread_binding()
    lane = tx % 32
    prefix = T.alloc_var("uint32")
    prefix_mask = T.alloc_var("uint32")
    rank = T.alloc_var("int32")
    bucket = T.alloc_var("uint32")
    counts = T.alloc_local((8,), "int32")
    suffix = T.alloc_var("int32")
    lane_total = T.alloc_var("int32")
    pivot_bucket = T.alloc_var("uint32")
    pivot_rank = T.alloc_var("uint32")
    whole_bucket = T.alloc_var("uint32")
    prefix = 0
    prefix_mask = 0
    rank = k
    # In streaming mode the histogram aliases the old candidate buffer.
    T.sync_threads()
    for digit in T.unroll(bits // 8):
        if not (msb_ready and digit == 0):
            for bucket_id in T.Parallel(257):
                histogram[bucket_id] = 0
            T.sync_threads()
            for j in T.unroll(keys.shape[0]):
                bucket = T.if_then_else(
                    (keys[j] & prefix_mask) == prefix,
                    (keys[j] >> (bits - 8 - digit * 8)) & 255,
                    T.uint32(256),
                )
                # Like DeepSelect, count nonmatching keys in an unread sink rather
                # than branching around the increment-one shared atomic.
                T.atomic_add(histogram[bucket], T.uint32(1))
        T.sync_threads()
        if tx < 32:
            # As in DeepSelect, one warp searches all 256 buckets: eight
            # consecutive counters per lane and an exclusive suffix scan.
            suffix = 0
            for j in T.unroll(8):
                counts[j] = T.cast(histogram[lane * 8 + j], "int32")
                suffix += counts[j]
            lane_total = suffix
            for s in T.unroll(5):
                value = T.shfl_down(suffix, 1 << s)
                if lane + (1 << s) < 32:
                    suffix += value
            suffix -= lane_total
            pivot_bucket = 0
            pivot_rank = 0
            whole_bucket = 0
            for j in T.unroll(8):
                if suffix < rank and rank <= suffix + counts[7 - j]:
                    pivot_bucket = T.cast(lane * 8 + 7 - j, "uint32")
                    pivot_rank = T.cast(rank - suffix, "uint32")
                    if bits == 32:
                        whole_bucket = T.cast(rank == suffix + counts[7 - j], "uint32")
                suffix += counts[7 - j]
            pivot_bucket = T.warp_reduce_sum(pivot_bucket)
            pivot_rank = T.warp_reduce_sum(pivot_rank)
            if bits == 32:
                whole_bucket = T.warp_reduce_sum(whole_bucket)
            if lane == 0:
                scratch[0, 0] = pivot_bucket
                scratch[1, 0] = pivot_rank
                if bits == 32:
                    scratch[0, 1] = whole_bucket
        T.sync_threads()
        prefix = prefix | (scratch[0, 0] << (bits - 8 - digit * 8))
        prefix_mask = prefix_mask | (T.uint32(255) << (bits - 8 - digit * 8))
        rank = T.cast(scratch[1, 0], "int32")
        # FP32 can finish at a whole-bucket boundary, using its conservative
        # lower edge as the threshold. BF16 always refines both bytes.
        if bits == 32:  # noqa: SIM102 - specialize before reading the FP32-only flag
            if scratch[0, 1] != 0:
                break
    # Do not decode an early-exit floor below -inf into a negative NaN.
    cutoff[0] = T.max(prefix, T.uint32(0x007FFFFF)) if bits == 32 else prefix
    T.sync_threads()


def _warp_exclusive_sum(count, max_count):
    """Exclusive sum of nonnegative lane counts bounded by max_count.

    Every lane of the calling warp must participate. Independent bit-plane
    ballots avoid a five-stage dependent shuffle/add scan, including when the
    inclusive bound is a power of two.
    """
    lower_lanes = (T.uint32(1) << (T.get_thread_binding() % 32)) - T.uint32(1)
    prefix = T.int32(0)
    for bit in range(int(max_count).bit_length()):
        votes = T.cast(T.ballot(((count >> bit) & 1) != 0), "uint32")
        prefix += T.cast(T.popcount(votes & lower_lanes), "int32") << bit
    return prefix


@T.macro
def _output_offsets(keys, cutoff, scratch, offsets, k):
    """CTA-wide selection: offsets receives an output start and equal-key quota.

    Pack greater/equal counts into two uint16 fields. The entire tile must have
    fewer than 65536 keys, so neither additions nor exclusive-prefix subtraction
    carry between the fields. All greater keys and exactly the remaining equal
    keys get disjoint positions, interleaved in thread order in a single pass.
    The cutoff must satisfy greater <= K <= greater + equal. All threads
    participate; scratch is uint32[2, warps]. Synchronize before reusing it.
    """
    tx = T.get_thread_binding()
    lane = tx % 32
    greater = T.alloc_var("int32")
    equal = T.alloc_var("int32")
    packed = T.alloc_var("uint32")
    inclusive = T.alloc_var("uint32")
    greater = 0
    equal = 0
    for j in T.unroll(keys.shape[0]):
        greater += T.cast(keys[j] > cutoff[0], "int32")
        equal += T.cast(keys[j] == cutoff[0], "int32")
    packed = (T.cast(greater, "uint32") << 16) | T.cast(equal, "uint32")
    inclusive = packed
    for bit in T.unroll(5):
        value = T.shfl_up(inclusive, 1 << bit)
        if lane >= (1 << bit):
            inclusive += value
    warp_total = T.warp_reduce_sum(packed)
    # Keep lane 0 as publisher (CUDA 13 miscompiles some lane-31 stores).
    if lane == 0:
        scratch[0, tx // 32] = warp_total
    T.sync_threads()
    stored = T.if_then_else(lane < scratch.shape[1], scratch[0, lane], T.uint32(0))
    total = T.warp_reduce_sum(stored)
    off_warp = T.warp_reduce_sum(T.if_then_else(lane < tx // 32, stored, T.uint32(0)))
    prefix = inclusive - packed + off_warp
    equal_budget = k - T.cast(total >> 16, "int32")
    preceding_equal = T.cast(prefix & 65535, "int32")
    offsets[0] = T.cast(prefix >> 16, "int32") + T.min(preceding_equal, equal_budget)
    offsets[1] = T.max(0, equal_budget - preceding_equal)


_PASS_CONFIG = {tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True}


@tilelang.jit(pass_configs=_PASS_CONFIG)
def _register_topk(m, n, k, dtype, index_dtype, splits=1, threads=256, indirect=False, return_values=True):
    """One register-resident tile; also used for exact hierarchical merging."""
    shard = (n + splits - 1) // splits
    slots = (shard + threads - 1) // threads

    @T.prim_func
    def kernel(
        X: T.Tensor((m, n), dtype),
        InputIndex: T.Tensor((m, n if indirect else splits * k), index_dtype),
        Values: T.Tensor((m, splits * k), dtype),
        Indices: T.Tensor((m, splits * k), index_dtype),
    ):
        with T.Kernel(m, splits, threads=threads) as (row, part):
            tx = T.get_thread_binding()
            keys = T.alloc_local((slots,), "uint32")
            scratch = T.alloc_shared((2, threads // 32), "uint32")
            histogram = T.alloc_shared((257,), "uint32")
            cutoff = T.alloc_local((1,), "uint32")
            offsets = T.alloc_local((2,), "int32")
            for j in T.unroll(slots):
                col = part * shard + j * threads + tx
                keys[j] = T.if_then_else(col < T.min((part + 1) * shard, n), _ordered_key(X[row, col], dtype), T.uint32(0))
            _radix_threshold(keys, histogram, scratch, cutoff, k, 16 if dtype == "bfloat16" else 32)
            _output_offsets(keys, cutoff, scratch, offsets, k)
            for j in T.unroll(slots):
                col = part * shard + j * threads + tx
                if keys[j] > cutoff[0] or (keys[j] == cutoff[0] and offsets[1] > 0):
                    if return_values:
                        Values[row, part * k + offsets[0]] = _key_value(keys[j], dtype)
                    Indices[row, part * k + offsets[0]] = InputIndex[row, col] if indirect else col
                    offsets[0] += 1
                    offsets[1] -= T.cast(keys[j] == cutoff[0], "int32")

    return kernel


@T.macro
def _prefetch(X, scan, loaded, row, part, shard, blocks, stride, seed, step):
    block = scan.shape[1]
    slot = step % scan.shape[0]
    b = (step * stride + seed % blocks + row * 13 + part * 7) % blocks
    T.tma_copy(
        X[row, part * shard + b * block : part * shard + (b + 1) * block],
        scan[slot, 0:block],
        barrier=loaded[slot],
        eviction_policy="evict_first",
        annotations={"emit_arrive": 1},
    )


# With automatic thread synchronization disabled, shared-memory liveness does
# not model outstanding TMA writes. Keep the ring disjoint from radix scratch.
@tilelang.jit(pass_configs={**_PASS_CONFIG, tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True})
def _stream_topk(
    m,
    n,
    k,
    dtype,
    index_dtype,
    splits=1,
    block=4096,
    threads=512,
    seed=0,
    return_values=True,
    vector_loads=False,
    tma_stages=0,
    init_size=0,
):
    """Scan/filter/compact; optional BF16 initialization overlaps TMA and radix."""
    shard = (n + splits - 1) // splits
    blocks = (shard + block - 1) // block
    init_blocks = min(blocks, init_size // block) if tma_stages else 0
    stride = max(1, int(blocks * 0.61803398875))
    while math.gcd(stride, blocks) != 1:
        stride += 1
    capacity = ((2 * k + block + threads - 1) // threads) * threads
    slots = capacity // threads
    packed_filter = dtype == "bfloat16" and (block // threads) % 2 == 0
    threshold_dtype = dtype if packed_filter else "float32"

    @T.prim_func
    def kernel(
        X: T.Tensor((m, n), dtype),
        Values: T.Tensor((m, splits * k), dtype),
        Indices: T.Tensor((m, splits * k), index_dtype),
    ):
        with T.Kernel(m, splits, threads=threads) as (row, part):
            tx = T.get_thread_binding()
            lane = tx % 32
            candidates = T.alloc_shared((capacity,), "uint32")
            candidate_indices = T.alloc_shared((capacity,), "int32")
            size = T.alloc_shared((1,), "int32")
            scratch = T.alloc_shared((2, threads // 32), "uint32")
            if tma_stages:
                scan = T.alloc_shared((tma_stages, block), dtype)
                loaded = T.alloc_barrier([1] * tma_stages)
            initial = T.alloc_local((max(1, init_blocks) * block // threads,), "uint32")
            values = T.alloc_local((block // threads,), dtype)
            hits = T.alloc_local((block // threads,), "int32")
            keys = T.alloc_local((slots,), "uint32")
            indices = T.alloc_local((slots,), "int32")
            cutoff = T.alloc_local((1,), "uint32")
            offsets = T.alloc_local((2,), "int32")
            threshold_value = T.alloc_var(threshold_dtype)
            mask = T.alloc_var("uint32")
            count = T.alloc_var("int32")
            prefix = T.alloc_var("int32")
            base = T.alloc_var("int32")
            length = T.alloc_var("int32")
            length = 0
            cutoff[0] = 0
            threshold_value = T.cast(-T.infinity("float32"), threshold_dtype)
            if tx == 0:
                size[0] = 0
            T.sync_threads()
            if tma_stages:
                for pre in T.unroll(min(tma_stages - 1, blocks)):
                    _prefetch(X, scan, loaded, row, part, shard, blocks, stride, seed, pre)
            if init_blocks:
                for bucket_id in T.Parallel(257):
                    candidates[bucket_id] = 0
                T.sync_threads()
                # Keep keys in registers; build MSB bins while later TMA tiles
                # are in flight. Reuse the same ring and candidate scratch.
                for step in T.unroll(init_blocks):
                    if step + tma_stages - 1 < blocks:
                        _prefetch(X, scan, loaded, row, part, shard, blocks, stride, seed, step + tma_stages - 1)
                    T.mbarrier_wait_parity(loaded[step % tma_stages], (step // tma_stages) % 2)
                    for j in T.vectorized(block // threads):
                        values[j] = scan[step % tma_stages, tx * (block // threads) + j]
                    for j in T.unroll(block // threads):
                        key = _ordered_key(values[j], dtype)
                        initial[step * (block // threads) + j] = key
                        T.atomic_add(candidates[key >> 8], T.uint32(1))
                    T.sync_threads()  # release the ring slot before reuse
                _radix_threshold(initial, candidates, scratch, cutoff, k, 16, msb_ready=True)
                _output_offsets(initial, cutoff, scratch, offsets, k)
                for j in T.unroll(init_blocks * block // threads):
                    if initial[j] > cutoff[0] or (initial[j] == cutoff[0] and offsets[1] > 0):
                        b = ((j // (block // threads)) * stride + seed % blocks + row * 13 + part * 7) % blocks
                        col = part * shard + b * block + tx * (block // threads) + j % (block // threads)
                        candidates[offsets[0]] = initial[j]
                        candidate_indices[offsets[0]] = col
                        offsets[0] += 1
                        offsets[1] -= T.cast(initial[j] == cutoff[0], "int32")
                threshold_value = T.cast(_key_value(cutoff[0], dtype), threshold_dtype)
                length = k
                if tx == 0:
                    size[0] = k
                T.sync_threads()
            for step in T.serial(init_blocks, blocks):
                # A bijection even for non-power-of-two block counts. This is
                # an affine traversal, not the paper's uniform random permutation.
                b = (step * stride + seed % blocks + row * 13 + part * 7) % blocks
                if tma_stages:
                    if step + tma_stages - 1 < blocks:
                        _prefetch(X, scan, loaded, row, part, shard, blocks, stride, seed, step + tma_stages - 1)
                    T.mbarrier_wait_parity(loaded[step % tma_stages], (step // tma_stages) % 2)
                mask = 0
                if tma_stages:
                    for j in T.vectorized(block // threads):
                        values[j] = scan[step % tma_stages, tx * (block // threads) + j]
                elif vector_loads:
                    for j in T.vectorized(block // threads):
                        col = part * shard + b * block + tx * (block // threads) + j
                        values[j] = T.if_then_else(col < T.min((part + 1) * shard, n), X[row, col], 0)
                else:
                    for j in T.unroll(block // threads):
                        col = part * shard + b * block + j * threads + tx
                        values[j] = T.if_then_else(col < T.min((part + 1) * shard, n), X[row, col], 0)
                if packed_filter:
                    # Keep the comparison vector-valued; the mask accumulation
                    # below is a separate scalar reduction. BF16 cutoff decoding
                    # is exact, including subnormals and signed zeros.
                    for pair in T.unroll(block // threads // 2):
                        for j in T.vectorized(2):
                            hits[pair * 2 + j] = T.cast(values[pair * 2 + j] > threshold_value, "int32")
                    if shard % block == 0 and n % splits == 0:
                        # Canonical 0/1 bits do not overlap, so addition is OR
                        # without carries and permits multiply-add lowering.
                        # On full blocks, select the all-hit case only once.
                        for j in T.unroll(block // threads):
                            mask = mask + (T.cast(hits[j], "uint32") << j)
                        if cutoff[0] == 0:
                            mask = T.uint32((1 << (block // threads)) - 1)
                    else:
                        # Fuse validity and hit selection on ragged shards;
                        # constructing then clearing bits adds avoidable work.
                        for j in T.unroll(block // threads):
                            offset = tx * (block // threads) + j if vector_loads else j * threads + tx
                            col = part * shard + b * block + offset
                            if col < T.min((part + 1) * shard, n) and (cutoff[0] == 0 or hits[j] != 0):
                                mask = mask | (T.uint32(1) << j)
                else:
                    for j in T.unroll(block // threads):
                        offset = tx * (block // threads) + j if vector_loads else j * threads + tx
                        col = part * shard + b * block + offset
                        if col < T.min((part + 1) * shard, n) and (cutoff[0] == 0 or T.cast(values[j], "float32") > threshold_value):
                            mask = mask | (T.uint32(1) << j)
                count = T.cast(T.popcount(mask), "int32")
                # Reserve once per nonempty warp, rather than once per survivor.
                if T.warp_reduce_sum(count) > 0:
                    first_value = T.alloc_var(dtype)
                    first_j = T.alloc_var("int32")
                    if tma_stages and packed_filter:
                        # Overlap the first shared load with the warp prefix.
                        # Empty lanes read their valid first slot but never store.
                        first_j = T.max(0, T.cast(31 - T.clz(mask & (T.uint32(0) - mask)), "int32"))
                        first_value = scan[step % tma_stages, tx * (block // threads) + first_j]
                    prefix = _warp_exclusive_sum(count, block // threads)
                    base = 0
                    if lane == 31:
                        base = T.atomic_add(size[0], prefix + count, return_prev=True)
                        length = base + prefix + count
                    base = T.shfl_sync(base, 31) + prefix
                    if tma_stages and packed_filter:
                        # Visit only hits. Reload from the live TMA slot rather
                        # than dynamically indexing the register array (spills).
                        # The round-end collective protects the slot from reuse.
                        if mask != 0:
                            candidates[base] = _ordered_key(first_value, dtype)
                            candidate_indices[base] = part * shard + b * block + tx * (block // threads) + first_j
                            base += 1
                            mask = mask & (mask - T.uint32(1))
                        while mask != 0:
                            j = T.cast(31 - T.clz(mask & (T.uint32(0) - mask)), "int32")
                            offset = tx * (block // threads) + j
                            candidates[base] = _ordered_key(scan[step % tma_stages, offset], dtype)
                            candidate_indices[base] = part * shard + b * block + offset
                            base += 1
                            mask = mask & (mask - T.uint32(1))
                    else:
                        for j in T.unroll(block // threads):
                            if (mask & (T.uint32(1) << j)) != 0:
                                offset = tx * (block // threads) + j if vector_loads else j * threads + tx
                                col = part * shard + b * block + offset
                                candidates[base] = _ordered_key(values[j], dtype)
                                candidate_indices[base] = col
                                base += 1
                # Before append, size < 2*k, so capacity >= 2*k+block is safe
                # for every distribution, including constant and ascending input.
                # Keep each warp's last reservation end until compaction. Their
                # maximum equals size, even across rounds with no new survivors.
                # The collective both publishes writes and tests that maximum,
                # avoiding a shared-size read and a second barrier on scan rounds.
                # It also releases this round's TMA slot before any future reuse.
                if T.syncthreads_or(length >= 2 * k or (step == blocks - 1 and length > k)) != 0:
                    length = size[0]
                    for j in T.unroll(slots):
                        pos = j * threads + tx
                        keys[j] = T.if_then_else(pos < length, candidates[pos], T.uint32(0))
                    _radix_threshold(keys, candidates, scratch, cutoff, k, 16 if dtype == "bfloat16" else 32)
                    for j in T.unroll(slots):
                        pos = j * threads + tx
                        indices[j] = T.if_then_else(pos < length, candidate_indices[pos], -1)
                    threshold_value = T.cast(_key_value(cutoff[0], dtype), threshold_dtype)
                    _output_offsets(keys, cutoff, scratch, offsets, k)
                    for j in T.unroll(slots):
                        if keys[j] > cutoff[0] or (keys[j] == cutoff[0] and offsets[1] > 0):
                            candidates[offsets[0]] = keys[j]
                            candidate_indices[offsets[0]] = indices[j]
                            offsets[0] += 1
                            offsets[1] -= T.cast(keys[j] == cutoff[0], "int32")
                    if tx == 0:
                        size[0] = k
                    length = k
                    T.sync_threads()
            for j in T.Parallel(k):
                Indices[row, part * k + j] = candidate_indices[j]
                if return_values:
                    Values[row, part * k + j] = _key_value(candidates[j], dtype)

    return kernel


@tilelang.jit
def _copy_all(m, n, dtype, index_dtype, threads=256, return_values=True):
    @T.prim_func
    def kernel(X: T.Tensor((m, n), dtype), Values: T.Tensor((m, n), dtype), Indices: T.Tensor((m, n), index_dtype)):
        with T.Kernel(T.ceildiv(m * n, 1024), threads=threads) as block:
            # Strided scalar accesses also support unaligned contiguous views.
            tx = T.get_thread_binding()
            for j in T.unroll(1024 // threads):
                pos = block * 1024 + j * threads + tx
                if pos < m * n:
                    Indices[pos // n, pos % n] = pos % n
                    if return_values:
                        Values[pos // n, pos % n] = X[pos // n, pos % n]

    return kernel


def prepare_deep_select(
    x,
    k,
    *,
    strategy="auto",
    splits=None,
    return_values=True,
    index_dtype=torch.int64,
    seed=0,
    threads=None,
    block_size=None,
    vector_loads=None,
    merge_fan_in=None,
    use_tma=True,
):
    """Compile and allocate once; return a no-argument, CUDA-graph-safe runner.

    The runner is bound to ``x`` and reuses its output/workspace tensors. It is
    not reentrant. Its ``configuration`` attribute describes the launch plan.
    Inputs must be contiguous CUDA BF16/FP32 matrices without NaNs. Infinities,
    subnormals, signed zeros, arbitrary tails, and ties are supported. Results
    are exact but unsorted, and tied indices have no stability guarantee.

    ``stream`` forces scan/filter/compact. ``hierarchical`` uses independent
    register tiles and exact merges. ``auto`` uses the former for long rows at
    large batch sizes and the latter to expose parallelism at small batches.
    The launch policy considers dtype, K tier, row length, SM count and shared
    memory. ``threads``, ``block_size``, ``vector_loads`` and ``merge_fan_in``
    are optional tuning overrides; block/load overrides require stream mode.
    Bulk TMA is enabled by default on SM90+ for aligned, full streaming tiles
    that fit shared memory; other configurations retain ordinary loads.
    ``use_tma=False`` disables it for comparison or debugging.
    Selection itself performs no timing, input sampling, or device launches.
    Every stage uses the same byte-radix selection. There is no compression
    algorithm switch, input-distribution detection, or approximate Top-K.
    """
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("This example requires an NVIDIA CUDA device")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous two-dimensional tensor")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("x must have dtype torch.bfloat16 or torch.float32")
    m, n = x.shape
    if type(k) is not int or not 1 <= k <= min(n, 4096):
        raise ValueError("k must be an integer in [1, min(n, 4096)]")
    if n >= 2**31:
        raise ValueError("Row lengths must fit int32")
    if index_dtype not in (torch.int32, torch.int64):
        raise ValueError("index_dtype must be torch.int32 or torch.int64")
    if strategy not in ("auto", "stream", "hierarchical"):
        raise ValueError("strategy must be auto, stream, or hierarchical")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(use_tma) is not bool:
        raise ValueError("use_tma must be a bool")
    dtype = str(x.dtype).split(".")[-1]
    idx_dtype = str(index_dtype).split(".")[-1]
    props = torch.cuda.get_device_properties(x.device)
    config = select_config(
        m,
        n,
        k,
        dtype,
        sm_count=props.multi_processor_count,
        shared_memory_limit=props.shared_memory_per_block_optin,
        strategy=strategy,
        splits=splits,
        threads=threads,
        block_size=block_size,
        vector_loads=vector_loads,
        merge_fan_in=merge_fan_in,
        use_tma=use_tma and props.major >= 9,
        input_aligned=x.data_ptr() % 16 == 0,
    )
    strategy, splits = config["strategy"], config["splits"]
    launches = []

    def allocate(rows, cols):
        return (
            torch.empty((rows, cols), dtype=x.dtype, device=x.device),
            torch.empty((rows, cols), dtype=index_dtype, device=x.device),
        )

    with torch.cuda.device(x.device):
        values, indices = allocate(m, splits * k)
        if m:
            if strategy == "copy":
                kernel = _copy_all(m, n, dtype, idx_dtype, threads=config["threads"], return_values=return_values)
                launches.append((kernel, (x, values, indices)))
            elif strategy == "stream":
                kernel = _stream_topk(
                    m,
                    n,
                    k,
                    dtype,
                    idx_dtype,
                    splits,
                    block=config["block_size"],
                    threads=config["threads"],
                    seed=seed,
                    return_values=return_values or splits > 1,
                    vector_loads=config["vector_loads"],
                    tma_stages=config["tma_stages"],
                    init_size=config["init_size"],
                )
                launches.append((kernel, (x, values, indices)))
            else:
                kernel = _register_topk(
                    m,
                    n,
                    k,
                    dtype,
                    idx_dtype,
                    splits,
                    threads=config["threads"],
                    return_values=return_values or splits > 1,
                )
                # InputIndex is unused in this specialization; its ABI shape
                # matches the output, so no dummy input-sized allocation is needed.
                launches.append((kernel, (x, indices, values, indices)))
            remaining = splits
            # Bound merge tiles independently of input length; large K uses a
            # wider merge to avoid a deep binary tree of low-parallelism stages.
            fan_in = config["merge_fan_in"]
            while remaining > 1:
                fan = min(remaining, fan_in)
                groups = remaining // fan
                input_values = values.reshape(m * groups, fan * k)
                input_indices = indices.reshape(m * groups, fan * k)
                values, indices = allocate(m * groups, k)
                kernel = _register_topk(
                    m * groups,
                    fan * k,
                    k,
                    dtype,
                    idx_dtype,
                    indirect=True,
                    return_values=return_values or groups > 1,
                )
                launches.append((kernel, (input_values, input_indices, values, indices)))
                remaining = groups
        result = (values.reshape(m, k) if return_values else None, indices.reshape(m, k))

    def run():
        for kernel, args in launches:
            kernel(*args)
        return result

    run.configuration = dict(config, launches=len(launches), index_dtype=idx_dtype)
    run.stages = launches
    return run


def deep_select(x, k, **kwargs):
    """Allocate and return ``(values, indices)``; see :func:`prepare_deep_select`."""
    return prepare_deep_select(x, k, **kwargs)()


if __name__ == "__main__":
    x = torch.randn(6, 131072, dtype=torch.bfloat16, device="cuda")
    values, indices = deep_select(x, 512)
    torch.testing.assert_close(values.sort().values, x.topk(512).values.sort().values, rtol=0, atol=0)
    print("Exact unsorted Top-K:", values.shape, indices.dtype)
