# NOTE: This script is designed for B=1 and extremely large S cases,
# For normal shape, e.g. (B, S) = (64, 32768) please compare this to `topk_selector.py
# to choose the optimal kernel.

import torch
import tilelang
import tilelang.language as T
import triton

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    # The interval analyzer cannot prove that dynamically-computed bin
    # indices fit inside the buffer shape, so the safe-memory legalizer
    # wraps every global BufferLoad inside our
    # `T.atomic_add(direct_counter[bx, l_bin_id32 + 1], ...)` calls with
    # an `if_then_else`. That converts `T.access_ptr(BufferLoad)` into
    # `T.access_ptr(Call)`, which breaks LowerAccessPtr. We manually
    # guarantee bin indices are in [0, RADIX-1] (see convert_to_uint16),
    # so it is safe to skip this pass entirely.
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}

RADIX = 1 << 8
BLOCK_SIZE = 1024
# Stage 1 / Stage 2 chunk size: each block in stages 1-2 owns this many input
# elements of one batch. Choose it as a multiple of BLOCK_SIZE so each block
# does CHUNK_SIZE/BLOCK_SIZE serial iterations over its slice.
# CHUNK_SIZE == BLOCK_SIZE => one element per thread, maximum block count and
# therefore maximum SM occupancy for the histogram passes.
CHUNK_SIZE = 4096
# Maximum number of threshold-bucket candidates carried from stage 2 to
# stage 3's tail pass. Assumes the threshold bucket size after the first
# pass is < 4K elements.
SMEM_INPUT_SIZE = 4096


def convert_to_uint16(x):
    hval = T.cast(x, T.float16)
    bits_uint = T.reinterpret(hval, T.uint16)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(x, T.uint32)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.cast((0xFFFFFFFF), T.uint32),
        bits_uint | T.cast((0x80000000), T.uint32),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_stage1_impl(in_dtype=T.float32, out_dtype=T.int32):
    """Stage 1: build per-batch global histogram (RADIX bins) over the high
    8 bits of every input value. Each batch is parallelized along seq_len:
    one block handles one (chunk, batch) pair, builds a local histogram in
    shared memory, then atomically merges it into the per-batch global
    histogram in HBM.

    Grid: (ceildiv(seq_len, CHUNK_SIZE), batch)
    """
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")

    @T.prim_func
    def stage1_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        starts: T.Tensor[(batch,), out_dtype],
        ends: T.Tensor[(batch,), out_dtype],
        global_histogram: T.Tensor[(batch, RADIX), T.int32],
    ):
        with T.Kernel(T.ceildiv(seq_len, CHUNK_SIZE), batch, threads=BLOCK_SIZE) as (cx, bx):
            tx = T.get_thread_binding()

            s_local_hist = T.alloc_shared([RADIX], T.int32)

            l_start_idx = T.alloc_var(T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_chunk_start = T.alloc_var(T.int32)

            l_start_idx = starts[bx]
            l_end_idx = ends[bx]
            l_chunk_start = cx * CHUNK_SIZE

            T.fill(s_local_hist, 0)
            T.sync_threads()

            for s in T.serial(T.ceildiv(CHUNK_SIZE, BLOCK_SIZE)):
                input_idx = l_chunk_start + s * BLOCK_SIZE + tx
                if input_idx < seq_len and input_idx >= l_start_idx and input_idx < l_end_idx:
                    bin_id = convert_to_uint16(input[bx, input_idx])
                    T.atomic_add(s_local_hist[bin_id], 1)
            T.sync_threads()

            if tx < RADIX and s_local_hist[tx] > 0:
                T.atomic_add(global_histogram[bx, tx], s_local_hist[tx])

    return stage1_kernel


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_stage2_impl(topk, in_dtype=T.float32, out_dtype=T.int32):
    """Stage 2: also multi-block per batch. Every block reads the merged
    global histogram, recomputes cumsum + threshold in its own shared memory
    (cheap: 256 entries), then re-scans ONLY its own chunk:

      * elements with `bin > threshold` are written straight into `index` at
        position ``s[bin+1] + atomic_add(direct_counter[bx, bin+1])``.
        ``s[bin+1]`` (the suffix-sum offset) is the same on every block, and
        the per-bin global counter ensures unique within-bin slots across
        all chunks.
      * elements with `bin == threshold` are appended to a per-batch global
        candidate list (`candidate_idx`, `candidate_count`) for stage 3.

    Grid: (ceildiv(seq_len, CHUNK_SIZE), batch)
    """
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")

    @T.prim_func
    def stage2_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        starts: T.Tensor[(batch,), out_dtype],
        ends: T.Tensor[(batch,), out_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
        global_histogram: T.Tensor[(batch, RADIX), T.int32],
        direct_counter: T.Tensor[(batch, RADIX + 1), T.int32],
        candidate_idx: T.Tensor[(batch, SMEM_INPUT_SIZE), out_dtype],
        candidate_count: T.Tensor[(batch,), T.int32],
    ):
        with T.Kernel(T.ceildiv(seq_len, CHUNK_SIZE), batch, threads=BLOCK_SIZE) as (cx, bx):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_histogram = T.alloc_shared([RADIX + 1], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_chunk_start = T.alloc_var(T.int32)
            l_start_idx = T.alloc_var(T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_bin_offset = T.alloc_var(T.int32)

            pos = T.alloc_var(T.int32)

            l_new_topk = topk
            l_start_idx = starts[bx]
            l_end_idx = ends[bx]
            l_chunk_start = cx * CHUNK_SIZE

            # Load merged global histogram into shared memory; index RADIX is
            # the sentinel that must remain 0 (used as cumsum tail).
            T.fill(s_histogram, 0)
            T.sync_threads()
            if tx < RADIX:
                s_histogram[tx] = global_histogram[bx, tx]
            T.sync_threads()

            # Cumsum (suffix sum) and threshold finding (same as the original
            # single-block kernel; runs identically on every block).
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # Re-scan ONLY this chunk and dispatch each element.
            for s in T.serial(T.ceildiv(CHUNK_SIZE, BLOCK_SIZE)):
                input_idx = l_chunk_start + s * BLOCK_SIZE + tx
                if input_idx < seq_len and input_idx >= l_start_idx and input_idx < l_end_idx:
                    bin_id = convert_to_uint16(input[bx, input_idx])
                    l_bin_id32 = T.cast(bin_id, T.int32)
                    if l_bin_id32 > l_threshold_bin_id:
                        # cumsum offset is consistent across blocks; use a
                        # per-(batch, bin) global counter for the within-bin slot.
                        l_bin_offset = s_histogram[l_bin_id32 + 1]
                        pos = T.atomic_add(direct_counter[bx, l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, l_bin_offset + pos] = input_idx
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        pos = T.atomic_add(candidate_count[bx], 1, return_prev=True)
                        if pos < SMEM_INPUT_SIZE:
                            candidate_idx[bx, pos] = input_idx

    return stage2_kernel


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_stage3_impl(topk, in_dtype=T.float32, out_dtype=T.int32):
    """Stage 3 (tail pass): single block per batch. Loads the threshold-bucket
    candidate list from HBM, recomputes the threshold from the global
    histogram to recover ``l_new_topk`` / ``l_start_pos``, and then runs up
    to 4 radix rounds (8 bits each) over the candidate set, writing the
    final indices into ``index``.

    Grid: (batch,)
    """
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")

    @T.prim_func
    def stage3_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
        global_histogram: T.Tensor[(batch, RADIX), T.int32],
        candidate_idx: T.Tensor[(batch, SMEM_INPUT_SIZE), out_dtype],
        candidate_count: T.Tensor[(batch,), T.int32],
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx,):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_histogram = T.alloc_shared([RADIX + 1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_num_input = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_start_pos = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)
            pos = T.alloc_var(T.int32)

            l_new_topk = topk

            # Recompute cumsum + threshold from the global histogram so we can
            # recover ``l_new_topk`` (= topk - s[t+1]); l_start_pos is then
            # topk - l_new_topk = s[t+1].
            T.fill(s_histogram, 0)
            T.sync_threads()
            if tx < RADIX:
                s_histogram[tx] = global_histogram[bx, tx]
            T.sync_threads()
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # Pull the threshold-bucket candidates that stage 2 collected in
            # HBM into shared memory s_input_idx[0]; this is the only piece
            # of state inherited from stage 2 (besides l_new_topk which we
            # just recomputed from the histogram).
            l_num_input = candidate_count[bx]
            for s in T.serial(T.ceildiv(SMEM_INPUT_SIZE, BLOCK_SIZE)):
                pos = s * BLOCK_SIZE + tx
                if pos < l_num_input:
                    s_input_idx[0, pos] = candidate_idx[bx, pos]
            if tx == 0:
                s_num_input[0] = l_num_input
            T.sync_threads()

            # Tail pass — identical to the original kernel's stage 2.
            for round in T.serial(4):
                if l_new_topk <= 0:
                    break

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.cast(
                            ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >> (24 - round * 8)) & 0xFF), T.int32
                        )
                        T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()

                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.cast(
                            ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >> (24 - round * 8)) & 0xFF), T.int32
                        )
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            if round == 3:
                                l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

    return stage3_kernel


def tl_topk(input, starts, ends, topk):
    batch, seq_len = input.shape
    indexes = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)
    global_histogram = torch.zeros(batch, RADIX, dtype=torch.int32, device=input.device)
    direct_counter = torch.zeros(batch, RADIX + 1, dtype=torch.int32, device=input.device)
    candidate_idx = torch.empty(batch, SMEM_INPUT_SIZE, dtype=torch.int32, device=input.device)
    candidate_count = torch.zeros(batch, dtype=torch.int32, device=input.device)

    stage1 = tl_topk_stage1_impl()
    stage2 = tl_topk_stage2_impl(topk)
    stage3 = tl_topk_stage3_impl(topk)

    stage1(input, starts, ends, global_histogram)
    stage2(input, starts, ends, indexes, global_histogram, direct_counter, candidate_idx, candidate_count)
    stage3(input, indexes, global_histogram, candidate_idx, candidate_count)
    return indexes


def test_topk_selector(batch=1, seq_len=131072, topk=4096):
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    print(f"{input.shape=}")

    indexes = tl_topk(input, starts, ends, topk)
    print(indexes)

    indexes_ref = torch.topk(input, topk, dim=-1)[1]
    print(indexes_ref)

    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        trt_np = indexes[i].cpu().to(torch.int32).numpy()
        set_ref = set(ref_np)
        set_trt = set(trt_np)
        intersection = set_ref & set_trt
        print("selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))

    torch.cuda.synchronize()

    for _ in range(5):
        _ = tl_topk(input, starts, ends, topk)
    torch.cuda.synchronize()

    # There's some minor gap between triton benchmark result and tilelang's
    # We choose to report both for clarity issues
    tl_time = tilelang.profiler.do_bench(lambda: tl_topk(input, starts, ends, topk), backend="cupti")
    print(f"Average tl_topk time: {tl_time:.3f} ms")
    tl_time = triton.testing.do_bench(lambda: tl_topk(input, starts, ends, topk))
    print(f"Average triton-benched tl_topk time: {tl_time:.3f} ms")

    torch_time = tilelang.profiler.do_bench(lambda: torch.topk(input, topk, dim=-1)[1], backend="cupti")
    print(f"Average torch.topk time: {torch_time:.3f} ms")


def run_regression_perf(batch=64, seq_len=32 * 1024, topk=2048):
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    from tilelang.profiler import do_bench

    def run_kernel_only():
        tl_topk(input, starts, ends, topk)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    test_topk_selector()
