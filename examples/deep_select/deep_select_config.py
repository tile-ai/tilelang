"""Deterministic, resource-aware launch policy; no GPU work or JIT during selection.

DeepSelect's dtype / K-tier / SM-wave dispatch is the starting point, not a
literal transplant of its TMA/cluster launch tuples. No input sampling is used.
"""


def _next_power_of_two(n):
    return 1 << (max(1, n) - 1).bit_length()


def stream_shared_bytes(k, block, threads, tma_stages=0, dtype="bfloat16"):
    """Conservative bound including TMA buffers, barriers and alignment padding."""
    capacity = ((2 * k + block + threads - 1) // threads) * threads
    stage_bytes = block * (2 if dtype == "bfloat16" else 4) + 8
    return capacity * 8 + 8 * (threads // 32) + 16 + ((tma_stages * stage_bytes + 15) // 16) * 16


def register_shared_bytes(threads):
    """256 radix bins, one sink and two counters per warp, with alignment."""
    return 1040 + 8 * (threads // 32)


def select_config(
    batch,
    n,
    k,
    dtype,
    *,
    sm_count,
    shared_memory_limit,
    strategy="auto",
    splits=None,
    threads=None,
    block_size=None,
    vector_loads=None,
    merge_fan_in=None,
    use_tma=False,
    input_aligned=True,
):
    """Return a launch plan; explicit overrides are validated, never silently changed.

    Shapes may be ragged. Every shard must contain K elements, including the
    last shard, otherwise exact local-Top-K merging would read unwritten slots.
    Shared memory is a feasibility bound, NOT an occupancy prediction.
    ``use_tma`` allows bulk TMA when shards and resources permit it; the caller
    must first check device support. ``input_aligned`` means a 16-byte aligned
    input pointer, required by bulk TMA and unmasked vectorized global loads.
    """
    if type(batch) is not int or batch < 0 or type(n) is not int or not 1 <= n < 2**31:
        raise ValueError("Invalid batch or row length")
    if type(k) is not int or not 1 <= k <= min(n, 4096):
        raise ValueError("k must be an integer in [1, min(n, 4096)]")
    if dtype not in ("bfloat16", "float32"):
        raise ValueError("dtype must be bfloat16 or float32")
    if type(sm_count) is not int or sm_count < 1 or shared_memory_limit < 1024:
        raise ValueError("Invalid device resource limits")
    if strategy not in ("auto", "stream", "hierarchical"):
        raise ValueError("strategy must be auto, stream, or hierarchical")
    if threads is not None and (type(threads) is not int or threads not in (256, 512)):
        raise ValueError("threads must be 256 or 512")
    if block_size is not None and (type(block_size) is not int or block_size not in (512, 1024, 2048, 4096, 8192)):
        raise ValueError("block_size must be 512, 1024, 2048, 4096, or 8192")
    if vector_loads is not None and type(vector_loads) is not bool:
        raise ValueError("vector_loads must be a bool")
    if type(use_tma) is not bool:
        raise ValueError("use_tma must be a bool")
    if type(input_aligned) is not bool:
        raise ValueError("input_aligned must be a bool")
    if splits is not None and (type(splits) is not int or splits < 1 or splits & (splits - 1)):
        raise ValueError("splits must be a positive power of two")
    if merge_fan_in is not None and (type(merge_fan_in) is not int or merge_fan_in not in (2, 4, 8, 16) or merge_fan_in * k > 16384):
        raise ValueError("merge_fan_in must be 2, 4, 8, or 16 with at most 16384 input candidates")

    requested_strategy = strategy
    waves = (batch + sm_count - 1) // sm_count
    k_tier = 512 if k <= 512 else 1024 if k <= 1024 else 4096
    # Short shards need fewer register tiles, not a streaming loop per tile.
    # Limit this choice to small grids: extra merge work loses at larger batches.
    short_register = (
        dtype == "bfloat16"
        and 1 < k <= 1024
        and n <= 65536
        and batch * 2 <= sm_count
        and block_size is None
        and vector_loads is None
        and (splits is None or (n + splits - 1) // splits <= 16384)
    )
    if strategy == "auto":
        strategy = "hierarchical" if n <= 16384 or batch <= 16 or short_register else "stream"
    reason = "short-row or small-batch register selection" if strategy == "hierarchical" else "dtype/K/SM-wave streaming"
    if strategy == "hierarchical":
        if block_size is not None or vector_loads is not None:
            raise ValueError("block_size and vector_loads require strategy='stream'")
        selected_threads = threads or (512 if dtype == "bfloat16" and 1 < k <= 1024 and n >= 8192 else 256)
        selected_block = None
        selected_vector = False
    else:
        if k == 1:
            # The general vector filter is not a specialized argmax. Preserve
            # the cheaper measured scalar configuration for this endpoint.
            selected_threads, selected_block = 512, 4096
        elif k_tier <= 1024:
            # Literal upstream single-wave 512t/B8192 loses to 256t/B4096
            # with this implementation's independent shard/merge kernels.
            # Wide FP32 scans pay off only once long rows have enough CTAs.
            use_wide = dtype == "float32" and waves > 1 and n >= 524288
            selected_threads = 512 if use_wide else 256
            selected_block = 8192 if use_wide else 4096
        else:
            # Large candidate buffers allow only one CTA on SM120. Keep 16
            # resident warps instead of halving that to 8 with 256 threads.
            selected_threads = 512
            selected_block = 4096
        selected_threads = threads or selected_threads
        selected_block = block_size or selected_block
        selected_vector = (k != 1 and input_aligned) if vector_loads is None else vector_loads
        if selected_vector and not input_aligned:
            raise ValueError("Vector loads require a 16-byte aligned input pointer; use vector_loads=False")
        while stream_shared_bytes(k, selected_block, selected_threads) > shared_memory_limit:
            if block_size is not None:
                raise ValueError("Requested stream configuration exceeds device shared-memory limit")
            if selected_block > selected_threads:
                selected_block //= 2
                reason = "stream block reduced to fit device shared memory"
            elif requested_strategy == "auto" and threads is None and vector_loads is None:
                strategy = "hierarchical"
                selected_threads, selected_block, selected_vector = 256, None, False
                reason = "register fallback: stream candidates exceed device shared memory"
                break
            else:
                raise ValueError("Stream candidates exceed device shared-memory limit; use hierarchical")
        if strategy == "stream" and not 1 <= selected_block // selected_threads <= 32:
            raise ValueError("Stream mask requires 1 to 32 elements per thread")

    if splits is None:
        if strategy == "hierarchical":
            # A full 16K register tile avoids a second selection and merge for
            # short BF16 rows in the K=1024 tier, even on multi-wave grids.
            wide_register = short_register or (dtype == "bfloat16" and 512 < k <= 1024 and n <= 16384)
            tile = 16384 if wide_register and selected_threads == 512 else 8192
            splits = _next_power_of_two((n + tile - 1) // tile)
        else:
            splits = _next_power_of_two((sm_count + max(batch, 1) - 1) // max(batch, 1))
            # Avoid a second wave plus a merge when fused TMA initialization
            # can keep each row in one CTA. Other paths retain the 80% rule.
            fused_row = dtype == "bfloat16" and use_tma and selected_vector and k <= 1024 and n >= 32768 and n % 2048 == 0
            if splits == 2 and (batch * 5 >= sm_count * 4 or (fused_row and batch * 2 > sm_count)):
                splits = 1
            # Avoid selecting almost every shard element, only to merge it again.
            while splits > 1 and (n + splits - 1) // splits < max(8192, 8 * k):
                splits //= 2
        while splits > 1 and n - (splits - 1) * ((n + splits - 1) // splits) < k:
            splits //= 2
    shard = (n + splits - 1) // splits
    if n - (splits - 1) * shard < k:
        raise ValueError("Every split, including the last one, must contain at least k elements")
    if strategy == "hierarchical" and shard > 16384:
        raise ValueError("Use more splits (or stream) to keep register tiles at most 16384 elements")
    tma_stages = 0
    if strategy == "stream" and use_tma and selected_vector and n % splits == 0:
        # Smaller scan blocks leave room for prefetching without enlarging the
        # candidate buffer. Preserve explicit block/thread overrides.
        # FP32 multi-wave scans need wider transfers. Single-wave BF16 benefits
        # from fewer rounds and more warps; split grids keep smaller candidates.
        wide_tma = k <= 1024 and (
            (dtype == "float32" and waves > 1) or (dtype == "bfloat16" and waves == 1 and splits == 1 and shard >= 32768)
        )
        tma_threads = threads or (512 if wide_tma else selected_threads)
        tma_block = block_size or min(selected_block, 4096 if wide_tma else 2048)
        if block_size is None and tma_block == 4096 and shard % tma_block:
            tma_block = 2048
        if tma_block >= tma_threads and shard % tma_block == 0:
            block_bytes = tma_block * (2 if dtype == "bfloat16" else 4)
            max_stages = 2
            if k <= 1024:
                buffer_budget = (32768 if dtype == "bfloat16" else 49152) if wide_tma else 24576 if k <= 512 else 16384
                max_stages = min(6 if k <= 512 else 4, max(2, buffer_budget // block_bytes))
            for stages in range(min(max_stages, shard // tma_block), 0, -1):
                if stream_shared_bytes(k, tma_block, tma_threads, stages, dtype) <= shared_memory_limit:
                    selected_threads, selected_block, tma_stages = tma_threads, tma_block, stages
                    reason = "dtype/K-tier bulk TMA streaming"
                    break
    shared_bytes = (
        stream_shared_bytes(k, selected_block, selected_threads, tma_stages, dtype)
        if strategy == "stream"
        else register_shared_bytes(selected_threads)
    )
    if shared_bytes > shared_memory_limit:
        raise ValueError("Radix histogram and reduction scratch exceed device shared-memory limit")

    # A larger register window amortizes selection, but hurts multi-wave grids.
    config = dict(
        strategy=strategy,
        splits=splits,
        threads=selected_threads,
        block_size=selected_block,
        vector_loads=selected_vector,
        tma_stages=tma_stages,
        init_size=min(shard, 16384) if tma_stages and dtype == "bfloat16" and batch * splits <= sm_count else 0,
        merge_fan_in=merge_fan_in or min(16, 1 << (((16384 if k > 1024 else 8192) // k).bit_length() - 1)),
        k_tier=k_tier,
        sm_count=sm_count,
        row_waves=waves,
        first_stage_ctas=batch * splits,
        shared_memory_bound=shared_bytes,
        shared_memory_limit=shared_memory_limit,
        reason=reason,
    )
    if k == n:
        config.update(
            strategy="copy",
            threads=threads or 256,
            block_size=1024,
            vector_loads=False,
            tma_stages=0,
            init_size=0,
            first_stage_ctas=(batch * n + 1023) // 1024,
            shared_memory_bound=0,
            reason="full selection: copy values and emit indices",
        )
    return config
