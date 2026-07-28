# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=invalid-name
"""A GEMV schedule rule for TileLang.

Uses TileSchedule primitives (cache_read_at, cache_write_at with reducer,
parallelize) to produce vectorized loads and efficient cross-thread reduction,
replacing the old s_tir.Schedule + LowerCrossThreadReduction approach.
"""

from __future__ import annotations

from tilelang import tvm

from ... import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule

from tvm import s_tir, tirx as tir
from tvm.target import Target
from tvm.s_tir.dlight import normalize_prim_func, try_inline_contiguous_spatial
from tvm.s_tir.dlight.analysis import SBlockInfo, is_gemv, normalize


def _as_const_int(expr: tir.PrimExpr) -> int | None:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _largest_power_of_two_at_most(value: int) -> int:
    result = 1
    while result << 1 <= value:
        result <<= 1
    return result


def _find_epilogue_name(block_infos: list[SBlockInfo], sch: TileSchedule) -> str | None:
    if len(block_infos) == 1:
        return None
    if len(block_infos) != 2 or not block_infos[1].is_injective():
        return None
    return sch.get(block_infos[1].block_rv).name_hint


def _choose_tile_gemv_params(
    target: Target,
    spatial_extent: tir.PrimExpr,
    reduction_extent: tir.PrimExpr,
    matrix_dtype: str,
) -> tuple[int, int, int, int]:
    """Choose (num_threads, tile_k, block_k, tile_m) for tile-based GEMV.

    The configuration maximizes memory bandwidth utilization by:
    - Using 128-bit vectorized loads (float4 for fp32, half8 for fp16)
    - Maximizing thread count for memory latency hiding (up to 256)
    - Ensuring BLOCK_K divides the reduction extent for regular shapes
    - Batching TILE_M rows per block to amortize AllReduce overhead

    Returns
    -------
    num_threads : int
        Number of cooperative reduction threads per block.
    tile_k : int
        Elements per vectorized load (128 bits / element bits).
    block_k : int
        Reduction chunk per outer iteration (num_threads * tile_k).
    tile_m : int
        Number of output rows per thread block.
    """
    dtype_bits = tvm.DataType(matrix_dtype).bits

    # 128-bit memory transactions: float4 for fp32, half8 for fp16
    tile_k = max(1, 128 // dtype_bits)

    max_threads = min(int(utils.max_threads_per_block(target)), 256)
    if max_threads <= 0:
        return 1, tile_k, tile_k, 1

    const_reduction = _as_const_int(reduction_extent)
    if const_reduction is None:
        return max_threads, tile_k, max_threads * tile_k, 1

    if const_reduction <= 0:
        return 1, 1, 1, 1

    # Each thread handles tile_k elements (one vector load) per iteration
    num_threads = min(max_threads, const_reduction // tile_k)
    num_threads = max(1, _largest_power_of_two_at_most(num_threads))

    block_k = num_threads * tile_k

    # Ensure reduction extent is divisible by block_k
    while block_k > tile_k and const_reduction % block_k != 0:
        num_threads //= 2
        block_k = num_threads * tile_k

    num_threads = max(1, num_threads)
    block_k = max(tile_k, num_threads * tile_k)

    # Choose TILE_M: batch multiple rows per block to amortize AllReduce.
    # Currently disabled (tile_m=1) because without loop reorder the B
    # vector is loaded redundantly for each row, hurting performance.
    tile_m = 1

    return num_threads, tile_k, block_k, tile_m


def _effective_rank(buf: tir.Buffer) -> int:
    """Count non-unit dimensions in a buffer shape.

    Unit leading dims (e.g. ``(1, K)`` from an HF LLaMA decode matmul
    with batch=1, seq=1) do not contribute spatial information and
    are dropped by ``normalize_prim_func`` at the block level.  We
    mirror that behaviour here so the canonical
    ``vector_rank + 1 == matrix_rank`` invariant still holds.
    """
    return sum(1 for dim in buf.shape if not (isinstance(dim, tir.IntImm) and int(dim.value) == 1))


def _can_use_tile_schedule(
    target: Target,
    matrix_buffer: tir.Buffer,
    vector_buffer: tir.Buffer,
    output_buffer: tir.Buffer,
) -> bool:
    """Check if the tile-based GEMV schedule can handle this workload.

    Uses effective (non-unit) rank for the vector and output so decode
    matmuls ``(1, K) × (N, K) → (1, N)`` are accepted even though the
    raw ranks of x and out are 2, not 1.
    """
    if target.kind.name != "cuda":
        return False
    if len(matrix_buffer.shape) not in (2, 3):
        return False
    if _effective_rank(vector_buffer) + 1 != len(matrix_buffer.shape):
        return False
    return _effective_rank(output_buffer) + 1 == len(matrix_buffer.shape)


class GEMV(GPUScheduleRule):
    """A GEMV schedule rule using TileSchedule with vectorized loads.

    Uses cache_read_at for vectorized input staging into local.fragment,
    cache_write_at with reduce_type="sum" for cross-thread reduction,
    and parallelize for cooperative thread work distribution.

    For optimal performance on NVIDIA GPUs, pass ``GEMV.COMPILE_FLAGS``
    when compiling the scheduled module::

        sch = GEMV().apply(func, target, False)
        mod = _lower_mod(sch.mod)
        kernel = tilelang.compile(mod["main"], compile_flags=GEMV.COMPILE_FLAGS)

    The flags instruct ptxas to use the streaming cache mode for global
    loads (``-dlcm=cs``), which avoids L2 cache pollution from the large
    matrix operand and improves memory bandwidth utilization by ~20%.
    """

    COMPILE_FLAGS: list[str] = ["-Xptxas", "-dlcm=cs"]
    """Recommended NVCC flags for GEMV kernels.

    ``-dlcm=cs`` sets the default load cache mode to "cache streaming"
    (evict-first), which prevents the large matrix operand from polluting
    the L2 cache.  This yields ~20% speedup on H100 for typical GEMV
    shapes, closing the gap to within 1-2% of Triton/cuBLAS.
    """

    def _apply_tile_schedule(
        self,
        func: tir.PrimFunc,
        target: Target,
        matrix_buffer: tir.Buffer,
        vector_buffer: tir.Buffer,
        output_buffer: tir.Buffer,
        fixed_spatial_tile=None,
    ) -> s_tir.Schedule | None:
        if not _can_use_tile_schedule(target, matrix_buffer, vector_buffer, output_buffer):
            return None

        try:
            sch = TileSchedule(func)
            block_infos = normalize_prim_func(sch)
            block_infos = try_inline_contiguous_spatial(sch, block_infos)
            if block_infos is None or len(block_infos) not in (1, 2):
                return None

            epilogue = None
            if len(block_infos) == 2:
                if not block_infos[1].is_injective():
                    return None
                epilogue = block_infos[1].block_rv

            block_info = block_infos[0]
            block = block_info.block_rv
            if normalize(sch, block_info) is not True:
                return None

            batch, s, r, c = sch.get_loops(block)

            num_threads, _tile_k, block_k, _tile_m = _choose_tile_gemv_params(
                target,
                sch.get(s).extent,
                sch.get(r).extent,
                matrix_buffer.dtype,
            )
            # ---- Loop transforms ----
            bx = sch.fuse(batch, s)

            # Split reduction into [outer_chunk, inner_chunk=BLOCK_K]
            ko, ki = sch.split(r, factors=[None, block_k])
            ki = sch.fuse(ki, c)  # absorb channel dim (extent 1)

            # ---- Decompose reduction to separate T.init() from update ----
            # cache_write_at detects the init subtree and replaces it with
            # T.fill in-place.
            block_stmt = sch.get(block)
            block_name = block_stmt.name_hint
            if block_stmt.init is not None:
                sch.decompose_reduction(block, ko)
                block_name = block_name + "_update"

            # ---- Identify read buffer indices ----
            block = sch.get_sblock(block_name)
            block_stmt = sch.get(block)
            matrix_read_idx = None
            vector_read_idx = None
            for idx, read_region in enumerate(block_stmt.reads):
                if read_region.buffer.same_as(matrix_buffer):
                    matrix_read_idx = idx
                elif read_region.buffer.same_as(vector_buffer):
                    vector_read_idx = idx
            if matrix_read_idx is None or vector_read_idx is None:
                return None

            # ---- Cache reads into local.fragment ----
            # Vectorized via layout inference + VectorizeLoop pass
            block = sch.get_sblock(block_name)
            sch.cache_read_at(ko, block, matrix_read_idx, "local.fragment")

            block = sch.get_sblock(block_name)
            sch.cache_read_at(ko, block, vector_read_idx, "local.fragment")

            # ---- Handle epilogue: move inside bx BEFORE cache_write_at ----
            # Place the epilogue inside the bx loop so that cache_write_at's
            # CacheBufferReplacer redirects its read of the intermediate
            # buffer to the accumulator fragment.  With write_back=False the
            # intermediate buffer is eliminated entirely.
            has_epilogue = epilogue is not None
            if has_epilogue:
                sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)

            # ---- Cache write with reducer for cross-thread reduction ----
            # Creates: T.fill + finalize_reducer (+ T.copy write-back when
            # there is no epilogue).  When the epilogue is present, the
            # epilogue itself writes the final output so no write-back needed.
            block = sch.get_sblock(block_name)
            sch.cache_write_at(
                bx,
                block,
                0,
                "local.fragment",
                reduce_type="sum",
                reducer_replication="all",
                write_back=not has_epilogue,
            )

            # ---- Parallelize inner reduction ----
            sch.parallelize(ki)

            # ---- Bind and launch ----
            sch.bind(bx, "blockIdx.x")
            root = sch.get_sblock("root")
            sch.launch_thread(root, num_threads)

            return sch
        except Exception:
            import traceback

            traceback.print_exc()
            return None

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = TileSchedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None:
            return None

        epilogue_name = _find_epilogue_name(block_infos, sch)
        if epilogue_name is None and len(block_infos) != 1:
            return None

        block_info = block_infos[0]
        if len(block_info.iters) not in [2, 3]:
            return None

        block = block_info.block_rv
        vector_input_buffers = is_gemv(sch, block_info)
        if vector_input_buffers is None or len(vector_input_buffers) != 1:
            return None

        is_inner_reduction = normalize(sch, block_info)
        if is_inner_reduction is None:
            return None
        if not is_inner_reduction:
            return None

        block_stmt = sch.get(block)
        if len(block_stmt.writes) != 1:
            return None

        vector_buffer = vector_input_buffers[0]
        output_buffer = block_stmt.writes[0].buffer
        matrix_buffer = None
        for read_region in block_stmt.reads:
            if not read_region.buffer.same_as(vector_buffer) and not read_region.buffer.same_as(output_buffer):
                matrix_buffer = read_region.buffer
                break
        if matrix_buffer is None:
            return None

        return self._apply_tile_schedule(
            func,
            target,
            matrix_buffer,
            vector_buffer,
            output_buffer,
        )

    def apply_config(
        self,
        func: tir.PrimFunc,
        target: Target,
        config,
        _: bool = False,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = TileSchedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None:
            return None

        epilogue_name = _find_epilogue_name(block_infos, sch)
        if epilogue_name is None and len(block_infos) != 1:
            return None

        block_info = block_infos[0]
        if len(block_info.iters) not in [2, 3]:
            return None

        block = block_info.block_rv
        vector_input_buffers = is_gemv(sch, block_info)
        if vector_input_buffers is None or len(vector_input_buffers) != 1:
            return None

        is_inner_reduction = normalize(sch, block_info)
        if is_inner_reduction is None or not is_inner_reduction:
            return None

        block_stmt = sch.get(block)
        if len(block_stmt.writes) != 1:
            return None

        vector_buffer = vector_input_buffers[0]
        output_buffer = block_stmt.writes[0].buffer
        matrix_buffer = None
        for read_region in block_stmt.reads:
            if not read_region.buffer.same_as(vector_buffer) and not read_region.buffer.same_as(output_buffer):
                matrix_buffer = read_region.buffer
                break
        if matrix_buffer is None:
            return None

        return self._apply_tile_schedule(
            func,
            target,
            matrix_buffer,
            vector_buffer,
            output_buffer,
            fixed_spatial_tile=config,
        )
