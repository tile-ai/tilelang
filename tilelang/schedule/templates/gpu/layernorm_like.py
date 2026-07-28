# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=invalid-name
"""LayerNorm-like reduction schedule using tile-level primitives."""

from __future__ import annotations


from ... import Schedule as TileSchedule
from tilelang.carver.common_schedules import get_output_blocks
from .base import GPUScheduleRule, spatial_tile_product_for_extents
from .reduction_utils import (
    _analyze_reduction_update,
    _as_const_int,
    _block_writes_buffer,
    _choose_num_threads,
    _choose_reduction_step,
    _collect_input_buffers,
    _find_buffer_index,
    _infer_init_value,
    _infer_reduce_dim,
    _is_square_of_buffer_load,
)

from tvm import s_tir, tirx as tir
from tvm.target import Target
from tvm.s_tir.dlight import normalize_prim_func
from tvm.s_tir.dlight.analysis import SBlockInfo


def _schedule_single_source_reduction(
    sch: TileSchedule,
    block_name: str,
    bx: s_tir.schedule.LoopRV,
    target: Target,
    write_back: bool = False,
    force_reduce_type: str | None = None,
    cache_read_consumer_blocks: list[s_tir.schedule.SBlockRV] | None = None,
    skip_cache_read: bool = False,
) -> tir.PrimExpr | None:
    block = sch.get_sblock(block_name)
    block_stmt = sch.get(block)
    if len(block_stmt.writes) != 1:
        return None

    update_info = _analyze_reduction_update(block_stmt)
    if update_info is None:
        return None
    reduce_type, rhs_expr = update_info

    write_buffer = block_stmt.writes[0].buffer
    write_buffer_index = _find_buffer_index(block_stmt.writes, write_buffer)
    if write_buffer_index is None:
        return None
    input_buffers = _collect_input_buffers(rhs_expr, write_buffer)
    if len(input_buffers) != 1:
        return None
    read_buffer_index = _find_buffer_index(block_stmt.reads, input_buffers[0])
    if read_buffer_index is None:
        return None

    reduction_loops = sch.get_loops(block)
    if len(reduction_loops) == 0:
        return None
    r_fused = reduction_loops[-1]

    reduce_step = _choose_reduction_step(target, sch.get(r_fused).extent)
    if reduce_step is not None:
        ro, ri = sch.split(r_fused, factors=[None, reduce_step], preserve_unit_iters=True)
        cache_read_loop = ro
        reduce_loop = ro
        thread_extent_expr = sch.get(ri).extent
    else:
        cache_read_loop = bx
        reduce_loop = r_fused
        thread_extent_expr = sch.get(r_fused).extent

    # When sharing the cache with sibling consumer blocks, the cache must live
    # at a loop that is an ancestor of ALL consumers — not the per-reduction
    # ``ro`` chunk, which only dominates its own reduction subtree.  Hoist to
    # ``bx`` (the common ancestor) so both reductions rewrite their reads to
    # the same fragment.
    if not skip_cache_read:
        shared_cache_loop = bx if cache_read_consumer_blocks else cache_read_loop
        sch.cache_read_at(
            shared_cache_loop,
            block,
            read_buffer_index,
            "local.fragment",
            consumer_blocks=cache_read_consumer_blocks,
        )

    block = sch.get_sblock(block_name)
    init_value = _infer_init_value(block_stmt, reduce_type)
    sch.cache_reduce_at(
        bx,
        block,
        write_buffer_index,
        "local.fragment",
        init_value,
        write_back=write_back,
    )

    block = sch.get_sblock(block_name)
    block_stmt = sch.get(block)
    if len(block_stmt.reads) != 1 or len(block_stmt.writes) != 1:
        return None
    reduce_dim = _infer_reduce_dim(
        block_stmt.reads[0].buffer,
        block_stmt.writes[0].buffer,
    )
    reduce_type_for_lower = force_reduce_type if force_reduce_type is not None else reduce_type
    sch.reduce_at(
        reduce_loop,
        block,
        read_buffer_index=0,
        write_buffer_index=0,
        reduce_type=reduce_type_for_lower,
        dim=reduce_dim,
        clear=False,
    )
    return thread_extent_expr


def _schedule_center_bridge(
    sch: TileSchedule,
    block_name: str,
    bx: s_tir.schedule.LoopRV,
) -> bool:
    block = sch.get_sblock(block_name)
    block_stmt = sch.get(block)
    if len(block_stmt.writes) != 1:
        return False

    write_buffer = block_stmt.writes[0].buffer
    for read_buffer_index, read_region in enumerate(block_stmt.reads):
        if read_region.buffer.same_as(write_buffer):
            continue
        # Skip buffers already in fast memory scope (produced by preceding
        # reduction stages in the same CTA).
        buf_scope = read_region.buffer.scope()
        if buf_scope and buf_scope != "global":
            continue
        sch.cache_read_at(bx, block, read_buffer_index, "local.fragment")
        block = sch.get_sblock(block_name)

    sch.cache_write_at(bx, block, 0, "local.fragment", write_back=False)
    block = sch.get_sblock(block_name)
    loops = sch.get_loops(block)
    if loops:
        sch.parallel(loops[-1])
    return True


class LayerNormLike(GPUScheduleRule):
    """LayerNorm-like two-reduction schedule rule."""

    def apply_config(
        self,
        func: tir.PrimFunc,
        target: Target,
        config,
        _: bool = False,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        return self._apply_impl(func, target, fixed_spatial_tile=config)

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        return self._apply_impl(func, target)

    def _apply_impl(
        self,
        func: tir.PrimFunc,
        target: Target,
        fixed_spatial_tile=None,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = TileSchedule(func)
        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None

        reduction_indices = [i for i, info in enumerate(block_infos) if info.is_reduction()]
        if len(reduction_indices) != 2:
            return None
        first_reduction_idx = reduction_indices[0]
        for info in block_infos[:first_reduction_idx]:
            if not info.is_injective():
                return None
            sch.compute_inline(info.block_rv)

        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None

        reduction_indices = [i for i, info in enumerate(block_infos) if info.is_reduction()]
        if len(reduction_indices) != 2:
            return None
        first_reduction_idx, second_reduction_idx = reduction_indices
        first_reduction_info = block_infos[first_reduction_idx]
        second_reduction_info = block_infos[second_reduction_idx]
        bridge_infos = block_infos[first_reduction_idx + 1 : second_reduction_idx]
        trailing_infos = block_infos[second_reduction_idx + 1 :]
        if len(trailing_infos) == 0:
            return None
        if not all(info.is_injective() for info in bridge_infos):
            return None
        if not all(info.is_injective() for info in trailing_infos):
            return None

        output_block_names = {sch.get(output_block).name_hint for output_block in get_output_blocks(sch, block_infos)}
        output_name = sch.get(trailing_infos[-1].block_rv).name_hint
        if output_name not in output_block_names:
            return None

        first_reduction_stmt = sch.get(first_reduction_info.block_rv)
        second_reduction_stmt = sch.get(second_reduction_info.block_rv)
        first_update_info = _analyze_reduction_update(first_reduction_stmt)
        second_update_info = _analyze_reduction_update(second_reduction_stmt)
        if first_update_info is None or second_update_info is None:
            return None
        first_reduce_type, _ = first_update_info
        second_reduce_type, second_rhs_expr = second_update_info
        if first_reduce_type != "sum" or second_reduce_type != "sum":
            return None
        if len(second_reduction_stmt.writes) != 1:
            return None
        second_input_buffers = _collect_input_buffers(
            second_rhs_expr,
            second_reduction_stmt.writes[0].buffer,
        )
        if len(second_input_buffers) != 1:
            return None
        center_buffer = second_input_buffers[0]
        if not _is_square_of_buffer_load(second_rhs_expr, center_buffer):
            return None

        # Look for a bridge block that writes to the second reduction's input
        # buffer (the classic LayerNorm "diff" block).  If there is none, we
        # fall into the second-moment variant: both reductions read the same
        # source (typically x) and no diff fragment is materialised.
        center_name: str | None = None
        for info in bridge_infos:
            block_stmt = sch.get(info.block_rv)
            if _block_writes_buffer(block_stmt, center_buffer):
                center_name = block_stmt.name_hint
                break

        first_reduction_name = first_reduction_stmt.name_hint
        second_reduction_name = second_reduction_stmt.name_hint

        for info in reversed(bridge_infos):
            name = sch.get(info.block_rv).name_hint
            if name == center_name:
                continue
            sch.compute_inline(info.block_rv)
        for info in reversed(trailing_infos[:-1]):
            sch.compute_inline(info.block_rv)

        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None
        block_infos_by_name: dict[str, SBlockInfo] = {sch.get(info.block_rv).name_hint: info for info in block_infos}
        if (
            first_reduction_name not in block_infos_by_name
            or second_reduction_name not in block_infos_by_name
            or output_name not in block_infos_by_name
        ):
            return None
        if center_name is not None and center_name not in block_infos_by_name:
            return None

        second_dom_kind = block_infos_by_name[second_reduction_name].dom_kind()
        num_leading_s = len(second_dom_kind) - len(second_dom_kind.lstrip("S"))
        if num_leading_s <= 0:
            return None

        # Derive bx from the OUTPUT block's leading loops so that the output is
        # automatically "inside" bx, breaking the circular dependency:
        #
        #   compute_at(second_reduction, bx): layernorm_out is under bx ✓
        #   compute_at(center/diff, bx):      sum_sq + output are under bx ✓
        #   compute_at(first_reduction, bx):  diff is under bx ✓
        #
        # For M>1: the output's first loop is M (extent M).  Split with
        # [None, 1] → bx(M), inner(1).  blockIdx.x = M.
        #
        # For M=1: TVM eliminates the unit M loop from injective blocks (diff,
        # layernorm_out) but preserves it in reduction blocks (sum_x, sum_sq).
        # The output's first loop is then N (extent N), not M (extent 1).
        # Splitting N with [1, None] → bx(1), N_inner(N) forces blockIdx.x=1
        # and keeps N_inner as the per-CTA parallel loop — no O(N²) regression.
        output_block_rv = sch.get_sblock(output_name)
        output_loops = sch.get_loops(output_block_rv)

        # Compare extents of the output's leading loops with the second
        # reduction's leading spatial loops to detect the M=1 case.
        second_info = block_infos_by_name[second_reduction_name]
        second_loops = sch.get_loops(second_info.block_rv)
        second_s_extents = [_as_const_int(sch.get(lp).extent) for lp in second_loops[:num_leading_s]]
        output_leading_extents = [_as_const_int(sch.get(lp).extent) for lp in output_loops[:num_leading_s]]
        m_loop_present = second_s_extents == output_leading_extents

        if m_loop_present:
            # Standard path: output has M loops → bx from M loops, split [None, 1].
            output_s_loops = output_loops[:num_leading_s]
            bx_s_fused = sch.fuse(*output_s_loops) if len(output_s_loops) > 1 else output_s_loops[0]
            spatial_tile = 1
            if fixed_spatial_tile is not None:
                spatial_tile = spatial_tile_product_for_extents(
                    fixed_spatial_tile,
                    [sch.get(loop).extent for loop in output_s_loops],
                )
                if spatial_tile is None:
                    return None
                spatial_extent = _as_const_int(sch.get(bx_s_fused).extent)
                if spatial_extent is not None:
                    spatial_tile = min(spatial_tile, spatial_extent)
            bx, _ = sch.split(
                bx_s_fused,
                factors=[None, spatial_tile],
                preserve_unit_iters=True,
            )
        else:
            # M=1 path: M loop eliminated from output; first output loop is N.
            # Split with [1, None] → bx(1) forces blockIdx.x=1.
            bx, _ = sch.split(output_loops[0], factors=[1, None], preserve_unit_iters=True)

        sch.bind(bx, "blockIdx.x")

        # Place second_reduction, (optional) center, first_reduction inside bx.
        # compute_at auto-inserts each block right before its first consumer,
        # giving the correct order: first_reduction → center → second_reduction → output.
        sch.compute_at(
            sch.get_sblock(second_reduction_name),
            bx,
            preserve_unit_loops=True,
        )
        if center_name is not None:
            sch.compute_at(
                sch.get_sblock(center_name),
                bx,
                preserve_unit_loops=True,
                index=0,
            )
        sch.compute_at(
            sch.get_sblock(first_reduction_name),
            bx,
            preserve_unit_loops=True,
            index=0,
        )

        # Second-moment variant: both reductions read the same source buffer
        # (typically x).  Share a single cache_read between them by passing
        # the second reduction as an extra consumer, so cache_read_at unions
        # their access regions and rewrites both blocks to use the same
        # fragment — saving ~16 regs/thread at large N (N=16384: 72→56).
        first_cache_consumers: list[s_tir.schedule.SBlockRV] | None = None
        if center_name is None:
            first_cache_consumers = [sch.get_sblock(second_reduction_name)]

        first_thread_extent = _schedule_single_source_reduction(
            sch,
            first_reduction_name,
            bx,
            target,
            write_back=False,
            cache_read_consumer_blocks=first_cache_consumers,
        )
        if first_thread_extent is None:
            return None

        if center_name is not None and not _schedule_center_bridge(sch, center_name, bx):
            return None

        # When we pre-created a shared x cache for both reductions, skip the
        # second reduction's own cache_read_at — its input buffer is already
        # in a local.fragment.  Creating another cache here would duplicate
        # the 16 regs/thread we just saved.
        second_thread_extent = _schedule_single_source_reduction(
            sch,
            second_reduction_name,
            bx,
            target,
            write_back=False,
            force_reduce_type="sumsq",
            skip_cache_read=(first_cache_consumers is not None),
        )
        if second_thread_extent is None:
            return None

        # Parallelise the output's innermost (N) loop and cache global inputs.
        output_block = sch.get_sblock(output_name)
        output_loops = sch.get_loops(output_block)
        if output_loops:
            sch.parallel(output_loops[-1])
            output_parallel_extent = sch.get(output_loops[-1]).extent
        else:
            output_parallel_extent = tir.IntImm("int32", 1)

        output_block = sch.get_sblock(output_name)
        output_stmt = sch.get(output_block)
        output_writes = [region.buffer for region in output_stmt.writes]
        bx_extent = _as_const_int(sch.get(bx).extent)
        output_parallel_extent_int = _as_const_int(output_parallel_extent)
        # Wide prefill rows (multiple CTAs over M, large N per CTA) are
        # register-limited by the persistent parameter fragments in the output
        # epilogue.  Let w/b come straight from global memory in that case:
        # every CTA reads the same vectors, so H100's L2 serves them cheaply,
        # while removing the fragments frees ~32 regs/thread at N=16384.
        skip_output_broadcast_cache = (
            bx_extent is not None and bx_extent > 1 and output_parallel_extent_int is not None and output_parallel_extent_int >= 8192
        )
        # The second-moment variant reads x twice in the reductions and once
        # more in the epilogue.  Caching x into a local.fragment for the
        # output would ADD a persistent x fragment on top of the reduction
        # caches, netting no register-pressure win.  Skip caching buffers
        # whose rank matches the output's rank (e.g. x at shape M×N).  For
        # wide prefill rows, also skip lower-rank broadcast inputs (w, b) so
        # those vectors do not become persistent fragments either.
        output_rank = len(output_stmt.writes[0].buffer.shape)
        for read_buffer_index, read_region in enumerate(output_stmt.reads):
            if any(read_region.buffer.same_as(write_buffer) for write_buffer in output_writes):
                continue
            # Skip buffers already in fast memory (produced by preceding stages).
            buf_scope = read_region.buffer.scope()
            if buf_scope and buf_scope != "global":
                continue
            # Skip same-rank (per-element) inputs — caching them creates a
            # persistent full-row fragment that dominates register pressure
            # at large N.  The compiler still coalesces the in-loop global
            # loads.
            if len(read_region.buffer.shape) == output_rank:
                continue
            if skip_output_broadcast_cache and len(read_region.buffer.shape) < output_rank:
                continue
            sch.cache_read_at(bx, output_block, read_buffer_index, "local.fragment")
            output_block = sch.get_sblock(output_name)

        num_threads = max(
            _choose_num_threads(target, first_thread_extent),
            _choose_num_threads(target, second_thread_extent),
            _choose_num_threads(target, output_parallel_extent),
        )
        sch.launch_thread(sch.get_sblock("root"), num_threads)
        return sch
