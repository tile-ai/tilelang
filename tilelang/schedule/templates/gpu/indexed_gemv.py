"""Indexed GEMV schedule rule.

Matches one-token MoE expert-table contractions of the form::

    O[b, n] += X[b, k] * W[index[b], n, k]

This is intentionally a sibling of the generic GEMV rule, not a runtime
fallback.  The generated TE/TIR contains a data-dependent load in the matrix
buffer index, which prevents the generic GEMV matcher from recognizing the
pattern.  Once matched, the loop structure is still GEMV-like, so this rule
applies the same split/cache/reduce strategy directly to the non-affine block.
"""

from __future__ import annotations

from tilelang import tvm

from tvm import s_tir, tirx as tir
from tvm.target import Target
from tvm.s_tir.dlight import normalize_prim_func, try_inline_contiguous_spatial
from tvm.s_tir.dlight.analysis import SBlockInfo

from .base import GPUScheduleRule
from .gemv import _choose_tile_gemv_params
from ... import Schedule as TileSchedule


def _is_int_buffer(buf: tir.Buffer) -> bool:
    dtype = tvm.DataType(buf.dtype)
    return dtype.type_code in (0, 1)  # int / uint


def _region_uses_buffer(region: tir.BufferRegion, target: tir.Buffer) -> bool:
    """True when a buffer-region index expression reads ``target``."""
    found = False

    def visit(expr: tir.PrimExpr) -> None:
        nonlocal found
        if isinstance(expr, tir.BufferLoad) and expr.buffer.same_as(target):
            found = True

    for rng in region.region:
        tir.stmt_functor.post_order_visit(rng.min, visit)
        tir.stmt_functor.post_order_visit(rng.extent, visit)
    return found


class IndexedGEMV(GPUScheduleRule):
    """Schedule indexed expert-table GEMV through the GEMV tile schedule."""

    def _find_epilogue(
        self,
        block_infos: list[SBlockInfo],
    ):
        if len(block_infos) == 1:
            return None
        if len(block_infos) != 2 or not block_infos[1].is_injective():
            return False
        return block_infos[1].block_rv

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        try:
            sch = TileSchedule(func)
            block_infos = normalize_prim_func(sch)
            block_infos = try_inline_contiguous_spatial(sch, block_infos)
            if block_infos is None or len(block_infos) not in (1, 2):
                return None
            epilogue = self._find_epilogue(block_infos)
            if epilogue is False:
                return None

            block_info = block_infos[0]
            if len(block_info.iters) != 3:
                return None
            block = block_info.block_rv

            block_stmt = sch.get(block)
            if len(block_stmt.writes) != 1:
                return None
            output_buffer = block_stmt.writes[0].buffer

            read_buffers: list[tir.Buffer] = []
            for region in block_stmt.reads:
                buf = region.buffer
                if buf.same_as(output_buffer):
                    continue
                if any(buf.same_as(existing) for existing in read_buffers):
                    continue
                read_buffers.append(buf)

            index_buffers = [buf for buf in read_buffers if _is_int_buffer(buf)]
            if len(index_buffers) != 1:
                return None

            data_buffers = [buf for buf in read_buffers if not _is_int_buffer(buf)]
            index_buffer = index_buffers[0]
            matrix_candidates: list[tir.Buffer] = []
            for region in block_stmt.reads:
                if _is_int_buffer(region.buffer):
                    continue
                if not _region_uses_buffer(region, index_buffer):
                    continue
                if not any(region.buffer.same_as(existing) for existing in matrix_candidates):
                    matrix_candidates.append(region.buffer)
            vector_candidates = [buf for buf in data_buffers if not any(buf.same_as(m) for m in matrix_candidates)]
            if len(matrix_candidates) != 1 or len(vector_candidates) != 1:
                return None

            loops = sch.get_loops(block)
            if len(loops) != 3:
                return None
            batch_loop, spatial_loop, reduce_loop = loops
            num_threads, _tile_k, block_k, _tile_m = _choose_tile_gemv_params(
                target,
                sch.get(spatial_loop).extent,
                sch.get(reduce_loop).extent,
                matrix_candidates[0].dtype,
            )

            bx = sch.fuse(batch_loop, spatial_loop)
            ko, ki = sch.split(reduce_loop, factors=[None, block_k])

            block_name = block_stmt.name_hint
            if block_stmt.init is not None:
                sch.decompose_reduction(block, ko)
                block_name = block_name + "_update"

            block = sch.get_sblock(block_name)
            block_stmt = sch.get(block)
            matrix_read_idx = None
            vector_read_idx = None
            for idx, read_region in enumerate(block_stmt.reads):
                if read_region.buffer.same_as(matrix_candidates[0]):
                    matrix_read_idx = idx
                elif read_region.buffer.same_as(vector_candidates[0]):
                    vector_read_idx = idx
            if matrix_read_idx is None or vector_read_idx is None:
                return None

            block = sch.get_sblock(block_name)
            sch.cache_read_at(ko, block, matrix_read_idx, "local.fragment")
            block = sch.get_sblock(block_name)
            sch.cache_read_at(ko, block, vector_read_idx, "local.fragment")

            has_epilogue = epilogue is not None
            if has_epilogue:
                sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)

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

            sch.parallelize(ki)
            sch.bind(bx, "blockIdx.x")
            root = sch.get_sblock("root")
            sch.launch_thread(root, num_threads)
            return sch
        except Exception:
            import traceback

            traceback.print_exc()
            return None
