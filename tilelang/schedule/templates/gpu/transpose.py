"""Tile-based transpose schedule rule."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

from tilelang import tvm
from tilelang import _ffi_api as tl_ffi

from ... import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule

from tvm import s_tir, tirx as tir
from tvm.target import Target
from tvm.s_tir.dlight import normalize_prim_func


def _largest_power_of_two_at_most(value: int) -> int:
    result = 1
    while (result << 1) <= value:
        result <<= 1
    return result


def _as_const_int(expr: tir.PrimExpr | int) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1 or value <= multiple:
        return value
    return max(multiple, (value // multiple) * multiple)


def _warp_size(target: Target) -> int:
    return int(target.attrs.get("thread_warp_size", 32))


def _exprs_structurally_equal(lhs: tir.PrimExpr, rhs: tir.PrimExpr) -> bool:
    return bool(tvm.ir.structural_equal(lhs, rhs))


def _same_index_multiset(lhs_indices, rhs_indices) -> bool:
    if len(lhs_indices) != len(rhs_indices):
        return False
    used = [False] * len(rhs_indices)
    for lhs in lhs_indices:
        matched = False
        for idx, rhs in enumerate(rhs_indices):
            if used[idx]:
                continue
            if _exprs_structurally_equal(lhs, rhs):
                used[idx] = True
                matched = True
                break
        if not matched:
            return False
    return True


@dataclass(frozen=True)
class _TransposeConfig:
    tile_m: int
    tile_n: int
    num_threads: int
    use_local_write_cache: bool = False


def _choose_transpose_config(
    target: Target,
    dtype: str,
    out_n_extent: tir.PrimExpr | int,
    out_m_extent: tir.PrimExpr | int,
) -> _TransposeConfig:
    """Choose a bandwidth-oriented transpose tile analytically.

    The rule keeps the tile square and transaction-aligned, then sizes it from
    a per-CTA shared-memory budget so the kernel stays occupancy-friendly.
    """

    elem_bits = int(tvm.DataType(dtype).bits)
    elem_bytes = max(elem_bits // 8, 1)
    warp = _warp_size(target)
    max_threads = min(int(utils.max_threads_per_block(target)), 256)

    sm_version = utils.get_sm_version(target)
    target_fast_bytes = 16 * 1024 if target.kind.name == "cuda" and sm_version >= 80 else 8 * 1024

    read_align_elems = max(1, 128 // elem_bytes) if target.kind.name == "cuda" else warp
    write_align_elems = max(1, 32 // elem_bytes) if target.kind.name == "cuda" else warp
    preferred_align = _largest_power_of_two_at_most(max(read_align_elems, write_align_elems))

    # Model both the shared read tile and the local write tile. For transpose,
    # smaller tiles with cached write-back typically win because they keep
    # occupancy high while still enabling vectorized copy paths.
    tile_limit = min(64, _largest_power_of_two_at_most(max(1, isqrt(max(1, target_fast_bytes // (2 * elem_bytes))))))

    out_n = _as_const_int(out_n_extent)
    out_m = _as_const_int(out_m_extent)
    if out_n is not None:
        tile_limit = min(tile_limit, _largest_power_of_two_at_most(max(1, out_n)))
    if out_m is not None:
        tile_limit = min(tile_limit, _largest_power_of_two_at_most(max(1, out_m)))

    tile_size = max(1, tile_limit)
    if tile_size >= preferred_align:
        tile_size = _round_down_to_multiple(tile_size, preferred_align)
    tile_size = _largest_power_of_two_at_most(max(1, tile_size))

    if tile_size >= 64:
        num_threads = 64
    elif tile_size >= 32:
        num_threads = 32
    else:
        num_threads = warp
    num_threads = _largest_power_of_two_at_most(min(max_threads, max(warp, num_threads)))

    return _TransposeConfig(
        tile_m=tile_size,
        tile_n=tile_size,
        num_threads=num_threads,
        use_local_write_cache=True,
    )


class Transpose(GPUScheduleRule):
    """Tile-based transpose schedule using shared-memory staging."""

    @staticmethod
    def _is_transpose(sch, block_rv) -> bool:
        block = sch.get(block_rv)
        if not isinstance(block.body, tir.BufferStore):
            return False
        rhs = block.body.value
        if not isinstance(rhs, tir.BufferLoad):
            return False
        lhs_indices = block.body.indices
        rhs_indices = rhs.indices
        same_order = len(lhs_indices) == len(rhs_indices) and all(
            _exprs_structurally_equal(lhs, rhs) for lhs, rhs in zip(lhs_indices, rhs_indices)
        )
        return (not same_order) and _same_index_multiset(lhs_indices, rhs_indices)

    def is_transpose(self, sch, block_rv):
        return Transpose._is_transpose(sch, block_rv)

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
        if block_infos is None:
            return None

        # Find the last transpose block (scan from end).
        transpose_idx = -1
        for idx, info in reversed(list(enumerate(block_infos))):
            if self._is_transpose(sch, info.block_rv):
                transpose_idx = idx
                break
            if not info.is_injective():
                return None
        if transpose_idx == -1:
            return None

        transpose_block = block_infos[transpose_idx].block_rv
        loops = sch.get_loops(transpose_block)
        if len(loops) != 2:
            return None

        # Inline injective producers before the transpose block.
        for info in block_infos[:transpose_idx]:
            try:
                sch.compute_inline(info.block_rv)
            except Exception:  # pylint: disable=broad-except
                import traceback

                traceback.print_exc()
                return None

        # Inline injective consumers after the transpose block (reversed).
        # May fail on output blocks — bail out and let another rule handle it.
        for info in reversed(block_infos[transpose_idx + 1 :]):
            try:
                sch.compute_inline(info.block_rv)
            except Exception:  # pylint: disable=broad-except
                import traceback

                traceback.print_exc()
                return None

        # Get the read buffer name to construct shared cache buffer name.
        block_stmt = sch.get(transpose_block)
        read_buf = block_stmt.reads[0].buffer
        shared_buf_name = read_buf.name + "_shared_dyn"

        i, j = loops
        i_ext = _as_const_int(sch.get(i).extent)
        j_ext = _as_const_int(sch.get(j).extent)

        config = _choose_transpose_config(
            target,
            read_buf.dtype,
            sch.get(i).extent,
            sch.get(j).extent,
        )

        # When one dimension is fully covered by the tile, the
        # cache_read_at read pattern becomes sparse in shared memory
        # (bounding box >> actual elements) and LayoutInference
        # cannot reshape the fragment to match.  Let the ElementWise
        # or Fallback rule handle these small transposes instead.
        if i_ext is not None and i_ext <= config.tile_n and j_ext is not None and j_ext > config.tile_m:
            return None
        if j_ext is not None and j_ext <= config.tile_m and i_ext is not None and i_ext > config.tile_n:
            return None

        bi, ii = sch.split(i, factors=[None, config.tile_n], preserve_unit_iters=True)
        bj, jj = sch.split(j, factors=[None, config.tile_m], preserve_unit_iters=True)
        sch.reorder(bi, bj, ii, jj)

        # Stage input through shared memory for coalesced global reads;
        # the transposed read from shared avoids uncoalesced global access.
        # SIMT copy with swizzled layout outperforms TMA for transpose.
        sch.cache_read_at(bj, transpose_block, 0, "shared.dyn", disable_tma=True)
        if config.use_local_write_cache:
            sch.cache_write_at(bj, transpose_block, 0, "local.fragment")

        # Annotate the shared buffer with a swizzled layout to avoid
        # bank conflicts on the transposed read.
        cache_buf = tir.decl_buffer((config.tile_m, config.tile_n), dtype=read_buf.dtype)
        swizzle_layout = tl_ffi.make_swizzled_layout(cache_buf, True, True)
        sch.annotate_layout(sch.get_sblock("root"), shared_buf_name, swizzle_layout)

        sch.parallel(ii)
        sch.parallel(jj)
        sch.bind(bi, "blockIdx.x")
        sch.bind(bj, "blockIdx.y")

        sch.launch_thread(sch.get_sblock("root"), config.num_threads)
        return sch
