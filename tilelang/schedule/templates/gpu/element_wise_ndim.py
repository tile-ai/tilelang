from __future__ import annotations

from ... import Schedule as TileSchedule
from .base import (
    GPUScheduleRule,
    align_spatial_tile_to_extents,
    spatial_tile_product_for_extents,
)

from tvm import s_tir, tirx as tir
from tvm.target import Target
from tvm.s_tir.dlight import normalize_prim_func, try_inline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_const_int(expr) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _largest_pow2_at_most(n: int) -> int:
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _choose_inner_factor(
    axis_extent: int,
    suffix_extent: int,
    max_factor: int,
    num_threads: int,
) -> int | None:
    """Choose a split factor that avoids predicated tail tiles when possible."""
    limit = min(axis_extent, max_factor)
    for factor in range(limit, 0, -1):
        if axis_extent % factor != 0:
            continue
        if (factor * suffix_extent) % num_threads != 0:
            continue
        return factor
    return None


def _choose_tile_and_threads(
    total: int,
    dtype_bits: int = 16,
    n_buffers: int = 2,
) -> tuple[int, int]:
    """Pick ``(tile, num_threads)`` anchored on a 128-bit per-thread load.

    Anchors:
      * ``vec = max(1, 128 // dtype_bits)`` — 128-bit vector width for the
        element type.  fp32 → 4, fp16/bf16 → 8, int8/fp8 → 16.
      * ``num_threads = 256`` — standard occupancy anchor on H100/A100.
      * ``ept = vec``; double to ``2 * vec`` when the tensor is large
        enough that per-block work warrants wider tiles, provided the
        register budget allows it.

    Register budget guard: each input / output fragment costs
    ``ept * dtype_bits`` bits per thread; total register usage scales with
    ``n_buffers``.  We cap the combined per-thread fragment budget at
    about 8 × 128 bits (8 ``uint4`` per thread) to avoid runaway spills.
    """
    if total <= 0:
        return 1, 1

    vec = max(1, 128 // dtype_bits)
    threads = 256
    if threads % total != 0:
        threads = total
        vec = 1
    else:
        # Fall back to fewer threads only when the tensor is smaller than a
        # single vectorised block; the outer guard (total >= 1024) makes this
        # almost never fire, but keep it for safety.

        while threads > 32 and threads * vec > total:
            threads //= 2

    # ept = vec locks one 128-bit (``uint4``-class) global load per
    # thread per input buffer.  Doubling to 2*vec was explored but
    # regressed compute-heavy unary ops (gelu/silu) due to register
    # pressure, while the big memory-bound ops are already bandwidth-
    # saturated at ept=vec.  A future cost model can reintroduce
    # adaptive scaling; until then, a single 128-bit width keeps the
    # heuristic predictable across dtypes.
    ept = vec

    tile = threads * ept
    if tile > total:
        tile = (total + threads - 1) // threads * threads
    return max(tile, threads), threads


def _dtype_bits(dtype: str) -> int:
    """Bits of one element; fallback to 16 for unknown dtypes."""
    try:
        return int(tir.DataType(dtype).bits)
    except Exception:
        return 16


def _has_cropped_read(block_stmt: tir.SBlock) -> bool:
    """Return True for elementwise crops/slices whose input is larger than output.

    The axis-walk schedule stages read buffers into local fragments.  For slice
    patterns such as ``input[:, offset:offset+n] -> output[:, :n]`` that creates
    fragment indices involving normalized expressions like ``v - block_offset``,
    which currently trips TileLang layout inference.  Let the simple Fallback
    rule handle these view/copy kernels without fragment caching.
    """
    if not block_stmt.writes:
        return False
    out_shape = block_stmt.writes[0].buffer.shape
    for read in block_stmt.reads:
        in_shape = read.buffer.shape
        if len(in_shape) != len(out_shape):
            continue
        for in_dim, out_dim in zip(in_shape, out_shape):
            in_extent = _as_const_int(in_dim)
            out_extent = _as_const_int(out_dim)
            if in_extent is not None and out_extent is not None and in_extent > out_extent:
                return True
    return False


def _has_broadcast_read(block_stmt: tir.SBlock) -> bool:
    """Return True when an input broadcasts a unit axis over the output.

    The axis-walk schedule can still bind the output iteration space for these
    kernels, but staging broadcasted reads into ``local.fragment`` may introduce
    vector ``tir.Broadcast`` index expressions during layout/legalization.  TVM's
    late ``RenormalizeSplitPattern`` prover expects scalar integer index
    arithmetic there and fails on those vector expressions.  Leave such reads in
    global memory and only use the axis-walk loop mapping.
    """
    if not block_stmt.writes:
        return False
    out_shape = block_stmt.writes[0].buffer.shape
    for read in block_stmt.reads:
        in_shape = read.buffer.shape
        if len(in_shape) != len(out_shape):
            continue
        for in_dim, out_dim in zip(in_shape, out_shape):
            in_extent = _as_const_int(in_dim)
            out_extent = _as_const_int(out_dim)
            if in_extent is not None and out_extent is not None and in_extent == 1 and out_extent > 1:
                return True
    return False


def _inline_to_single_block(sch):
    """Inline all blocks except the last, mirroring element_wise.py."""
    block_infos = normalize_prim_func(sch)
    if block_infos is None:
        return None
    block_infos = try_inline(sch, block_infos)
    if not block_infos:
        return None

    for bi in block_infos[:-1]:
        if bi.is_reduction() or not bi.is_injective():
            return None
        try:
            sch.compute_inline(bi.block_rv)
        except s_tir.ScheduleError:
            return None

    block_infos = normalize_prim_func(sch)
    if not block_infos or len(block_infos) != 1:
        return None

    info = block_infos[0]
    if info.is_reduction():
        return None
    if len(sch.get_loops(info.block_rv)) == 0:
        return None
    if any(it.kind not in ("S", "O") for it in info.iters):
        return None
    if not any(it.kind == "S" for it in info.iters):
        return None
    return info


# ---------------------------------------------------------------------------
# Schedule rule
# ---------------------------------------------------------------------------


class ElementWiseNDim(GPUScheduleRule):
    """Axis-walk schedule rule for N-D elementwise kernels."""

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        return self._apply_impl(func, target)

    def apply_config(
        self,
        func: tir.PrimFunc,
        target: Target,
        config,
        _: bool = False,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        return self._apply_impl(func, target, fixed_spatial_tile=config)

    def _apply_impl(
        self,
        func: tir.PrimFunc,
        target: Target,
        fixed_spatial_tile=None,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = TileSchedule(func)
        info = _inline_to_single_block(sch)
        if info is None:
            return None

        block = info.block_rv
        s_loop_rvs = [it.loop_rv for it in info.iters if it.kind == "S"]
        o_loop_rvs = [it.loop_rv for it in info.iters if it.kind == "O"]
        if not s_loop_rvs:
            return None

        # TODO: support dynamic shapes
        extents: list[int] = []
        for lrv in s_loop_rvs:
            ext = _as_const_int(sch.get(lrv).extent)
            if ext is None:
                return None
            extents.append(ext)

        total = 1
        for e in extents:
            total *= e
        # Skip tiny tensors — the simple fallback handles them fine and
        # the axis walk has no advantage.
        if total < 1024:
            return None

        # Anchor the heuristic on the element dtype so that per-thread
        # loads actually hit 128 bits.  Use the dominant read buffer's
        # dtype; for mixed-precision injective chains this is the
        # producer's precision, which dominates memory traffic.
        block_stmt = sch.get(block)
        has_cropped_read = _has_cropped_read(block_stmt)
        if _has_broadcast_read(block_stmt):
            return None
        if block_stmt.reads:
            dtype_bits = _dtype_bits(block_stmt.reads[0].buffer.dtype)
        elif block_stmt.writes:
            dtype_bits = _dtype_bits(block_stmt.writes[0].buffer.dtype)
        else:
            dtype_bits = 16
        n_buffers = len(block_stmt.reads) + len(block_stmt.writes)

        if fixed_spatial_tile is None:
            TILE, NUM_THREADS = _choose_tile_and_threads(
                total,
                dtype_bits=dtype_bits,
                n_buffers=n_buffers,
            )
        else:
            TILE = spatial_tile_product_for_extents(fixed_spatial_tile, extents)
            if TILE is None:
                return None
            TILE = min(TILE, total)
            _, NUM_THREADS = _choose_tile_and_threads(
                total,
                dtype_bits=dtype_bits,
                n_buffers=n_buffers,
            )

        # --- Axis walk: decide outer / inner split ---
        # Retry with a smaller tile when the largest candidate fails to
        # produce a clean split.  This handles shapes whose inner axis is
        # divisible by the base 128-bit-vector tile but not by the
        # 2x-scaled tile (e.g. (1024, 10240) fp16 with vec=8: 10240 % 2048
        # == 0 but 10240 % 4096 != 0).
        vec = max(1, 128 // dtype_bits)
        base_tile = NUM_THREADS * vec
        cutoff = len(s_loop_rvs)
        split_info: tuple[int, int] | None = None
        selected_tile = TILE

        while True:
            acc = 1
            cutoff = len(s_loop_rvs)
            split_info = None
            for i in range(len(s_loop_rvs) - 1, -1, -1):
                if acc * extents[i] <= selected_tile:
                    acc *= extents[i]
                    cutoff = i
                    continue
                need = selected_tile // acc
                factor = _choose_inner_factor(
                    extents[i],
                    acc,
                    need,
                    NUM_THREADS,
                )
                if factor is not None:
                    # Prefer factors that divide the split axis and make the
                    # fused inner work a whole number of thread blocks.  This
                    # avoids predicated tail tiles that TileLang layout
                    # inference currently struggles to assign a layout for.
                    split_info = (i, factor)
                break

            walk_ok = split_info is not None or cutoff < len(s_loop_rvs)
            # The schedule path additionally requires a non-empty outer
            # group (see the outer/inner check below); detect the failure
            # mode where everything swallows into "inner" as well.
            if walk_ok and split_info is None and cutoff == 0:
                walk_ok = False

            if walk_ok:
                TILE = selected_tile
                break
            if selected_tile <= base_tile:
                return None
            selected_tile //= 2

        try:
            if o_loop_rvs:
                sch.reorder(*s_loop_rvs, *o_loop_rvs)

            fixed_axis_tiles = None
            if fixed_spatial_tile is not None:
                fixed_axis_tiles = align_spatial_tile_to_extents(
                    fixed_spatial_tile,
                    extents,
                )
                if fixed_axis_tiles is None:
                    return None

            if fixed_axis_tiles is not None:
                outer_rvs = []
                inner_rvs = []
                inner_extents = []
                for loop_rv, extent, axis_tile in zip(s_loop_rvs, extents, fixed_axis_tiles):
                    axis_tile = max(1, min(int(axis_tile), int(extent)))
                    if axis_tile == extent:
                        inner_rvs.append(loop_rv)
                        inner_extents.append(axis_tile)
                    else:
                        outer, inner_axis = sch.split(
                            loop_rv,
                            factors=[None, axis_tile],
                            preserve_unit_iters=True,
                        )
                        outer_rvs.append(outer)
                        inner_rvs.append(inner_axis)
                        inner_extents.append(axis_tile)
            elif split_info is not None:
                axis, factor = split_info
                out_part, in_part = sch.split(
                    s_loop_rvs[axis],
                    factors=[None, factor],
                    preserve_unit_iters=True,
                )
                outer_rvs = list(s_loop_rvs[:axis]) + [out_part]
                inner_rvs = [in_part] + list(s_loop_rvs[axis + 1 :])
            else:
                outer_rvs = list(s_loop_rvs[:cutoff])
                inner_rvs = list(s_loop_rvs[cutoff:])
                inner_extents = extents[cutoff:]

            if not outer_rvs or not inner_rvs:
                return None

            if split_info is not None:
                axis, factor = split_info
                inner_extents = [factor] + extents[axis + 1 :]
            use_fragment_cache = not has_cropped_read and all(_is_power_of_two(ext) for ext in inner_extents)

            # Fuse outer loops into blockIdx.x
            bx = outer_rvs[0] if len(outer_rvs) == 1 else sch.fuse(*outer_rvs)

            # Fuse inner loops into the tile dimension
            inner = inner_rvs[0] if len(inner_rvs) == 1 else sch.fuse(*inner_rvs)

            if o_loop_rvs:
                sch.reorder(bx, inner, *o_loop_rvs)

            if use_fragment_cache:
                block_stmt = sch.get(block)
                for i in range(len(block_stmt.reads)):
                    sch.cache_read_at(bx, block, i, "local.fragment")
                sch.cache_write_at(bx, block, 0, "local.fragment")

            sch.parallel(inner)
            sch.bind(bx, "blockIdx.x")
            sch.launch_thread(sch.get_sblock("root"), NUM_THREADS)
            return sch
        except Exception:
            import traceback

            traceback.print_exc()
            return None
