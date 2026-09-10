"""Ascend persistent GEMM block schedulers.

Built on the shared ``BaseTileScheduler`` skeleton in
``tilelang.language.tile_schedule``; see that module's docstring for how
``@meta_class`` inlining and the ``T.alloc_var`` state protocol work.
"""

from __future__ import annotations

import tilelang.ascend.language as T
from tilelang.language.meta import meta_class
from tilelang.language.tile_schedule import BaseTileScheduler


@meta_class
class AscendBaseTileScheduler(BaseTileScheduler):
    """Base for persistent GEMM block schedulers (GROUP_M swizzle + N-first snake).

    Holds everything shared across GEMM variants: the grid geometry, the stateless
    ``_swizzle`` / ``coord`` decode, the persistent-block-index advance, and the
    ``init`` / ``valid`` / ``next_block`` loop protocol. Concrete variants subclass
    this and implement ``_step`` (advance one block + set ``m_idx`` / ``n_idx`` /
    ``valid_flag``); grouped variants add their own state and ``_init_state``.

    Each persistent core independently walks the global ``(m_block, n_block)`` grid.
    The block index handed out on iteration ``current_iter`` is
    ``current_iter * num_cores + core_id``, mapped to a tile by ``_swizzle``. Usage::

        sched = T.AscendTileScheduler(block_m=BM, block_n=BN,
                                num_cores=NUM_CORES, shape_m=M, shape_n=N)
        sched.init(bx)
        while sched.valid():
            m_tile, n_tile = sched.m_idx[0], sched.n_idx[0]
            # ... compute tile (m_tile, n_tile) ...
            sched.next_block()

    Parameters
    ----------
    block_m, block_n : int
        Tile sizes along M and N.
    num_cores : int | PrimExpr
        Persistent stride = number of resident cores (grid dim).
    shape_m, shape_n : int | PrimExpr
        Problem M and N.
    group_m : int
        Swizzle panel height along M (4 or 8).
    xor_n : bool
        XOR the in-panel n-index with the m-index (default True).
    snake : bool
        Reverse the N-group order on odd superrows (default True).
    stateful : bool
        If True (default) allocate state and enable ``init`` / ``valid`` /
        ``next_block``. If False expose only the store-free ``coord`` decode.
    name : str, optional
        Optional state-buffer name prefix. The default ``None`` uses generic
        auto-generated names.

    Notes
    -----
    State lives in single-element ``T.alloc_var`` buffers -- read with
    ``sched.m_idx[0]`` and (inside methods) write with ``self.x[0] = ...``.
    """

    def __init__(
        self,
        *,
        block_m,
        block_n,
        num_cores,
        shape_m,
        shape_n,
        group_m: int = 4,
        xor_n: bool = True,
        snake: bool = True,
        stateful: bool = True,
        name: str | None = None,
    ):
        super().__init__(stateful=stateful, name=name)
        assert group_m in (4, 8), "group_m must be 4 or 8"

        self._block_m = block_m
        self._block_n = block_n
        self._num_cores = num_cores
        self._shape_m = shape_m
        self._shape_n = shape_n
        self._group_m = group_m
        self._group_n = 32 // group_m
        self._xor_n = xor_n
        self._snake = snake

        # Compile-time grid geometry (Normal / Batched share the full grid).
        self._num_m_blocks = T.ceildiv(shape_m, block_m)
        self._num_n_blocks = T.ceildiv(shape_n, block_n)
        self._num_blocks = self._num_m_blocks * self._num_n_blocks

        # ``coord`` (stateless swizzle) is available regardless of statefulness.
        self._total_tiles = self._num_blocks

        if not stateful:
            return

        # ``core_id`` (set by init) and ``valid_flag`` (data-dependent validity).
        self.core_id = T.alloc_var(T.int32)
        self.valid_flag = T.alloc_var(T.int32)
        self._alloc_state()

    # ---- hooks for subclasses ---------------------------------------------

    def _alloc_state(self):
        """Allocate variant-specific cursor buffers (default: none)."""

    def _init_state(self):
        """Initialize variant-specific cursors in ``init`` (default: nothing)."""

    def _step(self):
        """Advance one persistent block: set ``m_idx`` / ``n_idx`` / ``valid_flag``."""
        raise NotImplementedError

    # ---- stateless swizzle -------------------------------------------------

    def _swizzle(self, block_idx, local_num_m_blocks):
        """Map a (group-local) ``block_idx`` to ``(m_block, n_block)`` (store-free).

        Faithful port of DeepGEMM ``get_swizzled_block``: small-M row-major fast
        path, tail row-major fallback for the last ``< group_m`` M-rows, and the
        GROUP_M superrow swizzle (+ optional xor_n / snake) on the aligned region.
        ``local_num_m_blocks`` is compile-time for Normal/Batched (the whole grid)
        and a runtime PrimExpr for the grouped variants (the current group).
        Runtime conditions use ``T.if_then_else``; compile-time toggles use ``if``.
        """
        group_m = self._group_m
        group_n = self._group_n
        num_n = self._num_n_blocks
        n_groups = num_n // group_n
        blocks_per_superrow = group_m * num_n
        full_blocks = n_groups * 32

        # small-M row-major
        sm_m = block_idx // num_n
        sm_n = block_idx % num_n

        # tail boundaries (compile-time for int local, runtime otherwise)
        if isinstance(local_num_m_blocks, int):
            small_m = local_num_m_blocks < group_m
            tail_m_start = local_num_m_blocks - local_num_m_blocks % group_m if not small_m else 0
        else:
            small_m = None
            tail_m_start = local_num_m_blocks - local_num_m_blocks % group_m
        tail_start = tail_m_start * num_n

        # tail row-major
        tail = block_idx - tail_start
        tl_m = tail_m_start + tail // num_n
        tl_n = tail % num_n

        # aligned swizzle region
        superrow = block_idx // blocks_per_superrow
        within = block_idx - superrow * blocks_per_superrow
        # full panel
        grp = within // 32
        local = within - grp * 32
        m_local = local % group_m
        n_local = local // group_m
        if self._xor_n:
            n_local = n_local ^ m_local
        ng = grp
        if self._snake:
            ng = T.if_then_else(superrow % 2 == 1, n_groups - 1 - grp, grp)
        full_m = superrow * group_m + m_local
        full_n = ng * group_n + n_local
        # remainder columns (n not covered by full 32-blocks)
        rem = within - full_blocks
        rem_m = superrow * group_m + rem % group_m
        rem_n = n_groups * group_n + rem // group_m
        sw_m = T.if_then_else(within < full_blocks, full_m, rem_m)
        sw_n = T.if_then_else(within < full_blocks, full_n, rem_n)

        # tail-vs-swizzle
        al_m = T.if_then_else(block_idx >= tail_start, tl_m, sw_m)
        al_n = T.if_then_else(block_idx >= tail_start, tl_n, sw_n)

        if small_m is True:
            return sm_m, sm_n
        if small_m is False:
            return al_m, al_n
        # runtime local_num_m_blocks (grouped)
        m = T.if_then_else(local_num_m_blocks < group_m, sm_m, al_m)
        n = T.if_then_else(local_num_m_blocks < group_m, sm_n, al_n)
        return m, n

    def coord(self, block_idx):
        """Decode a linear ``block_idx`` into ``(m_block, n_block)`` (Normal grid)."""
        return self._swizzle(block_idx, self._num_m_blocks)

    def update_current_idx(self, linear_idx):
        m, n = self.coord(linear_idx)
        self._m_idx[0] = m
        self._n_idx[0] = n

    # ---- stateful walk -----------------------------------------------------

    def valid(self):
        return self.valid_flag[0] != 0

    def init(self, core_id):
        self._current_iter[0] = -1
        self.core_id[0] = core_id
        self._init_state()
        self.next_block()

    def next_block(self):
        self._step()

    def _advance_block_idx(self):
        """Bump the persistent iteration and return the new global block index."""
        self._current_iter[0] = self._current_iter[0] + 1
        self._linear_idx[0] = self._current_iter[0] * self._num_cores + self.core_id[0]
        return self._linear_idx[0]

    def get_actual_m(self, m_block_idx):
        return T.min(self._shape_m - m_block_idx * self._block_m, self._block_m)

    def get_actual_n(self, n_block_idx):
        return T.min(self._shape_n - n_block_idx * self._block_n, self._block_n)


@meta_class
class AscendTileScheduler(AscendBaseTileScheduler):
    """Single flat M x N grid, GROUP_M swizzle + N-first snake; the last
    ``< group_m`` M-rows fall back to row-major. This is the plain (non-grouped,
    non-batched) GEMM scheduler."""

    def _step(self):
        block_idx = self._advance_block_idx()
        m, n = self._swizzle(block_idx, self._num_m_blocks)
        self._m_idx[0] = m
        self._n_idx[0] = n
        self.valid_flag[0] = T.if_then_else(block_idx < self._num_blocks, 1, 0)


@meta_class
class AscendBatchedTileScheduler(AscendBaseTileScheduler):
    """Standard batched matmul: the Normal grid replicated ``num_groups`` (= batch)
    times. ``get_batch_idx()`` exposes the batch of the current block.

    Extra parameter ``num_groups`` (= batch count) beyond the base scheduler.
    """

    def __init__(
        self,
        *,
        block_m,
        block_n,
        num_cores,
        shape_m,
        shape_n,
        num_groups: int,
        group_m: int = 4,
        xor_n: bool = True,
        snake: bool = True,
        stateful: bool = True,
        name: str | None = None,
    ):
        self._num_groups = num_groups
        super().__init__(
            block_m=block_m,
            block_n=block_n,
            num_cores=num_cores,
            shape_m=shape_m,
            shape_n=shape_n,
            group_m=group_m,
            xor_n=xor_n,
            snake=snake,
            stateful=stateful,
            name=name,
        )

    def _alloc_state(self):
        self.batch_idx = T.alloc_var(T.int32)

    def _step(self):
        block_idx = self._advance_block_idx()
        batch = block_idx // self._num_blocks
        self.batch_idx[0] = batch
        in_batch = block_idx - batch * self._num_blocks
        m, n = self._swizzle(in_batch, self._num_m_blocks)
        self._m_idx[0] = m
        self._n_idx[0] = n
        self.valid_flag[0] = T.if_then_else(batch < self._num_groups, 1, 0)

    def get_batch_idx(self):
        return self.batch_idx[0]


@meta_class
class AscendMGroupedTileScheduler(AscendBaseTileScheduler):
    """MoE m-grouped "contiguous" layout. ``grouped_layout`` is the prefix-sum-of-rows
    array (length ``num_groups``); group ``g``'s valid rows are
    ``[align(psum[g-1], alignment), psum[g])`` in the global M space (group 0 starts at
    0). ``get_group_idx()`` exposes the current group for per-group B/SF offsets.

    Extra parameters beyond the base: ``grouped_layout`` (GM ``int32`` prefix-sum
    buffer), ``num_groups``, ``alignment`` (group-start row alignment, default 256).
    """

    def __init__(
        self,
        *,
        block_m,
        block_n,
        num_cores,
        shape_m,
        shape_n,
        grouped_layout,
        num_groups: int,
        alignment: int = 256,
        group_m: int = 4,
        xor_n: bool = True,
        snake: bool = True,
        stateful: bool = True,
        name: str | None = None,
    ):
        self._grouped_layout = grouped_layout
        self._num_groups = num_groups
        self._alignment = alignment
        super().__init__(
            block_m=block_m,
            block_n=block_n,
            num_cores=num_cores,
            shape_m=shape_m,
            shape_n=shape_n,
            group_m=group_m,
            xor_n=xor_n,
            snake=snake,
            stateful=stateful,
            name=name,
        )

    def _alloc_state(self):
        self.group_idx = T.alloc_var(T.int32)
        self.cur_num_m_blocks = T.alloc_var(T.int32)
        self.last_psum_m = T.alloc_var(T.int32)
        self.cur_psum_m = T.alloc_var(T.int32)
        self.m_block_cumsum = T.alloc_var(T.int32)

    def _init_state(self):
        self.group_idx[0] = 0
        self.last_psum_m[0] = 0
        self.m_block_cumsum[0] = 0
        self.cur_psum_m[0] = self._grouped_layout[0]
        self.cur_num_m_blocks[0] = T.ceildiv(self.cur_psum_m[0], self._block_m)

    def _step(self):
        block_idx = self._advance_block_idx()
        num_n = self._num_n_blocks
        self.valid_flag[0] = 1
        # advance groups until block_idx lands in the current group's block range
        while self.group_idx[0] < self._num_groups:
            if block_idx < (self.m_block_cumsum[0] + self.cur_num_m_blocks[0]) * num_n:
                T.loop_break()
            self.group_idx[0] = self.group_idx[0] + 1
            if self.group_idx[0] == self._num_groups:
                self.valid_flag[0] = 0
                T.loop_break()
            self.last_psum_m[0] = T.ceildiv(self.cur_psum_m[0], self._alignment) * self._alignment
            self.cur_psum_m[0] = self._grouped_layout[self.group_idx[0]]
            self.m_block_cumsum[0] = self.m_block_cumsum[0] + self.cur_num_m_blocks[0]
            self.cur_num_m_blocks[0] = T.ceildiv(self.cur_psum_m[0] - self.last_psum_m[0], self._block_m)
        in_group = block_idx - self.m_block_cumsum[0] * num_n
        m, n = self._swizzle(in_group, self.cur_num_m_blocks[0])
        self._m_idx[0] = m + self.last_psum_m[0] // self._block_m
        self._n_idx[0] = n

    def get_group_idx(self):
        return self.group_idx[0]

    # m-grouped truncates the group's last m-block via the caller's own tail
    # handling; every scheduled m-block is a full BLOCK_M.
    def get_actual_m(self, m_block_idx):
        return self._block_m


@meta_class
class AscendKGroupedTileScheduler(AscendBaseTileScheduler):
    """MoE k-grouped. All groups share one M x N grid but differ in K length;
    ``grouped_layout[i]`` is the cumulative end-K of group ``i``. ``get_group_idx()``,
    ``get_k_idx_base()`` / ``get_sf_idx_base()`` give the group and its A/B and SF K
    offsets; ``get_shape_k()`` is the current group's K length.

    Extra parameters beyond the base: ``grouped_layout`` (GM ``int32`` prefix-sum
    buffer), ``num_groups``, ``alignment`` (group-start K alignment, default 256).
    """

    # One scale-factor pair covers this many K elements (MX FP hardware constant).
    _MX_SF_DIVISOR = 64

    def __init__(
        self,
        *,
        block_m,
        block_n,
        num_cores,
        shape_m,
        shape_n,
        grouped_layout,
        num_groups: int,
        alignment: int = 256,
        group_m: int = 4,
        xor_n: bool = True,
        snake: bool = True,
        stateful: bool = True,
        name: str | None = None,
    ):
        self._grouped_layout = grouped_layout
        self._num_groups = num_groups
        self._alignment = alignment
        super().__init__(
            block_m=block_m,
            block_n=block_n,
            num_cores=num_cores,
            shape_m=shape_m,
            shape_n=shape_n,
            group_m=group_m,
            xor_n=xor_n,
            snake=snake,
            stateful=stateful,
            name=name,
        )

    def _alloc_state(self):
        self.group_idx = T.alloc_var(T.int32)
        self.cur_shape_k = T.alloc_var(T.int32)
        self.last_psum_k = T.alloc_var(T.int32)
        self.cur_psum_k = T.alloc_var(T.int32)
        self.sf_k_cumsum = T.alloc_var(T.int32)
        self.k_block_cumsum = T.alloc_var(T.int32)

    def _init_state(self):
        self.group_idx[0] = 0
        self.last_psum_k[0] = 0
        self.sf_k_cumsum[0] = 0
        self.k_block_cumsum[0] = 0
        self.cur_psum_k[0] = self._grouped_layout[0]
        self.cur_shape_k[0] = self.cur_psum_k[0]

    def _step(self):
        block_idx = self._advance_block_idx()
        self.valid_flag[0] = 1
        # all groups share the M x N grid; each spans exactly num_blocks
        while self.group_idx[0] < self._num_groups:
            if self.cur_shape_k[0] > 0 and block_idx < self.k_block_cumsum[0] + self._num_blocks:
                T.loop_break()
            self.group_idx[0] = self.group_idx[0] + 1
            if self.group_idx[0] == self._num_groups:
                self.valid_flag[0] = 0
                T.loop_break()
            self.last_psum_k[0] = T.ceildiv(self.cur_psum_k[0], self._alignment) * self._alignment
            self.cur_psum_k[0] = self._grouped_layout[self.group_idx[0]]
            self.sf_k_cumsum[0] = self.last_psum_k[0] // self._MX_SF_DIVISOR
            if self.cur_shape_k[0] > 0:
                self.k_block_cumsum[0] = self.k_block_cumsum[0] + self._num_blocks
            self.cur_shape_k[0] = self.cur_psum_k[0] - self.last_psum_k[0]
        in_group = block_idx - self.k_block_cumsum[0]
        m, n = self._swizzle(in_group, self._num_m_blocks)
        self._m_idx[0] = m
        self._n_idx[0] = n

    def get_group_idx(self):
        return self.group_idx[0]

    def get_k_idx_base(self):
        return self.last_psum_k[0]

    def get_sf_idx_base(self):
        return self.sf_k_cumsum[0]

    def get_shape_k(self):
        return self.cur_shape_k[0]
