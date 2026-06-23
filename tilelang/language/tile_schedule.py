"""Tile schedulers built on TileLang ``meta_class``.

``@meta_class`` auto-inlines every method that emits a buffer store, so state
methods can freely read scalar state via ``self.x[0]`` and write via
``self.x[0] = ...`` (the latter lowers to a ``BufferStore``). Store-free methods
(``valid``, ``coord``) stay plain Python and just return ``PrimExpr`` values, so
they work in conditions and can be reused statelessly. State lives in
``T.alloc_var`` buffers allocated in ``__init__`` (only when ``stateful``). Works
in both eager (``@tilelang.jit``) and lazy (``@T.prim_func``) modes.

Note on compile-time branching: the lazy TVMScript parser does not constant-fold
a Python ``if`` on a compile-time value inside an inlined method (it would emit a
dead TIR branch). So all compile-time decisions (traversal order, clustering)
are made in ``__init__`` (plain Python); the inlined hot methods contain only
uniform runtime arithmetic. The store-free ``coord`` may use a compile-time
``if`` because, being plain Python, it is evaluated (not traced) per call.
"""

from __future__ import annotations

import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.language.meta import meta_class


@meta_class
class BaseTileScheduler:
    """Common state and tile-traversal skeleton.

    State (single-element ``T.alloc_var`` buffers): ``m_idx`` / ``n_idx`` are the
    current tile coordinates; ``linear_idx`` is the worker's global linear cursor;
    ``current_iter`` is the 0-based iteration count (how many tiles this worker has
    advanced past = the "wave" index). ``current_iter`` is the single iteration
    clock the kernel can read for pipeline/double-buffer state (e.g.
    ``sched.current_iter[0] & 1``), removing the need for a separate ``for w`` loop
    counter. Subclasses implement ``update_current_idx`` to decode ``linear_idx``
    into ``(m_idx, n_idx)`` and set ``self._total_tiles`` (used by ``valid``).
    """

    def __init__(self, prefix: str, stateful: bool = True):
        self._stateful = stateful
        if stateful:
            self.m_idx = T.alloc_var(T.int32)
            self.n_idx = T.alloc_var(T.int32)
            self.linear_idx = T.alloc_var(T.int32)
            self.current_iter = T.alloc_var(T.int32)
        self._total_tiles = 0

    def update_current_idx(self, linear_idx):
        raise NotImplementedError("Subclasses must implement update_current_idx")

    def init(self, linear_init):
        self.current_iter[0] = 0
        self.linear_idx[0] = linear_init
        self.update_current_idx(self.linear_idx[0])

    def next_tile(self, step):
        self.current_iter[0] = self.current_iter[0] + 1
        self.linear_idx[0] = self.linear_idx[0] + step
        self.update_current_idx(self.linear_idx[0])

    def valid(self):
        return self.linear_idx[0] < self._total_tiles


@meta_class
class PersistentTileScheduler(BaseTileScheduler):
    """Persistent, grid-strided tile scheduler with L2 swizzle and clustering.

    The 2D tile grid (``num_m_tiles`` x ``num_n_tiles``) is flattened into a
    linear order and distributed across ``num_workers`` persistent workers:
    worker ``w`` processes linear indices ``w, w + num_workers, ...`` until they
    run past the total. The linear-to-2D decode is controlled by three
    independent, interacting factors:

    1. ``column_major`` -- traversal order. With ``swizzle_size == 1`` this is
       pure column-major (``m`` varies fastest) when ``True``, or pure row-major
       (``n`` varies fastest) when ``False``.
    2. ``swizzle_size`` -- L2-locality panel width. The "fast" axis is widened
       into panels of ``swizzle_size`` tiles along the "slow" axis: a full strip
       of the fast axis is swept for each group of ``swizzle_size`` slow-axis
       tiles before advancing. ``swizzle_size == 1`` disables swizzling. This is
       the CUTLASS-style threadblock swizzle used by the SM100 persistent GEMM
       examples. Non-divisible tail panels are handled (narrower last panel).
    3. ``cluster_size`` -- block clustering along M (x) only. The M tiles are
       grouped into clusters of ``cluster_size`` rows, and the scheduler runs at
       *cluster* granularity: it produces the cluster-row in ``m_idx`` over a
       grid of ``ceildiv(num_m_tiles, cluster_size)`` cluster-rows x
       ``num_n_tiles`` columns. The caller turns the cluster-row into a real
       block row by adding the in-cluster rank::

           bx = sched.m_idx[0] * cluster_size + cta_rank_in_cluster

       (For ``cluster_size == 1`` this reduces to ``bx = sched.m_idx[0]``.)

    Interaction: clustering reshapes the grid to ``M' x N'`` (with
    ``M' = ceildiv(num_m_tiles, cluster_size)``) *first*; the swizzle and
    traversal order then operate on that reshaped grid. ``num_workers`` is the
    number of resident workers (= clusters when ``cluster_size > 1``).

    Parameters
    ----------
    prefix : str
        Name prefix for the scheduler's state buffers in the generated IR
        (``{prefix}_m_idx`` / ``{prefix}_n_idx`` / ``{prefix}_linear_idx``).
    num_m_tiles : int | PrimExpr
        Number of tiles along M (``ceildiv(M, block_M)``).
    num_n_tiles : int | PrimExpr
        Number of tiles along N (``ceildiv(N, block_N)``).
    num_workers : int | PrimExpr, optional
        Persistent stride = number of resident workers/clusters. Defaults to
        ``driver.get_num_sms() // cluster_size`` (one block per SM, grouped into
        clusters).
    swizzle_size : int
        L2 swizzle panel width (default ``1`` = no swizzle).
    column_major : bool
        Traversal order (default ``True`` = column-major / M fastest).
    cluster_size : int
        Block-cluster size along M only (default ``1`` = no clustering).
    stateful : bool
        If ``True`` (default), allocate the ``m_idx`` / ``n_idx`` / ``linear_idx``
        / ``current_iter`` state buffers and enable ``init`` / ``next_tile`` /
        ``valid`` for a ``while`` persistent loop. If ``False``, allocate no state
        and expose only the pure ``coord(tile_id) -> (m, n)`` decode (for
        ``for w in range(waves)`` loops and auto warp-specialization, where the
        loop owns the iteration clock and the WS pass owns the pipeline phase).

    Examples
    --------
    Plain persistent GEMM (one block per SM)::

        m_blocks, n_blocks = T.ceildiv(M, block_M), T.ceildiv(N, block_N)
        with T.Kernel(driver.get_num_sms(), threads=threads) as (block_id,):
            sched = T.PersistentTileScheduler("sched", m_blocks, n_blocks)
            sched.init(block_id)
            while sched.valid():
                bx, by = sched.m_idx[0], sched.n_idx[0]
                # ... compute tile (bx, by) ...
                sched.next_tile()

    With L2 swizzle (panel width 8)::

        sched = T.PersistentTileScheduler(
            "sched", m_blocks, n_blocks, swizzle_size=8)

    With a 2-CTA cluster along M (e.g. SM100 2-SM MMA)::

        sm_num = driver.get_num_sms()
        cluster_size = 2
        with T.ClusterKernel(sm_num, threads=256, cluster_dims=cluster_size) as (block_id):
            cta_rank = T.block_rank_in_cluster()
            sched = T.PersistentTileScheduler(
                "sched", m_blocks, n_blocks,
                swizzle_size=8, cluster_size=cluster_size)
            sched.init(block_id // cluster_size)   # init with cluster id
            while sched.valid():
                bx = sched.m_idx[0] * cluster_size + cta_rank
                by = sched.n_idx[0]
                # ... compute tile (bx, by) ...
                sched.next_tile()

    Stateless form (``stateful=False``) for ``for w`` loops / auto-WS, where the
    scheduler is only a tile-coordinate decoder::

        sched = T.PersistentTileScheduler(
            "sched", m_blocks, n_blocks, swizzle_size=8, stateful=False)
        for w in range(waves):
            bx, by = sched.coord(num_workers * w + worker_id)
            if bx * block_M < M and by * block_N < N:
                # ... pipelined inner loop (T.Pipelined / T.copy / T.gemm) ...
                pass

    Manual warp-specialized kernels use ``current_iter`` as the single iteration clock
    (no separate ``for w`` loop): each warp role runs its own
    ``while sched.valid()`` loop and reads ``w = sched.current_iter[0]`` for
    pipeline/double-buffer state (``w & 1`` etc.) while reading ``sched.m_idx[0]``
    / ``sched.n_idx[0]`` for the tile::

        sched.init(block_id)
        while sched.valid():
            w = sched.current_iter[0]
            bx, by = sched.m_idx[0], sched.n_idx[0]
            # ... use w for barrier phase / w & 1 double-buffering ...
            sched.next_tile()

    Notes
    -----
    State is held in single-element ``T.alloc_var`` buffers, so read with
    ``sched.m_idx[0]`` / ``sched.current_iter[0]`` and (inside methods) write with
    ``self.x[0] = ...``; the ``[0]`` is required for the write to lower to a
    ``BufferStore``."""

    def __init__(
        self,
        prefix: str,
        num_m_tiles,
        num_n_tiles,
        num_workers=None,
        swizzle_size: int = 1,
        column_major: bool = True,
        cluster_size: int = 1,
        stateful: bool = True,
    ):
        super().__init__(prefix, stateful=stateful)
        self.cluster_size = cluster_size
        if num_workers is None:
            num_workers = driver.get_num_sms() // cluster_size
        self.num_workers = num_workers
        self.swizzle_size = swizzle_size
        self._column_major = column_major

        # Cluster reshapes the grid along M first: M' x N'.
        m_clusters = T.ceildiv(num_m_tiles, cluster_size)
        self._total_tiles = m_clusters * num_n_tiles

        # "primary" axis is swept fully within a swizzle panel; "secondary" axis
        # is chunked into panels of width ``swizzle_size``.
        if column_major:  # M fastest
            self._primary = m_clusters
            self._secondary = num_n_tiles
        else:  # N fastest
            self._primary = num_n_tiles
            self._secondary = m_clusters

    def coord(self, tile_id):
        """Decode a linear ``tile_id`` into ``(m_idx, n_idx)`` (pure PrimExpr).

        Stateless: builds expressions only, touches no state buffer, so it is
        usable on a ``stateful=False`` instance and can be called independently
        from any loop / warp role. ``m_idx`` is the cluster row when
        ``cluster_size > 1`` (caller adds the in-cluster rank).
        """
        primary = self._primary
        secondary = self._secondary
        swizzle = self.swizzle_size

        group = tile_id // (primary * swizzle)
        base = group * swizzle
        # Panel width along the slow axis; narrower for a non-divisible tail, and
        # clamped to >= 1 so decoding an out-of-range tile_id never divides by 0.
        width = T.max(1, T.min(swizzle, secondary - base))
        in_group = tile_id - group * (primary * swizzle)
        fast = in_group // width
        slow = base + in_group % width
        if self._column_major:
            return fast, slow  # (m, n)
        return slow, fast  # (m, n)

    def update_current_idx(self, linear_idx):
        m, n = self.coord(linear_idx)
        self.m_idx[0] = m
        self.n_idx[0] = n

    def init(self, worker_id):
        self.current_iter[0] = 0
        self.linear_idx[0] = worker_id
        self.update_current_idx(self.linear_idx[0])

    def next_tile(self):
        self.current_iter[0] = self.current_iter[0] + 1
        self.linear_idx[0] = self.linear_idx[0] + self.num_workers
        self.update_current_idx(self.linear_idx[0])
