from __future__ import annotations
import tilelang.language as T
from dataclasses import dataclass
from collections.abc import Callable
from .mma_macro_generator import TensorCoreIntrinEmitter as MMAIntrinEmitter
from tvm import DataType
from tvm.tirx import PrimExpr, Buffer, Var, IndexMap, BufferRegion
from tilelang import tvm as tvm
from tilelang.utils import is_fragment, is_full_region
from math import gcd
from tilelang.layout import (
    Layout,
    SwizzleMode,
    cute,
)
from tvm.runtime import convert
from tilelang.cuda.intrinsics.layout.mma_layout import (
    shared_16x8_to_mma_32x4_layout_sr_a,
    shared_16x16_to_mma_32x8_layout_sr_a,
    shared_16x32_to_mma_32x16_layout_sr_a,
)

lift = convert


def _min_leaf_stride(stride) -> int:
    """Smallest leaf stride within a (possibly nested) stride tuple."""
    if isinstance(stride, tuple):
        return min(_min_leaf_stride(s) for s in stride)
    return int(stride)


@dataclass(frozen=True)
class WGMMADescriptorParams:
    """Pre-computed WGMMA descriptor parameters, produced by
    :func:`compute_gmma_descriptor` (a port of CuTe ``make_gmma_desc``) and
    consumed by ``init_wgmma_*_desc()`` and ``wgmma_*_atom()``.
    """

    swizzle_mode: SwizzleMode
    """Canonical swizzle mode; project to the descriptor field via ``wgmma_layout_type()``."""
    leading_byte_offset: int
    """LBO >> 4, ready to pass to ``T.initialize_wgmma_descriptor``."""
    stride_byte_offset: int
    """SBO >> 4, ready to pass to ``T.initialize_wgmma_descriptor``."""
    swizzle_atom_elems: int
    """Number of elements per swizzle atom along the non-K dimension."""
    k_atom_size: int
    """``max(swizzle_atom_elems // micro_size_k, 1)``."""
    elems_in_bytes: int
    """Byte width of a single element: ``DataType(dtype).bits // 8``."""
    is_k_major: bool
    """Whether the matrix is stored in K-major order (affects offset formula branching)."""
    slice_byte_offset: object = 0
    """Physical byte offset (raw bytes) of the operand slice origin within its
    buffer; passed to ``T.increase_descriptor_offset`` after building the
    descriptor from the buffer base. ``0`` for a whole-buffer / base-origin
    operand. Computed by :func:`compute_gmma_descriptor` from the CuTe layout."""


def compute_gmma_descriptor(tl_layout, buffer, transposed: bool, micro_size_k: int = 16, region=None) -> WGMMADescriptorParams:
    """Port of CuTe ``make_gmma_desc``.

    Decode an arbitrary shared-memory ``tl_layout`` for ``buffer`` and compute the
    WGMMA descriptor parameters, accepting *any* WGMMA-canonical layout -- not
    just the four "maker" layouts. A non-GMMA-canonical layout is a programming
    error, so the canonicity checks assert (mirroring CuTe's ``static_assert``s).

    ``transposed`` selects which logical axis is MN vs K (shape only): default
    ``[MN, K]``, transposed ``[K, MN]``. The operand is required to be row-major,
    i.e. K-major iff ``not transposed``; the contiguity detected from the layout
    (the GMMA mode owning the stride-1 sub-mode) is asserted to agree.

    ``region`` is the operand's per-axis ranges (one :class:`tvm.ir.Range` per
    logical buffer mode), used to restrict the decoded layout to a sliced
    operand (e.g. ``B[:, j*64:...]``). It may be ``None`` for a full-buffer
    operand or the atom-level API.
    """
    elems_in_bytes = int(DataType(buffer.dtype).bits) // 8
    bits = int(DataType(buffer.dtype).bits)
    # One decode in element space; the byte- and u128-address variants are pure
    # recasts of it (no second from_tilelang).
    composed_elem = cute.ComposedLayout.from_tilelang(tl_layout)
    assert composed_elem is not None, f"WGMMA operand layout is not decodable by the CuTe analyzer: {tl_layout}"
    # Swizzle is read in BYTE space (canonical atom Sw<b,4,3>, m_base==4).
    byte_swizzle = composed_elem.recast(bits, 8).swizzle
    swizzle_mode = byte_swizzle.to_swizzle_mode()

    # Restrict the (possibly hierarchical, swizzle-split) element-space layout to
    # the operand's slice. ``restrict`` reshapes each mode to its logical extent
    # via with_shape -- collapsing the split sub-modes back to the logical tile,
    # and skipping extent-1 modes (e.g. a software-pipeline stage the region pins
    # to one element) so the result is the bare (MN, K) operand. It also returns
    # the slice origin's physical element offset, which we scale to raw bytes for
    # increase_descriptor_offset: the descriptor is built from the buffer base, so
    # this advance lands it on the slice origin while keeping the cvta operand
    # (the base) loop-invariant => warp-uniform. With no region (atom API) the
    # layout is already the bare operand.
    tile = composed_elem.layout
    slice_byte_offset = 0
    if region is not None:
        slice_off_elems, tile = cute.restrict(composed_elem.layout, region)
        # A statically-zero origin is a plain int 0; a runtime origin is a PrimExpr.
        if slice_off_elems != 0:
            slice_byte_offset = tvm.arith.Analyzer().simplify(slice_off_elems * bits // 8)
    assert cute.rank(tile) == 2, f"WGMMA operand tile must be rank-2 (MN, K), got rank {cute.rank(tile)}"

    # Present in GMMA (MN, K) order, then recast the swizzled element layout to
    # uint128_t (exactly CuTe's recast<uint128_t const>(tensor)).
    mn_idx = 1 if transposed else 0
    k_idx = 1 - mn_idx
    mn_dim = int(cute.size(tile[mn_idx]))
    mnk = cute.make_layout([tile[mn_idx], tile[k_idx]])
    u128 = cute.ComposedLayout(composed_elem.swizzle, composed_elem.offset, mnk).recast(bits, 128)
    mn_mode, k_mode = u128.layout[0], u128.layout[1]

    # Row-major only: K-major iff not transposed. Assert the contiguity detected
    # from the layout (K mode owns the stride-1 sub-mode) agrees with that.
    k_major = not transposed
    detected_k_major = _min_leaf_stride(k_mode.stride) == 1
    assert detected_k_major == k_major, (
        f"WGMMA operand layout contiguity (k_major={detected_k_major}) disagrees with "
        f"the row-major expectation (k_major={k_major} for transposed={transposed}); "
        f"only row-major operand layouts are supported."
    )

    # W per CuTe LayoutType (INTERLEAVE->1, B32->2, B64->4, B128->8) = 1 << b_bits.
    W = 1 << byte_swizzle.b_bits
    swizzled = not swizzle_mode.is_none()

    # CuTe make_gmma_desc logical_divides each u128 (MN, K) mode by the canonical
    # tiler:
    #   MN-major: ((W,m),(8,k)):((1,LBO),(W,SBO))   [INTERLEAVE: ((1,m),(8,k)):((X,SBO),(1,LBO))]
    #   K-major : ((8,m),(2,k)):((8,SBO),(1,2))
    # Each divided mode is (tile, rest) and its strides read directly as scalars.
    if k_major:
        d_mn = cute.logical_divide(mn_mode, cute.make_layout(8, 1))
        d_k = cute.logical_divide(k_mode, cute.make_layout(2, 1))
    else:
        d_mn = cute.logical_divide(mn_mode, cute.make_layout(W, 1))
        d_k = cute.logical_divide(k_mode, cute.make_layout(8, 1))
    # The MN mode is always a clean (tile, rest) scalar pair. So is the K mode's
    # tile (stride<1,0>); only the K *rest* (stride<1,1>) may stay multi-atom for
    # a whole operand (CuTe's tensor is one K atom), and K-major never reads it.
    assert cute.congruent(d_mn.shape, (1, 1)) and cute.congruent(d_k[0].shape, 1), (
        f"WGMMA operand is not a canonical GMMA layout: divided MN={d_mn.shape}, K-tile={d_k[0].shape}"
    )
    s00, s01 = d_mn.stride
    s10 = d_k.stride[0]

    if k_major:
        # Canonical ((8,m),(2,k)):((8,SBO),(1,2)). stride<0,0>==W; stride<1,0> is
        # the INTERLEAVE pass-through or 1 when swizzled. SBO=stride<0,1>, LBO=1.
        assert s00 == W, f"Not a canonical GMMA_K layout: stride<0,0>={s00} != W={W}"
        assert not (swizzled and s10 != 1), f"Not a canonical GMMA_K layout: stride<1,0>={s10} != 1"
        sbo = s01
        lbo = s10
    else:
        # Canonical ((W,m),(8,k)). stride<1,0>==W, and stride<0,0>==1 when swizzled
        # (INTERLEAVE passes through). Rejects layouts CuTe itself rejects (e.g.
        # tilelang's K-oriented maker used as an MN operand).
        assert cute.congruent(d_k.shape, (1, 1)), f"WGMMA MN-major operand is not a canonical GMMA layout: divided K={d_k.shape}"
        s11 = d_k.stride[1]
        assert not (swizzled and s00 != 1), f"Not a canonical GMMA_MN layout: stride<0,0>={s00} != 1"
        assert s10 == W, f"Not a canonical GMMA_MN layout: stride<1,0>={s10} != W={W}"
        sbo = s11 if swizzled else s01
        lbo = s01 if swizzled else s11

    # Elements per swizzle atom along the non-K (MN) dimension; the unswizzled
    # case spans the whole MN tile.
    swizzle_atom_elems = mn_dim if swizzle_mode.is_none() else swizzle_mode.swizzle_byte_size() // elems_in_bytes
    return WGMMADescriptorParams(
        swizzle_mode=swizzle_mode,
        leading_byte_offset=int(lbo),
        stride_byte_offset=int(sbo),
        swizzle_atom_elems=swizzle_atom_elems,
        k_atom_size=max(swizzle_atom_elems // micro_size_k, 1),
        elems_in_bytes=elems_in_bytes,
        is_k_major=k_major,
        slice_byte_offset=slice_byte_offset,
    )


# derive from MMAIntrinEmitter as some layouts are the same
class TensorCoreIntrinEmitter(MMAIntrinEmitter):
    """
    To eliminate Python syntax within TIR Macro.
    """

    # should be rewritten to support dynamic k_dim
    wgmma_prefix: str

    # wgmma instruction M dimension
    wgmma_inst_m: int
    # wgmma instruction N dimension
    wgmma_inst_n: int

    a_shared_layout: Layout = None
    b_shared_layout: Layout = None

    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool | None = False,
        thread_var: Var | None = None,
    ):
        super().__init__(
            a_dtype,
            b_dtype,
            accum_dtype,
            a_transposed,
            b_transposed,
            block_row_warps,
            block_col_warps,
            warp_row_tiles,
            warp_col_tiles,
            chunk,
            reduce_k,
            num_elems_per_byte,
            is_m_first,
            thread_var,
        )
        self._initialize_wgmma_prefix(self.n_dim)

    def _assign_a_shared_layout(self, layout: Layout):
        self.a_shared_layout = layout
        return self

    def _assign_b_shared_layout(self, layout: Layout):
        self.b_shared_layout = layout
        return self

    def _initialize_wgmma_prefix(self, n_dim: int = 16):
        inst_m, inst_n = 64, gcd(self.warp_col_tiles, 256)
        assert inst_n % 8 == 0, (
            f"inst_n must be a multiple of 8, got {inst_n} (block_col_warps={self.block_col_warps}, warp_col_tiles={self.warp_col_tiles})"
        )
        # Validate inst_n: Hopper WGMMA supports n in [8, 256] and multiple of 8
        assert 8 <= inst_n <= 256, (
            f"inst_n must be within [8, 256], got {inst_n} (block_col_warps={self.block_col_warps}, warp_col_tiles={self.warp_col_tiles})"
        )
        # 256 bits per instruction
        inst_k = 256 // DataType(self.a_dtype).bits
        self.wgmma_inst_m = inst_m
        self.wgmma_inst_n = inst_n
        self.wgmma_prefix = f"m{inst_m}n{inst_n}k{inst_k}"

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        assert warp_row_tiles >= 16, f"warp_row_tiles must be greater than 16, got {warp_row_tiles}"
        assert warp_row_tiles % 16 == 0, f"warp_row_tiles must be divisible by 16, got {warp_row_tiles}"
        assert warp_col_tiles >= 8, f"warp_col_tiles must be greater than 8, got {warp_col_tiles}"
        assert warp_col_tiles % 8 == 0, f"warp_col_tiles must be divisible by 8, got {warp_col_tiles}"

        # four warps per block
        self.warp_rows = warp_row_tiles // m_dim
        if warp_col_tiles % 16 == 0:
            self.n_dim = 16
            self.micro_size_y = 16
            self.warp_cols = warp_col_tiles // 16
        else:
            # must be divisible by 8
            self.n_dim = 8
            self.micro_size_y = 8
            self.warp_cols = warp_col_tiles // 8

        self.micro_size_x = m_dim
        self.micro_size_k = k_dim

    def wgmma(
        self, A_region: BufferRegion, B_region: BufferRegion, C_region: BufferRegion, clear_accum: PrimExpr = False, wg_wait: int = 0
    ):
        if is_fragment(A_region):
            return self.wgmma_rs(A_region, B_region, C_region, clear_accum, wg_wait)

        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        assert is_full_region(C_region), "Fragment output C must be a full region"
        C_buf = C_region.buffer

        num_inst_m = self.wgmma_num_inst_m
        num_inst_n = self.wgmma_num_inst_n
        num_k_atoms = self.wgmma_num_k_atoms
        a_params = self.compute_wgmma_a_desc_params(A_region)
        b_params = self.compute_wgmma_b_desc_params(B_region)

        @T.macro
        def _warp_mma(C_buf):
            desc_a = T.alloc_wgmma_desc()
            desc_b = T.alloc_wgmma_desc()
            self.init_wgmma_a_desc(desc_a, A_region, a_params)
            self.init_wgmma_b_desc(desc_b, B_region, b_params)
            self.wgmma_fence_c(C_buf)
            self.wgmma_arrive()

            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(num_k_atoms):
                        self.wgmma_ss_atom(desc_a, desc_b, C_buf, i, j, ki, a_params, b_params, clear_accum)

            self.wgmma_commit()
            if wg_wait >= 0:
                self.wgmma_wait(wg_wait)
            self.wgmma_fence_c(C_buf)

        return _warp_mma(C_buf)

    def wgmma_rs(
        self, A_region: BufferRegion, B_region: BufferRegion, C_region: BufferRegion, clear_accum: PrimExpr = False, wg_wait: int = 0
    ):
        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        assert is_full_region(A_region), "Fragment input A must be a full region"
        assert is_full_region(C_region), "Fragment output C must be a full region"
        A_buf = A_region.buffer
        C_buf = C_region.buffer

        num_inst_m = self.wgmma_num_inst_m
        num_inst_n = self.wgmma_num_inst_n
        num_k_atoms = self.wgmma_num_k_atoms
        b_params = self.compute_wgmma_b_desc_params(B_region)

        @T.macro
        def _warp_mma(A_buf, C_buf):
            desc_b = T.alloc_wgmma_desc()
            self.init_wgmma_b_desc(desc_b, B_region, b_params)
            self.wgmma_fence_a(A_buf)
            self.wgmma_fence_c(C_buf)
            self.wgmma_arrive()

            for j in T.unroll(0, num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, num_k_atoms):
                        self.wgmma_rs_atom(A_buf, desc_b, C_buf, i, j, ki, b_params, clear_accum)

            self.wgmma_commit()
            if wg_wait >= 0:
                self.wgmma_wait(wg_wait)
            self.wgmma_fence_c(C_buf)
            self.wgmma_fence_a(A_buf)

        return _warp_mma(A_buf, C_buf)

    # ---- Atom-level interface ----

    @property
    def wgmma_num_inst_m(self) -> int:
        """Number of WGMMA instruction atoms along the M dimension."""
        return 4 * self.warp_row_tiles // self.wgmma_inst_m

    @property
    def wgmma_num_inst_n(self) -> int:
        """Number of WGMMA instruction atoms along the N dimension."""
        return self.warp_col_tiles // self.wgmma_inst_n

    @property
    def wgmma_num_k_atoms(self) -> int:
        """Number of K-dimension micro-steps (``chunk // micro_size_k``)."""
        return self.chunk // self.micro_size_k

    @property
    def wgmma_a_regs(self) -> int:
        """Number of 32-bit registers occupied by the A fragment (RS variant)."""
        a_bits = DataType(self.a_dtype).bits
        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        return ((self.warp_rows * self.local_size_a * (k_dim // micro_size_k)) * a_bits + 31) // 32

    @property
    def wgmma_accum_regs(self) -> int:
        """Number of 32-bit registers occupied by the accumulator fragment."""
        m_dim = self.block_row_warps * self.warp_row_tiles
        accum_bits = DataType(self.accum_dtype).bits
        return ((m_dim // 64) * self.warp_cols * self.local_size_out * accum_bits + 31) // 32

    # -- Descriptor parameter computation (pure Python, no TIR) --

    def compute_wgmma_b_desc_params(self, B_region: BufferRegion) -> WGMMADescriptorParams:
        """Compute B descriptor parameters from the B shared buffer region.

        Pure-Python helper (no TIR emitted); the returned ``WGMMADescriptorParams``
        is consumed by ``init_wgmma_b_desc()`` and ``wgmma_*_atom()``.
        """
        assert self.b_shared_layout is not None, "WGMMA B operand has no shared layout to decode"
        return compute_gmma_descriptor(
            self.b_shared_layout,
            B_region.buffer if isinstance(B_region, BufferRegion) else B_region,
            transposed=not self.b_transposed,
            micro_size_k=self.micro_size_k,
            region=list(B_region.region) if isinstance(B_region, BufferRegion) else None,
        )

    def compute_wgmma_a_desc_params(self, A_region: BufferRegion) -> WGMMADescriptorParams:
        """Compute A descriptor parameters from the A shared buffer region (SS variant).

        Pure-Python helper (no TIR emitted); the returned ``WGMMADescriptorParams``
        is consumed by ``init_wgmma_a_desc()`` and ``wgmma_ss_atom()``.
        """
        assert self.a_shared_layout is not None, "WGMMA A operand has no shared layout to decode"
        return compute_gmma_descriptor(
            self.a_shared_layout,
            A_region.buffer if isinstance(A_region, BufferRegion) else A_region,
            transposed=self.a_transposed,
            micro_size_k=self.micro_size_k,
            region=list(A_region.region) if isinstance(A_region, BufferRegion) else None,
        )

    # -- Descriptor initialization (emit TIR) --

    def init_wgmma_b_desc(
        self,
        desc_b: Buffer,
        B_region: BufferRegion,
        b_params: WGMMADescriptorParams,
    ):
        """Emit TIR to initialize a pre-allocated WGMMA B descriptor.

        Parameters
        ----------
        desc_b : Buffer
            A descriptor buffer allocated via ``T.alloc_wgmma_desc()``.
        B_region : BufferRegion
            The B operand shared memory region.
        b_params : WGMMADescriptorParams
            Pre-computed parameters from ``compute_wgmma_b_desc_params()``.
        """
        B_buf = B_region.buffer if isinstance(B_region, BufferRegion) else B_region
        B_base_ptr = B_buf.access_ptr("r")
        slice_off_bytes = b_params.slice_byte_offset
        swizzle_mode = b_params.swizzle_mode.wgmma_layout_type()
        lbo = b_params.leading_byte_offset
        sbo = b_params.stride_byte_offset

        @T.macro
        def _init_b_desc(desc_b, B_base_ptr):
            # Build from the buffer base (loop-invariant => uniform cvta), then
            # advance start_address_ to the slice origin via the descriptor's
            # in-place add. Keeps the descriptor warp-uniform (no per-thread cvta
            # of a slice pointer carrying an induction variable).
            T.initialize_wgmma_descriptor(desc_b, B_base_ptr, swizzle_mode, lbo, sbo)
            if slice_off_bytes != 0:
                T.increase_descriptor_offset(desc_b, slice_off_bytes)

        return _init_b_desc(desc_b, B_base_ptr)

    def init_wgmma_a_desc(
        self,
        desc_a: Buffer,
        A_region: BufferRegion,
        a_params: WGMMADescriptorParams,
    ):
        """Emit TIR to initialize a pre-allocated WGMMA A descriptor (SS variant).

        Parameters
        ----------
        desc_a : Buffer
            A descriptor buffer allocated via ``T.alloc_wgmma_desc()``.
        A_region : BufferRegion
            The A operand shared memory region.
        a_params : WGMMADescriptorParams
            Pre-computed parameters from ``compute_wgmma_a_desc_params()``.
        """
        A_buf = A_region.buffer if isinstance(A_region, BufferRegion) else A_region
        A_base_ptr = A_buf.access_ptr("r")
        slice_off_bytes = a_params.slice_byte_offset
        swizzle_mode = a_params.swizzle_mode.wgmma_layout_type()
        lbo = a_params.leading_byte_offset
        sbo = a_params.stride_byte_offset

        @T.macro
        def _init_a_desc(desc_a, A_base_ptr):
            # Build from the buffer base (uniform cvta), then advance to the slice
            # origin (see init_wgmma_b_desc).
            T.initialize_wgmma_descriptor(desc_a, A_base_ptr, swizzle_mode, lbo, sbo)
            if slice_off_bytes != 0:
                T.increase_descriptor_offset(desc_a, slice_off_bytes)

        return _init_a_desc(desc_a, A_base_ptr)

    # -- Fence / Arrive / Commit / Wait primitives --

    def wgmma_fence_a(self, A_buf: Buffer):
        """Emit ``warpgroup_fence_operand`` for the A fragment buffer."""
        a_regs = self.wgmma_a_regs

        @T.macro
        def _fence_a(A_buf):
            T.warpgroup_fence_operand(A_buf, num_regs=a_regs)

        return _fence_a(A_buf)

    def wgmma_fence_c(self, C_buf: Buffer):
        """Emit ``warpgroup_fence_operand`` for the accumulator buffer."""
        accum_regs = self.wgmma_accum_regs

        @T.macro
        def _fence_c(C_buf):
            T.warpgroup_fence_operand(C_buf, num_regs=accum_regs)

        return _fence_c(C_buf)

    def wgmma_arrive(self):
        """Emit ``warpgroup_arrive()``."""

        @T.macro
        def _arrive():
            T.warpgroup_arrive()

        return _arrive()

    def wgmma_commit(self):
        """Emit ``warpgroup_commit_batch()``."""

        @T.macro
        def _commit():
            T.warpgroup_commit_batch()

        return _commit()

    def wgmma_wait(self, n: int = 0):
        """Emit ``warpgroup_wait(n)``."""

        @T.macro
        def _wait():
            T.warpgroup_wait(n)

        return _wait()

    # -- Atom emission --

    def wgmma_rs_atom(
        self,
        A_buf: Buffer,
        desc_b: Buffer,
        C_buf: Buffer,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        b_params: WGMMADescriptorParams,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single WGMMA RS instruction for atom ``(inst_m_idx, inst_n_idx, ki)``.

        Must be called between a ``wgmma_fence_a``/``wgmma_fence_c``/``wgmma_arrive``
        sequence and a ``wgmma_commit``/``wgmma_wait`` sequence.

        Calling this for every ``(j, i, ki)`` in
        ``T.grid(wgmma_num_inst_n, wgmma_num_inst_m, wgmma_num_k_atoms)``
        produces identical TIR to ``wgmma_rs()``.

        Parameters
        ----------
        A_buf : Buffer
            Fragment buffer for operand A (in registers).
        desc_b : Buffer
            Initialized B descriptor (from ``init_wgmma_b_desc``).
        C_buf : Buffer
            Accumulator fragment buffer.
        inst_m_idx : int
            M-dimension atom index (0 .. wgmma_num_inst_m - 1).
        inst_n_idx : int
            N-dimension atom index (0 .. wgmma_num_inst_n - 1).
        ki : int
            K-dimension atom index (0 .. wgmma_num_k_atoms - 1).
        b_params : WGMMADescriptorParams
            Pre-computed B descriptor parameters.
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        local_size_a = self.local_size_a
        local_size_out = self.local_size_out
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        micro_size_k = self.micro_size_k
        n_dim = self.block_col_warps * self.warp_col_tiles
        k_dim = self.chunk
        wgmma_inst_n = self.wgmma_inst_n
        num_inst_n = self.wgmma_num_inst_n
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        wgmma_prefix = self.wgmma_prefix
        b_transposed = self.b_transposed
        elems_in_bytes = b_params.elems_in_bytes
        bk_atom_size = b_params.k_atom_size
        b_swizzle_atom_elems = b_params.swizzle_atom_elems

        thread_binding = self.get_thread_binding()

        A_offset = ki * warp_rows * local_size_a + inst_m_idx * local_size_a
        C_offset = inst_m_idx * warp_cols * local_size_out + inst_n_idx * warp_cols * local_size_out // num_inst_n

        @T.macro
        def _rs_atom(A_buf, desc_b, C_buf):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            warp_j = warp_n * num_inst_n + inst_n_idx
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))

            B_offset = (
                (ki // bk_atom_size) * n_dim * b_swizzle_atom_elems
                + warp_j * wgmma_inst_n * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                if b_params.is_k_major
                else (
                    ki * b_swizzle_atom_elems * micro_size_k + warp_j * wgmma_inst_n * (k_dim if n_dim // b_swizzle_atom_elems > 1 else 1)
                )
            )

            T.ptx_wgmma_rs(
                accum_dtype,
                wgmma_prefix,
                b_transposed,
                a_dtype_abbrv,
                b_dtype_abbrv,
                accum_dtype_abbrv,
                A_buf.data,
                A_offset,
                desc_b.data,
                (B_offset * elems_in_bytes) >> 4,
                C_buf.data,
                C_offset,
                scale_out,
                1,
                1,
            )

        return _rs_atom(A_buf, desc_b, C_buf)

    def wgmma_ss_atom(
        self,
        desc_a: Buffer,
        desc_b: Buffer,
        C_buf: Buffer,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        a_params: WGMMADescriptorParams,
        b_params: WGMMADescriptorParams,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single WGMMA SS instruction for atom ``(inst_m_idx, inst_n_idx, ki)``.

        Must be called between fence/arrive and commit/wait sequences.

        Parameters
        ----------
        desc_a : Buffer
            Initialized A descriptor (from ``init_wgmma_a_desc``).
        desc_b : Buffer
            Initialized B descriptor (from ``init_wgmma_b_desc``).
        C_buf : Buffer
            Accumulator fragment buffer.
        inst_m_idx : int
            M-dimension atom index (0 .. wgmma_num_inst_m - 1).
        inst_n_idx : int
            N-dimension atom index (0 .. wgmma_num_inst_n - 1).
        ki : int
            K-dimension atom index (0 .. wgmma_num_k_atoms - 1).
        a_params : WGMMADescriptorParams
            Pre-computed A descriptor parameters.
        b_params : WGMMADescriptorParams
            Pre-computed B descriptor parameters.
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        local_size_out = self.local_size_out
        warp_cols = self.warp_cols
        micro_size_k = self.micro_size_k
        m_dim = self.block_row_warps * self.warp_row_tiles
        n_dim = self.block_col_warps * self.warp_col_tiles
        k_dim = self.chunk
        wgmma_inst_n = self.wgmma_inst_n
        num_inst_m = self.wgmma_num_inst_m
        num_inst_n = self.wgmma_num_inst_n
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        wgmma_prefix = self.wgmma_prefix
        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed
        a_elems_in_bytes = a_params.elems_in_bytes
        b_elems_in_bytes = b_params.elems_in_bytes
        ak_atom_size = a_params.k_atom_size
        bk_atom_size = b_params.k_atom_size
        a_swizzle_atom_elems = a_params.swizzle_atom_elems
        b_swizzle_atom_elems = b_params.swizzle_atom_elems

        thread_binding = self.get_thread_binding()

        C_offset = inst_m_idx * warp_cols * local_size_out + inst_n_idx * warp_cols * local_size_out // num_inst_n

        @T.macro
        def _ss_atom(desc_a, desc_b, C_buf):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
            warp_i = (warp_m // 4) * num_inst_m + inst_m_idx
            warp_j = warp_n * num_inst_n + inst_n_idx

            A_offset = (
                (ki % ak_atom_size) * micro_size_k
                + warp_i * 64 * a_swizzle_atom_elems
                + (ki // ak_atom_size) * m_dim * a_swizzle_atom_elems
                if a_is_k_major
                else warp_i * 64 * k_dim + ki * a_swizzle_atom_elems * micro_size_k
            )
            B_offset = (
                (ki // bk_atom_size) * n_dim * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                + warp_j * wgmma_inst_n * b_swizzle_atom_elems
                if b_is_k_major
                else (
                    ki * b_swizzle_atom_elems * micro_size_k + warp_j * wgmma_inst_n * (k_dim if n_dim // b_swizzle_atom_elems > 1 else 1)
                )
            )

            T.ptx_wgmma_ss(
                accum_dtype,
                wgmma_prefix,
                a_is_k_major,
                b_is_k_major,
                a_dtype_abbrv,
                b_dtype_abbrv,
                accum_dtype_abbrv,
                desc_a.data,
                (A_offset * a_elems_in_bytes) >> 4,
                desc_b.data,
                (B_offset * b_elems_in_bytes) >> 4,
                C_buf.data,
                C_offset,
                scale_out,
                1,
                1,
            )

        return _ss_atom(desc_a, desc_b, C_buf)

    def make_mma_load_layout(self, local_buf: Buffer, matrix: str = "A") -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.
        This layout is used in conjunction with `inverse_mma_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        assert matrix in ["A"], "matrix should be A for WGMMA"
        dtype = self.a_dtype
        dtype_bits = DataType(dtype).bits
        transposed = self.a_transposed

        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        # sr also can represent a non-transposed basic layout
        # then rs also can represent a transposed basic layout
        transform_func_sr_a: Callable = None
        if dtype_bits == 32:
            transform_func_sr_a = shared_16x8_to_mma_32x4_layout_sr_a
        elif dtype_bits == 16:
            transform_func_sr_a = shared_16x16_to_mma_32x8_layout_sr_a
        elif dtype_bits == 8:
            transform_func_sr_a = shared_16x32_to_mma_32x16_layout_sr_a
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

        is_sr_conditions = [False]
        is_sr_conditions.append(not transposed)
        is_sr_axis_order = any(is_sr_conditions)

        # the layout of mma.sync is row.col.
        # so the b matrix expected a transposed basic layout
        transform_func: Callable = None
        transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)

        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"

        micro_size_s, micro_size_r = self.micro_size_x, self.micro_size_k

        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows = self.warp_rows
        chunk = self.chunk

        warp_s = warp_rows
        warp_r = chunk // micro_size_r
        block_s = block_row_warps
        replicate = block_col_warps

        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=False).replicate(replicate)
            block_fragment = warp_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
        else:
            # rs condition, transposed_a matrix
            warp_fragment = base_fragment.repeat([1, block_s], repeat_on_thread=True, lower_dim_first=False).replicate(replicate)
            block_fragment = warp_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)

        return block_fragment

    def make_mma_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.
        This layout is used in conjunction with `inverse_mma_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        inverse_mma_store_layout = self.get_store_index_map(inverse=True)
        assert is_fragment(local_buf), "local_buf must be a fragment"
        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a thread index according to `inverse_mma_store_layout`.
            """
            lane_id, _ = inverse_mma_store_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a local index in a single thread according
            to `inverse_mma_store_layout`.
            """
            _, local_id = inverse_mma_store_layout.map_indices([i, j])
            return local_id

        # reproduce src/layout/gemm_layouts.cc::MakeGemmFragmentCHopper
        base_fragment = T.Fragment(
            [micro_size_x, micro_size_y],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )
        warp_n_layout = base_fragment.repeat([1, warp_cols], False, False)
        block_layout = warp_n_layout.repeat([block_row_warps, block_col_warps], True, False)
        warp_m_layout = block_layout.repeat([warp_rows, 1], False, False)
        return warp_m_layout
