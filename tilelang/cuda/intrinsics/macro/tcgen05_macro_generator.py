from __future__ import annotations
from dataclasses import dataclass
import tilelang.language as T
import tilelang.language.dtypes as _dtypes
from .mma_macro_generator import TensorCoreIntrinEmitter as MMAIntrinEmitter
from tvm import DataType
from tvm.tirx import PrimExpr, Buffer, Var, BufferLoad, BufferRegion
from tilelang import tvm as tvm
from tilelang import _ffi_api
from tilelang.language.dtypes import get_tvm_dtype
from tilelang.utils import is_tensor_memory
from tilelang.layout import (
    Layout,
    SwizzleMode,
    cute,
)
from tvm.runtime import convert

lift = convert


def _min_leaf_stride(stride) -> int:
    """Smallest leaf stride within a (possibly nested) stride tuple."""
    if isinstance(stride, tuple):
        return min(_min_leaf_stride(s) for s in stride)
    return int(stride)


@dataclass(frozen=True)
class TCGEN05DescriptorParams:
    """Pre-computed parameters for TCGEN05 descriptor initialization and atom offset computation.

    Returned by ``compute_tcgen05_*_desc_params()`` and consumed by
    ``init_tcgen05_*_desc()`` and ``tcgen05_*_atom()`` methods.
    """

    swizzle_mode: SwizzleMode
    """Canonical swizzle mode; project to the descriptor field via ``tcgen05_layout_type()``."""
    leading_byte_offset: int
    """LBO >> 4, ready to pass to ``T.initialize_tcgen05_descriptor``."""
    stride_byte_offset: int
    """SBO >> 4, ready to pass to ``T.initialize_tcgen05_descriptor``."""
    swizzle_atom_elems: int
    """Number of elements per swizzle atom along the non-K dimension."""
    k_atom_size: int
    """``max(swizzle_atom_elems // micro_size_k, 1)``."""
    elem_bits: int
    """Bit width of a single logical element."""
    is_k_major: bool
    """Whether the matrix is stored in K-major order (affects offset formula branching)."""
    slice_byte_offset: object = 0
    """Physical byte offset (raw bytes) of the operand slice origin within its
    buffer; passed to ``T.increase_descriptor_offset`` after building the
    descriptor from the buffer base. ``0`` for a whole-buffer / base-origin
    operand. Computed by :func:`compute_umma_descriptor` from the CuTe layout."""


def _bytes_to_elements(byte_count: int, elem_bits: int) -> int:
    # Use bit widths for offsets so sub-byte dtypes such as FP4 stay packed.
    bits = byte_count * 8
    if bits % elem_bits != 0:
        raise ValueError(f"{byte_count} bytes cannot be represented as whole elements of {elem_bits}-bit dtype")
    return bits // elem_bits


def _elements_to_bytes(elem_count: int, elem_bits: int) -> int:
    bits = elem_count * elem_bits
    if isinstance(elem_count, int) and bits % 8 != 0:
        raise ValueError(f"{elem_count} elements of {elem_bits}-bit dtype do not end on a byte boundary")
    return bits // 8


def compute_umma_descriptor(tl_layout, buffer, transposed: bool, micro_size_k: int = 16, region=None) -> TCGEN05DescriptorParams:
    """Port of CuTe ``make_umma_desc`` (mma_traits_sm100.hpp).

    The Blackwell analog of :func:`compute_gmma_descriptor`: decode an arbitrary
    shared-memory ``tl_layout`` for ``buffer`` and compute the UMMA descriptor
    parameters, accepting any UMMA-canonical layout. A non-canonical layout is a
    programming error, so the canonicity checks assert (mirroring CuTe's
    ``static_assert``s).

    SBO/LBO are read in uint128 units from the canonical layout (dtype-agnostic,
    exactly like ``make_umma_desc``); ``elem_bits`` is carried for the atom-offset
    math. The ``SWIZZLE_128B_BASE32B`` UMMA mode is not produced by tilelang's
    shared-layout makers, so only the standard {NONE,32B,64B,128B} atoms appear.
    Unlike ``make_umma_desc`` (which sees one atom), the decoded layout here is the
    whole operand tile; 2SM/block-scaled atom splitting is handled by the caller.

    ``transposed`` selects which logical axis is MN vs K (shape only): default
    ``[MN, K]``, transposed ``[K, MN]``. The operand is required to be row-major,
    i.e. K-major iff ``not transposed``; the contiguity detected from the layout
    is asserted to agree.

    ``region`` is the operand's per-axis ranges (one :class:`tvm.ir.Range` per
    logical buffer mode), used to restrict the decoded layout to a sliced
    operand. It may be ``None`` for a full-buffer operand.
    """
    elem_bits = get_tvm_dtype(buffer.dtype).bits
    # One decode in element space; the byte- and u128-address variants are pure
    # recasts of it.
    composed_elem = cute.ComposedLayout.from_tilelang(tl_layout)
    assert composed_elem is not None, f"UMMA operand layout is not decodable by the CuTe analyzer: {tl_layout}"
    # Swizzle is read in BYTE space (canonical atom Sw<b,4,3>, m_base==4).
    byte_swizzle = composed_elem.recast(elem_bits, 8).swizzle
    swizzle_mode = byte_swizzle.to_swizzle_mode()

    # Restrict the (possibly hierarchical, swizzle-split) element-space layout to
    # the operand's slice; the descriptor is built from the buffer base and
    # advanced to the slice origin by this offset (warp-uniform).
    tile = composed_elem.layout
    slice_byte_offset = 0
    if region is not None:
        slice_off_elems, tile = cute.restrict(composed_elem.layout, region)
        if slice_off_elems != 0:
            slice_byte_offset = tvm.arith.Analyzer().simplify(slice_off_elems * elem_bits // 8)
    assert cute.rank(tile) == 2, f"UMMA operand tile must be rank-2 (MN, K), got rank {cute.rank(tile)}"
    # Present in UMMA (MN, K) order, then recast the swizzled element layout to
    # uint128_t (exactly CuTe's recast<uint128_t const>(tensor)).
    mn_idx = 1 if transposed else 0
    k_idx = 1 - mn_idx
    mn_dim = int(cute.size(tile[mn_idx]))
    mnk = cute.make_layout([tile[mn_idx], tile[k_idx]])
    u128 = cute.ComposedLayout(composed_elem.swizzle, composed_elem.offset, mnk).recast(elem_bits, 128)
    mn_mode, k_mode = u128.layout[0], u128.layout[1]

    # Row-major only: K-major iff not transposed. Assert the contiguity detected
    # from the layout agrees.
    k_major = not transposed
    detected_k_major = _min_leaf_stride(k_mode.stride) == 1
    assert detected_k_major == k_major, (
        f"UMMA operand layout contiguity (k_major={detected_k_major}) disagrees with "
        f"the row-major expectation (k_major={k_major} for transposed={transposed}); "
        f"only row-major operand layouts are supported."
    )

    # SwizzleAtomMNSize per UMMA LayoutType (NONE->1, B32->2, B64->4, B128->8) = 1 << b_bits.
    W = 1 << byte_swizzle.b_bits
    swizzled = not swizzle_mode.is_none()

    # CuTe make_umma_desc logical_divides each u128 (MN, K) mode by the canonical
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
    assert cute.congruent(d_mn.shape, (1, 1)) and cute.congruent(d_k[0].shape, 1), (
        f"UMMA operand is not a canonical layout: divided MN={d_mn.shape}, K-tile={d_k[0].shape}"
    )
    s00, s01 = d_mn.stride
    s10 = d_k.stride[0]

    if k_major:
        # Canonical ((8,m),(2,k)):((8,SBO),(1,2)). stride<0,0>==W; stride<1,0> is
        # the INTERLEAVE pass-through or 1 when swizzled. SBO=stride<0,1>, LBO=1.
        assert s00 == W, f"Not a canonical UMMA_K layout: stride<0,0>={s00} != W={W}"
        assert not (swizzled and s10 != 1), f"Not a canonical UMMA_K layout: stride<1,0>={s10} != 1"
        sbo = s01
        lbo = s10
    else:
        # Canonical ((W,m),(8,k)). stride<1,0>==W, and stride<0,0>==1 when swizzled.
        assert cute.congruent(d_k.shape, (1, 1)), f"UMMA MN-major operand is not a canonical layout: divided K={d_k.shape}"
        s11 = d_k.stride[1]
        assert not (swizzled and s00 != 1), f"Not a canonical UMMA_MN layout: stride<0,0>={s00} != 1"
        assert s10 == W, f"Not a canonical UMMA_MN layout: stride<1,0>={s10} != W={W}"
        sbo = s11 if swizzled else s01
        lbo = s01 if swizzled else s11

    # Elements per swizzle atom along the non-K (MN) dimension; the unswizzled
    # case spans the whole MN tile (bit-based so sub-byte dtypes stay packed).
    swizzle_atom_elems = mn_dim if swizzle_mode.is_none() else _bytes_to_elements(swizzle_mode.swizzle_byte_size(), elem_bits)
    return TCGEN05DescriptorParams(
        swizzle_mode=swizzle_mode,
        leading_byte_offset=int(lbo),
        stride_byte_offset=int(sbo),
        swizzle_atom_elems=swizzle_atom_elems,
        k_atom_size=max(swizzle_atom_elems // micro_size_k, 1),
        elem_bits=elem_bits,
        is_k_major=k_major,
        slice_byte_offset=slice_byte_offset,
    )


# derive from MMAIntrinEmitter as some layouts are the same
class TensorCoreIntrinEmitter(MMAIntrinEmitter):
    """Intrinsic emitter for Blackwell (SM100) TCGEN5MMA instructions.

    Generates TIR macros that lower to ``tcgen05.mma`` PTX instructions for
    both the SS (Shared-Shared) and TS (TensorMemory-Shared) GEMM variants.
    Also provides layout helpers for tensor-memory (TMEM) buffers.
    """

    # should be rewritten to support dynamic k_dim
    tcgen05_prefix: str

    a_shared_layout: Layout = None
    b_shared_layout: Layout = None

    @staticmethod
    def _smem_elems_in_bytes(dtype) -> int:
        """Byte width of one SMEM element for TCGEN05 descriptor offset math."""
        dt = get_tvm_dtype(dtype)
        return (dt.bits + 7) // 8

    def __init__(
        self,
        a_dtype: str = _dtypes.float16,
        b_dtype: str = _dtypes.float16,
        accum_dtype: str = _dtypes.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool = False,
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

    def _assign_a_shared_layout(self, layout: Layout):
        self.a_shared_layout = layout
        return self

    def _assign_b_shared_layout(self, layout: Layout):
        self.b_shared_layout = layout
        return self

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        # tcgen05 doesn't care about warp partitioning
        self.micro_size_x = m_dim
        self.micro_size_k = k_dim

    def _initialize_k_dim(self, a_dtype=_dtypes.float16):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        if a_dtype.bits == 6 or a_dtype.is_float4():
            if self.chunk % 32 != 0:
                raise ValueError(f"TCGEN5MMA FP{a_dtype.bits} requires chunk to be a multiple of 32, got {self.chunk}")
            self.k_dim = 32
            return
        super()._initialize_k_dim(a_dtype)

    @staticmethod
    def _as_buffer(buf_or_region):
        if isinstance(buf_or_region, BufferRegion):
            return buf_or_region.buffer
        return buf_or_region

    def tcgen05mma(self, A_buf: Buffer, B_buf: Buffer, C_local_buf: Buffer, mbar, clear_accum: PrimExpr = False):
        """Emit a TCGEN5MMA operation, dispatching to SS or TS variant based on A's memory scope.

        If *A_buf* resides in tensor memory (``shared.tmem``), the TS variant is
        emitted; otherwise the SS variant is used (both A and B from shared memory).

        Parameters
        ----------
        A_buf : Buffer
            Operand A — either in shared memory (SS) or tensor memory (TS).
        B_buf : Buffer
            Operand B in shared memory.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        mbar : PrimExpr
            Memory barrier used for MMA completion signalling.
        clear_accum : PrimExpr
            Whether to zero the accumulator before the first MMA.
        """
        if is_tensor_memory(A_buf):
            return self.tcgen05mma_ts(A_buf, B_buf, C_local_buf, mbar, clear_accum)
        return self.tcgen05mma_ss(A_buf, B_buf, C_local_buf, mbar, clear_accum)

    def tcgen05mma_ss(self, A_buf: Buffer, B_buf: Buffer, C_local_buf: Buffer, mbar, clear_accum: PrimExpr = False):
        """Emit the SS (Shared-Shared) variant of TCGEN5MMA.

        Reads operand A and B from shared memory via a descriptor.

        Parameters
        ----------
        A_buf : Buffer
            Operand A in shared memory.
        B_buf : Buffer
            Operand B in shared memory.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        mbar : PrimExpr
            Memory barrier for MMA completion signalling.
        clear_accum : PrimExpr
            Whether to zero the accumulator before the first MMA.
        """
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        num_inst_m = self.tcgen05_num_inst_m
        num_inst_n = self.tcgen05_num_inst_n
        num_k_atoms = self.tcgen05_num_k_atoms
        a_params = self.compute_tcgen05_a_desc_params(A_buf)
        b_params = self.compute_tcgen05_b_desc_params(B_buf)
        instr_desc = self.compute_tcgen05_instr_desc()

        @T.macro
        def _warp_mma_ss(A_buf, B_buf, C_local_buf, mbar):
            desc_a = T.alloc_tcgen05_smem_desc()
            desc_b = T.alloc_tcgen05_smem_desc()
            self.init_tcgen05_a_desc(desc_a, A_buf, a_params)
            self.init_tcgen05_b_desc(desc_b, B_buf, b_params)

            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, num_k_atoms):
                        self.tcgen05_ss_atom(desc_a, desc_b, C_local_buf, i, j, ki, a_params, b_params, instr_desc, clear_accum)
            self.tcgen05_atom_arrive(mbar)

        return _warp_mma_ss(A_buf, B_buf, C_local_buf, mbar)

    def tcgen05mma_ts(self, A_buf, B_buf, C_local_buf, mbar, clear_accum: PrimExpr = False):
        """Emit the TS (TensorMemory-Shared) variant of TCGEN5MMA.

        Reads operand A directly from tensor memory (TMEM) and operand B from
        shared memory via a descriptor.  The TMEM column offset for A is
        computed assuming packed storage (e.g. two ``bfloat16`` values per
        ``uint32`` column) to match the output of ``tcgen05.st``.

        Parameters
        ----------
        A_buf : Buffer
            Operand A residing in tensor memory (``shared.tmem``).
        B_buf : Buffer
            Operand B in shared memory.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        mbar : PrimExpr
            Memory barrier for MMA completion signalling.
        clear_accum : PrimExpr
            Whether to zero the accumulator before the first MMA.
        """
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        assert k_dim >= micro_size_k, f"k_dim must be >= {micro_size_k}, got {k_dim}"

        num_inst_m = self.tcgen05_num_inst_m
        num_inst_n = self.tcgen05_num_inst_n
        num_k_atoms = self.tcgen05_num_k_atoms
        b_params = self.compute_tcgen05_b_desc_params(B_buf)
        instr_desc = self.compute_tcgen05_instr_desc()

        # Resolve the TMEM data pointer for A
        if isinstance(A_buf, BufferRegion):
            a_tmem_data = A_buf.buffer.data
        elif isinstance(A_buf, Buffer):
            a_tmem_data = A_buf.data
        else:
            raise ValueError(f"Unsupported A_buf type for TS variant: {type(A_buf)}")

        @T.macro
        def _warp_mma_ts(a_data, B_buf, C_local_buf, mbar):
            desc_b = T.alloc_tcgen05_smem_desc()
            self.init_tcgen05_b_desc(desc_b, B_buf, b_params)

            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, num_k_atoms):
                        self.tcgen05_ts_atom(a_data, desc_b, C_local_buf, i, j, ki, b_params, instr_desc, clear_accum)
            self.tcgen05_atom_arrive(mbar)

        return _warp_mma_ts(a_tmem_data, B_buf, C_local_buf, mbar)

    def tcgen05mma_blockscaled(
        self,
        A_buf: Buffer,
        B_buf: Buffer,
        C_local_buf: Buffer,
        SFA_tmem,
        SFB_tmem,
        mbar,
        sf_k_start: PrimExpr,
        sf_a_granularity_k: int,
        sf_b_granularity_k: int,
        clear_accum: PrimExpr = False,
    ):
        """Emit a block-scaled TCGEN5MMA (SS variant with TMEM scale factors).

        Uses ``tcgen05.mma.cta_group::1|2.kind::mxf8f6f4.block_scale`` PTX instruction.
        Scale factors must already reside in tensor memory.
        """
        m_dim = self.block_row_warps * self.warp_row_tiles
        micro_size_k = self.micro_size_k
        k_dim, n_dim = self.chunk, self.block_col_warps * self.warp_col_tiles

        assert k_dim >= micro_size_k

        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed

        if len(self.meta) != 5:
            self.get_tcgen5_mma_meta(m_dim, n_dim, k_dim, disable_2cta=False, disable_ws=True)
        if len(self.meta) != 5:
            raise ValueError(
                f"Unsupported TCGEN5MMA configuration for block-scaled: M={m_dim}, N={n_dim}, "
                f"K={k_dim}, A dtype={self.a_dtype}, accum dtype={self.accum_dtype}"
            )
        atom_m, atom_n, _, _, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m

        # CuTe builds the block-scaled A descriptor with FrgTypeA =
        # UMMA::smem_desc<a_major>, i.e. the *same* make_umma_desc as the regular
        # SS path -- the descriptor describes the physical SMEM tile and is
        # independent of the MMA atom_m. So decode it via compute_umma_descriptor
        # (which also handles a sliced A operand) rather than re-deriving SBO/LBO
        # from atom_m_per_cta. The per-atom M stepping in tcgen05_blockscaled_atom
        # walks within the whole-tile descriptor exactly as tcgen05_ss_atom does.
        a_params = self.compute_tcgen05_a_desc_params(A_buf)
        b_params = self.compute_tcgen05_b_desc_params(B_buf)

        base_instr_desc = self.get_tcgen5_blockscaled_instr_desc(
            atom_m,
            atom_n,
            a_is_k_major,
            b_is_k_major,
            1,
            1,
            0,
            0,
        )

        num_inst_m = m_dim // atom_m_per_cta
        num_inst_n = n_dim // atom_n
        num_k_atoms = self.tcgen05_num_k_atoms

        if isinstance(SFA_tmem, BufferRegion):
            sfa_data = SFA_tmem.buffer.data
        elif isinstance(SFA_tmem, Buffer):
            sfa_data = SFA_tmem.data
        else:
            raise ValueError(f"Unsupported SFA_tmem type: {type(SFA_tmem)}")

        if isinstance(SFB_tmem, BufferRegion):
            sfb_data = SFB_tmem.buffer.data
        elif isinstance(SFB_tmem, Buffer):
            sfb_data = SFB_tmem.data
        else:
            raise ValueError(f"Unsupported SFB_tmem type: {type(SFB_tmem)}")

        @T.macro
        def _warp_mma_blockscaled(A_buf, B_buf, C_local_buf, sfa_data, sfb_data, mbar):
            desc_a = T.alloc_tcgen05_smem_desc()
            desc_b = T.alloc_tcgen05_smem_desc()
            self.init_tcgen05_a_desc(desc_a, A_buf, a_params)
            self.init_tcgen05_b_desc(desc_b, B_buf, b_params)

            _sf_k_start = tvm.tirx.const(sf_k_start, "int32") if isinstance(sf_k_start, int) else sf_k_start
            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, num_k_atoms):
                        runtime_sf_a_id = ((_sf_k_start + ki * micro_size_k) // int(sf_a_granularity_k)) % 4
                        runtime_sf_b_id = ((_sf_k_start + ki * micro_size_k) // int(sf_b_granularity_k)) % 4
                        runtime_instr_desc = base_instr_desc | (runtime_sf_a_id << 29) | (runtime_sf_b_id << 4)
                        self.tcgen05_blockscaled_atom(
                            desc_a,
                            desc_b,
                            C_local_buf,
                            sfa_data,
                            sfb_data,
                            i,
                            j,
                            ki,
                            a_params,
                            b_params,
                            runtime_instr_desc,
                            clear_accum,
                        )
            self.tcgen05_atom_arrive(mbar)

        return _warp_mma_blockscaled(A_buf, B_buf, C_local_buf, sfa_data, sfb_data, mbar)

    def get_tcgen5_blockscaled_instr_desc(
        self,
        atom_m: int,
        atom_n: int,
        a_is_k_major: bool,
        b_is_k_major: bool,
        scale_in_a: int,
        scale_in_b: int,
        a_sf_id: int,
        b_sf_id: int,
    ) -> PrimExpr:
        """Build the block-scaled instruction descriptor via FFI."""
        desc = _ffi_api.get_tcgen5_blockscaled_instr_desc(
            atom_m,
            atom_n,
            self.a_dtype,
            self.b_dtype,
            a_is_k_major,
            b_is_k_major,
            scale_in_a,
            scale_in_b,
            a_sf_id,
            b_sf_id,
        )
        return lift(desc)

    def make_mma_load_layout(self, local_buf: Buffer, matrix: str = "A") -> T.Fragment:
        raise NotImplementedError

    def make_mma_store_layout(self, tmem_buf: Buffer) -> Layout:
        """
        Create the TCGEN5 tensor-memory layout used to store MMA accumulators.

        Parameters
        ----------
        tmem_buf : tir.Buffer
            The local buffer representing tensormemory of a mma's output

        Returns
        -------
        Layout
            Layout object describing how logical (i, j) coordinates map to the
            swizzled tensor-memory offsets required by TCGEN5MMA.

        Raises
        ------
        AssertionError
            If `tmem_buf` is not detected to be a tensor-memory buffer.
        """
        assert is_tensor_memory(tmem_buf), "tmem_buf must reside in tensor memory (shared.tmem)"
        if len(tmem_buf.shape) != 2:
            raise ValueError(f"TCGEN5MMA expects a 2-D tensor-memory buffer, got shape {tmem_buf.shape}")

        m = int(tmem_buf.shape[0])
        n = int(tmem_buf.shape[1])
        k = int(self.chunk)

        meta = getattr(self, "meta", ())
        if len(meta) != 5:
            self.get_tcgen5_mma_meta(m, n, k, disable_2cta=True)
            meta = self.meta
        if len(meta) != 5:
            raise ValueError(
                f"Unsupported TCGEN5MMA configuration: M={m}, N={n}, K={k}, A dtype={self.a_dtype}, accum dtype={self.accum_dtype}"
            )
        atom_m, atom_n, _, _, enable_2cta = (int(x) for x in meta)
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m

        if m % atom_m_per_cta != 0 or n % atom_n != 0:
            raise ValueError(f"Invalid TCGEN5MMA store layout for shape ({m}, {n}) with atoms ({atom_m}, {atom_n})")

        def forward(i: PrimExpr, j: PrimExpr):
            atom_idx = (i // atom_m_per_cta) + (j // atom_n) * (m // atom_m_per_cta)
            ai = i % atom_m_per_cta
            aj = j % atom_n

            # NOTE: Currently not all 7 layout are supported
            if atom_m == 256:
                # Layout A (2 cta)
                assert enable_2cta, "atom_m=256 for TCGEN5MMA must use 2cta"
                return [
                    ai % 128,
                    aj + atom_idx * atom_n,
                ]
            if atom_m == 128:
                if enable_2cta:
                    # Layout B
                    half_atom_n = atom_n // 2
                    return [
                        ai + (aj // half_atom_n) * 64,
                        (aj % half_atom_n) + atom_idx * half_atom_n,
                    ]
                else:
                    # Layout D
                    return [
                        ai,
                        aj + atom_idx * atom_n,
                    ]
            if atom_m == 64:
                # Layout E (.ws variant)
                half_atom_n = atom_n // 2
                return [
                    (ai // 32) * 32 + ai % 32 + (aj // half_atom_n) * 64,
                    (aj % half_atom_n) + atom_idx * half_atom_n,
                ]
            if atom_m == 32:
                # Layout G
                quarter_atom_n = atom_n // 4
                return [
                    ai % 32 + (aj // quarter_atom_n) * 32,
                    (aj % quarter_atom_n) + atom_idx * quarter_atom_n,
                ]

            raise ValueError(f"Unsupported TCGEN5 atom_m={atom_m}")

        return Layout([m, n], forward)

    def get_tcgen5_mma_meta(self, m: int, n: int, k: int, disable_2cta: bool, disable_ws: bool = False):
        """Query the FFI for TCGEN5MMA atom metadata (atom_m, atom_n, atom_k, enable_ws, enable_2cta), and record them in `self.meta`."""
        self.meta = _ffi_api.get_tcgen5_mma_meta(
            int(m),
            int(n),
            int(k),
            self.a_dtype,
            self.accum_dtype,
            bool(disable_2cta),
            bool(disable_ws),
        )

    def get_tcgen5_instr_desc(
        self, atom_m: int, atom_n: int, atom_k: int, a_is_k_major: bool, b_is_k_major: bool, scale_in_a: int, scale_in_b: int
    ) -> PrimExpr:
        """Build the 64-bit instruction descriptor for a ``tcgen05.mma`` PTX call."""
        desc = _ffi_api.get_tcgen5_instr_desc(
            atom_m,
            atom_n,
            atom_k,
            self.a_dtype,
            self.b_dtype,
            self.accum_dtype,
            a_is_k_major,
            b_is_k_major,
            scale_in_a,
            scale_in_b,
        )
        return lift(desc)

    # ---- Atom-level interface ----

    @property
    def tcgen05_meta_unpacked(self) -> tuple:
        """Return ``(atom_m, atom_n, atom_k, enable_ws, enable_2cta)`` as ints.

        Requires ``self.meta`` to have been set via ``get_tcgen5_mma_meta()``.
        """
        assert len(self.meta) == 5, "TCGEN05 meta not initialized; call get_tcgen5_mma_meta() first"
        return tuple(int(x) for x in self.meta)

    @property
    def tcgen05_num_inst_m(self) -> int:
        """Number of TCGEN05MMA instruction atoms along M (SS variant)."""
        atom_m, _, _, _, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m
        return self.block_row_warps * self.warp_row_tiles // atom_m_per_cta

    @property
    def tcgen05_num_inst_n(self) -> int:
        """Number of TCGEN05MMA instruction atoms along N."""
        _, atom_n, _, _, _ = self.tcgen05_meta_unpacked
        return self.block_col_warps * self.warp_col_tiles // atom_n

    @property
    def tcgen05_num_k_atoms(self) -> int:
        """Number of K-dimension micro-steps (``chunk // micro_size_k``)."""
        return self.chunk // self.micro_size_k

    @staticmethod
    def _access_ptr_from(buffer_or_load_or_region, access_type: str = "r"):
        """Resolve an access pointer from a Buffer, BufferLoad, or BufferRegion."""
        if isinstance(buffer_or_load_or_region, Buffer):
            return buffer_or_load_or_region.access_ptr(access_type)
        elif isinstance(buffer_or_load_or_region, BufferLoad):
            buffer_load = buffer_or_load_or_region
            offset, stride = 0, 1
            buffer = buffer_load.buffer
            for i, shape in enumerate(reversed(buffer.shape)):
                indice = buffer_load.indices[len(buffer_load.indices) - i - 1]
                if isinstance(indice, tvm.tirx.Ramp):
                    offset += indice.base * stride
                elif isinstance(indice, (tvm.tirx.IntImm, tvm.tirx.PrimExpr)):
                    offset += indice * stride
                else:
                    raise ValueError(f"Unsupported index type: {type(indice)}")
                stride *= shape
            return buffer.access_ptr(access_type, offset=offset)
        elif isinstance(buffer_or_load_or_region, BufferRegion):
            buffer_region = buffer_or_load_or_region
            buffer = buffer_region.buffer
            offset, stride = 0, 1
            for i, shape in enumerate(reversed(buffer.shape)):
                offset += buffer_region.region[len(buffer_region.region) - i - 1].min * stride
                stride *= shape
            return buffer.access_ptr(access_type, offset=offset)
        else:
            raise ValueError(f"Unsupported buffer type: {type(buffer_or_load_or_region)}")

    # -- Descriptor parameter computation (pure Python, no TIR) --

    def compute_tcgen05_b_desc_params(self, B_buf) -> TCGEN05DescriptorParams:
        """Compute B descriptor parameters from the B shared buffer via the CuTe
        ``make_umma_desc`` port. The returned ``TCGEN05DescriptorParams`` is passed
        to ``init_tcgen05_b_desc()`` and ``tcgen05_*_atom()``.

        Parameters
        ----------
        B_buf : Buffer or BufferRegion
            The B operand in shared memory.
        """
        assert self.b_shared_layout is not None, "TCGEN05 B operand has no shared layout to decode"
        return compute_umma_descriptor(
            self.b_shared_layout,
            B_buf.buffer if isinstance(B_buf, BufferRegion) else B_buf,
            transposed=not self.b_transposed,
            micro_size_k=self.micro_size_k,
            region=list(B_buf.region) if isinstance(B_buf, BufferRegion) else None,
        )

    def compute_tcgen05_a_desc_params(self, A_buf) -> TCGEN05DescriptorParams:
        """Compute A descriptor parameters from the A shared buffer (SS variant)
        via the CuTe ``make_umma_desc`` port.

        Parameters
        ----------
        A_buf : Buffer or BufferRegion
            The A operand in shared memory.
        """
        assert self.a_shared_layout is not None, "TCGEN05 A operand has no shared layout to decode"
        return compute_umma_descriptor(
            self.a_shared_layout,
            A_buf.buffer if isinstance(A_buf, BufferRegion) else A_buf,
            transposed=self.a_transposed,
            micro_size_k=self.micro_size_k,
            region=list(A_buf.region) if isinstance(A_buf, BufferRegion) else None,
        )

    # -- Descriptor initialization (emit TIR) --

    def init_tcgen05_b_desc(self, desc_b, B_buf, b_params: TCGEN05DescriptorParams):
        """Emit TIR to initialize a pre-allocated TCGEN05 B descriptor.

        Parameters
        ----------
        desc_b : Buffer
            A descriptor buffer allocated via ``T.alloc_tcgen05_smem_desc()``.
        B_buf : Buffer or BufferRegion
            The B operand in shared memory.
        b_params : TCGEN05DescriptorParams
            Pre-computed parameters from ``compute_tcgen05_b_desc_params()``.
        """
        lbo = b_params.leading_byte_offset
        sbo = b_params.stride_byte_offset
        swizzle_mode = b_params.swizzle_mode.tcgen05_layout_type()
        slice_byte_offset = b_params.slice_byte_offset
        is_sliced = not isinstance(slice_byte_offset, int) or slice_byte_offset != 0
        B_base_ptr = self._as_buffer(B_buf).access_ptr("r")

        @T.macro
        def _init_b(desc_b, B_base_ptr):
            # Build from the buffer base (uniform cvta), then advance to the slice origin.
            T.initialize_tcgen05_descriptor(desc_b, B_base_ptr, lbo, sbo, 0, False, swizzle_mode)
            if is_sliced:
                T.increase_descriptor_offset(desc_b, slice_byte_offset)

        return _init_b(desc_b, B_base_ptr)

    def init_tcgen05_a_desc(self, desc_a, A_buf, a_params: TCGEN05DescriptorParams):
        """Emit TIR to initialize a pre-allocated TCGEN05 A descriptor (SS variant).

        Parameters
        ----------
        desc_a : Buffer
            A descriptor buffer allocated via ``T.alloc_tcgen05_smem_desc()``.
        A_buf : Buffer or BufferRegion
            The A operand in shared memory.
        a_params : TCGEN05DescriptorParams
            Pre-computed parameters from ``compute_tcgen05_a_desc_params()``.
        """
        lbo = a_params.leading_byte_offset
        sbo = a_params.stride_byte_offset
        swizzle_mode = a_params.swizzle_mode.tcgen05_layout_type()
        slice_byte_offset = a_params.slice_byte_offset
        is_sliced = not isinstance(slice_byte_offset, int) or slice_byte_offset != 0
        A_base_ptr = self._as_buffer(A_buf).access_ptr("r")

        @T.macro
        def _init_a(desc_a, A_base_ptr):
            # Build from the buffer base (uniform cvta), then advance to the slice origin.
            T.initialize_tcgen05_descriptor(desc_a, A_base_ptr, lbo, sbo, 0, False, swizzle_mode)
            if is_sliced:
                T.increase_descriptor_offset(desc_a, slice_byte_offset)

        return _init_a(desc_a, A_base_ptr)

    # -- Instruction descriptor computation --

    def compute_tcgen05_instr_desc(self) -> PrimExpr:
        """Compute the 64-bit instruction descriptor using current meta.

        Requires ``self.meta`` to have been set via ``get_tcgen5_mma_meta()``.
        """
        atom_m, atom_n, atom_k, _, _ = self.tcgen05_meta_unpacked
        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed
        return self.get_tcgen5_instr_desc(atom_m, atom_n, atom_k, a_is_k_major, b_is_k_major, 1, 1)

    # -- Arrive --

    def tcgen05_atom_arrive(self, mbar):
        """Emit ``tcgen05_mma_arrive(mbar)``."""
        _, _, _, _, enable_2cta = self.tcgen05_meta_unpacked

        @T.macro
        def _arrive(mbar):
            T.tcgen05_mma_arrive(mbar, arrive_2cta=bool(enable_2cta))

        return _arrive(mbar)

    # -- Atom emission --

    def tcgen05_ss_atom(
        self,
        desc_a,
        desc_b,
        C_local_buf: Buffer,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        a_params: TCGEN05DescriptorParams,
        b_params: TCGEN05DescriptorParams,
        instr_desc: PrimExpr,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single TCGEN05MMA SS instruction for atom ``(inst_m_idx, inst_n_idx, ki)``.

        Must be called after descriptor initialization and before ``tcgen05_atom_arrive()``.

        Parameters
        ----------
        desc_a, desc_b : Buffer
            Initialized A and B descriptors.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        inst_m_idx : int
            M-dimension atom index (0 .. tcgen05_num_inst_m - 1).
        inst_n_idx : int
            N-dimension atom index (0 .. tcgen05_num_inst_n - 1).
        ki : int
            K-dimension atom index (0 .. tcgen05_num_k_atoms - 1).
        a_params : TCGEN05DescriptorParams
            Pre-computed A descriptor parameters.
        b_params : TCGEN05DescriptorParams
            Pre-computed B descriptor parameters.
        instr_desc : PrimExpr
            Instruction descriptor from ``compute_tcgen05_instr_desc()``.
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        atom_m, atom_n, _, enable_ws, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m
        n_dim = self.block_col_warps * self.warp_col_tiles
        n_dim_per_cta = n_dim // 2 if enable_2cta else n_dim
        m_dim = self.block_row_warps * self.warp_row_tiles
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        accum_dtype_in_bits = get_tvm_dtype(self.accum_dtype).bits
        a_dtype_abbrv = self.a_dtype_abbrv
        a_elem_bits = a_params.elem_bits
        b_elem_bits = b_params.elem_bits
        ak_atom_size = a_params.k_atom_size
        bk_atom_size = b_params.k_atom_size
        a_swizzle_atom_elems = a_params.swizzle_atom_elems
        b_swizzle_atom_elems = b_params.swizzle_atom_elems
        mask_zero = T.cast(0, T.int32)

        # Pre-compute offsets
        if a_params.is_k_major:
            A_elem_offset = (
                (ki % ak_atom_size) * micro_size_k
                + inst_m_idx * atom_m_per_cta * a_swizzle_atom_elems
                + (ki // ak_atom_size) * m_dim * a_swizzle_atom_elems
            )
        else:
            A_elem_offset = inst_m_idx * atom_m_per_cta * k_dim + ki * a_swizzle_atom_elems * micro_size_k

        if b_params.is_k_major:
            B_elem_offset = (
                (ki // bk_atom_size) * n_dim_per_cta * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                + inst_n_idx * atom_n * b_swizzle_atom_elems
            )
        else:
            B_elem_offset = ki * b_swizzle_atom_elems * micro_size_k + inst_n_idx * atom_n * (
                k_dim if n_dim_per_cta // b_swizzle_atom_elems > 1 else 1
            )

        A_byte_offset = _elements_to_bytes(A_elem_offset, a_elem_bits)
        B_byte_offset = _elements_to_bytes(B_elem_offset, b_elem_bits)
        tmem_col_step = atom_n // (128 // atom_m_per_cta)
        C_offset = (inst_m_idx * n_dim + inst_n_idx * tmem_col_step) * accum_dtype_in_bits // 32

        @T.macro
        def _ss_atom(desc_a, desc_b, C_local_buf):
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
            T.ptx_tcgen05_mma_ss(
                a_dtype_abbrv,
                desc_a.data,
                A_byte_offset,
                desc_b.data,
                B_byte_offset,
                C_local_buf.data,
                C_offset,
                instr_desc,
                scale_out,
                mask_zero,
                mask_zero,
                mask_zero,
                mask_zero,
                enable_ws,
                enable_2cta,
            )

        return _ss_atom(desc_a, desc_b, C_local_buf)

    def tcgen05_ts_atom(
        self,
        a_tmem_data,
        desc_b,
        C_local_buf: Buffer,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        b_params: TCGEN05DescriptorParams,
        instr_desc: PrimExpr,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single TCGEN05MMA TS instruction for atom ``(inst_m_idx, inst_n_idx, ki)``.

        A resides in tensor memory; B in shared memory.

        Parameters
        ----------
        a_tmem_data : Var
            Data pointer for the A operand in tensor memory (e.g., ``A_buf.data``).
        desc_b : Buffer
            Initialized B descriptor.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        inst_m_idx : int
            M-dimension atom index.
        inst_n_idx : int
            N-dimension atom index.
        ki : int
            K-dimension atom index.
        b_params : TCGEN05DescriptorParams
            Pre-computed B descriptor parameters.
        instr_desc : PrimExpr
            Instruction descriptor from ``compute_tcgen05_instr_desc()``.
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        atom_m, atom_n, atom_k, _, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m
        n_dim = self.block_col_warps * self.warp_col_tiles
        n_dim_per_cta = n_dim // 2 if enable_2cta else n_dim
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        a_dtype_in_bits = get_tvm_dtype(self.a_dtype).bits
        accum_dtype_in_bits = get_tvm_dtype(self.accum_dtype).bits
        a_dtype_abbrv = self.a_dtype_abbrv
        b_elem_bits = b_params.elem_bits
        bk_atom_size = b_params.k_atom_size
        b_swizzle_atom_elems = b_params.swizzle_atom_elems
        mask_zero = T.cast(0, T.int32)

        # TMEM column geometry for A
        interleave = max(128 // atom_m, 1)
        a_tmem_cols_per_k_atom = atom_k * a_dtype_in_bits // 32 // interleave
        a_tmem_k_stride = k_dim * a_dtype_in_bits // 32 // interleave

        A_tmem_offset = inst_m_idx * a_tmem_k_stride + ki * a_tmem_cols_per_k_atom

        if b_params.is_k_major:
            B_elem_offset = (
                (ki // bk_atom_size) * n_dim_per_cta * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                + inst_n_idx * atom_n * b_swizzle_atom_elems
            )
        else:
            B_elem_offset = ki * b_swizzle_atom_elems * micro_size_k + inst_n_idx * atom_n * (
                k_dim if n_dim_per_cta // b_swizzle_atom_elems > 1 else 1
            )
        B_byte_offset = _elements_to_bytes(B_elem_offset, b_elem_bits)

        tmem_col_step = atom_n // (128 // atom_m_per_cta)
        C_offset = (inst_m_idx * n_dim + inst_n_idx * tmem_col_step) * accum_dtype_in_bits // 32

        @T.macro
        def _ts_atom(a_data, desc_b, C_local_buf):
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
            T.ptx_tcgen05_mma_ts(
                a_dtype_abbrv,
                a_data,
                A_tmem_offset,
                desc_b.data,
                B_byte_offset,
                C_local_buf.data,
                C_offset,
                instr_desc,
                scale_out,
                mask_zero,
                mask_zero,
                mask_zero,
                mask_zero,
                enable_2cta,
            )

        return _ts_atom(a_tmem_data, desc_b, C_local_buf)

    def tcgen05_blockscaled_atom(
        self,
        desc_a,
        desc_b,
        C_local_buf: Buffer,
        sfa_data,
        sfb_data,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        a_params: TCGEN05DescriptorParams,
        b_params: TCGEN05DescriptorParams,
        instr_desc: PrimExpr,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single TCGEN05MMA block-scaled SS instruction.

        Parameters
        ----------
        desc_a, desc_b : Buffer
            Initialized A and B descriptors.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        sfa_data, sfb_data : Var
            Scale factor data pointers in tensor memory.
        inst_m_idx, inst_n_idx, ki : int
            Atom indices.
        a_params, b_params : TCGEN05DescriptorParams
            Pre-computed descriptor parameters.
        instr_desc : PrimExpr
            Block-scaled instruction descriptor (with SF IDs already encoded).
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        atom_m, atom_n, _, _enable_ws, enable_2cta = self.tcgen05_meta_unpacked
        del _enable_ws  # block-scaled TCGEN05 does not support .ws
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m
        n_dim = self.block_col_warps * self.warp_col_tiles
        n_dim_per_cta = n_dim // 2 if enable_2cta else n_dim
        m_dim = self.block_row_warps * self.warp_row_tiles
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        accum_dtype_in_bits = get_tvm_dtype(self.accum_dtype).bits
        a_dtype_abbrv = self.a_dtype_abbrv
        a_elem_bits = a_params.elem_bits
        b_elem_bits = b_params.elem_bits
        ak_atom_size = a_params.k_atom_size
        bk_atom_size = b_params.k_atom_size
        a_swizzle_atom_elems = a_params.swizzle_atom_elems
        b_swizzle_atom_elems = b_params.swizzle_atom_elems

        if a_params.is_k_major:
            A_elem_offset = (
                (ki % ak_atom_size) * micro_size_k
                + inst_m_idx * atom_m_per_cta * a_swizzle_atom_elems
                + (ki // ak_atom_size) * m_dim * a_swizzle_atom_elems
            )
        else:
            A_elem_offset = inst_m_idx * atom_m_per_cta * k_dim + ki * a_swizzle_atom_elems * micro_size_k

        if b_params.is_k_major:
            B_elem_offset = (
                (ki // bk_atom_size) * n_dim_per_cta * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                + inst_n_idx * atom_n * b_swizzle_atom_elems
            )
        else:
            B_elem_offset = ki * b_swizzle_atom_elems * micro_size_k + inst_n_idx * atom_n * (
                k_dim if n_dim_per_cta // b_swizzle_atom_elems > 1 else 1
            )

        A_byte_offset = _elements_to_bytes(A_elem_offset, a_elem_bits)
        B_byte_offset = _elements_to_bytes(B_elem_offset, b_elem_bits)
        tmem_col_step = atom_n // (128 // atom_m_per_cta)
        C_offset = (inst_m_idx * n_dim + inst_n_idx * tmem_col_step) * accum_dtype_in_bits // 32

        @T.macro
        def _bs_atom(desc_a, desc_b, C_local_buf, sfa_data, sfb_data):
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
            T.ptx_tcgen05_mma_blockscaled_ss(
                a_dtype_abbrv,
                desc_a.data,
                A_byte_offset,
                desc_b.data,
                B_byte_offset,
                C_local_buf.data,
                C_offset,
                instr_desc,
                scale_out,
                sfa_data,
                0,
                sfb_data,
                0,
                0,
                0,
                enable_2cta,
            )

        return _bs_atom(desc_a, desc_b, C_local_buf, sfa_data, sfb_data)
