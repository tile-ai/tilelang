"""CuTe TMEM fragment layouts for dense SM100 MMA (``tmem_frg`` family).

A TMEM fragment is a :class:`cute.Layout` with :class:`cute.ScaledBasis`
strides, mapping logical buffer coordinates to hierarchical ``(datapath,
column)`` TMEM coordinates, with the column measured in the buffer's value
type.  Exactly as in CUTLASS ``mma_traits_sm100.hpp``, an atom in virtual
TMEM addressing (datapath stride one, column stride 128) is repeated over the
tile counts, then composed with ``tmem_restride`` into physical coordinates.
Both TMEM allocation and MMA address formation consume the same layout, and a
sliced operand is handled by ``cute.restrict`` like any other CuTe layout.

The atoms are the PTX "TMEM data path layout organization" variants
(https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-organization):

========  ==========================  ====================================
PTX       CUTLASS atom                shape:stride (virtual addressing)
========  ==========================  ====================================
Layout A  2SM M256 (M128/CTA)         ``(128, N):(1, 128)``
Layout B  2SM M128 (M64/CTA, "2x2")   ``(64, (N/2, 2)):(1, (128, 64))``
Layout C  2SM M64  (M32/CTA, "1x4")   ``(32, (N/4, 4)):(1, (128, 32))``
Layout D  1SM M128                    ``(128, N):(1, 128)``
Layout E  1SM M64  ``.ws`` C/D        ``(64, (N/2, 2)):(1, (128, 64))``
Layout F  1SM M64  half-datapath      ``((16, 4), N):((1, 32), 128)``
Layout G  1SM M32  ``.ws`` C/D        ``(32, (N/4, 4)):(1, (128, 32))``
========  ==========================  ====================================

Only the compiler's MMA lowering (this module and
``tcgen05_macro_generator``) sees CuTe layouts; everywhere else — the layout
map, TMEM allocation, and copy lowering — a fragment travels as its TileLang
``lambda *indices: [datapath, column]`` form via
:meth:`cute.Layout.to_tilelang`, and
:meth:`cute.Layout.from_tilelang_hierarchical` recovers the hierarchical
CuTe layout losslessly.

As everywhere else in the MMA lowering, the last two modes of a buffer are
the matrix modes; any leading modes are batch dimensions (for example a
software-pipeline stage) and repeat the fragment along TMEM columns.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from tvm.tirx import Buffer

from tilelang.language.dtypes import get_tvm_dtype
from tilelang.layout import cute


class TCGEN05TmemAllocMode(Enum):
    """Dense TMEM allocation modes from CUTLASS ``UMMA::TmemAllocMode``."""

    INTERLEAVED = "interleaved"
    NON_INTERLEAVED = "non_interleaved"
    DUPLICATED = "duplicated"


@dataclass(frozen=True)
class TCGEN05Meta:
    """The TCGEN5MMA instruction atom selected by ``get_tcgen5_mma_meta``."""

    atom_m: int
    atom_n: int
    atom_k: int
    enable_ws: bool
    enable_2cta: bool

    @classmethod
    def from_ffi(cls, values: Iterable[int]) -> TCGEN05Meta:
        atom_m, atom_n, atom_k, enable_ws, enable_2cta = (int(x) for x in values)
        return cls(atom_m, atom_n, atom_k, bool(enable_ws), bool(enable_2cta))

    @property
    def num_sms(self) -> int:
        return 2 if self.enable_2cta else 1

    @property
    def atom_m_per_cta(self) -> int:
        return self.atom_m // self.num_sms

    @property
    def b_atom_n_per_cta(self) -> int:
        """Each 2SM instruction reads an N/2 shard of B from each CTA."""
        return self.atom_n // self.num_sms


def make_tmem_frg_atom(
    num_sms: int,
    alloc_mode: TCGEN05TmemAllocMode,
    atom_m: int,
    atom_n: int,
) -> tuple[cute.Layout, int]:
    """Return CUTLASS ``tmem_frg``'s ``(tmem_atom, outer_tile_stride)`` pair.

    ``atom_m`` is the per-CTA M extent.  The 1SM M64 atom is PTX Layout F
    (half datapaths, 32-datapath halves interleaved); the 2SM atoms fold the
    peer CTA's N shard into the upper datapaths (PTX Layouts A/B/C).
    """

    if num_sms == 1:
        assert alloc_mode in {
            TCGEN05TmemAllocMode.INTERLEAVED,
            TCGEN05TmemAllocMode.NON_INTERLEAVED,
        }, f"1SM TMEM fragments do not support allocation mode {alloc_mode.value}"
        if atom_m == 64:
            # PTX Layout F: even 16-datapath halves; a second tile may
            # interleave into the odd halves (tile_stride one) or continue in
            # fresh columns (NonInterleaved, tile_stride two).
            atom = cute.make_layout(((16, 4), atom_n), ((1, 32), 128))
            tile_stride = 1 if alloc_mode is TCGEN05TmemAllocMode.INTERLEAVED else 2
            return atom, tile_stride
        if atom_m == 128:
            # PTX Layout D: all datapaths.
            return cute.make_layout((128, atom_n)), 1
        raise ValueError(f"1SM TMEM fragment M atom must be 64 or 128, got {atom_m}")

    if num_sms == 2:
        assert alloc_mode in {
            TCGEN05TmemAllocMode.INTERLEAVED,
            TCGEN05TmemAllocMode.DUPLICATED,
        }, f"2SM TMEM fragments do not support allocation mode {alloc_mode.value}"
        if atom_m == 32:
            # PTX Layout C ("1x4"): four N quarters stacked across datapaths.
            assert alloc_mode is TCGEN05TmemAllocMode.INTERLEAVED, "2SM M32 TMEM fragments only support interleaved allocation"
            assert atom_n % 4 == 0, f"2SM M32 TMEM fragment N atom must be divisible by 4, got {atom_n}"
            return cute.make_layout((32, (atom_n // 4, 4)), (1, (128, 32))), 1
        if atom_m == 64:
            if alloc_mode is TCGEN05TmemAllocMode.DUPLICATED:
                # Duplicated allocation has a physical M domain twice as large
                # as its logical domain.  ``make_tmem_frg_atom`` exposes the
                # legal CUTLASS atom for tests and capability queries;
                # ``make_tmem_frg`` rejects it because Layout is one-to-one.
                return cute.make_layout((128, atom_n)), 1
            # PTX Layout B ("2x2"): two N halves stacked across datapaths.
            assert atom_n % 2 == 0, f"2SM M64 TMEM fragment N atom must be even, got {atom_n}"
            return cute.make_layout((64, (atom_n // 2, 2)), (1, (128, 64))), 1
        if atom_m == 128:
            # PTX Layout A: all datapaths per CTA.
            return cute.make_layout((128, atom_n)), 1
        raise ValueError(f"2SM TMEM fragment M atom must be 32, 64, or 128, got {atom_m}")

    raise ValueError(f"TCGEN05 TMEM fragments require one or two SMs, got {num_sms}")


def make_tmem_frg_ws_atom(atom_m: int, atom_n: int) -> cute.Layout:
    """Return CUTLASS ``tmem_frg_ws``'s dense 1SM weight-stationary C/D atom.

    ``.ws`` instructions with M below 128 fold N shards into the upper
    datapaths instead of leaving them idle: PTX Layout E for M64 and Layout G
    for M32.
    """

    assert atom_n in (64, 128, 256), f"Weight-stationary 1SM TMEM fragment N atom must be 64, 128, or 256, got {atom_n}"
    if atom_m == 32:
        # PTX Layout G.
        return cute.make_layout((32, (atom_n // 4, 4)), (1, (128, 32)))
    if atom_m == 64:
        # PTX Layout E.
        return cute.make_layout((64, (atom_n // 2, 2)), (1, (128, 64)))
    if atom_m == 128:
        # Same as PTX Layout D; .ws with full datapaths has nothing to fold.
        return cute.make_layout((128, atom_n))
    raise ValueError(f"Weight-stationary 1SM TMEM fragment M atom must be 32, 64, or 128, got {atom_m}")


def _tile_tmem_frg_atom(
    buffer: Buffer,
    atom: cute.Layout,
    tile_stride: int,
    atom_m: int,
    atom_n: int,
    storage_bits: int,
) -> cute.Layout:
    """CUTLASS ``tmem_frg::make`` for one validated atom, in CuTe algebra."""

    shape = tuple(int(x) for x in buffer.shape)
    assert len(shape) >= 2, f"TMEM buffer {buffer.name} must have at least two modes, got shape {shape}"
    m_extent, n_extent = shape[-2:]
    assert m_extent % atom_m == 0, f"TMEM buffer {buffer.name} M extent {m_extent} must be divisible by fragment atom M={atom_m}"
    assert n_extent % atom_n == 0, f"TMEM buffer {buffer.name} second-mode extent {n_extent} must be divisible by fragment atom {atom_n}"

    value_dtype = get_tvm_dtype(buffer.dtype)
    value_bits = value_dtype.bits * value_dtype.lanes
    assert storage_bits >= value_bits and storage_bits % value_bits == 0, (
        f"TMEM buffer {buffer.name} storage width {storage_bits} must be an integer multiple of its value width {value_bits}"
    )

    batch_extents = tuple(reversed(shape[:-2]))
    batch_count = 1
    for extent in batch_extents:
        batch_count *= extent

    # CUTLASS checks storage capacity before constructing the fragment.
    # TMEM contains 128 datapaths by 512 b32 columns.
    assert batch_count * m_extent * n_extent * storage_bits <= 128 * 512 * 32, (
        f"TMEM buffer {buffer.name} shape {shape} exceeds the 128x512 b32 TMEM capacity"
    )

    # ``tmem_frg::make``: repeat the virtual atom over the tile counts, first
    # mode fastest, with the atom's requested base tile stride.  Batch modes
    # continue the same column-major tiling, so they extend the tiler
    # (row-major buffer axes reversed to keep the last batch axis fastest).
    tiler_shape = (m_extent // atom_m, n_extent // atom_n, *batch_extents)
    tiler_strides = []
    stride = tile_stride
    for extent in tiler_shape:
        tiler_strides.append(stride)
        stride *= extent
    tiler = cute.make_layout(tiler_shape, tuple(tiler_strides))
    tiled = cute.blocked_product(atom, tiler)

    # Present the top-level modes in buffer order (batch..., M, N) so the
    # layout consumes logical buffer coordinates directly.
    mode_order = [*range(cute.rank(tiled) - 1, 1, -1), 0, 1]
    ordered = cute.make_layout([tiled[mode] for mode in mode_order])

    # ``tmem_restride``: virtual addresses to ``(datapath, column)``
    # coordinates, the column scaled by StorageType/ValueType exactly as
    # CUTLASS's COL_ADDR stride (a float16 accumulator occupies the low half
    # of an int32 slot; tcgen05.ld reads it back with pack::16b).
    restride = cute.make_layout(
        (128, 16384),
        (cute.E(0), storage_bits // value_bits * cute.E(1)),
    )
    return cute.composition(restride, ordered)


def make_tmem_frg(
    buffer: Buffer,
    num_sms: int,
    alloc_mode: TCGEN05TmemAllocMode,
    atom_m: int,
    atom_n: int,
    storage_bits: int,
) -> cute.Layout:
    """Tile a legal ordinary CUTLASS atom over the buffer's matrix modes."""

    if alloc_mode is TCGEN05TmemAllocMode.DUPLICATED and atom_m == 64:
        raise ValueError(
            f"TMEM buffer {buffer.name} uses CUTLASS's duplicated 2SM M64 allocation, which cannot be represented by "
            "TileLang's one-to-one Layout; materialize the duplicated A storage explicitly"
        )
    atom, tile_stride = make_tmem_frg_atom(num_sms, alloc_mode, atom_m, atom_n)
    return _tile_tmem_frg_atom(buffer, atom, tile_stride, atom_m, atom_n, storage_bits)


def make_tmem_frg_a(buffer: Buffer, meta: TCGEN05Meta) -> cute.Layout:
    """Build the exact dense TS ``FrgTypeA`` selected by CUTLASS."""

    alloc_mode = TCGEN05TmemAllocMode.DUPLICATED if meta.enable_2cta else TCGEN05TmemAllocMode.NON_INTERLEAVED
    value_dtype = get_tvm_dtype(buffer.dtype)
    return make_tmem_frg(
        buffer,
        meta.num_sms,
        alloc_mode,
        meta.atom_m_per_cta,
        meta.atom_k,
        value_dtype.bits * value_dtype.lanes,
    )


def make_tmem_frg_c(buffer: Buffer, meta: TCGEN05Meta, *, is_ts: bool) -> cute.Layout:
    """Build the dense ``FrgTypeC`` paired with the selected instruction.

    Ordinary SS accumulators use the interleaved allocation; dense TS traits
    pin CUTLASS's NonInterleaved (1SM) / Interleaved (2SM) modes; ``.ws`` SS
    instructions use the weight-stationary fragment (PTX Layouts E/G).
    """

    if meta.enable_ws and not is_ts:
        assert not meta.enable_2cta, "Weight-stationary TCGEN5MMA is 1SM-only"
        return _tile_tmem_frg_atom(
            buffer,
            make_tmem_frg_ws_atom(meta.atom_m, meta.atom_n),
            1,
            meta.atom_m,
            meta.atom_n,
            32,
        )
    if is_ts:
        alloc_mode = TCGEN05TmemAllocMode.INTERLEAVED if meta.enable_2cta else TCGEN05TmemAllocMode.NON_INTERLEAVED
    else:
        alloc_mode = TCGEN05TmemAllocMode.INTERLEAVED
    return make_tmem_frg(
        buffer,
        meta.num_sms,
        alloc_mode,
        meta.atom_m_per_cta,
        meta.atom_n,
        32,
    )


def validate_tcgen05_ts_instruction(meta: TCGEN05Meta, a_dtype, transposed: bool) -> None:
    """Validate a dense TS instruction atom against CUTLASS SM100 wrappers."""

    if transposed:
        raise ValueError("TCGEN5MMA TS requires K-major TMEM A (transpose_A=False)")

    if meta.enable_2cta:
        if meta.atom_m not in (128, 256):
            raise ValueError(f"2CTA TCGEN5MMA TS instruction M must be 128 or 256, got {meta.atom_m}")
        if not (32 <= meta.atom_n <= 256 and meta.atom_n % 32 == 0):
            raise ValueError(f"2CTA TCGEN5MMA TS instruction N must be a multiple of 32 in [32, 256], got {meta.atom_n}")
        return

    if meta.atom_m not in (64, 128):
        raise ValueError(f"1CTA TCGEN5MMA TS instruction M must be 64 or 128, got {meta.atom_m}")
    dtype = get_tvm_dtype(a_dtype)
    is_int8 = str(dtype) in {"int8", "uint8"}
    if is_int8:
        legal_n = meta.atom_n == 8 or (16 <= meta.atom_n <= 256 and meta.atom_n % 16 == 0)
        requirement = "8 or a multiple of 16 in [16, 256]"
    elif meta.atom_m == 64:
        legal_n = 8 <= meta.atom_n <= 256 and meta.atom_n % 8 == 0
        requirement = "a multiple of 8 in [8, 256] for M=64"
    else:
        legal_n = 16 <= meta.atom_n <= 256 and meta.atom_n % 16 == 0
        requirement = "a multiple of 16 in [16, 256] for M=128"
    if not legal_n:
        raise ValueError(f"1CTA TCGEN5MMA TS instruction N must be {requirement}, got {meta.atom_n}")
