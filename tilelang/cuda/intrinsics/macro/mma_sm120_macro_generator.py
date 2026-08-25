from __future__ import annotations

from dataclasses import dataclass

import tilelang.language as T
from tilelang import tvm as tvm
from tvm.runtime import convert
from tvm.tirx import Buffer, BufferRegion, PrimExpr, Var

from ..layout.mma_layout import (
    ldmatrix_32x16_to_shared_8x64_layout_b,
    ldmatrix_32x32_to_shared_16x64_layout_a,
    ldmatrix_32x32_to_shared_16x64_layout_b,
)
from .mma_macro_generator import TensorCoreIntrinEmitter as MMAIntrinEmitter

lift = convert


@dataclass(frozen=True)
class BlockScaleMmaConfig:
    """Static SM120 warp-level block-scale MMA configuration."""

    kind: str
    mma_prefix: str
    atom_k: int
    scale_vec_size: int
    sf_vec_size: int
    scale_type: str
    a_dtype_abbrv: str
    b_dtype_abbrv: str
    # Accessing T.float32 here would create a circular import during language initialization.
    accum_dtype: str = "float32"
    active_sfa_threads: int = 16
    active_sfb_threads: int = 8


@dataclass(frozen=True)
class SM120BlockScaleTile:
    """Validated tile geometry for the SM120 packed-scale register pipeline."""

    tile_m: int
    tile_n: int
    tile_k: int
    block_row_warps: int
    block_col_warps: int
    warp_rows: int
    warp_cols: int
    warp_row_tiles: int
    warp_col_tiles: int
    kblocks: int
    micro_size_m: int
    micro_size_n: int
    micro_size_k: int
    sf_layout: str
    sfa_words: int
    sfb_words: int
    warp_issues: int
    warpgroup_issues: int
    # K elements covered by one scale byte (16 for 4X modes, 32 for 2X).
    sf_vec_size: int = 16

    @classmethod
    def from_emitter(
        cls,
        emitter: TensorCoreIntrinEmitterSM120,
        *,
        sf_layout: str,
        kblocks: int | None = None,
    ) -> SM120BlockScaleTile:
        if kblocks is None:
            kblocks = int(emitter.chunk // emitter.micro_size_k)
        tile = cls(
            tile_m=int(emitter.block_row_warps) * int(emitter.warp_row_tiles),
            tile_n=int(emitter.block_col_warps) * int(emitter.warp_col_tiles),
            tile_k=kblocks * int(emitter.micro_size_k),
            block_row_warps=int(emitter.block_row_warps),
            block_col_warps=int(emitter.block_col_warps),
            warp_rows=int(emitter.warp_rows),
            warp_cols=int(emitter.warp_cols),
            warp_row_tiles=int(emitter.warp_row_tiles),
            warp_col_tiles=int(emitter.warp_col_tiles),
            kblocks=kblocks,
            micro_size_m=int(emitter.micro_size_x),
            micro_size_n=int(emitter.micro_size_y),
            micro_size_k=int(emitter.micro_size_k),
            sf_layout=sf_layout,
            sfa_words=(int(emitter.warp_rows) + 1) // 2,
            sfb_words=(int(emitter.warp_cols) + 1) // 2,
            warp_issues=int(emitter.warp_rows) * int(emitter.warp_cols) * 2,
            warpgroup_issues=(
                int(emitter.warp_rows) * int(emitter.warp_cols) * 2 * int(emitter.block_row_warps) * int(emitter.block_col_warps)
            ),
            sf_vec_size=int(emitter.sf_vec_size),
        )
        tile.validate()
        return tile

    def validate(self) -> None:
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            raise ValueError(f"SM120 full-tile package dimensions must be positive, got {self.tile_m}x{self.tile_n}x{self.tile_k}")
        if self.block_row_warps <= 0 or self.block_col_warps <= 0:
            raise ValueError(
                f"SM120 full-tile package requires positive warp partition dimensions, got {self.block_row_warps}x{self.block_col_warps}"
            )
        if self.tile_m != self.block_row_warps * self.warp_row_tiles:
            raise ValueError(
                f"SM120 full-tile M shape mismatch: tile_m={self.tile_m}, "
                f"block_row_warps={self.block_row_warps}, warp_row_tiles={self.warp_row_tiles}"
            )
        if self.tile_n != self.block_col_warps * self.warp_col_tiles:
            raise ValueError(
                f"SM120 full-tile N shape mismatch: tile_n={self.tile_n}, "
                f"block_col_warps={self.block_col_warps}, warp_col_tiles={self.warp_col_tiles}"
            )
        # One compact scale word is shared by each adjacent pair of MMA atoms.
        if self.warp_rows <= 0 or self.warp_cols <= 0 or self.warp_rows % 2 != 0 or self.warp_cols % 2 != 0:
            raise ValueError(
                f"SM120 compact scale packages require a positive even MMA atom grid per warp, got {self.warp_rows}x{self.warp_cols}"
            )
        if self.warp_row_tiles != self.warp_rows * self.micro_size_m:
            raise ValueError(
                f"SM120 full-tile warp M shape mismatch: warp_row_tiles={self.warp_row_tiles}, "
                f"warp_rows={self.warp_rows}, micro_size_m={self.micro_size_m}"
            )
        if self.warp_col_tiles != self.warp_cols * self.micro_size_n:
            raise ValueError(
                f"SM120 full-tile warp N shape mismatch: warp_col_tiles={self.warp_col_tiles}, "
                f"warp_cols={self.warp_cols}, micro_size_n={self.micro_size_n}"
            )
        if self.tile_m % 128 != 0 or self.tile_n % 128 != 0:
            raise ValueError(
                "SM120 kmajor scale sources stack 128-row BlockScaledBasicChunk atoms, so "
                f"block M/N must be multiples of 128, got tile_m={self.tile_m}, tile_n={self.tile_n}"
            )
        if self.kblocks <= 0 or self.micro_size_k <= 0 or self.tile_k != self.kblocks * self.micro_size_k:
            raise ValueError(
                f"SM120 full-tile K shape mismatch: tile_k={self.tile_k}, kblocks={self.kblocks}, micro_size_k={self.micro_size_k}"
            )
        if self.sf_layout != "blockscaled_chunk_kmajor":
            raise ValueError("SM120 full-tile package contract requires sf_layout='blockscaled_chunk_kmajor'")
        if self.sf_vec_size <= 0 or self.tile_k % (self.sf_vec_size * 4) != 0:
            raise ValueError(
                f"SM120 kmajor scale words pack {self.sf_vec_size * 4} K elements; "
                f"tile_k must be a multiple of that, got tile_k={self.tile_k}"
            )
        expected_sfa_words = (self.warp_rows + 1) // 2
        expected_sfb_words = (self.warp_cols + 1) // 2
        if (self.sfa_words, self.sfb_words) != (expected_sfa_words, expected_sfb_words):
            raise ValueError(
                "SM120 compact scale package word count mismatch: expected "
                f"{expected_sfa_words} SFA and {expected_sfb_words} SFB words, got "
                f"{self.sfa_words}, {self.sfb_words}"
            )
        expected_issues_per_warp = self.warp_rows * self.warp_cols * 2
        expected_issues_per_warpgroup = expected_issues_per_warp * self.block_row_warps * self.block_col_warps
        if self.warp_issues != expected_issues_per_warp or self.warpgroup_issues != expected_issues_per_warpgroup:
            raise ValueError(
                "SM120 full-tile issue count mismatch: expected "
                f"{expected_issues_per_warp} per warp and {expected_issues_per_warpgroup} per warpgroup, got "
                f"{self.warp_issues}, {self.warpgroup_issues}"
            )

    @property
    def words_per_stage(self) -> int:
        """uint32 words per staged (tile_rows, tile_k) scale slice row span.

        One word packs four scale bytes of ``sf_vec_size`` K elements each,
        so a tile_k stage holds ``tile_k // (sf_vec_size * 4)`` words.
        """

        return self.tile_k // (self.sf_vec_size * 4)

    def _scale_word_offset(self, row: int, kblock: int, tile_rows: int) -> int:
        """Return the uint32 scale-word offset for the source/smem scale layout."""

        if tile_rows <= 0 or tile_rows % 128 != 0:
            raise ValueError(f"scale tile rows must be a positive multiple of 128, got {tile_rows}")
        if row < 0 or row >= tile_rows:
            raise ValueError(f"scale row must be in [0, {tile_rows}), got {row}")
        if kblock < 0 or kblock >= self.words_per_stage:
            raise ValueError(f"kblock must be a word column in [0, {self.words_per_stage}), got {kblock}")
        # The packer stacks 128-row K-major atoms vertically; inside one atom
        # rows are stored as [word][row % 32][row // 32], regardless of the
        # staged tile height.
        return (row // 128) * (self.words_per_stage * 128) + kblock * 128 + (row % 32) * 4 + (row % 128) // 32

    def compact_selector_scale_rows(self, lane: int, warp_m: int, warp_n: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return SFA/SFB semantic rows loaded by the current compact TV package."""

        if lane < 0 or lane >= 32:
            raise ValueError(f"lane must be in [0, 32), got {lane}")
        if warp_m < 0 or warp_m >= self.block_row_warps:
            raise ValueError(f"warp_m must be in [0, {self.block_row_warps}), got {warp_m}")
        if warp_n < 0 or warp_n >= self.block_col_warps:
            raise ValueError(f"warp_n must be in [0, {self.block_col_warps}), got {warp_n}")

        qlane = lane & 3
        sfa_row = 8 * (lane & 1) + (lane >> 2)
        sfb_col = lane >> 2
        a_owner_in_pair = qlane >> 1
        scale_m0 = warp_m * self.warp_row_tiles + a_owner_in_pair * 16 + sfa_row
        scale_n0 = warp_n * self.warp_col_tiles + qlane * 8 + sfb_col
        return (
            tuple(scale_m0 + g * 32 for g in range(self.sfa_words)),
            tuple(scale_n0 + g * 32 for g in range(self.sfb_words)),
        )

    def compact_selector_scale_word_offsets(
        self, lane: int, warp_m: int, warp_n: int, kblock: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        sfa_rows, sfb_rows = self.compact_selector_scale_rows(lane, warp_m, warp_n)
        return (
            tuple(self._scale_word_offset(row, kblock, self.tile_m) for row in sfa_rows),
            tuple(self._scale_word_offset(row, kblock, self.tile_n) for row in sfb_rows),
        )

    @staticmethod
    def sfa_selector_source_lane(lane: int, scale_a_thread_id: int) -> int:
        return (lane & ~3) | ((scale_a_thread_id & 1) << 1) | (lane & 1)

    @staticmethod
    def sfb_selector_source_lane(lane: int, scale_b_thread_id: int) -> int:
        return (lane & ~3) | (scale_b_thread_id & 3)

    def compact_selector_effective_rows(
        self, lane: int, warp_m: int, warp_n: int, issue: tuple[int, int, int, int, int, int, int]
    ) -> tuple[int, int]:
        """Return semantic SFA/SFB rows consumed by one compact-selector issue."""

        mma_i, mma_j, n8_half, sfa_word, sfb_word, scale_a_tid, scale_b_tid = issue
        del mma_i, mma_j, n8_half
        sfa_source_lane = self.sfa_selector_source_lane(lane, scale_a_tid)
        sfb_source_lane = self.sfb_selector_source_lane(lane, scale_b_tid)
        sfa_rows, _ = self.compact_selector_scale_rows(sfa_source_lane, warp_m, warp_n)
        _, sfb_rows = self.compact_selector_scale_rows(sfb_source_lane, warp_m, warp_n)
        return sfa_rows[sfa_word], sfb_rows[sfb_word]

    def package_pingpong_lifecycle(self) -> tuple[tuple[str, int, int], ...]:
        """Return the current copy/gemm package lifecycle from the CUDA helper.

        Each tuple is ``(op, register_package, kblock)``.  Scale and A/B packages
        share the same register package id in the current implementation.
        """

        lifecycle = [("copy", kblock, kblock) for kblock in range(min(self.kblocks, 2))]
        for kblock in range(self.kblocks):
            package_id = kblock & 1
            lifecycle.append(("gemm", package_id, kblock))
            next_kblock = kblock + 2
            if next_kblock < self.kblocks:
                lifecycle.append(("copy", package_id, next_kblock))
        return tuple(lifecycle)

    def omma_sf_issue_schedule_per_warp(self) -> tuple[tuple[int, int, int, int, int, int, int], ...]:
        """Return the current per-warp OMMA.SF issue schedule.

        Each tuple is ``(mma_i, mma_j, n8_half, sfa_word, sfb_word,
        scale_a_thread_id, scale_b_thread_id)``.  ``sfa_word`` and ``sfb_word``
        are indices inside the current compact-selector scale package, not
        source-memory offsets.
        """

        issues = []
        for mma_i in range(self.warp_rows):
            sfa_word = mma_i // 2
            scale_a_thread_id = mma_i & 1
            for mma_j in range(self.warp_cols):
                sfb_word = mma_j // 2
                for n8_half in range(2):
                    scale_b_thread_id = (mma_j & 1) * 2 + n8_half
                    issues.append(
                        (
                            mma_i,
                            mma_j,
                            n8_half,
                            sfa_word,
                            sfb_word,
                            scale_a_thread_id,
                            scale_b_thread_id,
                        )
                    )
        return tuple(issues)


_SUPPORTED_BLOCK_SCALE_MMA_CONFIGS = {
    ("mxf4nvf4", 4, "ue4m3"): BlockScaleMmaConfig(
        kind="mxf4nvf4",
        mma_prefix="m16n8k64",
        atom_k=64,
        scale_vec_size=4,
        sf_vec_size=16,
        scale_type="ue4m3",
        a_dtype_abbrv="e2m1",
        b_dtype_abbrv="e2m1",
    ),
    # MXFP4: 32-element scale granularity, power-of-two (ue8m0) scales. One
    # uint32 scale word covers K=128; each k64 MMA atom consumes a 2-byte
    # half selected by scale_*_byte_id.
    ("mxf4nvf4", 2, "ue8m0"): BlockScaleMmaConfig(
        kind="mxf4nvf4",
        mma_prefix="m16n8k64",
        atom_k=64,
        scale_vec_size=2,
        sf_vec_size=32,
        scale_type="ue8m0",
        a_dtype_abbrv="e2m1",
        b_dtype_abbrv="e2m1",
    ),
    # NVF4 granularity with ue8m0 scales; the instruction needs CUDA 13.1+
    # (see mma_block_scale.h), word/byte accounting matches the ue4m3 form.
    ("mxf4nvf4", 4, "ue8m0"): BlockScaleMmaConfig(
        kind="mxf4nvf4",
        mma_prefix="m16n8k64",
        atom_k=64,
        scale_vec_size=4,
        sf_vec_size=16,
        scale_type="ue8m0",
        a_dtype_abbrv="e2m1",
        b_dtype_abbrv="e2m1",
    ),
}


def _get_block_scale_mma_config(kind: str, scale_vec_size: int, scale_type: str) -> BlockScaleMmaConfig:
    key = (kind, scale_vec_size, scale_type)
    if key not in _SUPPORTED_BLOCK_SCALE_MMA_CONFIGS:
        supported = ", ".join(str(k) for k in sorted(_SUPPORTED_BLOCK_SCALE_MMA_CONFIGS))
        raise ValueError(f"Unsupported SM120 block-scale MMA config {key}; supported: {supported}")
    return _SUPPORTED_BLOCK_SCALE_MMA_CONFIGS[key]


class TensorCoreIntrinEmitterSM120(MMAIntrinEmitter):
    """Warp-level MMA emitter, with optional SM120 block-scale mode.

    The block-scale mode keeps scale-factor storage explicit, matching
    TileLang's TCGEN05 block-scaled style while targeting warp-level
    ``mma.sync``.
    """

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
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
        is_blockscaled: bool = False,
        kind: str = "mxf4nvf4",
        scale_vec_size: int = 4,
        stype: str = "ue4m3",
    ):
        self.is_blockscaled = is_blockscaled
        if is_blockscaled:
            self.block_scale_config = _get_block_scale_mma_config(kind, scale_vec_size, stype)
            a_dtype_abbrv = self._get_dtype_abbrv(str(a_dtype))
            b_dtype_abbrv = self._get_dtype_abbrv(str(b_dtype))
            if (
                a_dtype_abbrv != self.block_scale_config.a_dtype_abbrv
                or b_dtype_abbrv != self.block_scale_config.b_dtype_abbrv
                or str(accum_dtype) != self.block_scale_config.accum_dtype
            ):
                raise ValueError(
                    f"{self.block_scale_config.kind} expects a_dtype={self.block_scale_config.a_dtype_abbrv}, "
                    f"b_dtype={self.block_scale_config.b_dtype_abbrv}, "
                    f"accum_dtype={self.block_scale_config.accum_dtype}; "
                    f"got a_dtype={a_dtype}, b_dtype={b_dtype}, accum_dtype={accum_dtype}"
                )
            self.kind = self.block_scale_config.kind
            self.scale_vec_size = self.block_scale_config.scale_vec_size
            self.stype = self.block_scale_config.scale_type
            self.sf_vec_size = self.block_scale_config.sf_vec_size
            if int(chunk) % (self.sf_vec_size * 4) != 0:
                raise ValueError(
                    f"{self.kind} scale_vec::{self.scale_vec_size}X packs "
                    f"{self.sf_vec_size * 4} K elements per uint32 scale word; "
                    f"block_K must be a multiple of that, got block_K={chunk}"
                )
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            is_m_first=is_m_first,
            thread_var=thread_var,
        )

    def _initialize_k_dim(self, a_dtype="float16"):
        if self.is_blockscaled:
            self.k_dim = self.block_scale_config.atom_k
        else:
            super()._initialize_k_dim(a_dtype)

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        if self.is_blockscaled:
            self.a_dtype_abbrv = self.block_scale_config.a_dtype_abbrv
            self.b_dtype_abbrv = self.block_scale_config.b_dtype_abbrv
            self.accum_dtype_abbrv = self._get_dtype_abbrv(accum_dtype)
        else:
            super()._initialize_abbrev(a_dtype, b_dtype, accum_dtype)

    def _initialize_mma_prefix(self, k_dim: int = 16):
        if self.is_blockscaled:
            self.mma_prefix = self.block_scale_config.mma_prefix
        else:
            super()._initialize_mma_prefix(k_dim)

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        if not self.is_blockscaled:
            return super().ldmatrix_a(A_local_buf, A_shared_buf, ki, rk)
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_transposed = self.a_transposed

        thread_binding = self.get_thread_binding()
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]

        @T.macro
        def _warp_ld_a_e2m1(A_local_buf, A_shared_buf, ki, thread_binding, rk=0):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            for i in T.unroll(warp_rows):
                wi = warp_m * warp_row_tiles + i * micro_size_x
                wk = rk * chunk + ki * micro_size_k
                row_off, col_off = ldmatrix_32x32_to_shared_16x64_layout_a(tx)
                if a_transposed:
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4,
                        T.access_ptr(
                            A_buf[tuple(A_other) + (A_base0 + wk + row_off, A_base1 + wi + col_off)],
                            "r",
                            extent=local_size_a,
                        ),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=local_size_a),
                    )
                else:
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4,
                        T.access_ptr(
                            A_buf[tuple(A_other) + (A_base0 + wi + row_off, A_base1 + wk + col_off)],
                            "r",
                            extent=local_size_a,
                        ),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=local_size_a),
                    )

        return _warp_ld_a_e2m1(A_local_buf, A_region, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf: Buffer, B_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        if not self.is_blockscaled:
            return super().ldmatrix_b(B_local_buf, B_shared_buf, ki, rk)
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_transposed = self.b_transposed
        replicate_b = self.n_dim == 16

        thread_binding = self.get_thread_binding()
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]

        @T.macro
        def _warp_ld_b_e2m1(B_local_buf, B_shared_buf, ki, thread_binding, rk=0):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            for i in T.unroll(warp_cols):
                wi = warp_n * warp_col_tiles + i * micro_size_y
                wk = rk * chunk + ki * micro_size_k
                if replicate_b:
                    row_off, col_off = ldmatrix_32x32_to_shared_16x64_layout_b(tx)
                else:
                    row_off, col_off = ldmatrix_32x16_to_shared_8x64_layout_b(tx)
                if b_transposed:
                    T.ptx_ldmatrix(
                        T.bool(False),
                        4 if replicate_b else 2,
                        T.access_ptr(
                            B_buf[tuple(B_other) + (B_base0 + wi + row_off, B_base1 + wk + col_off)],
                            "r",
                            extent=local_size_b,
                        ),
                        T.access_ptr(B_local_buf[i * local_size_b], "w", extent=local_size_b),
                    )
                else:
                    T.ptx_ldmatrix(
                        T.bool(True),
                        4 if replicate_b else 2,
                        T.access_ptr(
                            B_buf[tuple(B_other) + (B_base0 + wk + row_off, B_base1 + wi + col_off)],
                            "r",
                            extent=local_size_b,
                        ),
                        T.access_ptr(B_local_buf[i * local_size_b], "w", extent=local_size_b),
                    )

        return _warp_ld_b_e2m1(B_local_buf, B_region, ki, thread_binding, rk)

    def _scale_region_parts(self, scale_buf: Buffer | BufferRegion):
        if isinstance(scale_buf, BufferRegion):
            scale_region = scale_buf
        elif isinstance(scale_buf, Buffer):
            scale_region = self._legalize_to_buffer_region(scale_buf)
        else:
            raise ValueError(f"Unsupported scale buffer type: {type(scale_buf)}")
        return (
            scale_region.buffer,
            [r.min for r in scale_region.region[:-2]],
            scale_region.region[-2].min,
            scale_region.region[-1].min,
        )

    @staticmethod
    def _sfa_row_in_atom(tx: PrimExpr):
        # CUTLASS SFALayout for k64 uses ((2,2,8),64), stride ((8,0,1),16).
        # With K-major flattening, the M coordinate is 8 * (lane % 2) + lane // 4.
        return 8 * (tx % 2) + (tx // 4)

    @staticmethod
    def _sfb_col_in_atom(tx: PrimExpr):
        # CUTLASS SFBLayout for k64 uses ((4,8),64), stride ((0,1),8), so the
        # logical N coordinate is lane // 4 with broadcast across four groups.
        return tx // 4

    def _scale_word_k(self, k_start: PrimExpr, ki: PrimExpr, sf_granularity_k: int):
        packed_word_k = int(sf_granularity_k) * 4
        if packed_word_k != self.sf_vec_size * 4:
            raise ValueError(
                f"{self.kind} expects packed scale words covering {self.sf_vec_size * 4} K elements, "
                f"got sf_granularity_k={sf_granularity_k}"
            )
        _k_start = tvm.tirx.const(k_start, "int32") if isinstance(k_start, int) else k_start
        return (_k_start + self.micro_size_k * ki) // packed_word_k

    def _scale_byte_id(self, k_start: PrimExpr, ki: PrimExpr):
        """Byte offset of the consumed scale bytes inside their uint32 word.

        scale_vec::4X consumes the whole word, so the byte id is always the
        literal 0 (keeping 4X codegen byte-identical). scale_vec::2X consumes
        a 2-byte half selected by the k64 atom parity inside the K=128 word.
        """

        if self.scale_vec_size == 4:
            return 0
        word_span = self.sf_vec_size * 4
        _k_start = tvm.tirx.const(k_start, "int32") if isinstance(k_start, int) else k_start
        return ((_k_start + self.micro_size_k * ki) % word_span) // self.sf_vec_size

    @staticmethod
    def _tile_kmajor_scale_word(idx: PrimExpr, word_k: PrimExpr, words_per_stage: int):
        """Return a tile-local compact K-major scale-word offset.

        The scale source stacks 128-row BlockScaledBasicChunk atoms
        vertically: the in-atom order is the packer's fixed 4-group split of
        128 rows regardless of the staged tile height, and rows beyond the
        first atom live ``words_per_stage * 128`` words further.
        """

        return (idx // 128) * (words_per_stage * 128) + word_k * 128 + (idx % 32) * 4 + (idx % 128) // 32

    def mma(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        k_inner: PrimExpr | None = 0,
        *,
        SFA_buf=None,
        SFB_buf=None,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        # Keep the base-class positional signature (A, B, C, k_inner): the
        # non-blockscaled gemm lowering calls mma(A_local, B_local, C_buf, ki).
        if not self.is_blockscaled:
            if SFA_buf is not None or SFB_buf is not None:
                raise ValueError("Scale buffers require TensorCoreIntrinEmitterSM120 block-scale mode")
            return super().mma(A_local_buf, B_local_buf, C_local_buf, k_inner)
        if SFA_buf is None or SFB_buf is None:
            raise ValueError("Block-scaled MMA requires SFA and SFB buffers")
        if sf_layout != "rowmajor":
            raise ValueError(
                "mma() supports sf_layout='rowmajor' only; blockscaled_chunk_kmajor scales are handled by mma_blockscaled_fulltile"
            )
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, k_inner, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, k_inner, sf_b_granularity_k)
        scale_byte_id = self._scale_byte_id(k_start, k_inner)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_block_scale(A_local_buf, B_local_buf, C_local_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            for i, j in T.grid(warp_rows, warp_cols):
                scale_m = warp_m * warp_row_tiles + i * micro_size_x + sfa_row
                scale_n = warp_n * warp_col_tiles + j * micro_size_y + sfb_col
                scale_a_ptr = T.access_ptr(
                    SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_k)],
                    "r",
                )
                scale_b_ptr = T.access_ptr(
                    SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n, SFB_base_k + scale_b_word_k)],
                    "r",
                )
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    scale_a_ptr,
                    scale_b_ptr,
                    scale_byte_id,
                    0,
                    scale_byte_id,
                    0,
                )
                if replicate_b:
                    scale_b_rep_ptr = T.access_ptr(
                        SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n + 8, SFB_base_k + scale_b_word_k)],
                        "r",
                    )
                    T.ptx_mma_block_scale(
                        accum_dtype,
                        mma_prefix,
                        "row",
                        "col",
                        kind,
                        scale_vec_size,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        stype,
                        A_local_buf.data,
                        i * local_size_a,
                        B_local_buf.data,
                        j * local_size_b + lift(local_size_b) // 2,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                        scale_a_ptr,
                        scale_b_rep_ptr,
                        scale_byte_id,
                        0,
                        scale_byte_id,
                        0,
                    )

        return _warp_mma_block_scale(A_local_buf, B_local_buf, C_local_buf, SFA_data, SFB_data, thread_binding)

    def ldscale(
        self,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
        SFA_buf,
        SFB_buf,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        if sf_layout != "rowmajor":
            raise ValueError(
                "ldscale() supports sf_layout='rowmajor' only; blockscaled_chunk_kmajor scales are handled by mma_blockscaled_fulltile"
            )
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        sf_vec_size = self.sf_vec_size
        sf_a_granularity_k = sf_vec_size if sf_a_granularity_k is None else sf_a_granularity_k
        sf_b_granularity_k = sf_vec_size if sf_b_granularity_k is None else sf_b_granularity_k
        scale_a_word_k = self._scale_word_k(k_start, ki, sf_a_granularity_k)
        scale_b_word_k = self._scale_word_k(k_start, ki, sf_b_granularity_k)
        thread_binding = self.get_thread_binding()
        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_ldscale_block_scale(SFA_local_buf, SFB_local_buf, SFB_rep_local_buf, SFA_data, SFB_data, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            for i in T.unroll(warp_rows):
                scale_m = warp_m * warp_row_tiles + i * micro_size_x + sfa_row
                SFA_local_buf[i] = SFA_data[tuple(SFA_other) + (SFA_base_m + scale_m, SFA_base_k + scale_a_word_k)]
            for j in T.unroll(warp_cols):
                scale_n = warp_n * warp_col_tiles + j * micro_size_y + sfb_col
                SFB_local_buf[j] = SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n, SFB_base_k + scale_b_word_k)]
                if replicate_b:
                    SFB_rep_local_buf[j] = SFB_data[tuple(SFB_other) + (SFB_base_n + scale_n + 8, SFB_base_k + scale_b_word_k)]

        return _warp_ldscale_block_scale(
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
            SFA_data,
            SFB_data,
            thread_binding,
        )

    def ldscale_fragment(
        self,
        SFA_fragment_buf,
        SFB_fragment_buf,
        SFB_rep_fragment_buf,
        SFA_buf,
        SFB_buf,
        ki: PrimExpr = 0,
        k_start: PrimExpr = 0,
        sf_a_granularity_k: int | None = None,
        sf_b_granularity_k: int | None = None,
        sf_layout: str = "rowmajor",
    ):
        """Load SM120 block-scale fragments into local registers.

        This is currently a thin wrapper over the existing scale-word load.
        The separate name gives the SM120 MMA lowering a stable hook for a
        CUTLASS-like scale-fragment copy path.
        """
        return self.ldscale(
            SFA_fragment_buf,
            SFB_fragment_buf,
            SFB_rep_fragment_buf,
            SFA_buf,
            SFB_buf,
            ki=ki,
            k_start=k_start,
            sf_a_granularity_k=sf_a_granularity_k,
            sf_b_granularity_k=sf_b_granularity_k,
            sf_layout=sf_layout,
        )

    def mma_blockscaled_fulltile(
        self,
        A_shared_buf: Buffer | BufferRegion,
        B_shared_buf: Buffer | BufferRegion,
        C_local_buf: Buffer,
        SFA_buf: Buffer | BufferRegion,
        SFB_buf: Buffer | BufferRegion,
        sf_layout: str = "rowmajor",
    ):
        """Emit an SM120 full-tile block-scaled MMA register micro-pipeline."""
        if self.n_dim != 16:
            raise ValueError("sm120 full-tile MMA requires replicated B n_dim=16")
        if not self.b_transposed:
            raise ValueError("sm120 full-tile MMA currently requires transpose_B=True")
        if sf_layout != "blockscaled_chunk_kmajor":
            raise ValueError("sm120 full-tile MMA currently requires sf_layout='blockscaled_chunk_kmajor'")
        k_blocks = int(self.chunk // self.micro_size_k)
        # Words per staged (tile_rows, block_K) scale slice: one uint32 packs
        # four scale bytes covering sf_vec_size K elements each.
        words_per_stage = int(self.chunk // (self.sf_vec_size * 4))
        # k64 MMA atoms served by one scale word (4X: 1, 2X: 2 via byte halves).
        atoms_per_word = int((self.sf_vec_size * 4) // self.micro_size_k)
        tile = SM120BlockScaleTile.from_emitter(
            self,
            sf_layout=sf_layout,
            kblocks=k_blocks,
        )

        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        sfa_words = tile.sfa_words
        sfb_words = tile.sfb_words
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        accum_dtype = self.accum_dtype
        mma_prefix = self.mma_prefix
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        stype = self.stype
        thread_binding = self.get_thread_binding()

        A_region = self._legalize_to_buffer_region(A_shared_buf)
        B_region = self._legalize_to_buffer_region(B_shared_buf)

        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)

        @T.macro
        def _load_kblock(A_local_buf, B_local_buf, SFA_local_buf, SFB_local_buf, k_block):
            self.ldmatrix_a(A_local_buf, A_region, k_block)
            self.ldmatrix_b(B_local_buf, B_region, k_block)

            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            qlane = tx % 4
            sfa_row = self._sfa_row_in_atom(tx)
            sfb_col = self._sfb_col_in_atom(tx)
            scale_m0 = warp_m * self.warp_row_tiles + (qlane // 2) * 16 + sfa_row
            scale_n0 = warp_n * self.warp_col_tiles + qlane * 8 + sfb_col
            k_word = k_block if atoms_per_word == 1 else k_block // atoms_per_word
            for g in T.unroll(sfa_words):
                scale_a_word = self._tile_kmajor_scale_word(scale_m0 + g * 32, k_word, words_per_stage)
                SFA_local_buf[g] = SFA_data[
                    tuple(SFA_other) + (SFA_base_m + scale_a_word // words_per_stage, SFA_base_k + scale_a_word % words_per_stage)
                ]
            for g in T.unroll(sfb_words):
                scale_b_word = self._tile_kmajor_scale_word(scale_n0 + g * 32, k_word, words_per_stage)
                SFB_local_buf[g] = SFB_data[
                    tuple(SFB_other) + (SFB_base_n + scale_b_word // words_per_stage, SFB_base_k + scale_b_word % words_per_stage)
                ]

        @T.macro
        def _mma_kblock(A_local_buf, B_local_buf, SFA_local_buf, SFB_local_buf, C_local_buf, k_block):
            scale_byte = 0 if atoms_per_word == 1 else (k_block % atoms_per_word) * 2
            for i in T.unroll(warp_rows):
                scale_a_ptr = T.access_ptr(SFA_local_buf[i // 2], "r")
                for j in T.unroll(warp_cols):
                    scale_b_ptr = T.access_ptr(SFB_local_buf[j // 2], "r")
                    for n8_half in T.unroll(2):
                        T.ptx_mma_block_scale(
                            accum_dtype,
                            mma_prefix,
                            "row",
                            "col",
                            kind,
                            scale_vec_size,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            stype,
                            A_local_buf.data,
                            i * local_size_a,
                            B_local_buf.data,
                            j * local_size_b + n8_half * (local_size_b // 2),
                            C_local_buf.data,
                            i * warp_cols * local_size_out + j * local_size_out + n8_half * (local_size_out // 2),
                            scale_a_ptr,
                            scale_b_ptr,
                            scale_byte,
                            i % 2,
                            scale_byte,
                            (j % 2) * 2 + n8_half,
                        )

        # Ping-pong normally preloads K blocks 0 and 1. A single block needs its
        # own path to avoid an out-of-range preload and unused register package.
        if k_blocks == 1:

            @T.macro
            def _warp_mma_blockscaled_fulltile(C_local_buf):
                A_local_0 = T.alloc_local((warp_rows * local_size_a,), a_dtype)
                B_local_0 = T.alloc_local((warp_cols * local_size_b,), b_dtype)
                SFA_local_0 = T.alloc_local((sfa_words,), "uint32")
                SFB_local_0 = T.alloc_local((sfb_words,), "uint32")

                _load_kblock(A_local_0, B_local_0, SFA_local_0, SFB_local_0, 0)
                _mma_kblock(A_local_0, B_local_0, SFA_local_0, SFB_local_0, C_local_buf, 0)

        else:

            @T.macro
            def _warp_mma_blockscaled_fulltile(C_local_buf):
                A_local_0 = T.alloc_local((warp_rows * local_size_a,), a_dtype)
                A_local_1 = T.alloc_local((warp_rows * local_size_a,), a_dtype)
                B_local_0 = T.alloc_local((warp_cols * local_size_b,), b_dtype)
                B_local_1 = T.alloc_local((warp_cols * local_size_b,), b_dtype)
                SFA_local_0 = T.alloc_local((sfa_words,), "uint32")
                SFA_local_1 = T.alloc_local((sfa_words,), "uint32")
                SFB_local_0 = T.alloc_local((sfb_words,), "uint32")
                SFB_local_1 = T.alloc_local((sfb_words,), "uint32")

                _load_kblock(A_local_0, B_local_0, SFA_local_0, SFB_local_0, 0)
                _load_kblock(A_local_1, B_local_1, SFA_local_1, SFB_local_1, 1)
                # Refill a package with k+2 only after its current K block has issued.
                for k_block in T.unroll(k_blocks):
                    if k_block % 2 == 0:
                        _mma_kblock(A_local_0, B_local_0, SFA_local_0, SFB_local_0, C_local_buf, k_block)
                        if k_block + 2 < k_blocks:
                            _load_kblock(A_local_0, B_local_0, SFA_local_0, SFB_local_0, k_block + 2)
                    else:
                        _mma_kblock(A_local_1, B_local_1, SFA_local_1, SFB_local_1, C_local_buf, k_block)
                        if k_block + 2 < k_blocks:
                            _load_kblock(A_local_1, B_local_1, SFA_local_1, SFB_local_1, k_block + 2)

        return _warp_mma_blockscaled_fulltile(C_local_buf)

    def mma_full_b_atom_with_scale_fragments(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_fragment_buf,
        SFB_fragment_buf,
        SFB_rep_fragment_buf,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
        *,
        scale_byte_id: PrimExpr | int = 0,
    ):
        """Issue one SM120 block-scaled MMA atom from a full B fragment tile."""
        return self.mma_full_b_atom_with_prefetched_scales(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_fragment_buf,
            SFB_fragment_buf,
            SFB_rep_fragment_buf,
            inst_m_idx,
            inst_n_idx,
            scale_byte_id=scale_byte_id,
        )

    def mma_full_b_atom_with_prefetched_scales(
        self,
        A_local_buf,
        B_local_buf,
        C_local_buf,
        SFA_local_buf,
        SFB_local_buf,
        SFB_rep_local_buf,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
        *,
        scale_byte_id: PrimExpr | int = 0,
    ):
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        kind = self.kind
        scale_vec_size = self.scale_vec_size
        stype = self.stype
        accum_dtype = self.accum_dtype
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        mma_prefix = self.mma_prefix
        warp_cols = self.warp_cols
        replicate_b = self.n_dim == 16

        @T.macro
        def _warp_mma_block_scale_full_b_atom_prefetched(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        ):
            scale_a_ptr = T.access_ptr(SFA_local_buf[inst_m_idx], "r")
            scale_b_ptr = T.access_ptr(SFB_local_buf[inst_n_idx], "r")
            T.ptx_mma_block_scale(
                accum_dtype,
                mma_prefix,
                "row",
                "col",
                kind,
                scale_vec_size,
                a_dtype_abbrv,
                b_dtype_abbrv,
                stype,
                A_local_buf.data,
                inst_m_idx * local_size_a,
                B_local_buf.data,
                inst_n_idx * local_size_b,
                C_local_buf.data,
                inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out,
                scale_a_ptr,
                scale_b_ptr,
                scale_byte_id,
                0,
                scale_byte_id,
                0,
            )
            if replicate_b:
                scale_b_rep_ptr = T.access_ptr(SFB_rep_local_buf[inst_n_idx], "r")
                T.ptx_mma_block_scale(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    kind,
                    scale_vec_size,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    stype,
                    A_local_buf.data,
                    inst_m_idx * local_size_a,
                    B_local_buf.data,
                    inst_n_idx * local_size_b + lift(local_size_b) // 2,
                    C_local_buf.data,
                    inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out + lift(local_size_out) // 2,
                    scale_a_ptr,
                    scale_b_rep_ptr,
                    scale_byte_id,
                    0,
                    scale_byte_id,
                    0,
                )

        return _warp_mma_block_scale_full_b_atom_prefetched(
            A_local_buf,
            B_local_buf,
            C_local_buf,
            SFA_local_buf,
            SFB_local_buf,
            SFB_rep_local_buf,
        )
