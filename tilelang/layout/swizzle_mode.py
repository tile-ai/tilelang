"""Shared memory swizzle mode."""

from __future__ import annotations

from typing import ClassVar

from tvm_ffi.dataclasses import Enum

from tilelang import _ffi_api


class SwizzleMode(Enum, type_key="tl.SwizzleMode"):
    # Bare ClassVar binders attach to the C++-registered variants.
    NONE: ClassVar[SwizzleMode]
    SWIZZLE_32B: ClassVar[SwizzleMode]
    SWIZZLE_64B: ClassVar[SwizzleMode]
    SWIZZLE_128B: ClassVar[SwizzleMode]

    def is_none(self) -> bool:
        return self.same_as(SwizzleMode.NONE)

    def is_swizzle_32b(self) -> bool:
        return self.same_as(SwizzleMode.SWIZZLE_32B)

    def is_swizzle_64b(self) -> bool:
        return self.same_as(SwizzleMode.SWIZZLE_64B)

    def is_swizzle_128b(self) -> bool:
        return self.same_as(SwizzleMode.SWIZZLE_128B)

    def wgmma_layout_type(self) -> int:
        """WGMMA descriptor ``layout_type_`` field (none->0, 32B->3, 64B->2, 128B->1)."""
        return int(_ffi_api.swizzle_mode_wgmma_layout_type(self))

    def tcgen05_layout_type(self) -> int:
        """TCGEN05 descriptor swizzle field (none->0, 32B->6, 64B->4, 128B->2)."""
        return int(_ffi_api.swizzle_mode_tcgen05_layout_type(self))

    def swizzle_byte_size(self) -> int:
        """Swizzle size in bytes (none->1, else 32/64/128)."""
        return int(_ffi_api.swizzle_mode_byte_width(self))

    def swizzle_atom_size(self) -> int:
        """Swizzle size in 16-byte vectors (none->1, else 2/4/8)."""
        return self.swizzle_byte_size() // 16 if not self.is_none() else 1

    def smem_alignment(self) -> int:
        """Required shared-memory base alignment in bytes (none->128, else 256/512/1024)."""
        return int(_ffi_api.swizzle_mode_smem_alignment(self))

    @staticmethod
    def from_ordinal(ordinal: int) -> SwizzleMode:
        return _ffi_api.swizzle_mode_from_ordinal(int(ordinal))
