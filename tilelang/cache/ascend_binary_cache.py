"""Cross-host caches for compiled Ascend and PTO device binaries."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from tilelang import __version__
from tilelang.cache.binary_cache import BinaryCache


class AscendBinaryCache(BinaryCache):
    """Cache executable CCE ELF bytes independently from host artifacts."""

    # Each key is an immutable directory containing metadata.json and a raw
    # kernel binary. Legacy flat `<key>.aibin` files are ignored.
    cache_root_dir = "ascend-binaries"
    cache_format = "tilelang.ascend-binary-cache.v1"
    binary_kind = "Ascend"
    binary_format = "aibin"

    @classmethod
    def make_key(
        cls,
        *,
        code: str,
        target_kind: str,
        target_arch: str,
        compile_format: str,
        options: list[str] | None = None,
        linker_options: list[str] | None = None,
    ) -> str:
        key_data: dict[str, Any] = {
            "tilelang_version": __version__,
            "code_hash": sha256(code.encode()).hexdigest(),
            "target_kind": target_kind,
            "target_arch": target_arch,
            "compile_format": compile_format,
            "options": tuple(options or []),
            "linker_options": tuple(linker_options or []),
        }
        return cls._finalize_key(key_data)


class PTOBinaryCache(BinaryCache):
    """Cache executable PTODSL-compiled CCE ELF bytes for the PTO target."""

    cache_root_dir = "pto-binaries"
    cache_format = "tilelang.pto-binary-cache.v1"
    binary_kind = "PTO"
    binary_format = "aibin"

    # Same fields as AscendBinaryCache.make_key (options = ptoas flags,
    # linker_options = bisheng link flags) plus a toolchain identity stamp.
    @classmethod
    def make_key(
        cls,
        *,
        code: str,
        target_kind: str,
        target_arch: str,
        compile_format: str,
        options: list[str] | None = None,
        linker_options: list[str] | None = None,
        toolchain: dict[str, str] | None = None,
    ) -> str:
        key_data: dict[str, Any] = {
            "tilelang_version": __version__,
            "code_hash": sha256(code.encode()).hexdigest(),
            "target_kind": target_kind,
            "target_arch": target_arch,
            "compile_format": compile_format,
            "options": tuple(options or []),
            "linker_options": tuple(linker_options or []),
            # The same PTODSL source produces different ELFs across ptoas or
            # bisheng versions; a toolchain upgrade must miss the cache.
            "toolchain": dict(toolchain or {}),
        }
        return cls._finalize_key(key_data)
