"""Cross-host cache for compiled Ascend device binaries."""

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
