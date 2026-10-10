"""Cross-host cache for compiled CUDA device binaries."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from tilelang import __version__
from tilelang.cache.binary_cache import BinaryCache


class CUDABinaryCache(BinaryCache):
    """Cache cubin/fatbin bytes independently from host executable artifacts."""

    # Each key is an immutable directory containing metadata.json and a raw
    # kernel binary. Legacy `<key>.<format>` files and sidecars are ignored.
    cache_root_dir = "cuda-binaries"
    cache_format = "tilelang.cuda-binary-cache.v1"
    binary_kind = "CUDA"

    @classmethod
    def make_key(
        cls,
        *,
        code: str,
        target_kind: str,
        target_arch: str,
        target_code: list[str],
        compile_format: str,
        options: list[str] | None = None,
        compiler: str = "nvcc",
        compiler_version: tuple[int, ...] | None = None,
    ) -> str:
        # Compiler options must be part of the key: flags like --use_fast_math
        # change the generated SASS without changing the CUDA source, so keying
        # on the code hash alone lets a fast-math binary satisfy a
        # precise-math compile (and vice versa).
        key_data: dict[str, Any] = {
            "tilelang_version": __version__,
            "code_hash": sha256(code.encode()).hexdigest(),
            "target_kind": target_kind,
            "target_arch": target_arch,
            "target_code": tuple(target_code),
            "compile_format": compile_format,
            "options": tuple(options or []),
            "compiler": compiler,
            "compiler_version": compiler_version,
        }
        return cls._finalize_key(key_data)
