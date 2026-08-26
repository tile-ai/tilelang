"""Cross-host cache for compiled CUDA device binaries."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import uuid
from hashlib import sha256
from typing import Any

from tilelang import __version__
from tilelang.env import env


class CUDABinaryCache:
    """Cache cubin/fatbin bytes independently from host executable artifacts."""

    cache_root_dir = "cuda-binaries"

    @staticmethod
    def _sanitize_path_component(component: str) -> str:
        sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in component)
        sanitized = sanitized.strip("._-")
        return sanitized or "unknown"

    @staticmethod
    def _format_version_namespace(version: str) -> str:
        public, sep, local = version.partition("+")
        public = CUDABinaryCache._sanitize_path_component(public)
        if not sep:
            return public
        local = "".join(ch if ch.isalnum() else "_" for ch in local).strip("_")
        return f"{public}_{local}" if local else public

    @classmethod
    def _get_namespace_root(cls) -> str:
        version = cls._format_version_namespace(__version__)
        return os.path.join(env.TILELANG_CACHE_DIR, version)

    @classmethod
    def _get_cache_root(cls) -> str:
        return os.path.join(cls._get_namespace_root(), cls.cache_root_dir)

    @staticmethod
    @functools.cache
    def _get_tilelang_lib_stamp() -> str | None:
        """Return a content hash for native TileLang libraries when requested."""
        import importlib

        lib_dirs: list[str] = []
        try:
            env_mod = importlib.import_module("tilelang.env")
            lib_dirs.extend(getattr(env_mod, "TL_LIBS", []) or [])
        except Exception:
            pass

        if sys.platform == "win32":
            lib_names = ["tvm_runtime.dll", "tvm_compiler.dll", "tvm_ffi.dll"]
        elif sys.platform == "darwin":
            lib_names = [
                "libtilelang.dylib",
                "libtilelang.so",
                "libtvm_runtime.dylib",
                "libtvm_compiler.dylib",
            ]
        else:
            lib_names = ["libtilelang.so", "libtvm_runtime.so", "libtvm_compiler.so"]

        stamps: list[str] = []
        seen_names: set[str] = set()
        for lib_dir in lib_dirs:
            for name in lib_names:
                if name in seen_names:
                    continue
                path = os.path.join(lib_dir, name)
                if os.path.exists(path):
                    file_hash = sha256()
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            file_hash.update(chunk)
                    stamps.append(f"{name}:{file_hash.hexdigest()}")
                    seen_names.add(name)
        if stamps:
            return "|".join(stamps)
        return None

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
        }
        if env.should_use_kernel_cache_lib_stamp():
            lib_stamp = cls._get_tilelang_lib_stamp()
            if lib_stamp:
                key_data["tilelang_lib"] = lib_stamp
        key_string = json.dumps(key_data, sort_keys=True)
        return sha256(key_string.encode()).hexdigest()

    @classmethod
    def get_path(cls, key: str, compile_format: str) -> str:
        filename = f"{key}.{compile_format}"
        return os.path.join(cls._get_cache_root(), filename)

    @classmethod
    def _sidecar_path(cls, path: str) -> str:
        return path + ".sha256"

    @classmethod
    def load(cls, key: str, compile_format: str) -> bytes | None:
        if not env.is_cache_enabled():
            return None
        path = cls.get_path(key, compile_format)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            return None
        if not env.should_verify_cache_hash():
            return data
        try:
            with open(cls._sidecar_path(path)) as f:
                expected_hash = f.read().strip()
        except OSError:
            # Entries written before content hashes were recorded.
            return data
        if sha256(data).hexdigest() == expected_hash:
            return data
        # Corrupted entry (e.g. truncated by a crashed writer): feeding it to
        # cuModuleLoadData would fail with CUDA_ERROR_INVALID_IMAGE on every
        # future run. Drop it so the caller recompiles and rewrites it.
        for stale in (path, cls._sidecar_path(path)):
            with contextlib.suppress(OSError):
                os.remove(stale)
        return None

    @classmethod
    def save(cls, key: str, compile_format: str, data: bytes) -> None:
        if not env.is_cache_enabled():
            return

        cache_root = cls._get_cache_root()
        os.makedirs(cache_root, exist_ok=True)
        path = cls.get_path(key, compile_format)
        # Sidecar first: a crash between the two renames then leaves a hash
        # without a payload (a plain cache miss) instead of an unverifiable
        # payload.
        cls._write_atomic(cls._sidecar_path(path), sha256(data).hexdigest().encode())
        cls._write_atomic(path, data)

    @classmethod
    def _write_atomic(cls, path: str, data: bytes) -> None:
        directory, filename = os.path.split(path)
        # Atomic replacement requires the temporary file and destination to be
        # on the same filesystem, so keep the temporary file next to the cache
        # entry.
        temp_path = os.path.join(directory, f".{filename}.{os.getpid()}_{uuid.uuid4().hex}.tmp")
        try:
            with open(temp_path, "wb") as f:
                f.write(data)
                # Without this barrier a crash can persist the rename below
                # before the file data, publishing a truncated binary.
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
