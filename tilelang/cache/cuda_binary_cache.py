"""Cross-host cache for compiled CUDA device binaries."""

from __future__ import annotations

import contextlib
import errno
import functools
import json
import os
import shutil
import sys
import uuid
from hashlib import sha256
from typing import Any

from tilelang import __version__
from tilelang.env import env


_CACHE_FORMAT = "tilelang.cuda-binary-cache.v1"


class CUDABinaryCache:
    """Cache cubin/fatbin bytes independently from host executable artifacts."""

    # Each key is an immutable directory containing metadata.json and a raw
    # kernel binary. Legacy `<key>.<format>` files and sidecars are ignored.
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

    @classmethod
    def _get_staging_root(cls) -> str:
        return os.path.join(cls._get_namespace_root(), ".staging", cls.cache_root_dir)

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
        return os.path.join(cls._get_cache_root(), key, f"kernel.{compile_format}")

    @classmethod
    def load(cls, key: str, compile_format: str) -> bytes | None:
        if not env.is_cache_enabled():
            return None
        path = cls.get_path(key, compile_format)
        try:
            with open(path, "rb") as f:
                data = f.read()
            with open(os.path.join(os.path.dirname(path), "metadata.json"), encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(metadata, dict) or metadata.get("format") != _CACHE_FORMAT:
            return None

        # Empty/short reads and missing metadata are always misses. Never delete
        # shared cache entries: on 3FS, unlinking a binary can invalidate another
        # reader's open fd.
        payload_size = metadata.get("size")
        if type(payload_size) is not int or payload_size <= 0 or len(data) != payload_size:
            return None
        if sha256(data).hexdigest() != metadata.get("sha256"):
            return None
        return data

    @classmethod
    def save(cls, key: str, compile_format: str, data: bytes) -> None:
        if not data:
            raise ValueError("Cannot cache an empty CUDA binary")
        if not env.is_cache_enabled():
            return

        cache_root = cls._get_cache_root()
        os.makedirs(cache_root, exist_ok=True)
        path = cls.get_path(key, compile_format)
        cache_path = os.path.dirname(path)
        # Published directories are immutable, even if a load found corruption.
        # The caller can use its fresh compilation; repairing shared entries
        # requires offline cleanup to avoid invalidating concurrent readers.
        if os.path.lexists(cache_path):
            return

        staging_root = cls._get_staging_root()
        os.makedirs(staging_root, exist_ok=True)
        staging_path = os.path.join(staging_root, f"{key}.{os.getpid()}.{uuid.uuid4().hex}")
        os.mkdir(staging_path)
        try:
            data = bytes(data)
            metadata = {"format": _CACHE_FORMAT, "size": len(data), "sha256": sha256(data).hexdigest()}
            with open(os.path.join(staging_path, os.path.basename(path)), "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            with open(os.path.join(staging_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            cls._fsync_dir(staging_path)

            try:
                # Both roots live in the same namespace/filesystem. Rename
                # publishes both files together and cannot overwrite another
                # writer's nonempty directory: the first publication wins.
                os.rename(staging_path, cache_path)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            else:
                cls._fsync_dir(cache_root)
                cls._fsync_dir(staging_root)
        finally:
            # Only remove this writer's private staging directory.
            shutil.rmtree(staging_path, ignore_errors=True)

    @staticmethod
    def _fsync_dir(path: str) -> None:
        """Best-effort durability barrier for the published directory entry."""
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            with contextlib.suppress(OSError):
                os.fsync(fd)
        finally:
            os.close(fd)
