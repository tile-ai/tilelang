"""Cross-host cache for compiled Ascend device binaries."""

from __future__ import annotations

import functools
import json
import os
import uuid
from hashlib import sha256
from typing import Any

from tilelang import __version__
from tilelang.env import env
import contextlib


class AscendBinaryCache:
    """Cache executable CCE ELF bytes independently from host artifacts."""

    cache_root_dir = "ascend-binaries"
    binary_format = "aibin"

    @staticmethod
    def _sanitize_path_component(component: str) -> str:
        sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in component)
        sanitized = sanitized.strip("._-")
        return sanitized or "unknown"

    @staticmethod
    def _format_version_namespace(version: str) -> str:
        public, sep, local = version.partition("+")
        public = AscendBinaryCache._sanitize_path_component(public)
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
                    with open(path, "rb") as file:
                        for chunk in iter(lambda: file.read(1 << 20), b""):
                            file_hash.update(chunk)
                    stamps.append(f"{name}:{file_hash.hexdigest()}")
                    seen_names.add(name)
        return "|".join(stamps) if stamps else None

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
        if env.should_use_kernel_cache_lib_stamp():
            lib_stamp = cls._get_tilelang_lib_stamp()
            if lib_stamp:
                key_data["tilelang_lib"] = lib_stamp
        key_string = json.dumps(key_data, sort_keys=True)
        return sha256(key_string.encode()).hexdigest()

    @classmethod
    def get_path(cls, key: str, compile_format: str = binary_format) -> str:
        return os.path.join(cls._get_cache_root(), f"{key}.{compile_format}")

    @classmethod
    def load(cls, key: str, compile_format: str = binary_format) -> bytes | None:
        if not env.is_cache_enabled():
            return None
        try:
            with open(cls.get_path(key, compile_format), "rb") as file:
                return file.read()
        except FileNotFoundError:
            return None

    @classmethod
    def save(cls, key: str, data: bytes, compile_format: str = binary_format) -> None:
        if not env.is_cache_enabled():
            return
        os.makedirs(env.TILELANG_CACHE_DIR, exist_ok=True)
        os.makedirs(cls._get_cache_root(), exist_ok=True)

        path = cls.get_path(key, compile_format)
        directory, filename = os.path.split(path)
        temp_path = os.path.join(directory, f".{filename}.{os.getpid()}_{uuid.uuid4().hex}.tmp")
        try:
            with open(temp_path, "wb") as file:
                file.write(data)
            os.replace(temp_path, path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
