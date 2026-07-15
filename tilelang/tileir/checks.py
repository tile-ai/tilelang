"""Dependency discovery for the TileIR execution backend.

The backend boundary intentionally depends only on public NVIDIA CUDA Tile
artifacts:

* CUDA Tile IR MLIR Python bindings from a local or bundled ``NVIDIA/cuda-tile``
  source build.
* ``tileiras`` from matching NVIDIA Python wheels, ``PATH``, or a CUDA Toolkit
  install.

This module must stay free of TileLang lowering logic. It is the single place
that turns missing external dependencies into actionable install errors.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess


TILEIRAS_PACKAGES = ("nvidia-cuda-tileiras", "nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-nvjitlink")
CUDA_TILE_IR_MLIR_MODULE = "cuda_tile._mlir.dialects.cuda_tile"
CUDA_TILE_IR_SUPPORTED_VERSION = "13.3"
CUDA_TILE_RUNTIME_SUPPORTED_VERSION = "1.5"
CUDA_TILE_RUNTIME_MODULE = "cuda.tile._cext"
CUDA_TILE_IR_REQUIRED_SYMBOLS = (
    "alloca",
    "atomic_red_view_tko",
    "Float4E2M1FN",
    "Int4",
    "make_strided_view",
    "mmaf_scaled",
    "pack",
    "StridedViewType",
    "unpack",
)
CUDA_TILE_RUNTIME_REQUIRED_SYMBOLS = ("TileDispatcher", "launch")


class TileIRDependencyError(ImportError):
    """Raised when the TileIR backend cannot find its public toolchain."""


@dataclass(frozen=True)
class TileIRToolchain:
    """Resolved public CUDA Tile toolchain components."""

    cuda_tile_ir_module: str
    tileiras_path: Path
    tileiras_version: str | None
    cuda_tile_ir_version: str = CUDA_TILE_IR_SUPPORTED_VERSION
    cuda_tile_runtime_version: str = CUDA_TILE_RUNTIME_SUPPORTED_VERSION


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _format_install_help() -> str:
    return (
        "Build/install the public NVIDIA/cuda-tile repository with "
        "`CUDA_TILE_ENABLE_BINDINGS_PYTHON=ON` and make its Python package "
        "visible. Install the assembler stack with "
        "`pip install tilelang[tileir]`, or install CUDA Toolkit 13.3 and make "
        "`tileiras` visible through `PATH`, `CUDA_HOME`, or `CUDA_PATH`. "
        "The `tileir` extra also installs the supported cuTile native dispatcher."
    )


def _missing_cuda_tile_ir_symbols() -> tuple[str, ...]:
    module = importlib.import_module(CUDA_TILE_IR_MLIR_MODULE)
    return tuple(symbol for symbol in CUDA_TILE_IR_REQUIRED_SYMBOLS if not hasattr(module, symbol))


def _missing_cuda_tile_runtime_symbols() -> tuple[str, ...]:
    module = importlib.import_module(CUDA_TILE_RUNTIME_MODULE)
    return tuple(symbol for symbol in CUDA_TILE_RUNTIME_REQUIRED_SYMBOLS if not hasattr(module, symbol))


def _find_pip_tileiras() -> Path | None:
    versions = {package: _package_version(package) for package in TILEIRAS_PACKAGES}
    if any(version is None for version in versions.values()):
        return None

    major_minor = {version.rsplit(".", 1)[0] for version in versions.values() if version is not None}
    if len(major_minor) != 1:
        details = ", ".join(f"{package}=={version}" for package, version in versions.items())
        raise TileIRDependencyError(
            "Mismatched NVIDIA TileIR toolchain wheels: "
            f"{details}. Install matching major.minor versions, for example `pip install tilelang[tileir]`."
        )

    try:
        import nvidia.cu13 as cu13_pkg
    except ImportError:
        return None

    package_paths = list(getattr(cu13_pkg, "__path__", []))
    if not package_paths:
        return None

    candidate = shutil.which("tileiras", path=str(Path(package_paths[0]) / "bin"))
    return Path(candidate) if candidate is not None else None


def is_pip_tileiras_path(tileiras_path: Path) -> bool:
    """Return whether ``tileiras_path`` resolves to the NVIDIA pip wheel binary."""

    try:
        import nvidia.cu13 as cu13_pkg
    except ImportError:
        return False

    try:
        resolved = tileiras_path.resolve()
    except OSError:
        resolved = tileiras_path

    for package_path in getattr(cu13_pkg, "__path__", []):
        try:
            resolved.relative_to((Path(package_path) / "bin").resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def tileiras_invocation_env(tileiras_path: Path) -> dict[str, str]:
    """Build an environment for invoking ``tileiras``.

    NVIDIA's pip ``tileiras`` bundle is self-contained. Leaving a system
    ``CUDA_HOME``/``CUDA_PATH`` in the environment can make it pick up an
    unrelated CUDA Toolkit and reject otherwise valid TileIR bytecode.
    """

    env = os.environ.copy()
    if is_pip_tileiras_path(tileiras_path):
        env.pop("CUDA_HOME", None)
        env.pop("CUDA_PATH", None)
    return env


def find_tileiras() -> Path:
    """Find the public ``tileiras`` binary used to assemble TileIR bytecode."""

    env_path = os.environ.get("TILELANG_TILEIRAS")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise TileIRDependencyError(f"TILELANG_TILEIRAS points to a non-executable file: {candidate}")

    pip_tileiras = _find_pip_tileiras()
    if pip_tileiras is not None:
        return pip_tileiras

    path_tileiras = shutil.which("tileiras")
    if path_tileiras is not None:
        return Path(path_tileiras)

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_root = os.environ.get(env_name)
        if not cuda_root:
            continue
        candidate = Path(cuda_root) / "bin" / "tileiras"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    raise TileIRDependencyError(f"`tileiras` was not found. {_format_install_help()}")


def _tileiras_version(tileiras_path: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(tileiras_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=tileiras_invocation_env(tileiras_path),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return output or None


def _validated_tileiras_version(tileiras_path: Path) -> str:
    """Return the declared assembler version after enforcing the supported ABI."""

    version = os.environ.get("TILELANG_TILEIRAS_VERSION")
    if version is None and is_pip_tileiras_path(tileiras_path):
        version = _package_version("nvidia-cuda-tileiras")
    if version is None:
        version = _tileiras_version(tileiras_path)

    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.\d+)?(?!\d)", version or "")
    if match is None:
        raise TileIRDependencyError(
            "TileLang could not determine the `tileiras` version. "
            "Use the NVIDIA 13.3 wheel stack from `pip install tilelang[tileir]`, "
            "or set TILELANG_TILEIRAS_VERSION=13.3 for a compatible explicitly managed binary."
        )
    major_minor = f"{match.group(1)}.{match.group(2)}"
    if major_minor != CUDA_TILE_IR_SUPPORTED_VERSION:
        raise TileIRDependencyError(
            f"TileLang requires tileiras {CUDA_TILE_IR_SUPPORTED_VERSION}.x, but `{tileiras_path}` reports {version!r}. "
            "Install the matching assembler with `pip install tilelang[tileir]`."
        )
    return version


def _validated_cuda_tile_runtime_version() -> str:
    """Validate the distribution version that owns the private launch ABI."""

    version = _package_version("cuda-tile")
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.\d+)?(?!\d)", version or "")
    if match is None:
        raise TileIRDependencyError(
            "TileLang could not determine the `cuda-tile` native dispatcher version. "
            "Install the supported runtime with `pip install tilelang[tileir]`."
        )
    major_minor = f"{match.group(1)}.{match.group(2)}"
    if major_minor != CUDA_TILE_RUNTIME_SUPPORTED_VERSION:
        raise TileIRDependencyError(
            f"TileLang requires cuda-tile {CUDA_TILE_RUNTIME_SUPPORTED_VERSION}.x for the native dispatcher ABI, "
            f"but found {version!r}. Install the supported runtime with `pip install tilelang[tileir]`."
        )
    return version


def has_cuda_tile_ir_bindings() -> bool:
    """Return whether CUDA Tile IR MLIR bindings are importable."""

    if not _module_exists(CUDA_TILE_IR_MLIR_MODULE):
        return False
    try:
        return not _missing_cuda_tile_ir_symbols()
    except ImportError:
        return False


def is_tileir_available() -> bool:
    """Return whether the TileIR backend dependencies appear usable."""

    try:
        check_tileir_available()
        return True
    except TileIRDependencyError:
        return False


def check_tileir_available() -> TileIRToolchain:
    """Validate that public CUDA Tile dependencies are installed.

    The backend does not lower through the cuTile Python DSL. It requires CUDA
    Tile IR Python bindings from ``NVIDIA/cuda-tile``, the public ``tileiras``
    assembler, and the native dispatcher shipped by the supported cuTile
    runtime package.
    """

    if not _module_exists(CUDA_TILE_IR_MLIR_MODULE):
        raise TileIRDependencyError(
            f"CUDA Tile IR MLIR Python bindings are not importable as `{CUDA_TILE_IR_MLIR_MODULE}`. {_format_install_help()}"
        )
    try:
        missing_symbols = _missing_cuda_tile_ir_symbols()
    except ImportError as exc:
        raise TileIRDependencyError(
            f"CUDA Tile IR MLIR Python bindings could not be loaded from `{CUDA_TILE_IR_MLIR_MODULE}`. {_format_install_help()}"
        ) from exc
    if missing_symbols:
        missing = ", ".join(missing_symbols)
        raise TileIRDependencyError(
            f"CUDA Tile IR MLIR Python bindings are older than {CUDA_TILE_IR_SUPPORTED_VERSION}; "
            f"missing required symbols: {missing}. {_format_install_help()}"
        )

    if not _module_exists(CUDA_TILE_RUNTIME_MODULE):
        raise TileIRDependencyError(
            f"The TileIR native dispatcher is not importable as `{CUDA_TILE_RUNTIME_MODULE}`. "
            "Install the supported runtime with `pip install tilelang[tileir]`."
        )
    try:
        missing_runtime_symbols = _missing_cuda_tile_runtime_symbols()
    except ImportError as exc:
        raise TileIRDependencyError(
            f"The TileIR native dispatcher `{CUDA_TILE_RUNTIME_MODULE}` could not be loaded. "
            "Install the supported runtime with `pip install tilelang[tileir]`."
        ) from exc
    if missing_runtime_symbols:
        missing = ", ".join(missing_runtime_symbols)
        raise TileIRDependencyError(
            f"The TileIR native dispatcher is incompatible; missing required symbols: {missing}. "
            "Install the supported runtime with `pip install tilelang[tileir]`."
        )
    cuda_tile_runtime_version = _validated_cuda_tile_runtime_version()

    tileiras_path = find_tileiras()
    tileiras_version = _validated_tileiras_version(tileiras_path)
    return TileIRToolchain(
        cuda_tile_ir_module=CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tileiras_path,
        tileiras_version=tileiras_version,
        cuda_tile_ir_version=CUDA_TILE_IR_SUPPORTED_VERSION,
        cuda_tile_runtime_version=cuda_tile_runtime_version,
    )
