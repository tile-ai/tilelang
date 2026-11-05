"""Load the matching TIRx compiler provider and its one FFI registry."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys

FFI_REF = "3e906bc685ea45624bc80e8d7b7d86497d492af3"
FFI_VERSION = "0.1.15.post3.dev30+g3e906bc68"
_dll_handles = []


def _install_error_namespace(provider):
    """Expose canonical TileLang errors without changing native IR objects."""
    from . import errors

    # Reject a namespace collision before registering any exception class.
    # tilelang.tvm remains the provider module itself; only the established
    # error namespace and TVMError spelling are added to that module.
    for name, value in (("error", errors), ("TVMError", errors.TVMError)):
        if name in vars(provider) and vars(provider)[name] is not value:
            raise ImportError(f"The TIRx provider already has an incompatible {name!r} binding")
    error_alias = "tilelang.tvm.error"
    if error_alias in sys.modules and sys.modules[error_alias] is not errors:
        raise ImportError(f"The {error_alias!r} module alias is already owned by another module")
    errors._ensure_error_registration()
    provider.error = errors
    provider.TVMError = errors.TVMError
    sys.modules[error_alias] = errors


def load_provider():
    """Return the native provider; never load another copy of its compiler DSO.

    A development checkout may bootstrap its adjacent TIRx Python source when
    no provider is installed.  Installed providers are otherwise authoritative,
    including when TIRx was imported before TileLang.
    """
    if "tvm" in sys.modules and getattr(sys.modules["tvm"], "__name__", None) != "tvm_tirx":
        raise ImportError("Legacy TVM and native TileLang must run in separate processes")
    try:
        version = importlib.metadata.version("apache-tvm-ffi")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(f"TileLang requires apache-tvm-ffi built from {FFI_REF}") from exc
    if version != FFI_VERSION:
        raise ImportError(f"TileLang requires apache-tvm-ffi {FFI_VERSION}; found {version}")

    if importlib.util.find_spec("tvm_tirx") is None:
        source_root = Path(__file__).resolve().parent.parent
        source_python = source_root / "3rdparty" / "tvm-tirx" / "python"
        if not (source_python / "tvm_tirx" / "__init__.py").is_file():
            raise ImportError("Install the matching TileLang TIRx provider wheel")
        sys.path.insert(0, str(source_python))
        build_lib = source_root / "build" / "lib"
        if build_lib.is_dir():
            os.environ.setdefault("TVM_TIRX_LIBRARY_PATH", str(build_lib))

    if sys.platform == "win32":
        from .env import get_runtime_library_dirs, get_cuda_dll_search_dirs

        for directory in dict.fromkeys([*get_runtime_library_dirs(), *get_cuda_dll_search_dirs()]):
            if Path(directory).is_dir():
                _dll_handles.append(os.add_dll_directory(directory))

    provider = importlib.import_module("tvm_tirx")
    from tvm_tirx import base
    import tvm_ffi

    if base._RUNTIME_ONLY:
        raise ImportError("TileLang requires a compiler provider, but TIRx loaded runtime-only libraries")
    build_info = tvm_ffi.get_global_func("tl.CompilerBuildInfo", allow_missing=True)
    if build_info is None:
        raise ImportError("The loaded TIRx compiler does not contain TileLang; install its matching provider wheel")
    info = dict(build_info())
    if info.get("ffi_ref") != FFI_REF or info.get("launch_schema") != "1":
        raise ImportError(f"Incompatible TileLang compiler provider: {info}")
    try:
        from ._build_info import COMPILER_INFO
    except ModuleNotFoundError as exc:
        if exc.name != "tilelang._build_info":
            raise
        source_root = Path(__file__).resolve().parent.parent
        if not (source_root / "CMakeLists.txt").is_file():
            raise ImportError("TileLang is missing its paired compiler identity") from exc
    else:
        if info.get("source_digest") != COMPILER_INFO["source_digest"]:
            raise ImportError("TileLang and the loaded compiler were built from different source sets")
        installed_version = importlib.metadata.version("apache-tvm-tirx")
        if installed_version != COMPILER_INFO["provider_version"]:
            raise ImportError(f"TileLang requires its exact compiler provider; found {installed_version}")
    _install_error_namespace(provider)
    # The alias exposes native TIRx interfaces; it does not install a top-level
    # tvm alias or emulate the removed PrimFunc/Buffer object model.
    sys.modules["tilelang.tvm"] = provider
    return provider, base._LOADED_LIBS["tvm_tirx_compiler"]
