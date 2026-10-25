import importlib.machinery
import os
import sys
from hashlib import sha256

from .env import TL_LIBS, get_cuda_dll_search_dirs


def native_library_paths():
    """Return the TileLang, standalone TIRx, and FFI binaries used by this process."""
    from tvm_ffi.libinfo import find_libtvm_ffi
    from tvm_tirx.base import _LOADED_LIBS

    paths = [find_lib_path("tilelang"), find_libtvm_ffi()]
    paths.extend(library._name for library in _LOADED_LIBS.values())
    return sorted(set(os.path.realpath(path) for path in paths))


def native_library_stamp():
    """Hash all compiler dependencies, without encoding installation paths."""
    stamps = []
    for path in native_library_paths():
        digest = sha256()
        with open(path, "rb") as library:
            for chunk in iter(lambda: library.read(1 << 20), b""):
                digest.update(chunk)
        stamps.append(f"{os.path.basename(path)}:{digest.hexdigest()}")
    return "|".join(sorted(stamps)) or None


def get_dll_directories():
    dll_dirs = list(TL_LIBS) + get_cuda_dll_search_dirs()
    return [os.path.abspath(path) for path in dll_dirs if os.path.isdir(path)]


def find_lib_path(name: str, py_ext=False):
    """Find tile lang library

    Parameters
    ----------
    name : str
        The name of the library

    optional: boolean
        Whether the library is required
    """
    if py_ext:
        lib_names = [f"{name}{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    elif sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_names = [f"lib{name}.so"]
    elif sys.platform.startswith("win32"):
        lib_names = [f"{name}.dll"]
    elif sys.platform.startswith("darwin"):
        lib_names = [f"lib{name}.dylib"]
    else:
        lib_names = [f"lib{name}.so"]

    for lib_root in TL_LIBS:
        for lib_name in lib_names:
            lib_dll_path = os.path.join(lib_root, lib_name)
            if os.path.exists(lib_dll_path) and os.path.isfile(lib_dll_path):
                return lib_dll_path
    else:
        message = f"Cannot find libraries: {', '.join(lib_names)}\n" + "List of candidates:\n" + "\n".join(TL_LIBS)
        raise RuntimeError(message)
