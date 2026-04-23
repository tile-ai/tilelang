import os
import sys

from .env import TL_LIBS, env


def get_dll_directories():
    dll_dirs = list(TL_LIBS)
    if sys.platform.startswith("win32"):
        cuda_home = getattr(env, "CUDA_HOME", "")
        dll_dirs.extend(
            path
            for path in (
                cuda_home,
                os.path.join(cuda_home, "bin"),
                os.path.join(cuda_home, "bin", "x86_64"),
                os.path.join(cuda_home, "lib", "x64"),
                os.path.join(cuda_home, "nvvm", "bin"),
            )
            if path
        )
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
        lib_names = [f"{name}.abi3.so"]
    elif sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_names = [f"lib{name}.so"]
    elif sys.platform.startswith("win32"):
        lib_names = [f"{name}.dll"]
        if name == "tilelang":
            lib_names.append("tvm.dll")
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
