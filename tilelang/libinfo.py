import sys
import os

from .env import TL_LIBS


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
        lib_name = f"{name}.abi3.so"
    elif sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_name = f"lib{name}.so"
    elif sys.platform.startswith("win32"):
        lib_name = f"{name}.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = f"lib{name}.dylib"
    else:
        lib_name = f"lib{name}.so"

    for lib_root in TL_LIBS:
        lib_dll_path = os.path.join(lib_root, lib_name)
        if os.path.exists(lib_dll_path) and os.path.isfile(lib_dll_path):
            return lib_dll_path
    else:
        message = f"Cannot find libraries: {lib_name}\n" + "List of candidates:\n" + "\n".join(TL_LIBS)
        raise RuntimeError(message)
