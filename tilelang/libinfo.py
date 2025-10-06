"""
As we're using sk-build-core,
all libraries will be installed in <site-packages>/tilelang/lib,
no matter if it's editable install.

We need to:
1. setup `TVM_LIBRARY_PATH` so tvm could be found;
"""

import sys
import os
import site

tl_lib = [os.path.join(i, 'tilelang/lib') for i in site.getsitepackages()]
tl_lib = [i for i in tl_lib if os.path.exists(i)]

os.environ['TVM_LIBRARY_PATH'] = ':'.join(tl_lib + [os.environ.get('TVM_LIBRARY_PATH', '')])


def find_lib_path(name: str, optional=False):
    """Find tile lang library

    Parameters
    ----------
    name : str
        The name of the library

    optional: boolean
        Whether the library is required
    """
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_name = f"lib{name}.so"
    elif sys.platform.startswith("win32"):
        lib_name = f"{name}.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = f"lib{name}.dylib"
    else:
        lib_name = f"lib{name}.so"

    for lib_root in tl_lib:
        lib_dll_path = [os.path.join(p, lib_name) for p in tl_lib]
        for lib in lib_dll_path:
            if os.path.exists(lib) and os.path.isfile(lib):
                return lib
    else:
        message = (f"Cannot find libraries: {lib_name}\n" + "List of candidates:\n" +
                   "\n".join(lib_dll_path))
        raise RuntimeError(message)
