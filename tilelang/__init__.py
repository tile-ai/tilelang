# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
import ctypes

import warnings
import functools
import logging
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that directs log output to tqdm progress bar to avoid interference."""

    def __init__(self, level=logging.NOTSET):
        """Initialize the handler with an optional log level."""
        super().__init__(level)

    def emit(self, record):
        """Emit a log record. Messages are written to tqdm to ensure output in progress bars isn't corrupted."""
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def set_log_level(level):
    """Set the logging level for the module's logger.

    Args:
        level (str or int): Can be the string name of the level (e.g., 'INFO') or the actual level (e.g., logging.INFO).
        OPTIONS: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)


def _init_logger():
    """Initialize the logger specific for this module with custom settings and a Tqdm-based handler."""
    logger = logging.getLogger(__name__)
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [TileLang:%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level("WARNING")


_init_logger()


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.
    """

    def decorator(func):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__} ({reason}).",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return new_func

    return decorator


logger = logging.getLogger(__name__)

# SETUP ENVIRONMENT VARIABLES
CUTLASS_NOT_FOUND_MESSAGE = ("CUTLASS is not installed or found in the expected path")
", which may lead to compilation bugs when utilize tilelang backend."
TL_TEMPLATE_NOT_FOUND_MESSAGE = ("TileLang is not installed or found in the expected path")
", which may lead to compilation bugs when utilize tilelang backend."
TVM_LIBRARY_NOT_FOUND_MESSAGE = ("TVM is not installed or found in the expected path")

SKIP_LOADING_TILELANG_SO = os.environ.get("SKIP_LOADING_TILELANG_SO", "0")

# Handle TVM_IMPORT_PYTHON_PATH to import tvm from the specified path
TVM_IMPORT_PYTHON_PATH = os.environ.get("TVM_IMPORT_PYTHON_PATH", None)

if TVM_IMPORT_PYTHON_PATH is not None:
    os.environ["PYTHONPATH"] = TVM_IMPORT_PYTHON_PATH + ":" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, TVM_IMPORT_PYTHON_PATH + "/python")
else:
    install_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rdparty", "tvm")
    install_tvm_library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    if os.path.exists(install_tvm_path) and install_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = install_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, install_tvm_path + "/python")

    develop_tvm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "tvm")
    develop_tvm_library_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "build", "tvm")
    if os.path.exists(develop_tvm_path) and develop_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = develop_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, develop_tvm_path + "/python")

    if os.environ.get("TVM_LIBRARY_PATH") is None:
        if os.path.exists(develop_tvm_library_path):
            os.environ["TVM_LIBRARY_PATH"] = develop_tvm_library_path
        elif os.path.exists(install_tvm_library_path):
            os.environ["TVM_LIBRARY_PATH"] = install_tvm_library_path
        else:
            logger.warning(TVM_LIBRARY_NOT_FOUND_MESSAGE)

if os.environ.get("TL_CUTLASS_PATH", None) is None:
    install_cutlass_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "3rdparty", "cutlass")
    develop_cutlass_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "cutlass")
    if os.path.exists(install_cutlass_path):
        os.environ["TL_CUTLASS_PATH"] = install_cutlass_path + "/include"
    elif (os.path.exists(develop_cutlass_path) and develop_cutlass_path not in sys.path):
        os.environ["TL_CUTLASS_PATH"] = develop_cutlass_path + "/include"
    else:
        logger.warning(CUTLASS_NOT_FOUND_MESSAGE)

if os.environ.get("TL_TEMPLATE_PATH", None) is None:
    install_tl_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    develop_tl_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    if os.path.exists(install_tl_template_path):
        os.environ["TL_TEMPLATE_PATH"] = install_tl_template_path
    elif (os.path.exists(develop_tl_template_path) and develop_tl_template_path not in sys.path):
        os.environ["TL_TEMPLATE_PATH"] = develop_tl_template_path
    else:
        logger.warning(TL_TEMPLATE_NOT_FOUND_MESSAGE)

import tvm
import tvm._ffi.base

from . import libinfo


def _load_tile_lang_lib():
    """Load Tile Lang lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    # pylint: disable=protected-access
    lib_name = "tilelang" if tvm._ffi.base._RUNTIME_ONLY else "tilelang_module"
    # pylint: enable=protected-access
    lib_path = libinfo.find_lib_path(lib_name, optional=False)
    return ctypes.CDLL(lib_path[0]), lib_path[0]


# only load once here
if SKIP_LOADING_TILELANG_SO == "0":
    _LIB, _LIB_PATH = _load_tile_lang_lib()

from .jit import jit  # noqa: F401
from .utils import (
    Profiler,  # noqa: F401
    TensorSupplyType,  # noqa: F401
)
from .layout import (
    Layout,  # noqa: F401
    Fragment,  # noqa: F401
)
from . import (
    transform,  # noqa: F401
    autotuner,  # noqa: F401
    language,  # noqa: F401
    engine,  # noqa: F401
)

from .engine import lower  # noqa: F401

from .version import __version__  # noqa: F401
