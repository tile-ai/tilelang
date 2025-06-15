# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import sys
import os
import ctypes

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
        fmt="%(asctime)s  [TileLang:%(name)s:%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level("INFO")


_init_logger()

logger = logging.getLogger(__name__)

from .env import SKIP_LOADING_TILELANG_SO
from .env import enable_cache, disable_cache, is_cache_enabled  # noqa: F401

import tvm
import tvm._ffi.base
from tvm import DataType  # noqa: F401

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

from .jit import jit, JITKernel, compile  # noqa: F401
from .profiler import Profiler  # noqa: F401
from .cache import cached, set_cache_dir, get_cache_dir  # noqa: F401

from .utils import (
    TensorSupplyType,  # noqa: F401
    deprecated,  # noqa: F401
)
from .layout import (
    Layout,  # noqa: F401
    Fragment,  # noqa: F401
)
from . import (
    transform,  # noqa: F401
    language,  # noqa: F401
    engine,  # noqa: F401
)
from .autotuner import autotune  # noqa: F401
from .transform import PassConfigKey  # noqa: F401

from .engine import lower, register_cuda_postproc, register_hip_postproc  # noqa: F401

from .version import __version__  # noqa: F401

from .math import *  # noqa: F403
