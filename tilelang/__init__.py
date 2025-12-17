import sys
import os
import ctypes

import logging
import warnings
from pathlib import Path
from tqdm.auto import tqdm


def _compute_version() -> str:
    """Return the package version without being polluted by unrelated installs.

    Preference order:
    1) If running from a source checkout (VERSION file present at repo root),
       use the dynamic version from version_provider (falls back to plain VERSION).
    2) Otherwise, use importlib.metadata for the installed distribution.
    3) As a last resort, return a dev sentinel.
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent
        version_file = repo_root / "VERSION"
        if version_file.is_file():
            try:
                from version_provider import dynamic_metadata  # type: ignore

                return dynamic_metadata("version")
            except Exception:
                # Fall back to the raw VERSION file if provider isn't available.
                return version_file.read_text().strip()
    except Exception:
        # If any of the above fails, fall through to installed metadata.
        pass

    try:
        from importlib.metadata import version as _dist_version  # py3.8+

        return _dist_version("tilelang")
    except Exception as exc:
        warnings.warn(
            f"tilelang version metadata unavailable ({exc!r}); using development version.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "0.0.dev0"


__version__ = _compute_version()


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

from .env import enable_cache, disable_cache, is_cache_enabled  # noqa: F401
from .env import env as env  # noqa: F401

import tvm
import tvm.base  # noqa: F401
from tvm import DataType  # noqa: F401

# Setup tvm search path before importing tvm
from . import libinfo


def _load_tile_lang_lib():
    """Load Tile Lang lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    # pylint: disable=protected-access
    lib_name = "tilelang" if tvm.base._RUNTIME_ONLY else "tilelang_module"
    # pylint: enable=protected-access
    lib_path = libinfo.find_lib_path(lib_name)
    return ctypes.CDLL(lib_path), lib_path


# only load once here
if env.SKIP_LOADING_TILELANG_SO == "0":
    _LIB, _LIB_PATH = _load_tile_lang_lib()

from .jit import jit, lazy_jit, JITKernel, compile, par_compile  # noqa: F401
from .profiler import Profiler  # noqa: F401
from .cache import clear_cache  # noqa: F401

from .utils import (
    TensorSupplyType,  # noqa: F401
    deprecated,  # noqa: F401
)
from .layout import (
    Layout,  # noqa: F401
    Fragment,  # noqa: F401
)
from . import (
    analysis,  # noqa: F401
    transform,  # noqa: F401
    language,  # noqa: F401
    engine,  # noqa: F401
    tools,  # noqa: F401
)
from .language.v2 import dtypes  # noqa: F401
from .autotuner import autotune  # noqa: F401
from .transform import PassConfigKey  # noqa: F401

from .engine import lower, register_cuda_postproc, register_hip_postproc  # noqa: F401

from .math import *  # noqa: F403

from . import ir  # noqa: F401

from . import tileop  # noqa: F401
