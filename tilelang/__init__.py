import logging
import warnings
from pathlib import Path


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
del _compute_version


logger = logging.getLogger(__name__)


def set_log_level(level):
    """Set the logging level for the module's logger.

    Args:
        level (str or int): Can be the string name of the level (e.g., 'INFO') or the actual level (e.g., logging.INFO).
        OPTIONS: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)


def _init_logger():
    """Initialize the logger specific for this module with custom settings and a Tqdm-based handler."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    class TqdmLoggingHandler(logging.Handler):
        """Custom logging handler that directs log output to tqdm progress bar to avoid interference."""

        def __init__(self, level=logging.NOTSET):
            """Initialize the handler with an optional log level."""
            super().__init__(level)

        def emit(self, record):
            """Emit a log record. Messages are written to tqdm to ensure output in progress bars isn't corrupted."""
            try:
                msg = self.format(record)
                if tqdm is not None:
                    tqdm.write(msg)
            except Exception:
                self.handleError(record)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s  [TileLang:%(name)s:%(levelname)s] (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level("INFO")


from .env import env as env  # noqa: F401

# Skip logger initialization in light import mode
if not env.is_light_import():
    _init_logger()

del _init_logger


# The provider owns all compiler registration and library loading.  Python
# consumers are loaded on demand, so imports cannot initialize a second compiler
# or require an unrelated device SDK before a backend is selected.
if not env.is_light_import():
    from ._provider import load_provider as _load_provider
    from tvm_ffi import dtype as DataType  # noqa: F401
    from .env import enable_cache as enable_cache, disable_cache as disable_cache, is_cache_enabled as is_cache_enabled

    _LIB = _load_provider()[1]
    _LIB_PATH = _LIB._name
    del _load_provider
    from ._extensions import install_frontend_extension_finder as _install_frontend_extension_finder

    _install_frontend_extension_finder()
    del _install_frontend_extension_finder
    # Importing a submodule assigns it to its parent package. Bind these public
    # callables explicitly so language's jit.exceptions import cannot replace
    # @tilelang.jit with the tilelang.jit module object.
    from .jit import jit as jit, compile as compile, par_compile as par_compile

    # Discover the built-in target descriptors in their established priority
    # order. These modules register detection/normalization callbacks without
    # loading compiler pipelines or requiring their optional device SDKs.
    # Delaying them until backend resolution is too late for target="auto".
    from importlib import import_module as _import_target_module

    for _target_package in ("cuda", "rocm", "ascend", "metal"):
        _import_target_module(f"tilelang.{_target_package}.target")
    del _import_target_module, _target_package

_PUBLIC_ATTRIBUTES = {
    "jit": ("jit", "jit"),
    "JITKernel": ("jit", "JITKernel"),
    "compile": ("jit", "compile"),
    "par_compile": ("jit", "par_compile"),
    "Profiler": ("profiler", "Profiler"),
    "TensorSupplyType": ("utils", "TensorSupplyType"),
    "deprecated": ("utils", "deprecated"),
    "build_date": ("utils", "build_date"),
    "Layout": ("layout", "Layout"),
    "Fragment": ("layout", "Fragment"),
    "autotune": ("autotuner", "autotune"),
    "PassConfigKey": ("transform", "PassConfigKey"),
    "lower": ("engine", "lower"),
    "register_cuda_postproc": ("engine", "register_cuda_postproc"),
    "register_hip_postproc": ("engine", "register_hip_postproc"),
    "register_c_postproc": ("engine", "register_c_postproc"),
    "dtypes": ("language", "dtypes"),
}
_PUBLIC_MODULES = {
    "layout",
    "analysis",
    "driver",
    "transform",
    "language",
    "engine",
    "tools",
    "ir",
    "tileop",
    "cpu",
    "ascend",
    "cuda",
    "rocm",
    "metal",
    "webgpu",
}


def __getattr__(name):
    import importlib

    if name in _PUBLIC_ATTRIBUTES:
        module_name, attribute = _PUBLIC_ATTRIBUTES[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attribute)
    elif name in _PUBLIC_MODULES:
        value = importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _PUBLIC_ATTRIBUTES.keys() | _PUBLIC_MODULES)


from .math import next_power_of_2 as next_power_of_2, cdiv as cdiv
