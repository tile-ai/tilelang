import sys
import os
import pathlib
import logging
import shutil
import glob
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# SETUP ENVIRONMENT VARIABLES
CUTLASS_NOT_FOUND_MESSAGE = ("CUTLASS is not installed or found in the expected path")
", which may lead to compilation bugs when utilize tilelang backend."
COMPOSABLE_KERNEL_NOT_FOUND_MESSAGE = (
    "Composable Kernel is not installed or found in the expected path")
", which may lead to compilation bugs when utilize tilelang backend."
TL_TEMPLATE_NOT_FOUND_MESSAGE = ("TileLang is not installed or found in the expected path")
", which may lead to compilation bugs when utilize tilelang backend."
TVM_LIBRARY_NOT_FOUND_MESSAGE = ("TVM is not installed or found in the expected path")


def _find_cuda_home() -> str:
    """Find the CUDA install path.

    Adapted from https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py
    """
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None and "cuda" in nvcc_path.lower():
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            if sys.platform == 'win32':
                cuda_homes = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                cuda_home = '' if len(cuda_homes) == 0 else cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home if cuda_home is not None else ""


def _find_rocm_home() -> str:
    """Find the ROCM install path."""
    rocm_home = os.environ.get('ROCM_PATH') or os.environ.get('ROCM_HOME')
    if rocm_home is None:
        rocmcc_path = shutil.which("hipcc")
        if rocmcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(rocmcc_path))
        else:
            rocm_home = '/opt/rocm'
            if not os.path.exists(rocm_home):
                rocm_home = None
    return rocm_home if rocm_home is not None else ""


# Cache control
class CacheState:
    """Class to manage global kernel caching state."""
    _enabled = True

    @classmethod
    def enable(cls):
        """Enable kernel caching globally."""
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Disable kernel caching globally."""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """Return current cache state."""
        return cls._enabled


@dataclass
class EnvVar:
    """
    Descriptor for managing access to a single environment variable.

    Purpose
    -------
    In many projects, access to environment variables is scattered across the codebase:
        * `os.environ.get(...)` calls are repeated everywhere
        * Default values are hard-coded in multiple places
        * Overriding env vars for tests/debugging is messy
        * There's no central place to see all environment variables a package uses

    This descriptor solves those issues by:
        1. Centralizing the definition of the variable's **key** and **default value**
        2. Allowing *dynamic* reads from `os.environ` so changes take effect immediately
        3. Supporting **forced overrides** at runtime (for unit tests or debugging)
        4. Logging a warning when a forced value is used (helps detect unexpected overrides)
        5. Optionally syncing forced values back to `os.environ` if global consistency is desired

    How it works
    ------------
    - This is a `dataclass` implementing the descriptor protocol (`__get__`, `__set__`)
    - When used as a class attribute, `instance.attr` triggers `__get__()`
        → returns either the forced override or the live value from `os.environ`
    - Assigning to the attribute (`instance.attr = value`) triggers `__set__()`
        → stores `_forced_value` for future reads
    - You may uncomment the `os.environ[...] = value` line in `__set__` if you want
      the override to persist globally in the process

    Example
    -------
    ```python
    class Environment:
        TILELANG_PRINT_ON_COMPILATION = EnvVar("TILELANG_PRINT_ON_COMPILATION", "0")

    env = Environment()
    print(cfg.TILELANG_PRINT_ON_COMPILATION)  # Reads from os.environ (with default fallback)
    cfg.TILELANG_PRINT_ON_COMPILATION = "1"   # Forces value to "1" until changed/reset
    ```

    Benefits
    --------
    * Centralizes all env-var keys and defaults in one place
    * Live, up-to-date reads (no stale values after `import`)
    * Testing convenience (override without touching the real env)
    * Improves IDE discoverability and type hints
    * Avoids hardcoding `os.environ.get(...)` in multiple places
    """

    key: str  # Environment variable name (e.g. "TILELANG_PRINT_ON_COMPILATION")
    default: str  # Default value if the environment variable is not set
    _forced_value: Optional[str] = None  # Temporary runtime override (mainly for tests/debugging)

    def get(self):
        if self._forced_value is not None:
            return self._forced_value
        return os.environ.get(self.key, self.default)

    def __get__(self, instance, owner):
        """
        Called when the attribute is accessed.
        1. If a forced value is set, return it and log a warning
        2. Otherwise, look up the value in os.environ; return the default if missing
        """
        return self.get()

    def __set__(self, instance, value):
        """
        Called when the attribute is assigned to.
        Stores the value as a runtime override (forced value).
        Optionally, you can also sync this into os.environ for global effect.
        """
        self._forced_value = value
        # Uncomment the following line if you want the override to persist globally:
        # os.environ[self.key] = value


# Cache control API (wrap CacheState)
enable_cache = CacheState.enable
disable_cache = CacheState.disable
is_cache_enabled = CacheState.is_enabled


# Utility function for environment variables with defaults
# Assuming EnvVar and CacheState are defined elsewhere
class Environment:
    """
    Environment configuration for TileLang.
    Handles CUDA/ROCm detection, integration paths, template/cache locations,
    auto-tuning configs, and build options.
    """

    # CUDA/ROCm home directories
    CUDA_HOME = _find_cuda_home()
    ROCM_HOME = _find_rocm_home()

    # Path to the TileLang package root
    TILELANG_PACKAGE_PATH = pathlib.Path(__file__).resolve().parent

    # External library include paths
    CUTLASS_INCLUDE_DIR = EnvVar("TL_CUTLASS_PATH", None)
    COMPOSABLE_KERNEL_INCLUDE_DIR = EnvVar("TL_COMPOSABLE_KERNEL_PATH", None)

    # TVM integration
    TVM_PYTHON_PATH = EnvVar("TVM_IMPORT_PYTHON_PATH", None)
    TVM_LIBRARY_PATH = EnvVar("TVM_LIBRARY_PATH", None)

    # TileLang resources
    TILELANG_TEMPLATE_PATH = EnvVar("TL_TEMPLATE_PATH", None)
    TILELANG_CACHE_DIR = EnvVar("TILELANG_CACHE_DIR", os.path.expanduser("~/.tilelang/cache"))
    TILELANG_TMP_DIR = EnvVar("TILELANG_TMP_DIR", os.path.join(TILELANG_CACHE_DIR.get(), "tmp"))

    # Kernel Build options
    TILELANG_PRINT_ON_COMPILATION = EnvVar("TILELANG_PRINT_ON_COMPILATION",
                                           "1")  # print kernel name on compile
    TILELANG_CLEAR_CACHE = EnvVar("TILELANG_CLEAR_CACHE", "0")  # clear cache automatically if set

    # Auto-tuning settings
    TILELANG_AUTO_TUNING_CPU_UTILITIES = EnvVar("TILELANG_AUTO_TUNING_CPU_UTILITIES",
                                                "0.9")  # percent of CPUs used
    TILELANG_AUTO_TUNING_CPU_COUNTS = EnvVar("TILELANG_AUTO_TUNING_CPU_COUNTS",
                                             "-1")  # -1 means auto
    TILELANG_AUTO_TUNING_MAX_CPU_COUNT = EnvVar("TILELANG_AUTO_TUNING_MAX_CPU_COUNT",
                                                "-1")  # -1 means no limit

    # TVM integration
    SKIP_LOADING_TILELANG_SO = EnvVar("SKIP_LOADING_TILELANG_SO", "0")
    TVM_IMPORT_PYTHON_PATH = EnvVar("TVM_IMPORT_PYTHON_PATH", None)

    def _initialize_torch_cuda_arch_flags(self) -> None:
        """
        Detect target CUDA architecture and set TORCH_CUDA_ARCH_LIST
        to ensure PyTorch extensions are built for the proper GPU arch.
        """
        from tilelang.contrib import nvcc
        from tilelang.utils.target import determine_target

        target = determine_target(return_object=True)  # get target GPU
        compute_version = nvcc.get_target_compute_version(target)  # e.g. "8.6"
        major, minor = nvcc.parse_compute_version(compute_version)  # split to (8, 6)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"  # set env var for PyTorch

    # Cache control API (wrap CacheState)
    def is_cache_enabled(self) -> bool:
        return CacheState.is_enabled()

    def enable_cache(self) -> None:
        CacheState.enable()

    def disable_cache(self) -> None:
        CacheState.disable()

    def is_print_on_compilation_enabled(self) -> bool:
        return self.TILELANG_PRINT_ON_COMPILATION.lower() in ("1", "true", "yes", "on")


# Instantiate as a global configuration object
env = Environment()

# Export CUDA_HOME and ROCM_HOME, both are static variables
# after initialization.
CUDA_HOME = env.CUDA_HOME
ROCM_HOME = env.ROCM_HOME

# Initialize TVM paths
if env.TVM_IMPORT_PYTHON_PATH is not None:
    os.environ["PYTHONPATH"] = env.TVM_IMPORT_PYTHON_PATH + ":" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, env.TVM_IMPORT_PYTHON_PATH)
else:
    install_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rdparty", "tvm")
    if os.path.exists(install_tvm_path) and install_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = (
            install_tvm_path + "/python:" + os.environ.get("PYTHONPATH", ""))
        sys.path.insert(0, install_tvm_path + "/python")
        env.TVM_IMPORT_PYTHON_PATH = install_tvm_path + "/python"

    develop_tvm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "tvm")
    if os.path.exists(develop_tvm_path) and develop_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = (
            develop_tvm_path + "/python:" + os.environ.get("PYTHONPATH", ""))
        sys.path.insert(0, develop_tvm_path + "/python")
        env.TVM_IMPORT_PYTHON_PATH = develop_tvm_path + "/python"

    develop_tvm_library_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "build", "tvm")
    install_tvm_library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    if os.environ.get("TVM_LIBRARY_PATH") is None:
        if os.path.exists(develop_tvm_library_path):
            os.environ["TVM_LIBRARY_PATH"] = develop_tvm_library_path
        elif os.path.exists(install_tvm_library_path):
            os.environ["TVM_LIBRARY_PATH"] = install_tvm_library_path
        else:
            logger.warning(TVM_LIBRARY_NOT_FOUND_MESSAGE)
        # pip install build library path
        lib_path = os.path.join(env.TILELANG_PACKAGE_PATH, "lib")
        existing_path = os.environ.get("TVM_LIBRARY_PATH")
        if existing_path:
            os.environ["TVM_LIBRARY_PATH"] = f"{existing_path}:{lib_path}"
        else:
            os.environ["TVM_LIBRARY_PATH"] = lib_path
        env.TVM_LIBRARY_PATH = os.environ.get("TVM_LIBRARY_PATH", None)

# Initialize CUTLASS paths
if os.environ.get("TL_CUTLASS_PATH", None) is None:
    install_cutlass_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "3rdparty", "cutlass")
    develop_cutlass_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "cutlass")
    if os.path.exists(install_cutlass_path):
        os.environ["TL_CUTLASS_PATH"] = install_cutlass_path + "/include"
        env.CUTLASS_INCLUDE_DIR = install_cutlass_path + "/include"
    elif (os.path.exists(develop_cutlass_path) and develop_cutlass_path not in sys.path):
        os.environ["TL_CUTLASS_PATH"] = develop_cutlass_path + "/include"
        env.CUTLASS_INCLUDE_DIR = develop_cutlass_path + "/include"
    else:
        logger.warning(CUTLASS_NOT_FOUND_MESSAGE)

# Initialize COMPOSABLE_KERNEL paths
if os.environ.get("TL_COMPOSABLE_KERNEL_PATH", None) is None:
    install_ck_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "3rdparty", "composable_kernel")
    develop_ck_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "composable_kernel")
    if os.path.exists(install_ck_path):
        os.environ["TL_COMPOSABLE_KERNEL_PATH"] = install_ck_path + "/include"
        env.COMPOSABLE_KERNEL_INCLUDE_DIR = install_ck_path + "/include"
    elif (os.path.exists(develop_ck_path) and develop_ck_path not in sys.path):
        os.environ["TL_COMPOSABLE_KERNEL_PATH"] = develop_ck_path + "/include"
        env.COMPOSABLE_KERNEL_INCLUDE_DIR = develop_ck_path + "/include"
    else:
        logger.warning(COMPOSABLE_KERNEL_NOT_FOUND_MESSAGE)

# Initialize TL_TEMPLATE_PATH
if os.environ.get("TL_TEMPLATE_PATH", None) is None:
    install_tl_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    develop_tl_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    if os.path.exists(install_tl_template_path):
        os.environ["TL_TEMPLATE_PATH"] = install_tl_template_path
        env.TILELANG_TEMPLATE_PATH = install_tl_template_path
    elif (os.path.exists(develop_tl_template_path) and develop_tl_template_path not in sys.path):
        os.environ["TL_TEMPLATE_PATH"] = develop_tl_template_path
        env.TILELANG_TEMPLATE_PATH = develop_tl_template_path
    else:
        logger.warning(TL_TEMPLATE_NOT_FOUND_MESSAGE)

# Export static variables after initialization.
CUTLASS_INCLUDE_DIR = env.CUTLASS_INCLUDE_DIR
COMPOSABLE_KERNEL_INCLUDE_DIR = env.COMPOSABLE_KERNEL_INCLUDE_DIR
TILELANG_TEMPLATE_PATH = env.TILELANG_TEMPLATE_PATH
