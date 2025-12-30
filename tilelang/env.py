from __future__ import annotations
import sys
import os
import pathlib
import logging
import shutil
import glob
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# SETUP ENVIRONMENT VARIABLES
CUTLASS_NOT_FOUND_MESSAGE = "CUTLASS is not installed or found in the expected path"
", which may lead to compilation bugs when utilize tilelang backend."
COMPOSABLE_KERNEL_NOT_FOUND_MESSAGE = "Composable Kernel is not installed or found in the expected path"
", which may lead to compilation bugs when utilize tilelang backend."
TL_TEMPLATE_NOT_FOUND_MESSAGE = "TileLang is not installed or found in the expected path"
", which may lead to compilation bugs when utilize tilelang backend."
TVM_LIBRARY_NOT_FOUND_MESSAGE = "TVM is not installed or found in the expected path"

TL_ROOT = os.path.dirname(os.path.abspath(__file__))
# Only expose the internal lib directory to sys.path to avoid shadowing
# common top-level module names (e.g., utils, analysis) from user projects.
TL_LIBS = [os.path.join(TL_ROOT, "lib")]
TL_LIBS = [i for i in TL_LIBS if os.path.exists(i)]

DEV = False
THIRD_PARTY_ROOT = os.path.join(TL_ROOT, "3rdparty")
if not os.path.exists(THIRD_PARTY_ROOT):
    DEV = True
    tl_dev_root = os.path.dirname(TL_ROOT)

    dev_lib_root = os.path.join(tl_dev_root, "build")
    # In dev builds, place artifacts under build/lib and point search path there
    # to avoid adding the entire build root to sys.path.
    TL_LIBS = [os.path.join(dev_lib_root, "lib"), os.path.join(dev_lib_root, "tvm")]
    THIRD_PARTY_ROOT = os.path.join(tl_dev_root, "3rdparty")
    logger.warning(f"Loading tilelang libs from dev root: {dev_lib_root}")

assert TL_LIBS and all(os.path.exists(i) for i in TL_LIBS), f"tilelang lib root do not exists: {TL_LIBS}"

for lib in TL_LIBS:
    if lib not in sys.path:
        sys.path.insert(0, lib)


def _find_cuda_home() -> str:
    """Find the CUDA install path.

    Adapted from https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py
    """
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            # Standard CUDA pattern
            if "cuda" in nvcc_path.lower():
                cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
            # NVIDIA HPC SDK pattern
            elif "hpc_sdk" in nvcc_path.lower():
                # Navigate to the root directory of nvhpc
                cuda_home = os.path.dirname(os.path.dirname(os.path.dirname(nvcc_path)))
            # Generic fallback for non-standard or symlinked installs
            else:
                cuda_home = os.path.dirname(os.path.dirname(nvcc_path))

        else:
            # Guess #3
            if sys.platform == "win32":
                cuda_homes = glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")
                cuda_home = "" if len(cuda_homes) == 0 else cuda_homes[0]
            else:
                # Linux/macOS
                if os.path.exists("/usr/local/cuda"):
                    cuda_home = "/usr/local/cuda"
                elif os.path.exists("/opt/nvidia/hpc_sdk/Linux_x86_64"):
                    cuda_home = "/opt/nvidia/hpc_sdk/Linux_x86_64"

            # Validate found path
            if cuda_home is None or not os.path.exists(cuda_home):
                cuda_home = None

    return cuda_home if cuda_home is not None else ""


def _find_rocm_home() -> str:
    """Find the ROCM install path."""
    rocm_home = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if rocm_home is None:
        rocmcc_path = shutil.which("hipcc")
        if rocmcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(rocmcc_path))
        else:
            rocm_home = "/opt/rocm"
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
    _forced_value: str | None = None  # Temporary runtime override (mainly for tests/debugging)

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
    TILELANG_PRINT_ON_COMPILATION = EnvVar("TILELANG_PRINT_ON_COMPILATION", "1")  # print kernel name on compile
    TILELANG_DISABLE_CACHE = EnvVar(
        "TILELANG_DISABLE_CACHE", "0"
    )  # disable kernel cache, usually for unit testing / debugging, high priority
    TILELANG_CLEAR_CACHE = EnvVar("TILELANG_CLEAR_CACHE", "0")  # DEPRECATED! clear cache automatically if set

    # Kernel selection options
    # Default to GEMM v2; set to "1"/"true"/"yes"/"on" to force v1
    TILELANG_USE_GEMM_V1 = EnvVar("TILELANG_USE_GEMM_V1", "0")

    # Auto-tuning settings
    TILELANG_AUTO_TUNING_DISABLE_CACHE = EnvVar("TILELANG_AUTO_TUNING_DISABLE_CACHE", "0")
    TILELANG_AUTO_TUNING_CPU_UTILITIES = EnvVar("TILELANG_AUTO_TUNING_CPU_UTILITIES", "0.9")  # percent of CPUs used
    TILELANG_AUTO_TUNING_CPU_COUNTS = EnvVar("TILELANG_AUTO_TUNING_CPU_COUNTS", "-1")  # -1 means auto
    TILELANG_AUTO_TUNING_MAX_CPU_COUNT = EnvVar("TILELANG_AUTO_TUNING_MAX_CPU_COUNT", "-1")  # -1 means no limit

    # Compilation defaults (for jit, autotune, compile)
    # These allow overriding default compilation parameters via environment variables
    TILELANG_DEFAULT_TARGET = EnvVar("TILELANG_TARGET", "auto")
    TILELANG_DEFAULT_EXECUTION_BACKEND = EnvVar("TILELANG_EXECUTION_BACKEND", "auto")
    TILELANG_DEFAULT_VERBOSE = EnvVar("TILELANG_VERBOSE", "0")

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
        return not self.is_cache_globally_disabled() and CacheState.is_enabled()

    def enable_cache(self) -> None:
        CacheState.enable()

    def disable_cache(self) -> None:
        CacheState.disable()

    def is_cache_globally_disabled(self) -> bool:
        return self.TILELANG_DISABLE_CACHE.lower() in ("1", "true", "yes", "on")

    def is_autotune_cache_disabled(self) -> bool:
        return self.TILELANG_AUTO_TUNING_DISABLE_CACHE.lower() in ("1", "true", "yes", "on")

    def is_print_on_compilation_enabled(self) -> bool:
        return self.TILELANG_PRINT_ON_COMPILATION.lower() in ("1", "true", "yes", "on")

    def use_gemm_v1(self) -> bool:
        """Return True if GEMM v1 should be used based on env.

        Controlled by `TILELANG_USE_GEMM_V1`. Truthy values are one of
        {"1", "true", "yes", "on"} (case-insensitive).
        """
        return str(self.TILELANG_USE_GEMM_V1).lower() in ("1", "true", "yes", "on")

    def get_default_target(self) -> str:
        """Get default compilation target from environment."""
        return self.TILELANG_DEFAULT_TARGET

    def get_default_execution_backend(self) -> str:
        """Get default execution backend from environment."""
        return self.TILELANG_DEFAULT_EXECUTION_BACKEND

    def get_default_verbose(self) -> bool:
        """Get default verbose flag from environment."""
        return self.TILELANG_DEFAULT_VERBOSE.lower() in ("1", "true", "yes", "on")


# Instantiate as a global configuration object
env = Environment()

# Cache control API (wrap env, which is managed by CacheState and Environment Variables jointly)
enable_cache = env.enable_cache  # CacheState.enable
disable_cache = env.disable_cache  # CacheState.disable
is_cache_enabled = env.is_cache_enabled  # CacheState.is_enabled

# Export CUDA_HOME and ROCM_HOME, both are static variables
# after initialization.
CUDA_HOME = env.CUDA_HOME
ROCM_HOME = env.ROCM_HOME


def prepend_pythonpath(path):
    if not os.environ.get("PYTHONPATH", None):
        os.environ["PYTHONPATH"] = path
    else:
        os.environ["PYTHONPATH"] = path + os.pathsep + os.environ["PYTHONPATH"]

    sys.path.insert(0, path)


# Initialize TVM paths
if env.TVM_IMPORT_PYTHON_PATH is not None:
    prepend_pythonpath(env.TVM_IMPORT_PYTHON_PATH)
else:
    tvm_path = os.path.join(THIRD_PARTY_ROOT, "tvm", "python")
    assert os.path.exists(tvm_path), tvm_path
    if tvm_path not in sys.path:
        prepend_pythonpath(tvm_path)
        env.TVM_IMPORT_PYTHON_PATH = tvm_path
# By default, the built TVM-related libraries are stored in TL_LIBS.
if os.environ.get("TVM_LIBRARY_PATH") is None:
    os.environ["TVM_LIBRARY_PATH"] = env.TVM_LIBRARY_PATH = os.pathsep.join(TL_LIBS)

# Initialize CUTLASS paths
if os.environ.get("TL_CUTLASS_PATH", None) is None:
    cutlass_inc_path = os.path.join(THIRD_PARTY_ROOT, "cutlass", "include")
    if os.path.exists(cutlass_inc_path):
        os.environ["TL_CUTLASS_PATH"] = env.CUTLASS_INCLUDE_DIR = cutlass_inc_path
    else:
        logger.warning(CUTLASS_NOT_FOUND_MESSAGE)

# Initialize COMPOSABLE_KERNEL paths
if os.environ.get("TL_COMPOSABLE_KERNEL_PATH", None) is None:
    ck_inc_path = os.path.join(THIRD_PARTY_ROOT, "composable_kernel", "include")
    if os.path.exists(ck_inc_path):
        os.environ["TL_COMPOSABLE_KERNEL_PATH"] = env.COMPOSABLE_KERNEL_INCLUDE_DIR = ck_inc_path
    else:
        logger.warning(COMPOSABLE_KERNEL_NOT_FOUND_MESSAGE)

# Initialize TL_TEMPLATE_PATH
if os.environ.get("TL_TEMPLATE_PATH", None) is None:
    tl_template_path = os.path.join(THIRD_PARTY_ROOT, "..", "src")
    if os.path.exists(tl_template_path):
        os.environ["TL_TEMPLATE_PATH"] = env.TILELANG_TEMPLATE_PATH = tl_template_path
    else:
        logger.warning(TL_TEMPLATE_NOT_FOUND_MESSAGE)

# Export static variables after initialization.
CUTLASS_INCLUDE_DIR = env.CUTLASS_INCLUDE_DIR
COMPOSABLE_KERNEL_INCLUDE_DIR = env.COMPOSABLE_KERNEL_INCLUDE_DIR
TILELANG_TEMPLATE_PATH = env.TILELANG_TEMPLATE_PATH
