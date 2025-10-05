import fcntl
import functools
import hashlib
import io
import subprocess
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from typing import List, Optional
import re
import tarfile
from io import BytesIO
from pathlib import Path
import os
import sys
import site
import sysconfig
import urllib.request
from packaging.version import Version
import platform
import multiprocessing
from setuptools.command.build_ext import build_ext
import importlib
import logging

# Configure logging with basic settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

# Environment variables False/True
PYPI_BUILD = os.environ.get("PYPI_BUILD", "False").lower() == "true"
PACKAGE_NAME = "tilelang"
ROOT_DIR = os.path.dirname(__file__)

# Add LLVM control environment variable
USE_LLVM = os.environ.get("USE_LLVM", "False").lower() == "true"
# Add ROCM control environment variable
USE_ROCM = os.environ.get("USE_ROCM", "False").lower() == "true"
# Build with Debug mode
DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() == "true"
# Include commit ID in wheel filename and package metadata
WITH_COMMITID = os.environ.get("WITH_COMMITID", "True").lower() == "true"


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


envs = load_module_from_path('env', os.path.join(ROOT_DIR, PACKAGE_NAME, 'env.py'))

CUDA_HOME = envs.CUDA_HOME
ROCM_HOME = envs.ROCM_HOME

# Check if both CUDA and ROCM are enabled
if USE_ROCM and not ROCM_HOME:
    raise ValueError(
        "ROCM support is enabled (USE_ROCM=True) but ROCM_HOME is not set or detected.")

# For LLVM-only builds, skip CUDA/ROCM validation
if not USE_LLVM:
    if not USE_ROCM and not CUDA_HOME:
        raise ValueError(
            "CUDA support is enabled by default (USE_ROCM=False) but CUDA_HOME is not set or detected."
        )

    # Ensure one of CUDA or ROCM is available
    if not (CUDA_HOME or ROCM_HOME):
        raise ValueError(
            "Failed to automatically detect CUDA or ROCM installation. Please set the CUDA_HOME or ROCM_HOME environment variable manually (e.g., export CUDA_HOME=/usr/local/cuda or export ROCM_HOME=/opt/rocm)."
        )

# TileLang only supports Linux platform
assert sys.platform.startswith("linux"), "TileLang only supports Linux platform (including WSL)."


def _is_linux_like():
    return (sys.platform == "darwin" or sys.platform.startswith("linux") or
            sys.platform.startswith("freebsd"))


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements(file_path: str = "requirements.txt") -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path(file_path)) as f:
        requirements = f.read().strip().split("\n")
    return requirements


def find_version(version_file_path: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    # Read and store the version information from the VERSION file
    # Use 'strip()' to remove any leading/trailing whitespace or newline characters
    if not os.path.exists(version_file_path):
        raise FileNotFoundError(f"Version file not found at {version_file_path}")
    with open(version_file_path, "r") as version_file:
        version = version_file.read().strip()
    return version


def get_nvcc_cuda_version():
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_path = os.path.join(CUDA_HOME, "bin", "nvcc")
    nvcc_output = subprocess.check_output([nvcc_path, "-V"], universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = Version(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_rocm_version():
    """Get the ROCM version from rocminfo."""
    rocm_output = subprocess.check_output(["rocminfo"], universal_newlines=True)
    # Parse ROCM version from output
    # Example output: ROCM version: x.y.z-...
    match = re.search(r'ROCm Version: (\d+\.\d+\.\d+)', rocm_output)
    if match:
        return Version(match.group(1))
    else:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        rocm_version_file = os.path.join(rocm_path, "lib", "cmake", "rocm",
                                         "rocm-config-version.cmake")
        if os.path.exists(rocm_version_file):
            with open(rocm_version_file, "r") as f:
                content = f.read()
                match = re.search(r'set\(PACKAGE_VERSION "(\d+\.\d+\.\d+)"', content)
                if match:
                    return Version(match.group(1))
    # return a default
    return Version("5.0.0")


def get_tilelang_version(with_cuda=True, with_system_info=True, with_commit_id=False) -> str:
    version = find_version(get_path(".", "VERSION"))
    local_version_parts = []
    if with_system_info:
        local_version_parts.append(get_system_info().replace("-", "."))

    if with_cuda:
        if USE_ROCM:
            if ROCM_HOME:
                rocm_version = str(get_rocm_version())
                rocm_version_str = rocm_version.replace(".", "")[:3]
                local_version_parts.append(f"rocm{rocm_version_str}")
        else:
            if CUDA_HOME:
                nvcc_path = os.path.join(CUDA_HOME, "bin", "nvcc")
                if os.path.exists(nvcc_path):
                    cuda_version = str(get_nvcc_cuda_version())
                    cuda_version_str = cuda_version.replace(".", "")[:3]
                    local_version_parts.append(f"cu{cuda_version_str}")

    if local_version_parts:
        version += f"+{'.'.join(local_version_parts)}"

    if with_commit_id:
        commit_id = None
        try:
            commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                stderr=subprocess.DEVNULL,
                                                encoding='utf-8').strip()
        except subprocess.SubprocessError as error:
            logger.warning(f"Ignore commit id because failed to get git commit id: {str(error)}")
        if commit_id:
            # Truncate commit ID to 8 characters to keep version string reasonable
            short_commit_id = commit_id[:8]
            if local_version_parts:
                version += f".{short_commit_id}"
            else:
                version += f"+{short_commit_id}"

    return version


@functools.lru_cache(maxsize=None)
def get_cplus_compiler():
    """Return the path to the default C/C++ compiler.

    Returns
    -------
    out: Optional[str]
        The path to the default C/C++ compiler, or None if none was found.
    """

    if not _is_linux_like():
        return None

    env_cxx = os.environ.get("CXX") or os.environ.get("CC")
    if env_cxx:
        return env_cxx
    cc_names = ["g++", "clang++", "c++"]
    dirs_in_path = os.get_exec_path()
    for cc in cc_names:
        for d in dirs_in_path:
            cc_path = os.path.join(d, cc)
            if os.path.isfile(cc_path) and os.access(cc_path, os.X_OK):
                return cc_path
    return None


@functools.lru_cache(maxsize=None)
def get_cython_compiler() -> Optional[str]:
    """Return the path to the Cython compiler.

    Returns
    -------
    out: Optional[str]
        The path to the Cython compiler, or None if none was found.
    """

    cython_names = ["cython", "cython3"]

    # Check system PATH
    dirs_in_path = list(os.get_exec_path())

    # Add user site-packages bin directory
    user_base = site.getuserbase()
    if user_base:
        user_bin = os.path.join(user_base, "bin")
        if os.path.exists(user_bin):
            dirs_in_path = [user_bin] + dirs_in_path

    # If in a virtual environment, add its bin directory
    if sys.prefix != sys.base_prefix:
        venv_bin = os.path.join(sys.prefix, "bin")
        if os.path.exists(venv_bin):
            dirs_in_path = [venv_bin] + dirs_in_path

    for cython_name in cython_names:
        for d in dirs_in_path:
            cython_path = os.path.join(d, cython_name)
            if os.path.isfile(cython_path) and os.access(cython_path, os.X_OK):
                return cython_path
    return None


@functools.lru_cache(maxsize=None)
def get_cmake_path() -> str:
    """Return the path to the CMake compiler.
    """
    # found which cmake is used
    cmake_path = shutil.which("cmake")
    if not os.path.exists(cmake_path):
        raise Exception("CMake is not installed, please install it first.")
    return cmake_path


def get_system_info():
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
            version_id_match = re.search(r'VERSION_ID="(\d+\.\d+)"', os_release)
            if version_id_match:
                version_id = version_id_match.group(1)
                distro = "ubuntu"
                return f"{distro}-{version_id}"
        except FileNotFoundError:
            pass
    return system


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def download_and_extract_llvm(version, is_aarch64=False, extract_path="3rdparty"):
    """
    Downloads and extracts the specified version of LLVM for the given platform.
    Args:
        version (str): The version of LLVM to download.
        is_aarch64 (bool): True if the target platform is aarch64, False otherwise.
        extract_path (str): The directory path where the archive will be extracted.

    Returns:
        str: The path where the LLVM archive was extracted.
    """
    ubuntu_version = "16.04"
    if version >= "16.0.0":
        ubuntu_version = "20.04"
    elif version >= "13.0.0":
        ubuntu_version = "18.04"

    base_url = (f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{version}")
    file_name = f"clang+llvm-{version}-{'aarch64-linux-gnu' if is_aarch64 else f'x86_64-linux-gnu-ubuntu-{ubuntu_version}'}.tar.xz"

    download_url = f"{base_url}/{file_name}"

    # Download the file
    logger.info(f"Downloading {file_name} from {download_url}")
    with urllib.request.urlopen(download_url) as response:
        if response.status != 200:
            raise Exception(f"Download failed with status code {response.status}")
        file_content = response.read()
    # Ensure the extract path exists
    os.makedirs(extract_path, exist_ok=True)

    # if the file already exists, remove it
    if os.path.exists(os.path.join(extract_path, file_name)):
        os.remove(os.path.join(extract_path, file_name))

    # Extract the file
    logger.info(f"Extracting {file_name} to {extract_path}")
    with tarfile.open(fileobj=BytesIO(file_content), mode="r:xz") as tar:
        tar.extractall(path=extract_path)

    logger.info("Download and extraction completed successfully.")
    return os.path.abspath(os.path.join(extract_path, file_name.replace(".tar.xz", "")))


package_data = {
    "tilelang": ["py.typed", "*pyx"],
}

LLVM_VERSION = "10.0.1"
IS_AARCH64 = False  # Set to True if on an aarch64 platform
EXTRACT_PATH = "3rdparty"  # Default extraction path


def update_submodules():
    """Updates git submodules if in a git repository."""

    def is_git_repo():
        try:
            # Check if current directory is a git repository
            subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"],
                                    stderr=subprocess.STDOUT)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    if not is_git_repo():
        logger.info("Info: Not a git repository, skipping submodule update.")
        return

    try:
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to update submodules") from error


def setup_llvm_for_tvm():
    """Downloads and extracts LLVM, then configures TVM to use it.

    If system llvm-config is available, uses that instead of downloading LLVM.
    """
    # Check if system llvm-config is available
    system_llvm_config = shutil.which("llvm-config")
    if system_llvm_config:
        logger.info(f"Using system llvm-config at {system_llvm_config}")
        extract_path = os.path.dirname(os.path.dirname(system_llvm_config))  # Go up to LLVM root
        return extract_path, system_llvm_config

    # Otherwise download LLVM
    logger.info(f"System llvm-config not found, downloading LLVM {LLVM_VERSION}")
    extract_path = download_and_extract_llvm(LLVM_VERSION, IS_AARCH64, EXTRACT_PATH)
    llvm_config_path = os.path.join(extract_path, "bin", "llvm-config")
    return extract_path, llvm_config_path


def patch_libs(libpath):
    """
    tvm and tilelang libs are copied from elsewhere into wheels
    and have a hard-coded rpath.
    Set rpath to the directory of libs so auditwheel works well.
    """
    # check if patchelf is installed
    # find patchelf in the system
    patchelf_path = shutil.which("patchelf")
    if not patchelf_path:
        logger.warning(
            "patchelf is not installed, which is required for auditwheel to work for compatible wheels."
        )
        return
    subprocess.run([patchelf_path, '--set-rpath', '$ORIGIN', libpath])


class TileLangBuilPydCommand(build_py):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        build_py.run(self)
        self.run_command("build_ext")
        build_ext_cmd = self.get_finalized_command("build_ext")
        build_temp_dir = build_ext_cmd.build_temp
        ext_modules = build_ext_cmd.extensions
        for ext in ext_modules:
            extdir = build_ext_cmd.get_ext_fullpath(ext.name)
            logger.info(f"Extension {ext.name} output directory: {extdir}")

        ext_output_dir = os.path.dirname(extdir)
        logger.info(f"Extension output directory (parent): {ext_output_dir}")
        logger.info(f"Build temp directory: {build_temp_dir}")

        # copy cython files
        CYTHON_SRC = [
            "tilelang/jit/adapter/cython/cython_wrapper.pyx",
            "tilelang/jit/adapter/cython/.cycache",
        ]
        for item in CYTHON_SRC:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                self.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        # copy the tl_templates
        TILELANG_SRC = [
            "src/tl_templates",
        ]
        for item in TILELANG_SRC:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                self.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        TVM_PREBUILD_ITEMS = [
            "libtvm_runtime.so",
            "libtvm.so",
            "libtilelang.so",
            "libtilelang_module.so",
        ]

        potential_dirs = [
            ext_output_dir,
            self.build_lib,
            build_temp_dir,
            os.path.join(ROOT_DIR, "build"),
        ]

        for item in TVM_PREBUILD_ITEMS:
            source_lib_file = None
            for dir in potential_dirs:
                candidate = os.path.join(dir, item)
                if os.path.exists(candidate):
                    source_lib_file = candidate
                    break

            if source_lib_file:
                patch_libs(source_lib_file)
                target_dir_release = os.path.join(self.build_lib, PACKAGE_NAME, "lib")
                target_dir_develop = os.path.join(PACKAGE_NAME, "lib")
                os.makedirs(target_dir_release, exist_ok=True)
                os.makedirs(target_dir_develop, exist_ok=True)
                shutil.copy2(source_lib_file, target_dir_release)
                logger.info(f"Copied {source_lib_file} to {target_dir_release}")
                shutil.copy2(source_lib_file, target_dir_develop)
                logger.info(f"Copied {source_lib_file} to {target_dir_develop}")
                os.remove(source_lib_file)
            else:
                logger.info(f"WARNING: {item} not found in any expected directories!")

        TVM_CONFIG_ITEMS = [
            f"{build_temp_dir}/config.cmake",
        ]
        for item in TVM_CONFIG_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            # only copy the file
            file_name = os.path.basename(item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, file_name)
            target_dir = os.path.dirname(target_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if os.path.exists(source_dir):
                shutil.copy2(source_dir, target_dir)
            else:
                logger.info(f"INFO: {source_dir} does not exist.")

        TVM_PACAKGE_ITEMS = [
            "3rdparty/tvm/src",
            "3rdparty/tvm/python",
            "3rdparty/tvm/licenses",
            "3rdparty/tvm/conftest.py",
            "3rdparty/tvm/CONTRIBUTORS.md",
            "3rdparty/tvm/KEYS",
            "3rdparty/tvm/LICENSE",
            "3rdparty/tvm/README.md",
            "3rdparty/tvm/mypy.ini",
            "3rdparty/tvm/pyproject.toml",
            "3rdparty/tvm/version.py",
        ]
        for item in TVM_PACAKGE_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                self.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        # Copy CUTLASS to the package directory
        CUTLASS_PREBUILD_ITEMS = [
            "3rdparty/cutlass/include",
            "3rdparty/cutlass/tools",
        ]
        for item in CUTLASS_PREBUILD_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                self.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)
        # copy compoable kernel to the package directory
        CK_PREBUILD_ITEMS = [
            "3rdparty/composable_kernel/include",
            "3rdparty/composable_kernel/library",
        ]
        for item in CK_PREBUILD_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                self.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        # copy compoable kernel to the package directory
        TL_CONFIG_ITEMS = ["CMakeLists.txt", "VERSION", "README.md", "LICENSE"]
        for item in TL_CONFIG_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            # if is VERSION file, replace the content with the new version with commit id
            if not PYPI_BUILD and item == "VERSION":
                version = get_tilelang_version(
                    with_cuda=False, with_system_info=False, with_commit_id=WITH_COMMITID)
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                with open(os.path.join(target_dir, item), "w") as f:
                    print(f"Writing {version} to {os.path.join(target_dir, item)}")
                    f.write(version)
                continue

            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                self.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)


class TileLangSdistCommand(sdist):
    """Customized setuptools sdist command - includes the pyproject.toml file."""

    def make_distribution(self):
        self.distribution.metadata.name = PACKAGE_NAME
        self.distribution.metadata.version = get_tilelang_version(
            with_cuda=False, with_system_info=False, with_commit_id=WITH_COMMITID)
        super().make_distribution()


class CMakeExtension(Extension):
    """
    A specialized setuptools Extension class for building a CMake project.

    :param name: Name of the extension module.
    :param sourcedir: Directory containing the top-level CMakeLists.txt.
    """

    def __init__(self, name, sourcedir=""):
        # We pass an empty 'sources' list because
        # the actual build is handled by CMake, not setuptools.
        super().__init__(name=name, sources=[])

        # Convert the source directory to an absolute path
        # so that CMake can correctly locate the CMakeLists.txt.
        self.sourcedir = os.path.abspath(sourcedir)


class CythonExtension(Extension):
    """
    A specialized setuptools Extension class for building a Cython project.
    """

    def __init__(self, name, sourcedir=""):
        super().__init__(name=name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class TilelangExtensionBuild(build_ext):
    """
    Custom build_ext command for CMake-based projects.

    This class overrides the 'run' method to ensure that CMake is available,
    and then iterates over all extensions defined as CMakeExtension,
    delegating the actual build logic to 'build_cmake'.
    """

    def run(self):
        # Check if CMake is installed and accessible by attempting to run 'cmake --version'.
        try:
            cmake_path = get_cmake_path()
            if not cmake_path:
                raise Exception("CMake is not installed, please install it first.")
            subprocess.check_output([cmake_path, "--version"])
        except OSError as error:
            # If CMake is not found, raise an error.
            raise RuntimeError(
                "CMake must be installed to build the following extensions") from error

        update_submodules()

        # Build each extension (of type CMakeExtension) using our custom method.
        for ext in self.extensions:
            if isinstance(ext, CythonExtension):
                self.build_cython(ext)
            elif isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
            else:
                raise ValueError(f"Unsupported extension type: {type(ext)}")

        # To make it works with editable install,
        # we need to copy the lib*.so files to the tilelang/lib directory
        import glob
        files = glob.glob("*.so")
        if os.path.exists(PACKAGE_NAME):
            target_lib_dir = os.path.join(PACKAGE_NAME, "lib")
            for file in files:
                if not os.path.exists(target_lib_dir):
                    os.makedirs(target_lib_dir)
                shutil.copy(file, target_lib_dir)
                # remove the original file
                os.remove(file)

    def build_cython(self, ext):
        """
        Build a single Cython-based extension.

        :param ext: The extension (an instance of CythonExtension).
        """
        cython_compiler = get_cython_compiler()
        if not cython_compiler:
            logger.info("Cython compiler not found, install it first")
            subprocess.check_call(["pip", "install", "cython"])
            cython_compiler = get_cython_compiler()
            if not cython_compiler:
                raise Exception("Cython is not installed, please install it first.")

        logger.info(f"Using Cython compiler: {cython_compiler}")
        cython_warpper_dir = os.path.join(ext.sourcedir, "tilelang", "jit", "adapter", "cython")
        cython_wrapper_path = os.path.join(cython_warpper_dir, "cython_wrapper.pyx")
        py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
        cache_dir = Path(cython_warpper_dir) / ".cycache" / py_version
        os.makedirs(cache_dir, exist_ok=True)

        with open(cython_wrapper_path, "r") as f:
            cython_wrapper_code = f.read()
            source_path = cache_dir / "cython_wrapper.cpp"
            library_path = cache_dir / "cython_wrapper.so"
            md5_path = cache_dir / "md5.txt"
            code_hash = hashlib.sha256(cython_wrapper_code.encode()).hexdigest()
            cache_path = cache_dir / f"{code_hash}.so"
            lock_file = cache_path.with_suffix('.lock')

            # Check if cached version exists and is valid
            need_compile = True
            if md5_path.exists() and library_path.exists():
                with open(md5_path, "r") as f:
                    cached_hash = f.read().strip()
                    if cached_hash == code_hash:
                        logger.info("Cython JIT adapter is up to date, no need to compile...")
                        need_compile = False
                    else:
                        logger.info("Cython JIT adapter is out of date, need to recompile...")
            else:
                logger.info("No cached version found for Cython JIT adapter, need to compile...")

            if need_compile:
                logger.info("Waiting for lock to compile Cython JIT adapter...")
                with open(lock_file, 'w') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    try:
                        # After acquiring the lock, check again if the file has been compiled by another process
                        if md5_path.exists() and library_path.exists():
                            with open(md5_path, "r") as f:
                                cached_hash = f.read().strip()
                                if cached_hash == code_hash:
                                    logger.info(
                                        "Another process has already compiled the file, using it..."
                                    )
                                    need_compile = False

                        if need_compile:
                            logger.info("Compiling Cython JIT adapter...")
                            temp_path = cache_dir / f"temp_{code_hash}.so"

                            with open(md5_path, "w") as f:
                                f.write(code_hash)

                            # compile the cython_wrapper.pyx file into .cpp
                            cython = get_cython_compiler()
                            if cython is None:
                                raise Exception("Cython is not installed, please install it first.")
                            os.system(f"{cython} {cython_wrapper_path} --cplus -o {source_path}")
                            python_include_path = sysconfig.get_path("include")
                            cc = get_cplus_compiler()
                            command = f"{cc} -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I{python_include_path} {source_path} -o {temp_path}"
                            os.system(command)

                            # rename the temp file to the library file
                            temp_path.rename(library_path)
                    except Exception as e:
                        if 'temp_path' in locals() and temp_path.exists():
                            temp_path.unlink()
                        raise Exception(f"Failed to compile Cython JIT adapter: {e}") from e
                    finally:
                        if lock_file.exists():
                            lock_file.unlink()

            # add the .so file to the sys.path
            cache_dir_str = str(cache_dir)
            if cache_dir_str not in sys.path:
                sys.path.append(cache_dir_str)

    def build_cmake(self, ext):
        """
        Build a single CMake-based extension by generating a CMake config and invoking CMake/Ninja.

        Generates or updates a config.cmake in the build directory (based on the extension's sourcedir),
        injecting LLVM/CUDA/ROCm and Python settings, then runs CMake to configure and build the target.
        When running an in-place build the resulting library is placed under ./tilelang/lib; otherwise the
        standard extension output directory is used.

        Parameters:
            ext: The CMakeExtension to build; its `sourcedir` should contain the TVM/CMake `config.cmake`
                 template under `3rdparty/tvm/cmake/`.

        Raises:
            subprocess.CalledProcessError: If the CMake configuration or build commands fail.
            OSError: If filesystem operations (read/write) fail.
        """
        # Only setup LLVM if it's enabled
        llvm_config_path = "OFF"
        if USE_LLVM:
            # Setup LLVM for TVM and retrieve the path to llvm-config
            _, llvm_config_path = setup_llvm_for_tvm()

        # Determine the directory where the final .so or .pyd library should go.
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # To make it compatible with in-place build and avoid redundant link during incremental build,
        # we need to change the build destination to tilelang/lib, where it's actually loaded
        if self.inplace:
            extdir = os.path.abspath('./tilelang/lib/')

        # Prepare arguments for the CMake configuration step.
        # -DCMAKE_LIBRARY_OUTPUT_DIRECTORY sets where built libraries go
        # -DPYTHON_EXECUTABLE ensures that the correct Python is used
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if DEBUG_MODE else 'Release'}",
            "-G",
            "Ninja",
        ]
        if not USE_ROCM:
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={os.path.join(CUDA_HOME, 'bin', 'nvcc')}")

        # Create the temporary build directory (if it doesn't exist).
        if self.inplace:
            build_temp = os.path.abspath('./build')
        else:
            build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        # Paths to the source and destination config.cmake files
        src_config = Path(ext.sourcedir) / "3rdparty" / "tvm" / "cmake" / "config.cmake"
        dst_config = Path(build_temp) / "config.cmake"

        # Read the default config template
        content_lines = src_config.read_text().splitlines()

        # Add common LLVM configuration
        content_lines.append(f"set(USE_LLVM {llvm_config_path})")

        # Append GPU backend configuration based on environment
        if USE_ROCM:
            content_lines += [
                f"set(USE_ROCM {ROCM_HOME})",
                "set(USE_CUDA OFF)",
            ]
        else:
            content_lines += [
                f"set(USE_CUDA {CUDA_HOME})",
                "set(USE_ROCM OFF)",
            ]

        # Create the final file content
        new_content = "\n".join(content_lines) + "\n"

        # Write the file only if it does not exist or has changed
        if not dst_config.exists() or dst_config.read_text() != new_content:
            dst_config.write_text(new_content)
            print(f"[Config] Updated: {dst_config}")
        else:
            print(f"[Config] No changes: {dst_config}")

        cmake_path = get_cmake_path()
        # Run CMake to configure the project with the given arguments.
        if not os.path.exists(os.path.join(build_temp, "build.ninja")):
            logger.info(
                f"[CMake] Generating build.ninja: {cmake_path} {ext.sourcedir} {' '.join(cmake_args)}"
            )
            subprocess.check_call([cmake_path, ext.sourcedir] + cmake_args, cwd=build_temp)
        else:
            logger.info(f"[CMake] build.ninja already exists in {build_temp}")

        num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
        logger.info(
            f"[Build] Using {num_jobs} jobs | cmake: {cmake_path} (exists: {os.path.exists(cmake_path)}) | build dir: {build_temp}"
        )

        subprocess.check_call(
            [cmake_path, "--build", ".", "--config", "Release", "-j",
             str(num_jobs)],
            cwd=build_temp)


setup(
    name=PACKAGE_NAME,
    version=(get_tilelang_version(with_cuda=False, with_system_info=False, with_commit_id=False)
             if PYPI_BUILD else get_tilelang_version(with_commit_id=WITH_COMMITID)),
    packages=find_packages(where="."),
    package_dir={"": "."},
    author="Tile-AI",
    description="A tile level programming language to generate high performance code.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    platforms=[
        "Environment :: GPU :: NVIDIA CUDA" if not USE_ROCM else "Environment :: GPU :: AMD ROCm",
        "Operating System :: POSIX :: Linux",
    ],
    license="MIT",
    keywords="BLAS, CUDA, HIP, Code Generation, TVM",
    url="https://github.com/tile-ai/tilelang",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    package_data=package_data,
    include_package_data=False,
    ext_modules=[
        CMakeExtension("TileLangCXX", sourcedir="."),
        CythonExtension("TileLangCython", sourcedir="."),
    ],
    cmdclass={
        "build_py": TileLangBuilPydCommand,
        "sdist": TileLangSdistCommand,
        "build_ext": TilelangExtensionBuild,
    },
)
