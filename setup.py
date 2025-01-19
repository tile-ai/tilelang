# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import subprocess
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
import distutils.dir_util
from typing import List
import re
import tarfile
from io import BytesIO
import os
import sys
import urllib.request
from distutils.version import LooseVersion
import platform
import multiprocessing
from setuptools.command.build_ext import build_ext

# Environment variables False/True
PYPI_BUILD = os.environ.get("PYPI_BUILD", "False").lower() == "true"
PACKAGE_NAME = "tilelang"
ROOT_DIR = os.path.dirname(__file__)

# TileLang only supports Linux platform
assert sys.platform.startswith("linux"), "TileLang only supports Linux platform (including WSL)."


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
    nvcc_output = subprocess.check_output(["nvcc", "-V"], universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = LooseVersion(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_tilelang_version(with_cuda=True, with_system_info=True) -> str:
    version = find_version(get_path(".", "VERSION"))
    local_version_parts = []
    if with_system_info:
        local_version_parts.append(get_system_info().replace("-", "."))
    if with_cuda:
        cuda_version = str(get_nvcc_cuda_version())
        cuda_version_str = cuda_version.replace(".", "")[:3]
        local_version_parts.append(f"cu{cuda_version_str}")
    if local_version_parts:
        version += f"+{'.'.join(local_version_parts)}"
    return version


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
    print(f"Downloading {file_name} from {download_url}")
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
    print(f"Extracting {file_name} to {extract_path}")
    with tarfile.open(fileobj=BytesIO(file_content), mode="r:xz") as tar:
        tar.extractall(path=extract_path)

    print("Download and extraction completed successfully.")
    return os.path.abspath(os.path.join(extract_path, file_name.replace(".tar.xz", "")))


package_data = {
    "tilelang": ["py.typed"],
}

LLVM_VERSION = "10.0.1"
IS_AARCH64 = False  # Set to True if on an aarch64 platform
EXTRACT_PATH = "3rdparty"  # Default extraction path


def update_submodules():
    """Updates git submodules."""
    try:
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to update submodules") from error


def build_csrc(llvm_config_path):
    """Configures and builds TVM."""

    if not os.path.exists("build"):
        os.makedirs("build")
    os.chdir("build")
    # Copy the config.cmake as a baseline
    if not os.path.exists("config.cmake"):
        shutil.copy("../3rdparty/tvm/cmake/config.cmake", "config.cmake")
    # Set LLVM path and enable CUDA in config.cmake
    with open("config.cmake", "a") as config_file:
        config_file.write(f"set(USE_LLVM {llvm_config_path})\n")
        config_file.write("set(USE_CUDA /usr/local/cuda)\n")
    # Run CMake and make
    try:
        subprocess.check_call(["cmake", ".."])
        num_jobs = multiprocessing.cpu_count()
        subprocess.check_call(["make", f"-j{num_jobs}"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to build TileLang C Source") from error


def setup_llvm_for_tvm():
    """Downloads and extracts LLVM, then configures TVM to use it."""
    # Assume the download_and_extract_llvm function and its dependencies are defined elsewhere in this script
    extract_path = download_and_extract_llvm(LLVM_VERSION, IS_AARCH64, EXTRACT_PATH)
    llvm_config_path = os.path.join(extract_path, "bin", "llvm-config")
    return extract_path, llvm_config_path


class TileLangBuilPydCommand(build_py):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        build_py.run(self)
        self.run_command("build_ext")
        build_ext_cmd = self.get_finalized_command("build_ext")
        build_temp_dir = build_ext_cmd.build_temp
        ext_modules = build_ext_cmd.extensions  # 列出所有扩展模块
        for ext in ext_modules:
            extdir = build_ext_cmd.get_ext_fullpath(ext.name)  # 获取扩展模块的完整路径
            print(f"Extension {ext.name} output directory: {extdir}")

        ext_output_dir = os.path.dirname(extdir)
        print(f"Extension output directory (parent): {ext_output_dir}")

        TILELANG_SRC = [
            "src/tl_templates",
        ]
        for item in TILELANG_SRC:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)
        # Copy the built TVM to the package directory
        TVM_PREBUILD_ITEMS = [
            f"{ext_output_dir}/libtvm_runtime.so",
            f"{ext_output_dir}/libtvm.so",
            f"{ext_output_dir}/libtilelang.so",
            f"{ext_output_dir}/libtilelang_module.so",
        ]
        for item in TVM_PREBUILD_ITEMS:
            source_lib_file = os.path.join(ROOT_DIR, item)
            # only copy the file
            file_name = os.path.basename(item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, file_name)
            target_dir = os.path.dirname(target_dir)
            target_dir = os.path.join(target_dir, "lib")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if os.path.exists(source_lib_file):
                shutil.copy2(source_lib_file, target_dir)
                # remove the original file
                os.remove(source_lib_file)
            else:
                print(f"INFO: {source_lib_file} does not exist.")

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
                print(f"INFO: {source_dir} does not exist.")

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
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

        # Copy CUTLASS to the package directory
        CUTLASS_PREBUILD_ITEMS = [
            "3rdparty/cutlass",
        ]
        for item in CUTLASS_PREBUILD_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)
        # copy compoable kernel to the package directory
        CK_PREBUILD_ITEMS = [
            "3rdparty/composable_kernel",
        ]
        for item in CK_PREBUILD_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
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
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
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
            with_cuda=False, with_system_info=False)
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


class CMakeBuild(build_ext):
    """
    Custom build_ext command for CMake-based projects.

    This class overrides the 'run' method to ensure that CMake is available,
    and then iterates over all extensions defined as CMakeExtension,
    delegating the actual build logic to 'build_cmake'.
    """

    def run(self):
        # Check if CMake is installed and accessible by attempting to run 'cmake --version'.
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            # If CMake is not found, raise an error.
            raise RuntimeError("CMake must be installed to build the following extensions") from e

        update_submodules()

        # Build each extension (of type CMakeExtension) using our custom method.
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        """
        Build a single CMake-based extension.

        :param ext: The extension (an instance of CMakeExtension).
        """
        # Setup LLVM for TVM and retrieve the path to llvm-config.
        # We assume the function returns (_, llvm_config_path).
        _, llvm_config_path = setup_llvm_for_tvm()

        # Determine the directory where the final .so or .pyd library should go.
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Prepare arguments for the CMake configuration step.
        # -DCMAKE_LIBRARY_OUTPUT_DIRECTORY sets where built libraries go
        # -DPYTHON_EXECUTABLE ensures that the correct Python is used
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        # Create the temporary build directory (if it doesn't exist).
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        # Copy the default 'config.cmake' from the source tree into our build directory.
        src_config_cmake = os.path.join(ext.sourcedir, "3rdparty", "tvm", "cmake", "config.cmake")
        dst_config_cmake = os.path.join(build_temp, "config.cmake")
        shutil.copy(src_config_cmake, dst_config_cmake)

        # Append some configuration variables to 'config.cmake'.
        # Here, we set USE_LLVM and USE_CUDA, for example.
        with open(dst_config_cmake, "a") as config_file:
            config_file.write(f"set(USE_LLVM {llvm_config_path})\n")
            config_file.write("set(USE_CUDA /usr/local/cuda)\n")

        # Run CMake to configure the project with the given arguments.
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)

        # Build the project in "Release" mode with all available CPU cores ("-j").
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release", "-j"],
                              cwd=build_temp)


setup(
    name=PACKAGE_NAME,
    version=(get_tilelang_version(with_cuda=False, with_system_info=False)
             if PYPI_BUILD else get_tilelang_version()),
    packages=find_packages(where="."),
    package_dir={"": "."},
    author="Microsoft Research",
    description="A tile level programming language to generate high performance code.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    platforms=[
        "Environment :: GPU :: NVIDIA CUDA",
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
    ext_modules=[CMakeExtension("TileLangCXX", sourcedir=".")],
    cmdclass={
        "build_py": TileLangBuilPydCommand,
        "sdist": TileLangSdistCommand,
        "build_ext": CMakeBuild,
    },
)
