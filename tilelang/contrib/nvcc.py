# pylint: disable=invalid-name
# modified from apache tvm python/tvm/contrib/nvcc.py
"""Utility to invoke nvcc compiler in the system"""

from __future__ import annotations

import os
import subprocess
import warnings
import contextlib
from tilelang.env import CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
import shutil
import tempfile
import tvm_ffi
from tilelang import tvm as tvm
from tvm.target import Target

from tvm.base import py_str
from tvm.contrib import utils


def compile_cuda(code, target_format="ptx", arch=None, options=None, path_target=None, verbose=False):
    """Compile cuda code with NVCC from env.

    Parameters
    ----------
    code : str
        The cuda code.

    target_format : str
        The target format of nvcc compiler.

    arch : str
        The cuda architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    cubin : bytearray
        The bytearray of the cubin
    """
    if arch is None:
        # If None, then it will use `tvm.target.Target.current().arch`.
        # Target arch could be a str like "sm_xx", or a list, such as
        # [
        #   "-gencode", "arch=compute_52,code=sm_52",
        #   "-gencode", "arch=compute_70,code=sm_70"
        # ]
        compute_version = get_target_compute_version(Target.current(allow_none=True))
        target_arch = get_target_arch(compute_version)
        arch = ["-gencode", f"arch=compute_{target_arch},code=sm_{target_arch}"]

    temp = utils.tempdir()
    file_name = "tvm_kernels"
    if target_format not in ["cubin", "ptx", "fatbin"]:
        raise ValueError("target_format must be in cubin, ptx, fatbin")
    temp_code = temp.relpath(f"{file_name}.cu")
    temp_target = temp.relpath(f"{file_name}.{target_format}")

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = pass_context.config.get("cuda.kernels_output_dir", None)
    if kernels_output_dir is not None:
        if not os.path.isdir(kernels_output_dir):
            os.makedirs(kernels_output_dir)
        temp_code = os.path.join(kernels_output_dir, f"{file_name}.cu")
        temp_target = os.path.join(kernels_output_dir, f"{file_name}.{target_format}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = [get_nvcc_compiler()]
    cmd += [f"--{target_format}", "-O3"]
    # Always include line info for better profiling and mapping
    cmd += ["-lineinfo"]
    if isinstance(arch, list):
        cmd += arch
    elif isinstance(arch, str):
        cmd += ["-arch", arch]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]

    # NOTE: ccbin option can be used to tell nvcc where to find the c++ compiler
    # just in case it is not in the path. On Windows it is not in the path by default.
    # However, we cannot use TVM_CXX_COMPILER_PATH because the runtime env.
    # Because it is hard to do runtime compiler detection, we require nvcc is configured
    # correctly by default.
    # if cxx_compiler_path != "":
    #    cmd += ["-ccbin", cxx_compiler_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if verbose:
        print(py_str(out))

    if proc.returncode != 0:
        msg = f"{code}\nCompilation error:\n{py_str(out)}\nCommand: {' '.join(cmd)}\n"
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def default_compile_options(compile_flags: list[str] | None = None) -> list[str]:
    """
    Build a set of default NVCC compile options for TileLang generated sources.

    Includes C++ standard and common include paths (TileLang templates, CUTLASS,
    CUDA include). Merges user-provided compile flags if given.

    Parameters
    ----------
    compile_flags : Optional[List[str]]
        Additional flags to include. Items are split on whitespace.

    Returns
    -------
    List[str]
        A list of flags suitable for NVCC's command line.
    """
    options: list[str] = ["-std=c++17"]
    try:
        if TILELANG_TEMPLATE_PATH:
            options.append(f"-I{TILELANG_TEMPLATE_PATH}")
    except Exception:
        pass
    try:
        if CUTLASS_INCLUDE_DIR:
            options.append(f"-I{CUTLASS_INCLUDE_DIR}")
    except Exception:
        pass
    try:
        if CUDA_HOME:
            options.append(f"-I{os.path.join(CUDA_HOME, 'include')}")
    except Exception:
        pass

    # Preserve user flags exactly, including repeated tokens required by NVCC
    # (e.g., multiple "-gencode" pairs or repeated "-Xcompiler" entries).
    if compile_flags:
        import shlex

        for flag in compile_flags:
            # Split each string like a shell would, preserving quoted args
            tokens = shlex.split(flag) if isinstance(flag, str) else [str(flag)]
            options.extend(tokens)
    return options


def get_ptx_from_source(code: str, compile_flags: list[str] | None = None, verbose: bool = False) -> str:
    """
    Compile CUDA C++ source to PTX using NVCC and return as text.

    Parameters
    ----------
    code : str
        CUDA C++ kernel source code.
    compile_flags : Optional[List[str]]
        Additional flags merged with defaults.
    verbose : bool
        Print NVCC output when True.

    Returns
    -------
    str
        PTX text.
    """
    opts = default_compile_options(compile_flags)
    ptx_bytes = compile_cuda(code, target_format="ptx", options=opts, verbose=verbose)
    try:
        return ptx_bytes.decode("utf-8")
    except Exception:
        return str(ptx_bytes)


def _find_tool(name: str) -> str | None:
    """Find a CUDA binary in PATH or under CUDA_HOME/bin."""
    path = shutil.which(name)
    if path:
        return path
    if CUDA_HOME:
        candidate = os.path.join(CUDA_HOME, "bin", name)
        if os.path.exists(candidate):
            return candidate
    return None


def get_sass_from_source(code: str, compile_flags: list[str] | None = None, verbose: bool = False) -> str:
    """
    Compile CUDA C++ source to CUBIN and disassemble to SASS.

    Uses nvdisasm if available; otherwise falls back to cuobjdump.

    Parameters
    ----------
    code : str
        CUDA C++ kernel source code.
    compile_flags : Optional[List[str]]
        Additional flags merged with defaults.
    verbose : bool
        Print tool outputs when True.

    Returns
    -------
    str
        SASS text.
    """
    opts = default_compile_options(compile_flags)
    cubin_bytes = compile_cuda(code, target_format="cubin", options=opts, verbose=verbose)

    # Write to a temp .cubin file
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as tmp:
        tmp.write(cubin_bytes)
        cubin_path = tmp.name

    # Try disassembly tools (prefer nvdisasm, fallback cuobjdump)
    cand_nvdisasm = _find_tool("nvdisasm")
    cand_cuobjdump = _find_tool("cuobjdump")
    if not cand_nvdisasm and not cand_cuobjdump:
        raise RuntimeError("Cannot find 'nvdisasm' or 'cuobjdump'. Please ensure CUDA toolkit is installed and in PATH.")
    last_err: str | None = None
    try:
        # Attempt nvdisasm first
        tools_to_try = []
        if cand_nvdisasm:
            tools_to_try.append(("nvdisasm", [cand_nvdisasm, cubin_path]))
        if cand_cuobjdump:
            tools_to_try.append(("cuobjdump", [cand_cuobjdump, "--dump-sass", cubin_path]))

        for tool_name, cmd in tools_to_try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out, _ = proc.communicate()
            text = py_str(out)
            if verbose:
                print(f"[{tool_name}] output:\n{text}")
            if proc.returncode == 0 and text.strip():
                return text
            last_err = f"{tool_name} rc={proc.returncode}, output:\n{text}"
        # If we reach here, all attempts failed
        raise RuntimeError(f"SASS disassembly failed. Tried tools: {', '.join(name for name, _ in tools_to_try)}\n{last_err or ''}")
    finally:
        with contextlib.suppress(Exception):
            os.remove(cubin_path)


def find_cuda_path():
    """Utility function to find cuda path

    Returns
    -------
    path : str
        Path to cuda root.
    """
    if CUDA_HOME:
        return CUDA_HOME
    raise RuntimeError(
        "Failed to automatically detect CUDA installation. Please set the CUDA_HOME environment variable manually (e.g., export CUDA_HOME=/usr/local/cuda)."
    )


def get_cuda_version(cuda_path=None):
    """Utility function to get cuda version

    Parameters
    ----------
    cuda_path : Optional[str]

        Path to cuda root.  If None is passed, will use
        `find_cuda_path()` as default.

    Returns
    -------
    version : float
        The cuda version

    """
    if cuda_path is None:
        cuda_path = find_cuda_path()

    version_file_path = os.path.join(cuda_path, "version.txt")
    if not os.path.exists(version_file_path):
        # Debian/Ubuntu repackaged CUDA path
        version_file_path = os.path.join(cuda_path, "lib", "cuda", "version.txt")
    try:
        with open(version_file_path) as f:
            version_str = f.read().strip().split()[-1]
            return tuple(int(field) for field in version_str.split("."))
    except FileNotFoundError:
        pass

    cmd = [os.path.join(cuda_path, "bin", "nvcc"), "--version"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        release_line = [l for l in out.split("\n") if "release" in l][0]
        release_fields = [s.strip() for s in release_line.split(",")]
        version_str = [f[1:] for f in release_fields if f.startswith("V")][0]
        return tuple(int(field) for field in version_str.split("."))
    raise RuntimeError("Cannot read cuda version file")


@tvm_ffi.register_global_func("tilelang_callback_libdevice_path", override=True)
def find_libdevice_path(arch):
    """Utility function to find libdevice

    Parameters
    ----------
    arch : int
        The compute architecture in int

    Returns
    -------
    path : str
        Path to libdevice.
    """
    cuda_path = find_cuda_path()
    lib_path = os.path.join(cuda_path, "nvvm/libdevice")
    if not os.path.exists(lib_path):
        # Debian/Ubuntu repackaged CUDA path
        lib_path = os.path.join(cuda_path, "lib/nvidia-cuda-toolkit/libdevice")
    selected_ver = 0
    selected_path = None
    cuda_ver = get_cuda_version(cuda_path)
    major_minor = (cuda_ver[0], cuda_ver[1])
    if major_minor in (
        (9, 0),
        (9, 1),
        (10, 0),
        (10, 1),
        (10, 2),
        (11, 0),
        (11, 1),
        (11, 2),
        (11, 3),
    ):
        path = os.path.join(lib_path, "libdevice.10.bc")
    else:
        for fn in os.listdir(lib_path):
            if not fn.startswith("libdevice"):
                continue

            try:
                # expected pattern: libdevice.${ARCH}.10.bc
                #             e.g., libdevice.compute_20.10.bc
                ver = int(fn.split(".")[-3].split("_")[-1])
                if selected_ver < ver <= arch:
                    selected_ver = ver
                    selected_path = fn
            except ValueError:
                # it can just be `libdevice.10.bc` in CUDA 10
                selected_path = fn

        if selected_path is None:
            raise RuntimeError(f"Cannot find libdevice for arch {arch}")
        path = os.path.join(lib_path, selected_path)
    return path


def callback_libdevice_path(arch):
    try:
        return find_libdevice_path(arch)
    except RuntimeError:
        warnings.warn("Cannot find libdevice path", stacklevel=2)
        return ""


@tvm_ffi.register_global_func("tvm.contrib.nvcc.get_compute_version", override=True)
def get_target_compute_version(target=None):
    """Utility function to get compute capability of compilation target.

    Looks for the target arch in three different places, first in the target input, then the
    Target.current() scope, and finally the GPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        compute capability of a GPU (e.g. "8.6" or "9.0")
    """
    # 1. input target object
    # 2. Target.current()
    target = target or Target.current()
    if target and target.arch:
        arch = target.arch.split("_")[1]
        if len(arch) == 2:
            major, minor = arch
            # Handle old format like sm_89
            return major + "." + minor
        elif len(arch) == 3:
            major = int(arch[0])
            if major < 2:
                major = arch[0:2]
                minor = arch[2]
                return major + "." + minor
            else:
                # This is for arch like "sm_90a"
                major, minor, suffix = arch
            return major + "." + minor + "." + suffix

    # 3. GPU compute version
    if tvm.cuda(0).exist:
        return tvm.cuda(0).compute_version

    raise ValueError("No CUDA architecture was specified or GPU detected.Try specifying it by adding '-arch=sm_xx' to your target.")


def parse_compute_version(compute_version) -> tuple[int, int]:
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error") from err


def get_target_arch(compute_version) -> str:
    major, minor = parse_compute_version(compute_version)
    target_arch = str(major * 10 + minor)
    if major >= 9:
        target_arch += "a"
    return target_arch


def have_fp16(compute_version):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version: str
        compute capability of a GPU (e.g. "6.0")
    """
    major, minor = parse_compute_version(compute_version)
    # fp 16 support in reference to:
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions
    conditions = [False]
    conditions.append(major == 5 and minor >= 3)
    conditions.append(major >= 6)
    return any(conditions)


def have_int8(compute_version):
    """Either int8 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.1")
    """
    major, _ = parse_compute_version(compute_version)
    return major >= 6


def have_tensorcore(compute_version=None, target=None):
    """Either TensorCore support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str, optional
        compute capability of a GPU (e.g. "7.0").

    target : tvm.target.Target, optional
        The compilation target, will be used to determine arch if compute_version
        isn't specified.
    """
    if compute_version is None:
        if tvm.cuda(0).exist:
            compute_version = tvm.cuda(0).compute_version
        else:
            if target is None or "arch" not in target.attrs:
                warnings.warn(
                    "Tensorcore will be disabled due to no CUDA architecture specified."
                    "Try specifying it by adding '-arch=sm_xx' to your target.",
                    stacklevel=2,
                )
                return False
            compute_version = target.attrs["arch"]
            # Compute version will be in the form "sm_{major}{minor}"
            major, minor = compute_version.split("_")[1]
            compute_version = major + "." + minor
    major, _ = parse_compute_version(compute_version)
    return major >= 7


def have_cudagraph():
    """Either CUDA Graph support is provided"""
    try:
        cuda_ver = get_cuda_version()
        return not cuda_ver < (10, 0)
    except RuntimeError:
        return False


@tvm_ffi.register_global_func("tvm.contrib.nvcc.supports_bf16", override=True)
def have_bf16(compute_version):
    """Either bf16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "8.0")
    """
    major, _ = parse_compute_version(compute_version)
    return major >= 8


@tvm_ffi.register_global_func("tvm.contrib.nvcc.supports_fp8", override=True)
def have_fp8(compute_version):
    """Whether fp8 support is provided in the specified compute capability or not

    Parameters
    ----------
    compute_version : str
        GPU capability
    """
    major, minor = parse_compute_version(compute_version)
    # fp8 is supported in Ada Lovelace (8.9) or later architectures.
    conditions = [False]
    conditions.append(major == 8 and minor >= 9)
    conditions.append(major >= 9)
    return any(conditions)


@tvm_ffi.register_global_func("tvm.contrib.nvcc.supports_tma", override=True)
def have_tma(target):
    """Whether TMA support is provided in the specified compute capability or not

    Parameters
    ----------
    target : tvm.target.Target
        The compilation target
    """
    if target.kind.name != "cuda":
        return False
    compute_version = get_target_compute_version(target)
    major, minor = parse_compute_version(compute_version)
    # TMA is supported in Ada Lovelace (9.0) or later architectures.
    conditions = [False]
    conditions.append(major >= 9)
    return any(conditions)


def is_hopper(target):
    if target.kind.name != "cuda":
        return False
    compute_version = get_target_compute_version(target)
    major, minor = parse_compute_version(compute_version)
    return major == 9 and minor == 0


def get_nvcc_compiler() -> str:
    """Get the path to the nvcc compiler"""
    return os.path.join(find_cuda_path(), "bin", "nvcc")
