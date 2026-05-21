# pylint: disable=invalid-name
"""Utility to invoke hipcc compiler in the system"""
# File is copied from a modified version of hipcc.py to support
# compilation of HIP code with hipcc compiler
# Source Path:
# https://github1s.com/TileLang/tvm/blob/upstream/python/tvm/contrib/hipcc.py

from __future__ import absolute_import as _abs

import os
import subprocess
import tempfile

import tvm_ffi

from tilelang import tvm as tvm
from tilelang.env import env
from tvm.contrib import utils
from tvm.base import py_str
from tvm.contrib.rocm import get_rocm_arch, find_rocm_path


def _resolve_artifact_paths(temp, file_name, target_format, kernels_output_dir=None):
    if kernels_output_dir is None:
        return temp.relpath(f"{file_name}.cc"), temp.relpath(f"{file_name}.{target_format}")

    os.makedirs(kernels_output_dir, exist_ok=True)
    fd, temp_code = tempfile.mkstemp(prefix=f"{file_name}_", suffix=".cc", dir=kernels_output_dir)
    os.close(fd)
    file_stem, _ = os.path.splitext(os.path.basename(temp_code))
    temp_target = os.path.join(kernels_output_dir, f"{file_stem}.{target_format}")
    return temp_code, temp_target


def compile_hip(code, target_format="hsaco", arch=None, options=None, path_target=None, verbose=False):
    """Compile HIP code with hipcc.

    Parameters
    ----------
    code : str
        The HIP code.

    target_format : str
        The target format of hipcc compiler.

    arch : str
        The AMD GPU architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    hsaco : bytearray
        The bytearray of the hsaco
    """
    if arch is None:
        rocm_path = find_rocm_path()
        arch = get_rocm_arch(rocm_path)

    temp = utils.tempdir(keep_for_debug=not env.should_cleanup_temp_files())
    file_name = "my_kernel"
    if target_format not in ["hsaco"]:
        raise ValueError("target_format must be hsaco")
    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = pass_context.config.get("hip.kernels_output_dir", None)
    temp_code, temp_target = _resolve_artifact_paths(temp, file_name, target_format, kernels_output_dir=kernels_output_dir)

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = ["hipcc"]
    cmd += ["-O3", "-c"]
    # Always include line info for better profiling and mapping
    cmd += ["-gline-tables-only"]
    if isinstance(arch, str):
        cmd += [f"--offload-arch={arch}"]
    if target_format == "hsaco":
        cmd += ["--genco"]
    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()
    if verbose:
        print(py_str(out))

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


@tvm_ffi.register_global_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target):
    """use hipcc to generate fatbin code for better optimization"""
    from tilelang.utils.target import target_get_mcpu

    hsaco = compile_hip(code, target_format="hsaco", arch=target_get_mcpu(target))
    return hsaco
