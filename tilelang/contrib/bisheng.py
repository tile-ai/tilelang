# pylint: disable=invalid-name
"""Utility to invoke bisheng compiler for Ascend NPU targets."""

from __future__ import absolute_import as _abs

import contextlib
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tilelang.env import env
from tvm.base import py_str
from tvm.contrib import utils
from tvm.target import Target


def find_bisheng_path() -> str:
    """Find the bisheng compiler binary.

    Searches ``BISHENG_HOME/bin/bisheng``, then ``PATH``.

    Returns
    -------
    path : str
        Full path to the ``bisheng`` executable.

    Raises
    ------
    RuntimeError
        If the compiler cannot be found.
    """
    bisheng_home = os.environ.get("BISHENG_HOME", "")
    if bisheng_home:
        candidate = os.path.join(bisheng_home, "bin", "bisheng")
        if os.path.isfile(candidate):
            return candidate

    path = shutil.which("bisheng")
    if path is not None:
        return path

    raise RuntimeError(
        "Cannot find the bisheng compiler.Please install it and make sure it is in PATH, or set the BISHENG_HOME environment variable."
    )


def find_ld_lld_path() -> str:
    """Find the CCE-capable ``ld.lld`` shipped with the Ascend toolkit."""
    candidates: list[Path] = []
    for env_name in ("BISHENG_HOME", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        root = os.environ.get(env_name)
        if not root:
            continue
        root_path = Path(root)
        candidates.extend((root_path / "bin" / "ld.lld", root_path / "ld.lld"))

    with contextlib.suppress(RuntimeError):
        candidates.append(Path(find_bisheng_path()).with_name("ld.lld"))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    path = shutil.which("ld.lld")
    if path is not None:
        return path
    raise RuntimeError(
        "Cannot find ld.lld. Install the Ascend toolkit and put its bin directory in PATH, or set ASCEND_HOME_PATH/BISHENG_HOME."
    )


def normalize_options(options: str | Sequence[str] | None) -> list[str]:
    """Normalize shell-like compiler options into individual argv tokens."""
    if options is None:
        return []
    if isinstance(options, str):
        return shlex.split(options)
    if not isinstance(options, Sequence):
        raise ValueError("options must be a string or a sequence of strings")

    normalized: list[str] = []
    for option in options:
        if not isinstance(option, str):
            raise ValueError("all compiler options must be strings")
        normalized.extend(shlex.split(option))
    return normalized


def get_npu_arch(npu_arch: str | None = None) -> str:
    """Resolve the Bisheng NPU architecture option."""
    return npu_arch or os.environ.get("ASCEND_NPU_ARCH", "dav-3510")


def get_target_npu_arch(target: Target) -> str:
    """Resolve the Bisheng NPU architecture from an Ascend target."""
    for attr_name in ("arch", "mcpu"):
        value = target.attrs.get(attr_name)
        if value is not None:
            return get_npu_arch(str(value))
    return get_npu_arch()


def get_bisheng_compile_options(
    npu_arch: str | None = None,
    options: str | Sequence[str] | None = None,
) -> list[str]:
    """Return common Bisheng options shared by tvm-ffi and Cython paths."""
    resolved_arch = get_npu_arch(npu_arch)
    result = ["-O2", "-fPIC", "-std=c++20", "-mllvm", "-cce-aicore-dcpreload-args=false"]
    if resolved_arch:
        result.append(f"--npu-arch={resolved_arch}")
    result.extend(normalize_options(options))
    return result


def get_aibin_linker_options() -> list[str]:
    """Return options used to turn relocatable CCE code into executable ELF."""
    return ["-m", "aicorelinux", "-Ttext", "0", "--no-mmap-output-file"]


def _run_command(command: list[str], code: str, verbose: bool, stage: str) -> None:
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = py_str(proc.stdout)
    if verbose and output:
        print(output)
    if proc.returncode != 0:
        raise RuntimeError(f"Ascend {stage} failed.\nCommand: {shlex.join(command)}\nCompiler output:\n{output}\nSource:\n{code}")


def compile_ascend(
    code,
    target_format="o",
    npu_arch=None,
    options=None,
    path_target=None,
    verbose=False,
):
    """Compile Ascend source code with the bisheng compiler.

    Parameters
    ----------
    code : str
        The Ascend kernel source code.

    target_format : str
        Output format: ``"o"`` for a relocatable host object, ``"so"`` for a
        host shared library, or ``"aibin"`` for an executable CCE ELF suitable
        for ``aclrtBinaryLoadFromData``.

    npu_arch : str, optional
        NPU architecture string passed via ``--npu-arch=<arch>`` (e.g.
        ``"dav-3510"``).  If *None*, ``ASCEND_NPU_ARCH`` env-var is used.

    options : str or list of str, optional
        Extra flags forwarded to bisheng.

    path_target : str, optional
        Explicit output path.  A temporary file is used when not given.

    verbose : bool
        Print compiler output.

    Returns
    -------
    data : bytearray
        Contents of the compiled output file.
    """
    if target_format not in {"o", "so", "aibin"}:
        raise ValueError(f"Unsupported Ascend target format: {target_format}")

    temp = utils.tempdir(keep_for_debug=not env.should_cleanup_temp_files())
    # Use .asc suffix so bisheng auto-detects the language without ``-x asc``
    temp_code = temp.relpath("tl_kernel.asc")
    suffix = "so" if target_format == "so" else ("aibin" if target_format == "aibin" else "o")
    temp_target = temp.relpath(f"tl_kernel.{suffix}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = os.fspath(path_target) if path_target else temp_target
    output_dir = os.path.dirname(file_target)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    compile_options = get_bisheng_compile_options(npu_arch, options)
    if target_format == "aibin":
        # Bisheng emits a relocatable CCE ELF. ACL requires a linked executable
        # CCE ELF, so mirror the toolchain's two-stage flow explicitly.
        rel_object = temp.relpath("tl_kernel.rel.o")
        compile_command = [
            find_bisheng_path(),
            *compile_options,
            "--cce-aicore-only",
            temp_code,
            "-o",
            rel_object,
        ]
        _run_command(compile_command, code, verbose, "device compilation")

        link_command = [
            find_ld_lld_path(),
            *get_aibin_linker_options(),
            rel_object,
            "-o",
            file_target,
        ]
        _run_command(link_command, code, verbose, "device link")
    else:
        command = [
            find_bisheng_path(),
            *compile_options,
            # Avoid mmap output writes on distributed filesystems.
            "-Wl,--no-mmap-output-file",
        ]
        if target_format == "so":
            command.append("-shared")
        else:
            command.append("-c")
        command.extend([temp_code, "-o", file_target])
        _run_command(command, code, verbose, "compilation")

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data
