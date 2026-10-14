"""Utilities to invoke the PTO toolchain (ptodsl -> ptoas -> bisheng) for Ascend NPU targets.

``compile_ptodsl_to_aibin`` lowers one TileLang-generated PTODSL module into
an executable CCE ELF (aibin): the PTODSL frontend script (ptodsl_compile.py,
run as a subprocess) lowers the source to MLIR, ptoas assembles a fat object,
bisheng links it, and the .aicore_binary section of the linked image is
extracted as the device binary consumed by the Ascend runtime module.
"""

from __future__ import annotations

import ast
import functools
import json
import logging
import os
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from tvm.contrib import utils

from tilelang.env import env

logger = logging.getLogger(__name__)

# The fat object links alone into a shared image: the Ascend runtime launches
# the extracted device binary directly with host-side packed arguments, so no
# host launcher object is needed.
BISHENG_FATOBJ_LINK_FLAGS = ("-fPIC", "-shared", "--cce-fatobj-link")


def resolve_pto_options() -> tuple[str, list[str]]:
    """Resolve the ptoas arch and extra flags from the environment."""
    pto_arch = os.environ.get("PTO_ARCH", "a5").strip().lower()
    if pto_arch != "a5":
        raise RuntimeError(f"Unsupported PTO_ARCH for PTO JIT: {pto_arch}")
    pto_flags = shlex.split(os.environ.get("PTO_FLAGS", ""))
    if not any(flag == "--pto-backend" or flag.startswith("--pto-backend=") for flag in pto_flags):
        pto_flags.append("--pto-backend=vpto")
    return pto_arch, pto_flags


def parse_pto_kernel_names(ptodsl_source: str) -> list[str]:
    """Return the top-level ``@pto.jit`` kernel names in definition order."""
    try:
        module = ast.parse(ptodsl_source)
    except SyntaxError as err:
        raise RuntimeError("Failed to parse PTODSL source for PTO kernels.") from err

    names: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "jit"
                and isinstance(target.value, ast.Name)
                and target.value.id == "pto"
            ):
                names.append(node.name)
                break
    if not names:
        raise RuntimeError("PTODSL source defines no @pto.jit kernels.")
    return names


def _compile_ptodsl_to_pto(
    ptodsl_source: str,
    kernel_names: list[str],
    src_path: str | Path,
    out_path: str | Path,
    verbose: bool = False,
) -> None:
    """Lower one PTODSL module to MLIR text.

    ptodsl_compile.py runs in a subprocess so the DSL's tracing globals
    (pto.init_core and friends) never enter this interpreter.
    """
    Path(src_path).write_text(ptodsl_source, encoding="utf-8")

    compile_script = Path(__file__).with_name("ptodsl_compile.py")
    command = [sys.executable, str(compile_script), str(src_path), json.dumps(kernel_names), str(out_path)]
    if verbose:
        print(f"PTODSL compile command: {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode == 2:
        raise RuntimeError(
            f"PTODSL compiler frontend is unavailable in the current environment.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if result.returncode != 0:
        raise RuntimeError(
            "PTODSL compile-only lowering failed.\n"
            f"Kernels: {kernel_names}\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def run_ptoas(
    pto_path: str | Path,
    fatobj_path: str | Path,
    pto_arch: str,
    pto_flags: list[str],
    verbose: bool = False,
) -> None:
    """Assemble the MLIR text into a device fat object."""
    command = ["ptoas", f"--pto-arch={pto_arch}", *pto_flags, str(pto_path), "-o", str(fatobj_path)]
    if verbose:
        print(f"PTO compile command: {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"PTO lowering failed.\nCommand: {' '.join(command)}\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}")


def bisheng_fatobj_link(fatobj_path: str | Path, out_so: str | Path, verbose: bool = False) -> None:
    """Link the fat object alone into a shared image carrying the device ELF."""
    from tilelang.contrib.bisheng import find_bisheng_path

    command = [find_bisheng_path(), *BISHENG_FATOBJ_LINK_FLAGS, "-o", str(out_so), str(fatobj_path)]
    if verbose:
        print(f"bisheng command: {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"PTO device link failed.\nCommand: {' '.join(command)}\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}")


def extract_aicore_binary(so_path: str | Path) -> bytearray:
    """Extract the ``.aicore_binary`` section bytes from a linked image.

    Parses the ELF64 section table directly instead of shelling out to
    ``readelf``: tokenizing its text output can silently truncate the
    extract, and a bad extract is only rejected later by
    ``aclrtBinaryLoadFromData``.
    """
    with open(so_path, "rb") as file:
        image = file.read()
    if image[:4] != b"\x7fELF" or image[4] != 2 or image[5] != 1:
        raise RuntimeError(f"{so_path} is not a little-endian ELF64 image.")

    e_shoff = struct.unpack_from("<Q", image, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", image, 0x3A)

    def section_header(index: int) -> tuple[int, int, int]:
        base = e_shoff + index * e_shentsize
        name_offset, _, _, _, offset, size = struct.unpack_from("<IIQQQQ", image, base)
        return name_offset, offset, size

    strtab_offset = section_header(e_shstrndx)[1]
    for index in range(e_shnum):
        name_offset, offset, size = section_header(index)
        name_start = strtab_offset + name_offset
        name_end = image.index(b"\x00", name_start)
        if image[name_start:name_end] == b".aicore_binary":
            if size == 0:
                raise RuntimeError(f"{so_path} has an empty .aicore_binary section.")
            return bytearray(image[offset : offset + size])
    raise RuntimeError(f"{so_path} has no .aicore_binary section.")


def _first_output_line(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True)
    except OSError:
        return "unavailable"
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    return output.splitlines()[0] if output else "unknown"


@functools.cache
def toolchain_stamp() -> dict[str, str]:
    """Identity of the tools that produce a PTO aibin, for cache keys."""
    from tilelang.contrib.bisheng import find_bisheng_path

    ptoas_path = shutil.which("ptoas")
    if ptoas_path is None:
        raise RuntimeError("Cannot find the ptoas assembler. Install ptoas-vmi and make sure `ptoas` is on PATH.")
    bisheng_path = find_bisheng_path()
    return {
        "ptoas": _first_output_line([ptoas_path, "--version"]),
        "bisheng": _first_output_line([bisheng_path, "--version"]),
    }


def compile_ptodsl_to_aibin(
    ptodsl_source: str,
    pto_arch: str,
    pto_flags: list[str],
    verbose: bool = False,
) -> bytearray:
    """Compile one PTODSL module into an executable CCE ELF (aibin)."""
    if not ptodsl_source.strip():
        raise RuntimeError("PTO compilation requires a non-empty PTODSL kernel source.")

    keep_temp = not env.should_cleanup_temp_files()
    tempdir = utils.tempdir(keep_for_debug=keep_temp)
    work = Path(tempdir.temp_dir)
    try:
        source_path = work / "kernel.ptodsl.py"
        pto_path = work / "kernel.pto"
        fatobj_path = work / "kernel.fatobj.o"
        so_path = work / "lib_kernel.so"

        kernel_names = parse_pto_kernel_names(ptodsl_source)
        _compile_ptodsl_to_pto(ptodsl_source, kernel_names, source_path, pto_path, verbose)
        run_ptoas(pto_path, fatobj_path, pto_arch, pto_flags, verbose)
        bisheng_fatobj_link(fatobj_path, so_path, verbose)
        return extract_aicore_binary(so_path)
    except Exception:
        if keep_temp:
            logger.warning("PTO temporary directory retained after failure: %s", work)
        else:
            tempdir.remove()
        raise
    if not keep_temp:
        tempdir.remove()
