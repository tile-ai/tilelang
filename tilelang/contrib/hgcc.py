# pylint: disable=invalid-name
"""Utility to invoke hgcc (HGGC compiler) in the system."""

from __future__ import annotations

import os
import subprocess
import tempfile

from tilelang import tvm as tvm
from tilelang.env import TILELANG_TEMPLATE_PATH, ACTLIZE_INCLUDE_DIR, env
from tvm.target import Target


def _get_compile_timeout_seconds() -> float | None:
    return env.get_compile_timeout_seconds()


def _find_ppu_sdk() -> str:
    """Find PPU SDK root from PPU_SDK env var."""
    ppu_sdk = os.environ.get("PPU_SDK")
    if not ppu_sdk:
        raise RuntimeError(
            "PPU_SDK environment variable is not set. Please source the PPU SDK envsetup.sh (e.g. source $PPU_SDK/envsetup.sh ppu)"
        )
    return ppu_sdk


def get_hgcc_compiler() -> str:
    """Get the path to the hgcc compiler binary."""
    return os.path.join(_find_ppu_sdk(), "bin", "hgcc")


def get_target_compute_version(target=None):
    """Get compute capability of the PPU compilation target.

    Looks for the target arch in the target object, then Target.current(),
    and finally the PPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        Compute capability of the PPU
    """
    # 1. input target object
    # 2. Target.current()
    target = target or Target.current()
    if target and "arch" in target.attrs:
        arch = str(target.attrs["arch"]).split("_")[1].rstrip("af")
        if len(arch) == 2:
            major, minor = arch
            return major + "." + minor
        elif len(arch) == 3:
            major = arch[0:2]
            minor = arch[2]
            return major + "." + minor
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    # 3. PPU device compute version
    ppu_dev_type = getattr(tvm.ffi.DLDeviceType, "kDLPPU", 19)
    if tvm.device(ppu_dev_type, 0).exist:
        raw_version = tvm.device(ppu_dev_type, 0).compute_version
        ppu_version_map = {
            "8.0": "1.0",
            "8.9": "1.5",
        }
        return ppu_version_map.get(raw_version, raw_version)

    raise ValueError(
        "No PPU architecture was specified or GPU detected. Specify it with a target config such as {'kind': 'ppu', 'arch': 'ppu0015'}."
    )


def parse_compute_version(compute_version) -> tuple[int, int]:
    """Parse compute capability string to divide major and minor version.

    Parameters
    ----------
    compute_version : str
        Compute capability of a PPU

    Returns
    -------
    major : int
        Major version number
    minor : int
        Minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        raise RuntimeError("Compute version parsing error") from err


def get_target_arch(compute_version: str | tuple[int, int]) -> str:
    """Convert compute version to PPU architecture string.

    Parameters
    ----------
    compute_version : str or tuple
        Compute capability

    Returns
    -------
    arch : str
        PPU architecture string: "ppu_15" and "ppu_10"
    """
    if isinstance(compute_version, str):
        major, minor = parse_compute_version(compute_version)
    else:
        major, minor = compute_version
    arch_int = major * 10 + minor
    if arch_int >= 15:
        return "ppu_15"
    else:
        return "ppu_10"


def get_target_arch_int(compute_version: str | tuple[int, int]) -> int:
    """Convert compute version to integer."""
    if isinstance(compute_version, str):
        major, minor = parse_compute_version(compute_version)
    else:
        major, minor = compute_version
    return major * 10 + minor


def compile_ppu(
    code: str,
    target: Target | None = None,
    options: list[str] | None = None,
    verbose: bool = False,
) -> bytes:
    """Compile PPU code with hgcc.

    Parameters
    ----------
    code : str
        The PPU C++ source code.
    target : tvm.target.Target, optional
        The compilation target. If None, uses Target.current().
    options : list of str, optional
        Additional compiler options.
    verbose : bool
        Whether to print the compile command.

    Returns
    -------
    binary : bytes
        The compiled .hgbin binary data.
    """
    compute_version = get_target_compute_version(target)
    ppu_arch = get_target_arch(compute_version)

    ppu_sdk = _find_ppu_sdk()
    hgcc = get_hgcc_compiler()

    # Base options
    # Template path: use env var if set, otherwise fall back to repo src/
    import os.path as osp

    tl_path = TILELANG_TEMPLATE_PATH
    if tl_path is None:
        tl_path = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "src"))
    cmd_options = [
        "-std=c++20",
        "-I" + tl_path,
    ]

    # PPU SDK include path
    ppu_inc = os.path.join(ppu_sdk, "targets", "x86_64-linux", "include")
    if os.path.isdir(ppu_inc):
        cmd_options += ["-I" + ppu_inc]

    # actlize (pure PPU library, replaces cutlass)
    if ACTLIZE_INCLUDE_DIR:
        cmd_options += ["-I" + ACTLIZE_INCLUDE_DIR, "-DPPU_DEFAULT"]
    else:
        cmd_options += ["-DPPU_COMPATIBLE"]

    # Merge extra options
    if options:
        cmd_options += options

    # Write source to temp file and compile
    tmp_src = tempfile.NamedTemporaryFile(mode="w", suffix=".hg", delete=False)
    hgbin_path = None
    try:
        tmp_src.write(code)
        tmp_src.close()
        hgbin_path = tmp_src.name.replace(".hg", ".hgbin")

        cmd = [hgcc, "-x", "hg", "-hgbin", "-O3", "-lineinfo", "-arch=" + ppu_arch] + cmd_options + ["-o", hgbin_path, tmp_src.name]

        if verbose:
            print("PPU compile command:", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_get_compile_timeout_seconds())
        if proc.returncode != 0:
            raise RuntimeError(f"hgcc compilation failed:\n{proc.stderr}\n{proc.stdout}\nCommand: {' '.join(cmd)}\n{code}")

        with open(hgbin_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_src.name):
            os.unlink(tmp_src.name)
        if hgbin_path and os.path.exists(hgbin_path):
            os.unlink(hgbin_path)


# ---------------------------------------------------------------------------
# Feature detection (all PPU variants support these)
# ---------------------------------------------------------------------------

_PPU_PDL_SUPPORTED_ARCHS: frozenset[int] = frozenset()


def have_fp16(compute_version=None):
    return True


def have_int8(compute_version=None):
    return True


def have_tensorcore(compute_version=None, target=None):
    return True


def have_bf16(compute_version=None):
    return True


def have_fp8(compute_version=None):
    return True


def have_mbarrier(target=None):
    return True


def have_tma(target=None):
    return False


def have_pdl(target=None):
    target = target or Target.current()
    if target is not None and target.kind.name != "ppu":
        return False
    compute_version = get_target_compute_version(target)
    arch_int = get_target_arch_int(compute_version)
    return arch_int in _PPU_PDL_SUPPORTED_ARCHS
