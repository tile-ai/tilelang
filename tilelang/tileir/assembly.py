"""Helpers for optimizing and assembling CUDA Tile IR modules."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from collections.abc import Callable

from tvm.target import Target

from .artifact import TileIRLaunchMetadata, TileIRLoweringResult
from .checks import TileIRToolchain, check_tileir_available, tileiras_invocation_env
from .errors import TileIRAssemblyError, TileIRLoweringError


def _tileiras_gpu_name(arch: str) -> str:
    """Map an nvcc-style arch to the base SM token that ``tileiras --gpu-name`` accepts.

    A CUDA target's arch may carry an architecture-/family-specific suffix
    (``sm_90a``, ``sm_100a``, ``sm_100f``) — nvcc/ptxas use these to enable
    arch-exclusive features (wgmma, tcgen, ...). ``tileiras``, however, only
    accepts the base SM tokens (its ``--help`` lists ``sm_80``..``sm_124`` with
    no suffix; the cuda-tile README uses ``--gpu-name sm_100``) and selects
    arch-specific features in the Tile IR compiler itself. So strip the suffix:
    ``sm_90a`` -> ``sm_90``, ``sm_100f`` -> ``sm_100``. Without this, an
    auto-detected target on Hopper/Blackwell (arch ``sm_90a``/``sm_100a``) fails
    assembly with ``tileiras: Cannot find option named 'sm_90a'``.
    """
    match = re.fullmatch(r"(sm_\d+)[a-z]*", arch.strip())
    return match.group(1) if match else arch


def target_arch(target: Target) -> str:
    arch = getattr(target, "arch", None)
    if not arch:
        attrs = getattr(target, "attrs", None)
        if attrs and "arch" in attrs:
            arch = attrs["arch"]
    if not arch:
        raise TileIRLoweringError(
            f"TileIR target requires an explicit CUDA architecture, for example `tileir -arch=sm_120`; got target={target}."
        )
    return _tileiras_gpu_name(str(arch))


def run_tool(cmd: list[str], *, stage: str) -> subprocess.CompletedProcess[str]:
    try:
        env = tileiras_invocation_env(Path(cmd[0])) if Path(cmd[0]).name == "tileiras" else None
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TileIRAssemblyError(f"TileIR {stage} failed to run `{cmd[0]}`: {exc}") from exc

    if result.returncode != 0:
        details = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        raise TileIRAssemblyError(
            f"TileIR {stage} failed with exit code {result.returncode}: {' '.join(cmd)}" + (f"\n{details}" if details else "")
        )
    return result


def write_tileir_bytecode(tileir_module: Any, output_path: Path) -> None:
    try:
        from cuda_tile._mlir._mlir_libs._cuda_tile import writeBytecode
    except ImportError as exc:
        raise TileIRAssemblyError("CUDA Tile IR bytecode writer is unavailable.") from exc

    operation = getattr(tileir_module, "operation", tileir_module)
    with output_path.open("wb") as file:
        ok = writeBytecode(file, operation)
    if not ok:
        raise TileIRAssemblyError("CUDA Tile IR bytecode writer rejected the structured module.")


def optimize_tileir_module(tileir_module: Any, opt_level: int = 3) -> None:
    try:
        from cuda_tile._mlir._mlir_libs._cuda_tile import (
            TileIROptimizationsOpts,
            applyTileIROptimizations,
        )
    except ImportError as exc:
        raise TileIRAssemblyError("CUDA Tile IR optimizer is unavailable.") from exc

    operation = getattr(tileir_module, "operation", tileir_module)
    try:
        opts = TileIROptimizationsOpts()
        opts.opt_level = opt_level
        if not applyTileIROptimizations(operation, opts):
            raise TileIRAssemblyError("CUDA Tile IR optimizer rejected the structured module.")
    except Exception as exc:
        raise TileIRAssemblyError(f"CUDA Tile IR optimizer failed: {exc}") from exc


def assemble_tileir_module(
    tileir_module: Any,
    *,
    kernel_name: str,
    target: Target,
    toolchain: TileIRToolchain | None = None,
    launch_metadata: TileIRLaunchMetadata | None = None,
    argument_names: tuple[str, ...] = (),
    opt_level: int = 3,
    optimize: Callable[[Any], None] | None = None,
    write_bytecode: Callable[[Any, Path], None] | None = None,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> TileIRLoweringResult:
    """Assemble a structured CUDA Tile IR module into a cubin."""

    if toolchain is None:
        toolchain = check_tileir_available()

    arch = target_arch(target)
    if optimize is None:
        optimize = optimize_tileir_module
    if write_bytecode is None:
        write_bytecode = write_tileir_bytecode
    optimize(tileir_module)
    tileir_source = str(tileir_module)
    with tempfile.TemporaryDirectory(prefix="tilelang_tileir_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bytecode_path = tmp_path / "kernel.tileir"
        cubin_path = tmp_path / "kernel.cubin"

        write_bytecode(tileir_module, bytecode_path)
        cmd = [
            str(toolchain.tileiras_path),
            f"--gpu-name={arch}",
            f"--opt-level={opt_level}",
            str(bytecode_path),
            "-o",
            str(cubin_path),
        ]
        if run is None:
            run_tool(cmd, stage="assembly")
        else:
            run(cmd)

        return TileIRLoweringResult(
            kernel_name=kernel_name,
            cubin=cubin_path.read_bytes(),
            tileir_source=tileir_source,
            launch_metadata=launch_metadata or TileIRLaunchMetadata(),
            argument_names=argument_names,
        )
