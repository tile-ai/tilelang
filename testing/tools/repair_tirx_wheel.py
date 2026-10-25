"""Repair one wheel without bundling independently distributed compiler dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile


DEPENDENCY_STEMS = ("tvm_ffi", "tvm_tirx_compiler", "tvm_runtime")


def verify_boundary(wheel, kind):
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            basename = Path(name).name
            if name.startswith("tvm/") or "3rdparty/tvm/" in name:
                raise ValueError(f"Legacy TVM is bundled in {wheel}: {name}")
            stems = DEPENDENCY_STEMS if kind == "tilelang" else ("tvm_ffi",)
            if any(re.match(rf"(?:lib)?{stem}(?:[._-]|$)", basename) for stem in stems):
                if basename.endswith((".dll", ".dylib", ".lib")) or ".so" in basename:
                    raise ValueError(f"Independent dependency is bundled in {wheel}: {name}")


def repair_command(wheel, output, kind, platform=sys.platform, archs=None):
    stems = DEPENDENCY_STEMS if kind == "tilelang" else ("tvm_ffi",)
    if platform == "linux":
        excludes = [
            *(f"lib{stem}*.so*" for stem in stems),
            "libz3.so*",
            "libcuda.so.1",
            "libcudart.so*",
            "libamdhip64.so*",
            "libhsa-runtime64.so*",
            "/usr/local/cuda*",
        ]
        return [
            "auditwheel",
            "-v",
            "repair",
            *(argument for name in excludes for argument in ("--exclude", name)),
            "-w",
            str(output),
            str(wheel),
        ]
    if platform == "darwin":
        command = ["delocate-wheel", "--verbose", "--ignore-missing-dependencies", "--no-sanitize-rpaths"]
        command.extend(argument for name in (*stems, "z3") for argument in ("--exclude", f"lib{name}"))
        if archs:
            command.extend(["--require-archs", archs])
        return [*command, "-w", str(output), str(wheel)]
    if platform == "win32":
        names = ["tvm_ffi.dll", "libz3.dll"]
        if kind == "tilelang":
            names.extend(
                [
                    "tvm_tirx_compiler.dll",
                    "tvm_runtime.dll",
                    *(f"tvm_runtime_{backend}.dll" for backend in ("cuda", "rocm", "metal", "opencl", "vulkan")),
                ]
            )
        return ["delvewheel", "repair", "--exclude", ";".join(names), "-w", str(output), str(wheel)]
    raise ValueError(f"Unsupported wheel repair platform: {platform}")


def repair(wheel, output, kind, archs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = repair_command(wheel, output, kind, archs=archs)
    if not shutil.which(command[0]):
        tool = "delocate" if command[0] == "delocate-wheel" else command[0]
        uv = shutil.which("uv")
        install = [uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]
        subprocess.run([*install, tool], check=True)
    executable = shutil.which(command[0]) or str(
        Path(sysconfig.get_path("scripts")) / (command[0] + (".exe" if sys.platform == "win32" else ""))
    )
    with tempfile.TemporaryDirectory(prefix="tilelang-wheel-repair-") as temporary:
        command = repair_command(wheel, temporary, kind, archs=archs)
        command[0] = executable
        subprocess.run(command, check=True)
        wheels = list(Path(temporary).glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Repair must produce exactly one wheel")
        verify_boundary(wheels[0], kind)
        destination = output / wheels[0].name
        shutil.copy2(wheels[0], destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("tilelang", "tirx"), default="tilelang")
    parser.add_argument("--archs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    repair(arguments.wheel, arguments.output_dir, arguments.kind, arguments.archs)


if __name__ == "__main__":
    main()
