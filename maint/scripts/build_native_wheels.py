#!/usr/bin/env python3
"""Build the matched FFI, compiler provider and frontend wheel bundle.

The FFI has a CPython extension and needs one wheel per supported interpreter.
The provider is independent of the Python ABI; TileLang uses the stable ABI.
Build FFI wheels first so the other wheels can be tested without unpublished
dependencies from a package index. Linux builds retain the manylinux containers
and backend SDK setup in pyproject.toml. No artifacts are published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

from build_native_artifacts import build, stage


def _run(command, *, cwd, environment, record_path):
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        raise FileExistsError(record_path)
    record = {"command": command, "cwd": str(cwd), "status": "running", "exit_code": None}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=environment, check=False)
    record.update(status="completed", exit_code=result.returncode, elapsed_seconds=time.monotonic() - started)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage", type=Path, default=Path(".native-stage"))
    parser.add_argument("--output", type=Path, default=Path("wheelhouse"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--ffi-build", help="Override the CPython wheel selectors; does not change the FFI source revision")
    parser.add_argument("--print-build-identifiers", action="store_true")
    parser.add_argument("--platform", choices=["linux", "macos", "windows"], help="Select a platform for --print-build-identifiers")
    args = parser.parse_args()
    root, staged, output = args.source.resolve(), args.stage.resolve(), args.output.resolve()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not staged.is_relative_to(root) or not output.is_relative_to(root):
        parser.error("--stage and --output must be inside --source so cibuildwheel can copy the complete bundle into its container")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be empty; keep previous wheel bundles separately")
    actual_platform = "windows" if sys.platform == "win32" else "macos" if sys.platform == "darwin" else "linux"
    platform_name = args.platform or actual_platform
    if platform_name != actual_platform and not args.print_build_identifiers:
        parser.error("--platform can only inspect another platform; run native builds on that platform")
    identity = stage(root, staged)
    output.mkdir(parents=True, exist_ok=True)
    relative_stage = staged.relative_to(root).as_posix()
    relative_output = output.relative_to(root).as_posix()
    project_wheels = '"{project}/' + relative_output + '"'
    project_manifest = '"{project}/' + relative_stage + '/manifest.json"'
    logs = staged / "wheel-build-logs"
    # cibuildwheel's own host interpreter may be newer than the wheel target.
    tool_python = sys.executable if sys.version_info >= (3, 11) else "3.13"
    cibw = ["uv", "tool", "run", "--python", tool_python, "--from", "cibuildwheel==4.3.0", "cibuildwheel"]
    base_environment = os.environ.copy()
    base_environment.update(CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), PYTHONNOUSERSITE="1")
    base_environment.pop("PYTHONPATH", None)
    # This release still follows the repository's Windows ARM64 Python 3.13
    # preview matrix. Other supported hosts need CPython 3.10 through 3.14 FFI.
    ffi_build = args.ffi_build or (
        "cp313-*"
        if platform_name == "windows" and os.environ.get("RUNNER_ARCH", platform.machine()).upper() in ("ARM64", "AARCH64")
        else "cp3{10,11,12,13,14}-*"
    )

    for role in ("ffi", "provider", "frontend"):
        if platform_name == "windows" and role != "ffi" and not args.print_build_identifiers:
            break
        environment = base_environment.copy()
        environment.setdefault("CIBW_ARCHS", "auto64")
        environment["CIBW_BUILD_FRONTEND"] = "build[uv]"
        command = cibw + [str(staged / role), "--output-dir", str(output), "--platform", platform_name]
        if role == "ffi":
            environment["CIBW_BUILD"] = ffi_build
            environment["CIBW_TEST_SKIP"] = ""
            environment["CIBW_TEST_GROUPS"] = ""
            environment["CIBW_TEST_REQUIRES"] = "pytest numpy ml-dtypes packaging"
            environment["CIBW_BEFORE_TEST"] = ""
            environment["CIBW_TEST_COMMAND"] = 'python -m pytest --noconftest -o addopts= -q "{package}/tests/python/test_dtype.py"'
            # Keep this fixed source version even in a gitless staging tree.
            # It is passed to the build process, not written over FFI sources.
            ffi_environment = (
                f'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI="{identity["ffi_version"]}" CMAKE_BUILD_PARALLEL_LEVEL="{args.jobs}"'
            )
            environment["CIBW_ENVIRONMENT_" + platform_name.upper()] = ffi_environment
        else:
            command += ["--config-file", str(root / "pyproject.toml")]
            environment.setdefault("CIBW_BUILD", "cp310-*")
            requirement = (
                f"apache-tvm-ffi=={identity['ffi_version']}" if role == "provider" else f"apache-tvm-tirx=={identity['provider_version']}"
            )
            environment["CIBW_BEFORE_TEST"] = f'uv pip install --python python --find-links {project_wheels} "{requirement}"'
            if role == "provider":
                code = (
                    "import tvm_tirx, tvm_ffi; "
                    "info = dict(tvm_ffi.get_global_func('tl.CompilerBuildInfo')()); "
                    f"assert info['source_digest'] == '{identity['source_digest']}'; "
                    f"assert info['ffi_ref'] == '{identity['ffi_ref']}'"
                )
                environment["CIBW_TEST_COMMAND"] = "python -c " + shlex.quote(code)
            else:
                verify = 'python "{project}/maint/scripts/verify_native_install.py" --expected-source ' + project_manifest
                environment["CIBW_TEST_COMMAND"] = (
                    verify
                    + " --import-order tilelang-first --output native-tilelang-first.json && "
                    + verify
                    + " --import-order tirx-first --output native-tirx-first.json"
                )
                # Its Cython extensions do not configure CUDA or another compiler.
                environment["CIBW_BEFORE_ALL_LINUX"] = ""
        if args.print_build_identifiers:
            command.append("--print-build-identifiers")
        _run(command, cwd=root, environment=environment, record_path=logs / f"{role}.json")

    if platform_name == "windows" and not args.print_build_identifiers:
        # Keep the provider compiler and Cython build directories stable for
        # Windows compiler caches. Only FFI uses the per-interpreter builds.
        raw = staged / "raw-wheelhouse"
        build(staged, raw, args.python, args.jobs, [], roles=("provider", "frontend"), formats=("wheel",), frontend="uv")
        for wheel in sorted(raw.glob("*.whl")):
            command = [
                "uv",
                "tool",
                "run",
                "--from",
                "delvewheel",
                "delvewheel",
                "repair",
                "--analyze-existing",
                "--ignore-existing",
                "--exclude",
                "*tvm_ffi*.dll;libz3.dll;z3.dll",
                "-w",
                str(output),
                str(wheel),
            ]
            _run(command, cwd=root, environment=base_environment, record_path=logs / f"repair-{wheel.stem}.json")

    (output / "native-wheel-bundle.json").write_text(
        json.dumps(
            {
                "source_digest": identity["source_digest"],
                "ffi_ref": identity["ffi_ref"],
                "provider_version": identity["provider_version"],
                "ffi_version": identity["ffi_version"],
                "platform": platform_name,
                "preview_only": args.print_build_identifiers,
                "files": [
                    {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(output.glob("*.whl"))
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
