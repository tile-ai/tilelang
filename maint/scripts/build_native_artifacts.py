#!/usr/bin/env python3
"""Stage and build matching FFI, TIRx provider and TileLang distributions.

The staging directories contain source snapshots, never borrowed compiler
libraries. Each can be rebuilt from its sdist without the original checkout.
Run in a dedicated Python environment. With --install, install each built
wheel into the selected --python environment. Source checkouts and submodules
are never changed; builds use the staging copies and do not publish artifacts.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile


def _manifest_module(root):
    spec = importlib.util.spec_from_file_location("_compiler_manifest", root / "tilelang_packaging/compiler_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_tree(source, destination):
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache"),
        )
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def stage_frontend(root, destination, identity):
    """Create the frontend source distribution, including its exact provider pin."""
    destination.mkdir(parents=True, exist_ok=False)
    names = [
        "CMakeLists.txt",
        "pyproject.toml",
        "VERSION",
        "LICENSE",
        "README.md",
        "THIRDPARTYNOTICES.txt",
        "version_provider.py",
        "provider_metadata.py",
        "tilelang_build_backend.py",
        "cmake",
        "src",
        "tilelang",
        "tilelang_packaging",
        "testing",
        "examples",
        "maint/scripts/build_native_artifacts.py",
        "maint/scripts/verify_native_install.py",
        "3rdparty/cutlass/include",
        "3rdparty/cutlass/tools",
        "3rdparty/cutlass/LICENSE.txt",
    ]
    names.extend(path.name for path in root.glob("requirements*.txt"))
    for name in names:
        _copy_tree(root / name, destination / name)
    projection = dict(identity, frontend_only=True)
    (destination / "tilelang_packaging/native-source.json").write_text(json.dumps(projection, indent=2) + "\n")
    (destination / ".git_commit.txt").write_text(identity["repositories"]["tilelang"]["head"] + "\n")


def stage(root, destination):
    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise RuntimeError(f"Choose a new staging directory: {destination}")
    module = _manifest_module(root)
    identity = module.source_identity(root, check=True)
    destination.mkdir(parents=True)
    provider = destination / "provider"
    source = provider / "tilelang-source"
    source.mkdir(parents=True)
    for item in identity["source_files"]:
        _copy_tree(root / item["path"], source / item["path"])
    # Runtime source identity above deliberately omits frontend metadata. These
    # extra files are needed by the embedding project and its license records.
    for name in ("LICENSE", "THIRDPARTYNOTICES.txt", "VERSION", "README.md"):
        _copy_tree(root / name, source / name)
    (source / "tilelang_packaging/native-source.json").write_text(json.dumps(identity, indent=2) + "\n")
    module.verify_sources(source, identity)
    templates = root / "tilelang_packaging/tirx_companion"
    for name in ("CMakeLists.txt", "pyproject.toml"):
        content = (templates / (name + ".in")).read_text()
        content = content.replace("@PROVIDER_VERSION@", identity["provider_version"]).replace("@FFI_VERSION@", identity["ffi_version"])
        (provider / name).write_text(content)
    (provider / "native-source.json").write_text(json.dumps(identity, indent=2) + "\n")
    stage_frontend(root, destination / "frontend", identity)
    # FFI is a separately owned distribution. Build it in its own copy so
    # Cython/SCM output cannot alter the provider's frozen compiler inputs.
    ffi_prefix = Path("3rdparty/tvm-tirx/3rdparty/tvm-ffi")
    ffi = destination / "ffi"
    ffi.mkdir()
    for item in identity["source_files"]:
        relative = Path(item["path"])
        if relative.is_relative_to(ffi_prefix):
            _copy_tree(root / relative, ffi / relative.relative_to(ffi_prefix))
    # build() supplies the checked version to setuptools-scm for this gitless
    # snapshot. The backend writes the sdist's PKG-INFO, which then carries the
    # version when rebuilding that archive. Precreating it here would cause
    # scikit-build-core to include two different entries at the same tar path.
    # This outer manifest records both source ownership and transport bytes.
    files = [
        {"path": path.relative_to(destination).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    (destination / "manifest.json").write_text(json.dumps({"identity": module.build_projection(identity), "files": files}, indent=2) + "\n")
    return identity


def _artifact_metadata(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise RuntimeError(f"Expected one wheel METADATA: {path}")
            content = archive.read(names[0])
    else:
        with tarfile.open(path) as archive:
            entries = [
                item
                for item in archive.getmembers()
                if item.isfile() and len(Path(item.name).parts) == 2 and item.name.endswith("/PKG-INFO")
            ]
            if len(entries) != 1:
                raise RuntimeError(f"Expected one root sdist PKG-INFO: {path}")
            content = archive.extractfile(entries[0]).read()
    return BytesParser().parsebytes(content)


def _check_artifact(path, role, identity):
    metadata = _artifact_metadata(path)
    name = {"ffi": "apache-tvm-ffi", "provider": "apache-tvm-tirx", "frontend": "tilelang"}[role]
    if metadata["Name"].lower().replace("_", "-") != name:
        raise RuntimeError(f"Unexpected distribution in {path}: {metadata['Name']}")
    version = identity.get({"ffi": "ffi_version", "provider": "provider_version"}.get(role, ""))
    if version and metadata["Version"] != version:
        raise RuntimeError(f"Wrong fixed version in {path}: {metadata['Version']} != {version}")
    if role != "ffi":
        dependencies = {
            value.split(";", 1)[0].replace(" ", "").replace("(", "").replace(")", "") for value in metadata.get_all("Requires-Dist", [])
        }
        required = f"apache-tvm-ffi=={identity['ffi_version']}"
        if required not in dependencies:
            raise RuntimeError(f"Missing fixed FFI dependency in {path}")
        if role == "frontend" and f"apache-tvm-tirx=={identity['provider_version']}" not in dependencies:
            raise RuntimeError(f"Missing matched compiler provider dependency in {path}")
    return {"distribution": name, "version": metadata["Version"]}


def build(stage_root, output, python, jobs, cmake_defines, *, roles=None, formats=("sdist", "wheel"), frontend="python", install=False):
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    if install and "wheel" not in formats:
        raise ValueError("--install requires building wheels")
    stage_root, output = stage_root.resolve(), output.resolve()
    manifest = json.loads((stage_root / "manifest.json").read_text())
    for item in manifest["files"]:
        path = stage_root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Staged source was modified: {item['path']}")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "build-logs"
    logs.mkdir(exist_ok=True)
    artifact_manifest = output / "artifacts.json"
    artifacts = []
    if artifact_manifest.exists():
        previous = json.loads(artifact_manifest.read_text())
        if previous["identity"] != manifest["identity"]:
            raise RuntimeError("Choose a new output directory for a different source identity")
        artifacts = previous["files"]
        for item in artifacts:
            if hashlib.sha256((output / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Previous artifact was modified: {item['path']}")
    selected_roles = [name for name in ("ffi", "provider", "frontend") if roles is None or name in roles]
    for role in selected_roles:
        if (logs / f"{role}.json").exists():
            raise RuntimeError(f"A {role} build was already attempted here; choose a new output directory")
    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    # PEP 517 build environments and runtime dependency resolution both need
    # the matching local artifacts. These source-specific versions need not
    # exist on a package index, and --install must not select another provider.
    for key in ("UV_FIND_LINKS", "PIP_FIND_LINKS"):
        env[key] = output.as_uri() + (" " + env[key] if env.get(key) else "")
    # A repeated --role flag selects a subset but cannot reverse dependencies.
    for role in selected_roles:
        before = set(output.glob("*.whl")) | set(output.glob("*.tar.gz"))
        build_env = env.copy()
        if role == "ffi":
            build_env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI"] = manifest["identity"]["ffi_version"]
        if frontend == "uv":
            command = ["uv", "build", "--python", str(python), "--out-dir", str(output)]
        else:
            command = [str(python), "-m", "build", "--no-isolation", "--outdir", str(output)]
        # Both frontends default to sdist followed by a wheel rebuilt from that
        # archive. Passing both --sdist and --wheel instead builds each directly
        # from the checkout, which would miss incomplete source distributions.
        if len(formats) == 1:
            command.append(f"--{formats[0]}")
        for setting in cmake_defines if role != "ffi" else ():
            if "=" not in setting:
                raise ValueError("--cmake-define requires NAME=VALUE")
            command += ["--config-setting", "cmake.define." + setting]
        start = time.time()
        record = {"command": command, "cwd": str(stage_root / role), "jobs": jobs, "role": role, "status": "running", "exit_code": None}
        path = logs / f"{role}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        with (logs / f"{role}.log").open("w") as stream:
            result = subprocess.run(command, cwd=stage_root / role, env=build_env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(status="completed", exit_code=result.returncode, elapsed_seconds=time.time() - start)
        path.write_text(json.dumps(record, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{role} build failed; see {logs / (role + '.log')}")
        produced = (set(output.glob("*.whl")) | set(output.glob("*.tar.gz"))) - before
        if not produced or any(
            not any(path.suffix == ".whl" if kind == "wheel" else path.name.endswith(".tar.gz") for path in produced) for kind in formats
        ):
            raise RuntimeError(f"No new requested {role} artifacts in {output}")
        for artifact in sorted(produced):
            info = _check_artifact(artifact, role, manifest["identity"])
            artifacts.append({"path": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), **info})
        temporary = artifact_manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": manifest["identity"], "files": artifacts}, indent=2) + "\n")
        temporary.replace(artifact_manifest)
        if install:
            wheels = sorted(str(path) for path in produced if path.suffix == ".whl")
            command = [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--find-links",
                str(output),
                "--reinstall-package",
                info["distribution"],
                *wheels,
            ]
            record = {"command": command, "cwd": str(output), "status": "running", "exit_code": None}
            path = logs / f"{role}-install.json"
            path.write_text(json.dumps(record, indent=2) + "\n")
            with (logs / f"{role}-install.log").open("w") as stream:
                result = subprocess.run(command, cwd=output, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record.update(status="completed", exit_code=result.returncode)
            path.write_text(json.dumps(record, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"{role} installation failed; see {logs / (role + '-install.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("stage")
    prepare.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    prepare.add_argument("--destination", type=Path, required=True)
    compile_parser = subparsers.add_parser("build")
    compile_parser.add_argument("--stage", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    compile_parser.add_argument("--jobs", type=int, default=4)
    compile_parser.add_argument("--cmake-define", action="append", default=[])
    compile_parser.add_argument("--role", action="append", choices=["ffi", "provider", "frontend"])
    compile_parser.add_argument(
        "--frontend",
        choices=["python", "uv"],
        default="python",
        help="python uses an existing build environment; uv creates isolated PEP 517 build environments",
    )
    compile_parser.add_argument("--install", action="store_true", help="Install built wheels into --python after validating their metadata")
    formats = compile_parser.add_mutually_exclusive_group()
    formats.add_argument("--sdist-only", action="store_true")
    formats.add_argument("--wheel-only", action="store_true")
    args = parser.parse_args()
    if args.action == "stage":
        identity = stage(args.source, args.destination)
        print(json.dumps({"source_digest": identity["source_digest"], "provider_version": identity["provider_version"]}))
    else:
        formats = ("sdist",) if args.sdist_only else ("wheel",) if args.wheel_only else ("sdist", "wheel")
        build(
            args.stage,
            args.output,
            args.python,
            args.jobs,
            args.cmake_define,
            roles=args.role,
            formats=formats,
            frontend=args.frontend,
            install=args.install,
        )


if __name__ == "__main__":
    main()
