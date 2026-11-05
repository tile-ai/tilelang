"""Source identity shared by the frontend and its canonical compiler provider.

This module only reads files and Git identities. It never imports either compiler
and does not use a pre-existing build to infer the selected source version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

TIRX_REF = "7e49a148b8f7ce00c0e0d5629d1b4c4d46d1c3f9"
FFI_REF = "3e906bc685ea45624bc80e8d7b7d86497d492af3"
FFI_VERSION = "0.1.15.post3.dev30+g3e906bc68"
TIRX_VERSION = "0.1.0a2"
FROZEN_IDENTITY = "tilelang_packaging/native-source.json"

# These are compiler inputs, not all first-party consumers. Frontend-only Python
# edits need not rebuild the native provider. Device template headers are part of
# src/ and therefore remain in this identity.
_INPUTS = (
    "CMakeLists.txt",
    "cmake",
    "src",
    "testing/cpp",
    "tilelang_packaging/compiler_manifest.py",
    "tilelang_packaging/tirx_companion",
    "3rdparty/tvm-tirx/CMakeLists.txt",
    "3rdparty/tvm-tirx/cmake",
    "3rdparty/tvm-tirx/include",
    "3rdparty/tvm-tirx/src",
    "3rdparty/tvm-tirx/python",
    "3rdparty/tvm-tirx/licenses",
    "3rdparty/tvm-tirx/LICENSE",
    "3rdparty/tvm-tirx/NOTICE",
    "3rdparty/tvm-tirx/3rdparty/compiler-rt",
    "3rdparty/tvm-tirx/3rdparty/tvm-ffi",
    "3rdparty/hip-headers/include",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _listed_files(path: Path):
    """Use Git's file inventory, expanding initialized submodule gitlinks."""
    repository = Path(_git(path if path.is_dir() else path.parent, "rev-parse", "--show-toplevel"))
    relative = path.relative_to(repository).as_posix()
    output = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", relative]
    )
    for raw in output.split(b"\0"):
        if not raw:
            continue
        candidate = repository / os.fsdecode(raw)
        if candidate.is_dir() and (candidate / ".git").exists():
            yield from _listed_files(candidate)
        elif candidate.is_file():
            yield candidate


def _source_files(root: Path):
    files = set()
    for relative in _INPUTS:
        path = root / relative
        if not path.exists():
            raise RuntimeError(f"Missing compiler source input: {path}")
        for candidate in _listed_files(path):
            rel = candidate.relative_to(root)
            # A compiler source distribution must be self-contained. In-tree
            # source symlinks are accepted only when their referent is inside
            # the selected source tree, and are materialized when staging.
            if candidate.is_symlink() and not candidate.resolve().is_relative_to(root):
                raise RuntimeError(f"Source symlink escapes the checkout: {rel}")
            if candidate.is_file():
                files.add(rel.as_posix())
    return [{"path": relative, "sha256": _hash(root / relative)} for relative in sorted(files)]


def verify_sources(root: Path, manifest: dict, *, files_present: bool = True) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("ffi_ref") != FFI_REF:
        raise RuntimeError("Unsupported compiler source identity or FFI revision")
    files = manifest["source_files"]
    if len({item["path"] for item in files}) != len(files):
        raise RuntimeError("Duplicate compiler source paths")
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Invalid source path: {relative}")
        if files_present:
            path = root / relative
            if not path.is_file() or _hash(path) != item["sha256"]:
                raise RuntimeError(f"Compiler source differs from its frozen identity: {relative}")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest["source_digest"] != digest or manifest["provider_version"] != f"{TIRX_VERSION}+tilelang.{digest[:16]}":
        raise RuntimeError("Invalid compiler source digest/version")
    if files_present:
        # CMake source globs can consume newly added files as well as changed
        # ones. A frozen provider must reject additions outside its manifest;
        # checking only the originally recorded hashes would miss those inputs.
        actual = set()
        generated_directories = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
        for name in _INPUTS:
            entry = root / name
            candidates = entry.rglob("*") if entry.is_dir() else (entry,)
            for candidate in candidates:
                relative = candidate.relative_to(root)
                if generated_directories.intersection(relative.parts) or candidate.suffix == ".pyc":
                    continue
                if candidate.is_file():
                    actual.add(relative.as_posix())
        unexpected = actual - {item["path"] for item in files}
        if unexpected:
            raise RuntimeError(f"Unrecorded compiler source inputs: {sorted(unexpected)}")


def source_identity(root: Path, *, check: bool = False) -> dict:
    root = root.resolve()
    frozen = root / FROZEN_IDENTITY
    if frozen.is_file():
        manifest = json.loads(frozen.read_text())
        # A frontend sdist intentionally does not carry compiler sources. Its
        # dependency pin is the projection recorded by the paired source build.
        verify_sources(root, manifest, files_present=check or not manifest.get("frontend_only", False))
        return manifest
    tirx = root / "3rdparty/tvm-tirx"
    ffi = tirx / "3rdparty/tvm-ffi"
    if _git(tirx, "rev-parse", "HEAD") != TIRX_REF:
        raise RuntimeError(f"The compiler provider requires upstream TIRx {TIRX_REF}")
    if _git(tirx, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("The compiler provider requires an unmodified upstream TIRx checkout")
    ffi_gitlink = _git(tirx, "ls-tree", "HEAD", "--", "3rdparty/tvm-ffi").split()
    if ffi_gitlink[:2] != ["160000", "commit"] or ffi_gitlink[2] != FFI_REF:
        raise RuntimeError("The selected FFI must match the fixed upstream TIRx gitlink")
    if _git(ffi, "rev-parse", "HEAD") != FFI_REF:
        raise RuntimeError(f"The compiler provider requires FFI {FFI_REF}")
    if _git(ffi, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Commit and explicitly pin FFI changes before building paired artifacts")
    files = _source_files(root)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1,
        "source_digest": digest,
        "provider_distribution": "apache-tvm-tirx",
        "provider_version": f"{TIRX_VERSION}+tilelang.{digest[:16]}",
        "ffi_ref": FFI_REF,
        "ffi_version": FFI_VERSION,
        "repositories": {
            name: {"head": _git(path, "rev-parse", "HEAD"), "commit_time": _git(path, "show", "-s", "--format=%ci", "HEAD")}
            for name, path in (("tilelang", root), ("tirx", tirx), ("ffi", ffi))
        },
        "source_files": files,
    }


def build_projection(manifest: dict) -> dict:
    return {
        key: manifest[key]
        for key in ("schema_version", "source_digest", "provider_distribution", "provider_version", "ffi_ref", "ffi_version")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = source_identity(args.root, check=args.check)
    args.output.mkdir(parents=True, exist_ok=True)

    def write_if_changed(name, content):
        path = args.output / name
        if not path.exists() or path.read_text() != content:
            path.write_text(content)

    write_if_changed("native-source.json", json.dumps(manifest, indent=2) + "\n")
    projection = build_projection(manifest)
    write_if_changed(
        "_build_info.py", '"""Generated identity of the paired compiler sources."""\nCOMPILER_INFO = ' + repr(projection) + "\n"
    )
    watched = [args.root.resolve() / item["path"] for item in manifest["source_files"]]
    watched += [args.root.resolve() / FROZEN_IDENTITY]
    # Ninja/Make must rerun configuration when a native source changes, so the
    # embedded digest cannot silently lag behind a rebuilt object. The CMake
    # source globs also detect added translation units; their headers are read
    # by the next identity calculation.
    dependencies = "".join(
        'set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "'
        + str(path).replace("\\", "/").replace('"', '\\"').replace(";", "\\;")
        + '")\n'
        for path in watched
        if path.is_file()
    )
    write_if_changed(
        "source-info.cmake",
        f'set(TILELANG_COMPILER_SOURCE_DIGEST "{manifest["source_digest"]}")\n'
        f'set(TILELANG_PROVIDER_VERSION "{manifest["provider_version"]}")\n'
        f'set(TVM_TIRX_SOURCE_REVISION "{manifest["repositories"]["tirx"]["head"]}")\n'
        f'set(TVM_TIRX_SOURCE_REVISION_TIME "{manifest["repositories"]["tirx"].get("commit_time", "NOT-FOUND")}")\n' + dependencies,
    )
    print(json.dumps(projection))


if __name__ == "__main__":
    main()
