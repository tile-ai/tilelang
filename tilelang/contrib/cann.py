"""Identify the CANN installation and templates used to compile an Ascend kernel.

This module uses only the standard library so its environment checks can be
tested without importing TVM or initializing an NPU.
"""

from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re


def _hash_contents(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _hash_toolchain(path, stamp):
    # Installed compiler/linker binaries are large and not development headers.
    return _hash_contents(path)


def _file_identity(path, *, toolchain=False):
    path = Path(path).resolve()
    # Mutable headers/templates and version metadata need content identity.
    # Same-size writes can share mtime/ctime on a coarse-timestamp filesystem.
    if not toolchain:
        return _hash_contents(path)
    stat = path.stat()
    stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return _hash_toolchain(str(path), stamp)


def _roots(sdk):
    return list(dict.fromkeys([sdk, sdk / (platform.machine() + "-linux")]))


def _metadata_file(sdk, component):
    for root in _roots(sdk):
        path = root / "share/info" / component / "version.info"
        if path.is_file():
            return path
    raise RuntimeError(f"CANN component metadata is missing: {component} under {sdk}")


def _sdk_root(compiler):
    configured = [Path(os.environ[key]).resolve() for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if os.environ.get(key)]
    if configured:
        # One variable may name the architecture subdirectory of the other.
        candidates = [p for p in configured if all(p == q or p in q.parents for q in configured)]
        if not candidates:
            raise RuntimeError("ASCEND_HOME_PATH and ASCEND_TOOLKIT_HOME select different SDKs")
        sdk = candidates[0]
    else:
        sdk = next((p for p in compiler.parents if any((r / "share/info/asc-devkit/version.info").is_file() for r in _roots(p))), None)
        if sdk is None:
            raise RuntimeError("Cannot identify CANN; source the selected SDK's set_env.sh")
    if not compiler.is_relative_to(sdk):
        raise RuntimeError(f"Bisheng is outside the selected CANN installation: {compiler}")
    return sdk


def _header_identity(root):
    """Hash header content, including edits made in place with unchanged versions."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".h", ".hpp", ".hh", ".inc"))
    return [(p.relative_to(root).as_posix(), _file_identity(p)) for p in files]


def compile_context(compiler, template_root):
    compiler = Path(compiler).resolve()
    if not compiler.is_file():
        raise RuntimeError(f"Bisheng does not exist: {compiler}")
    sdk = _sdk_root(compiler)
    linker = compiler.with_name("ld.lld")
    if not linker.is_file():
        raise RuntimeError(f"Selected Bisheng has no sibling ld.lld: {compiler}")

    components, metadata = {}, {}
    for name in ("asc-devkit", "bisheng-compiler", "runtime"):
        path = _metadata_file(sdk, name)
        text = path.read_text(encoding="utf-8")
        matches = re.findall(r"^Version=([^\r\n]+)", text, re.M)
        if len(matches) != 1:
            raise RuntimeError(f"Invalid CANN version metadata: {path}")
        components[name] = matches[0].strip()
        metadata[name] = _file_identity(path)
    if len(set(components.values())) != 1:
        raise RuntimeError(f"Mixed CANN component versions: {components}")

    includes = []
    header_roots = []
    for root in _roots(sdk):
        for path in (root / "include", root / "asc/include", root / "asc"):
            if path.is_dir() and path.resolve() not in includes:
                includes.append(path.resolve())
        for path in (root / "include", root / "asc"):
            if path.is_dir() and path.resolve() not in header_roots:
                header_roots.append(path.resolve())
    version_header = next((p / "version/asc_devkit_version.h" for p in includes if (p / "version/asc_devkit_version.h").is_file()), None)
    if version_header is None or not any((p / "c_api/asc_simd.h").is_file() for p in includes):
        raise RuntimeError("The selected CANN installation lacks its SIMD/version headers")
    definitions = version_header.read_text(encoding="utf-8")
    numbers = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        found = re.findall(r"^#define\s+ASC_DEVKIT_" + part + r"\s+(\d+)\s*$", definitions, re.M)
        if len(found) != 1:
            raise RuntimeError(f"Invalid numeric SDK version in {version_header}")
        numbers.append(int(found[0]))
    declared = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:$|[~.+-])", components["asc-devkit"])
    if declared is None or tuple(numbers) != tuple(map(int, declared.groups())):
        raise RuntimeError("CANN numeric header version disagrees with component metadata")

    templates = Path(template_root).resolve() / "tl_templates/ascend"
    if not (templates / "cann_compat.h").is_file():
        raise RuntimeError(f"Selected TileLang templates lack cann_compat.h: {templates}")
    identity = {
        "schema": 1,
        "components": components,
        "metadata": metadata,
        "numeric_version": numbers,
        "compiler": _file_identity(compiler, toolchain=True),
        "linker": _file_identity(linker, toolchain=True),
        "sdk_headers": [_header_identity(p) for p in header_roots],
        "templates": _header_identity(templates),
    }
    fingerprint = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "sdk_root": str(sdk),
        "compiler": str(compiler),
        "linker": str(linker),
        "include_dirs": [str(p) for p in includes],
        "components": components,
        "numeric_version": tuple(numbers),
        "fingerprint": fingerprint,
    }
