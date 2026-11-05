"""PEP 517 metadata for the exact companion compiler built from these sources."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _identity():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("_tilelang_compiler_manifest", root / "tilelang_packaging/compiler_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module.source_identity(root)


def dynamic_metadata(field, settings=None):
    if field != "dependencies":
        raise ValueError(f"Unsupported dynamic metadata field: {field}")
    root, identity = _identity()
    requirements = [
        line.strip() for line in (root / "requirements.txt").read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    return [*requirements, f"apache-tvm-tirx=={identity['provider_version']}"]
