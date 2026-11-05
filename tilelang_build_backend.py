"""Scikit-build hooks, freezing the compiler dependency into frontend sdists."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile

from scikit_build_core.build import (
    build_editable as build_editable,
    build_wheel as build_wheel,
    get_requires_for_build_editable as get_requires_for_build_editable,
    get_requires_for_build_sdist as get_requires_for_build_sdist,
    get_requires_for_build_wheel as get_requires_for_build_wheel,
    prepare_metadata_for_build_editable as prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel as prepare_metadata_for_build_wheel,
)
from scikit_build_core.build import build_sdist as _build_sdist


def build_sdist(sdist_directory, config_settings=None):
    root = Path(__file__).resolve().parent
    if (root / "tilelang_packaging/native-source.json").is_file():
        return _build_sdist(sdist_directory, config_settings)
    target = Path(sdist_directory).resolve()
    spec = importlib.util.spec_from_file_location("_tilelang_artifacts", root / "maint/scripts/build_native_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity = module._manifest_module(root).source_identity(root)
    with tempfile.TemporaryDirectory(prefix="tilelang-frontend-sdist-") as temporary:
        source = Path(temporary) / "source"
        module.stage_frontend(root, source, identity)
        previous = Path.cwd()
        try:
            os.chdir(source)
            return _build_sdist(str(target), config_settings)
        finally:
            os.chdir(previous)
