"""The four packaged frontend extensions use normal top-level imports."""

import importlib
from importlib import machinery, util
from pathlib import Path
import sys

import pytest

from tilelang import _extensions


@pytest.mark.parametrize("name", sorted(_extensions._FRONTEND_EXTENSION_NAMES))
def test_frontend_extension_import_and_identity(name):
    before = list(sys.path)
    module = importlib.import_module(name)
    assert module is importlib.import_module(name)
    assert sys.modules[name] is module and module.__name__ == name
    assert any(Path(module.__file__).resolve().parent == path for path in _extensions._extension_directories())
    assert isinstance(module.__loader__, machinery.ExtensionFileLoader)
    assert sys.path == before


def test_finder_registration_is_idempotent():
    before = list(sys.path)
    _extensions.install_frontend_extension_finder()
    _extensions.install_frontend_extension_finder()
    assert sys.meta_path.count(_extensions._finder) == 1
    assert sys.path == before


@pytest.mark.parametrize("name", ["tvm", "tvm_ffi", "tvm_tirx", "tilelang.not_an_extension"])
def test_finder_does_not_claim_other_modules(name, tmp_path):
    finder = _extensions._FrontendExtensionFinder((tmp_path,))
    assert finder.find_spec(name) is None


def test_missing_extension_does_not_fall_back_to_another_distribution(tmp_path):
    finder = _extensions._FrontendExtensionFinder((tmp_path,))
    with pytest.raises(ModuleNotFoundError) as error:
        finder.find_spec("tilelang_cython_wrapper")
    assert error.value.name == "tilelang_cython_wrapper"
    assert finder.find_spec("tilelang_cython_wrapper", path=[str(tmp_path)]) is None


def test_directory_and_interpreter_suffix_order(tmp_path):
    primary, fallback = tmp_path / "primary", tmp_path / "fallback"
    primary.mkdir()
    fallback.mkdir()
    name = "tilelang_pto_wrapper"
    first = primary / (name + machinery.EXTENSION_SUFFIXES[-1])
    second = fallback / (name + machinery.EXTENSION_SUFFIXES[0])
    first.write_bytes(b"discovery-only primary")
    second.write_bytes(b"discovery-only fallback")
    finder = _extensions._FrontendExtensionFinder((primary, fallback))
    spec = finder.find_spec(name)
    assert spec.origin == str(first)
    assert isinstance(spec.loader, machinery.ExtensionFileLoader)
    preferred = primary / (name + machinery.EXTENSION_SUFFIXES[0])
    preferred.write_bytes(b"discovery-only preferred suffix")
    assert finder.find_spec(name).origin == str(preferred)


def test_invalid_selected_binary_preserves_loader_error(tmp_path):
    name = "tilelang_deepgemm_heuristics"
    damaged = tmp_path / (name + machinery.EXTENSION_SUFFIXES[0])
    damaged.write_bytes(b"not a native extension")
    finder = _extensions._FrontendExtensionFinder((tmp_path, *_extensions._extension_directories()))
    spec = finder.find_spec(name)
    assert spec.origin == str(damaged)
    with pytest.raises(ImportError):
        util.module_from_spec(spec)
