import importlib

import pytest

try:
    import tvm
    from tvm.target import Target
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip("TVM not available", allow_module_level=True)

_target_mod = importlib.import_module("tilelang.utils.target")
_tt_lower = importlib.import_module("tilelang.engine.tt.lower")
CompiledArtifact = importlib.import_module("tilelang.engine.param").CompiledArtifact


@pytest.fixture
def toggle_tt_backend(monkeypatch):
    original = getattr(_target_mod, "_HAS_TENSTORRENT_BACKEND", False)

    def setter(value: bool):
        monkeypatch.setattr(_target_mod, "_HAS_TENSTORRENT_BACKEND", value, raising=False)

    setter(original)
    try:
        yield setter
    finally:
        setter(original)


def test_available_targets_contains_tt():
    assert _target_mod.TENSTORRENT_TARGET in _target_mod.AVALIABLE_TARGETS


def test_determine_target_returns_target_when_backend_enabled(toggle_tt_backend):
    toggle_tt_backend(True)
    scope_name = _target_mod.determine_target(_target_mod.TENSTORRENT_TARGET)
    assert scope_name == _target_mod.TENSTORRENT_TARGET

    target_obj = _target_mod.determine_target(_target_mod.TENSTORRENT_TARGET, return_object=True)
    assert isinstance(target_obj, Target)
    assert target_obj.kind.name == _target_mod.TENSTORRENT_TARGET


def test_determine_target_raises_when_backend_disabled(toggle_tt_backend):
    toggle_tt_backend(False)
    with pytest.raises(ValueError, match="Tenstorrent backend requires"):
        _target_mod.determine_target(_target_mod.TENSTORRENT_TARGET)


def test_tenstorrent_engine_lower_returns_placeholder(toggle_tt_backend):
    toggle_tt_backend(True)
    artifact = _tt_lower.lower(
        tvm.IRModule(),
        params=None,
        target=_target_mod.TENSTORRENT_TARGET,
        target_host=None,
        runtime_only=False,
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    assert isinstance(artifact, CompiledArtifact)
    assert artifact.kernel_source.startswith("// Tenstorrent backend lowering")
