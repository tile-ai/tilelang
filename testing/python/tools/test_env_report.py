import importlib.util
import sys
import types
from pathlib import Path


def _load_env_report():
    module_path = Path(__file__).resolve().parents[3] / "tilelang" / "tools" / "env_report.py"
    spec = importlib.util.spec_from_file_location("unit_env_report", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_env_report_includes_tilelang_and_selected_env(monkeypatch):
    env = types.SimpleNamespace(
        CUDA_HOME="/opt/cuda",
        ROCM_HOME="",
        get_default_target=lambda: {"kind": "cuda", "arch": "sm_90"},
        get_default_execution_backend=lambda: "auto",
        get_default_verbose=lambda: True,
    )
    tilelang_module = types.ModuleType("tilelang")
    tilelang_module.env = env
    target_module = types.ModuleType("tilelang.backend.target")
    target_module.list_target_detectors = lambda: ("cuda", "hip")

    monkeypatch.setitem(sys.modules, "tilelang", tilelang_module)
    monkeypatch.setitem(sys.modules, "tilelang.backend", types.ModuleType("tilelang.backend"))
    monkeypatch.setitem(sys.modules, "tilelang.backend.target", target_module)
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", '{kind: "cuda", arch: "sm_90"}')
    monkeypatch.setenv("MACA_HOME", "/opt/maca")

    report = _load_env_report().collect_env_report(extra_env_keys=["MACA_HOME"])

    assert report["tilelang"]["cuda_home"] == "/opt/cuda"
    assert report["tilelang"]["default_target"] == {"kind": "cuda", "arch": "sm_90"}
    assert report["tilelang"]["target_detectors"] == ["cuda", "hip"]
    assert report["environment"]["TILELANG_DEFAULT_TARGET"] == '{kind: "cuda", arch: "sm_90"}'
    assert report["environment"]["MACA_HOME"] == "/opt/maca"
