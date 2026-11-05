"""Cold public imports install real target callbacks before backend resolution."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry", ["tilelang", "tilelang.backend.target", "tilelang.cuda.target", "tilelang.ascend.target"])
def test_public_target_registration_from_cold_import(entry):
    source = f"""
import importlib
import sys
importlib.import_module({entry!r})
from tvm_tirx.target import Target
from tilelang.backend.target import (
    determine_target, list_target_detectors, register_target_detector,
)
# Frozen old-stack import: probe-target-detectors-old-01. Preserve priority on
# hosts exposing more than one backend, independently of the first public import.
assert list_target_detectors() == ("cuda", "hip", "ascend", "metal")
target = determine_target({{"kind": "cuda", "arch": "sm_90a"}}, return_object=True)
assert target.kind.name == "cuda" and target.attrs["arch"] == "sm_90a"
assert determine_target("pto", return_object=True).kind.name == "ascend"
# User registration keeps its existing override semantics after a cold import.
register_target_detector("cuda", lambda: Target("llvm"), override=True)
assert determine_target("auto", return_object=True).kind.name == "llvm"
assert "tilelang.cuda.pipeline" not in sys.modules
assert "tilelang.jit.adapter.nvrtc" not in sys.modules
assert "tvm" not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("entry", ["tilelang.engine.lower", "tilelang.engine.param", "tilelang.jit.kernel"])
def test_lower_function_survives_direct_submodule_import(entry):
    source = f"""
import importlib
importlib.import_module({entry!r})
import tilelang
import tilelang.engine
module = importlib.import_module("tilelang.engine.lower")
assert callable(tilelang.lower)
assert tilelang.lower is module.lower
assert tilelang.engine.lower is module.lower
assert tilelang.engine.is_device_call is module.is_device_call
"""
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
