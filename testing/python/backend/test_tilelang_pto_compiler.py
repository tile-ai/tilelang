"""Hardware-free PTO compiler contract tests (also run on non-Ascend builds)."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tilelang.ascend import pto
from tilelang.backend import get_backend
from tilelang.backend.execution_backend import normalize_execution_alias
from tilelang.cache.ascend_binary_cache import AscendBinaryCache


def _elf(kind=2):
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    data[16:18] = kind.to_bytes(2, "little")
    return data


def test_pto_reuses_ascend_host_ir_contract():
    ascend = get_backend("ascend")
    backend = get_backend("pto")
    assert backend.pipelines == ascend.pipelines
    assert backend.host_codegens == ascend.host_codegens
    (execution,) = backend.execution_backends
    assert execution is ascend.execution_backends[0]
    assert execution.name == "tvm_ffi"
    assert execution.enable_host_codegen and execution.enable_device_compile
    assert execution.supports_callee_allocated_outputs


def test_pto_alias_normalizes_without_modifying_configs(monkeypatch):
    from tilelang.backend import target as target_module
    from tilelang.ascend import target as ascend_target

    target = object()
    monkeypatch.setattr(target_module, "determine_target", lambda *a, **kw: target)
    monkeypatch.setattr(ascend_target, "target_is_pto", lambda obj: obj is target)
    configs = {"tl.enable_fast_math": True}
    with pytest.warns(DeprecationWarning, match="legacy wrapper is no longer used"):
        actual, execution, result_configs = normalize_execution_alias("pto", "pto", configs)
    assert actual is target and execution == "tvm_ffi"
    assert result_configs is configs
    monkeypatch.setattr(ascend_target, "target_is_pto", lambda obj: False)
    with pytest.raises(ValueError, match="requires target='pto'"):
        normalize_execution_alias("ascend", "pto", configs)


def test_device_extraction_rejects_missing_or_ambiguous_elf(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "relocatable.o").write_bytes(_elf(1))
    (tmp_path / "host.so").write_bytes(_elf(3))
    with pytest.raises(RuntimeError, match="got 0"):
        pto._read_device_binary(tmp_path)
    (tmp_path / "device.o").write_bytes(_elf())
    assert pto._read_device_binary(tmp_path) == _elf()
    (tmp_path / "other.o").write_bytes(_elf())
    with pytest.raises(RuntimeError, match="got 2"):
        pto._read_device_binary(tmp_path)


def test_compile_ptodsl_entry_in_subprocess(tmp_path):
    source = """
class Kernel:
    def compile(self):
        return self
    def mlir_text(self):
        return "module {}"
entry = Kernel()
"""
    output = tmp_path / "kernel.pto"
    pto.compile_ptodsl_source_to_pto(source, "entry", tmp_path / "kernel.py", output)
    assert output.read_text() == "module {}"
    with pytest.raises(RuntimeError, match="PTODSL compile-only lowering failed"):
        pto.compile_ptodsl_source_to_pto(source, "missing", tmp_path / "kernel.py", output)


def test_compile_pto_builds_only_device_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("PTO_ARCH", "a5")
    monkeypatch.setenv("PTO_FLAGS", "--pto-backend=vpto")
    monkeypatch.setattr(AscendBinaryCache, "load", lambda *a: None)
    saved = []
    monkeypatch.setattr(AscendBinaryCache, "save", lambda *a: saved.append(a))
    monkeypatch.setattr(pto.bisheng, "find_bisheng_path", lambda: "bisheng")
    monkeypatch.setattr(pto, "_find_msobjdump", lambda: "msobjdump")
    monkeypatch.setattr(pto.utils, "tempdir", lambda **kw: SimpleNamespace(temp_dir=str(tmp_path)))
    frontend = []
    monkeypatch.setattr(pto, "compile_ptodsl_source_to_pto", lambda *a: frontend.append(a))
    commands = []

    def run(command, stage):
        commands.append(command)
        if command[0] == "msobjdump":
            (Path(command[-1]) / "kernel.o").write_bytes(_elf())

    monkeypatch.setattr(pto, "_run", run)
    result = pto.compile_pto("ptodsl source", "entry", "ascend", {"tl.device_compile_flags": ["--foo"]})
    assert result == _elf()
    assert frontend[0][:2] == ("ptodsl source", "entry")
    assert [cmd[0] for cmd in commands] == ["ptoas", "bisheng", "msobjdump"]
    assert "--foo" in commands[0]
    assert "--cce-fatobj-link" in commands[1]
    assert not any("launch.cpp" in arg for cmd in commands for arg in cmd)
    assert saved[0][1:] == ("aibin", result)
    monkeypatch.setattr(AscendBinaryCache, "load", lambda *a: bytes(result))
    monkeypatch.setattr(pto, "_run", lambda *a: pytest.fail("cache hit must not invoke the toolchain"))
    assert pto.compile_pto("ptodsl source", "entry", "ascend") == result


def test_compile_pto_rejects_unsupported_arch(monkeypatch):
    monkeypatch.setenv("PTO_ARCH", "a3")
    with pytest.raises(ValueError, match="only supports PTO_ARCH=a5"):
        pto.compile_pto("source", "entry", "ascend")


def test_tool_failure_reports_stage_and_output(monkeypatch):
    monkeypatch.setattr(pto.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, "out", "err"))
    with pytest.raises(RuntimeError, match="device link failed") as error:
        pto._run(["bisheng", "--cce-fatobj-link"], "device link")
    assert "out" in str(error.value) and "err" in str(error.value)
