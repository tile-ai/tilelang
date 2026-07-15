"""Assembly contracts for the TileIR JIT backend."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import types

import pytest

from tilelang.tileir import assembly as tileir_assembly
from tilelang.tileir import checks
from tilelang.tileir.artifact import TileIRLaunchMetadata
from tilelang.tileir.assembly import assemble_tileir_module
from tilelang.tileir.errors import TileIRAssemblyError

from tileir_jit_test_utils import cuda_target_for_test as _cuda_target_for_test


def test_tileir_assemble_module_writes_bytecode_without_text_translation(monkeypatch, tmp_path):
    calls = []
    writes = []
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "bin" / "tileiras",
        tileiras_version="tileiras 13.3",
    )

    class FakeModule:
        operation = object()

        def __str__(self):
            return "module { cuda_tile.module @fake {} }"

    def fake_write_tileir_bytecode(tileir_module, output_path):
        writes.append((tileir_module, output_path))
        output_path.write_bytes(b"tileir-bytecode")

    def fake_run(cmd, check, capture_output, text, timeout, env=None):
        del check, capture_output, text, timeout, env
        calls.append(cmd)
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_bytes(b"cubin")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(tileir_assembly, "write_tileir_bytecode", fake_write_tileir_bytecode)
    monkeypatch.setattr(tileir_assembly, "optimize_tileir_module", lambda _: None)
    monkeypatch.setattr(tileir_assembly.subprocess, "run", fake_run)

    fake_module = FakeModule()
    result = assemble_tileir_module(
        fake_module,
        kernel_name="kernel",
        target=_cuda_target_for_test(),
        toolchain=toolchain,
        launch_metadata=TileIRLaunchMetadata(grid=(4, 1, 1)),
    )

    assert writes[0][0] is fake_module
    assert result.kernel_name == "kernel"
    assert result.tileir_source == "module { cuda_tile.module @fake {} }"
    assert result.cubin == b"cubin"
    assert result.launch_metadata.grid == (4, 1, 1)
    assert len(calls) == 1
    assert calls[0][0] == str(toolchain.tileiras_path)
    assert "--gpu-name=sm_120" in calls[0]


def test_tileir_optimizer_forwards_opt_level(monkeypatch):
    assigned = []
    optimized = []

    class FakeOptions:
        def __setattr__(self, name, value):
            assigned.append((name, value))
            super().__setattr__(name, value)

    class FakeModule:
        operation = object()

    def fake_apply(operation, options):
        optimized.append((operation, options))
        return True

    fake_cuda_tile = types.SimpleNamespace(
        TileIROptimizationsOpts=FakeOptions,
        applyTileIROptimizations=fake_apply,
    )
    monkeypatch.setitem(sys.modules, "cuda_tile._mlir._mlir_libs._cuda_tile", fake_cuda_tile)

    tileir_assembly.optimize_tileir_module(FakeModule())

    assert len(optimized) == 1
    assert isinstance(optimized[0][1], FakeOptions)
    # The optimizer forwards the TileIR opt_level (default 3) onto the opts.
    assert assigned == [("opt_level", 3)]

    # A caller-provided opt_level is forwarded verbatim.
    assigned.clear()
    optimized.clear()
    tileir_assembly.optimize_tileir_module(FakeModule(), opt_level=0)
    assert assigned == [("opt_level", 0)]


def test_tileir_assemble_module_cleans_cuda_home_for_pip_tileiras(monkeypatch, tmp_path):
    captured_env = {}
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "nvidia" / "cu13" / "bin" / "tileiras",
        tileiras_version="tileiras 13.3",
    )

    class FakeModule:
        operation = object()

        def __str__(self):
            return "module { cuda_tile.module @fake {} }"

    def fake_write_tileir_bytecode(tileir_module, output_path):
        del tileir_module
        output_path.write_bytes(b"tileir-bytecode")

    def fake_run(cmd, check, capture_output, text, timeout, env=None):
        del check, capture_output, text, timeout
        captured_env.update(env or {})
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_bytes(b"cubin")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv("CUDA_PATH", "/usr/local/cuda")
    monkeypatch.setattr(checks, "is_pip_tileiras_path", lambda _: True)
    monkeypatch.setattr(tileir_assembly, "write_tileir_bytecode", fake_write_tileir_bytecode)
    monkeypatch.setattr(tileir_assembly, "optimize_tileir_module", lambda _: None)
    monkeypatch.setattr(tileir_assembly.subprocess, "run", fake_run)

    assemble_tileir_module(
        FakeModule(),
        kernel_name="kernel",
        target=_cuda_target_for_test(),
        toolchain=toolchain,
    )

    assert "CUDA_HOME" not in captured_env
    assert "CUDA_PATH" not in captured_env


def test_tileir_assemble_module_raises_on_tileiras_failure(monkeypatch, tmp_path):
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "bin" / "tileiras",
        tileiras_version="tileiras 13.3",
    )

    class FakeModule:
        operation = object()

        def __str__(self):
            return "module { cuda_tile.module @fake {} }"

    def fake_write_tileir_bytecode(tileir_module, output_path):
        del tileir_module
        output_path.write_bytes(b"tileir-bytecode")

    def fake_run(cmd, check, capture_output, text, timeout, env=None):
        del check, capture_output, text, timeout, env
        return subprocess.CompletedProcess(cmd, 1, "", "tileiras failed")

    monkeypatch.setattr(tileir_assembly, "write_tileir_bytecode", fake_write_tileir_bytecode)
    monkeypatch.setattr(tileir_assembly, "optimize_tileir_module", lambda _: None)
    monkeypatch.setattr(tileir_assembly.subprocess, "run", fake_run)

    with pytest.raises(TileIRAssemblyError, match="tileiras failed"):
        assemble_tileir_module(
            FakeModule(),
            kernel_name="kernel",
            target=_cuda_target_for_test(),
            toolchain=toolchain,
            launch_metadata=TileIRLaunchMetadata(grid=(2, 1, 1)),
        )
