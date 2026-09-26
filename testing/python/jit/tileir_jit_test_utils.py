"""Shared helpers for TileIR JIT integration tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tilelang import tvm as tvm
from tilelang.tileir import checks
from tilelang.tileir.artifact import TileIRArtifactCompatibility
from tilelang.backend.target import determine_target
from tvm import tirx
from tvm.target import Target

Range = tvm.ir.Range


def make_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    path.chmod(0o755)


def cuda_target_for_test() -> Target:
    return determine_target({"kind": "cuda", "arch": "sm_120"}, return_object=True)


def artifact_compatibility(
    *,
    runtime_version: str = "1.5.0",
    tileiras_version: str = "13.4.0",
) -> TileIRArtifactCompatibility:
    return TileIRArtifactCompatibility(
        target_arch="sm_120",
        cuda_tile_ir_version="13.4",
        cuda_tile_runtime_version=runtime_version,
        tileiras_version=tileiras_version,
    )


def load_mla_ws_example():
    path = Path(__file__).parents[3] / "examples" / "deepseek_mla" / "example_mla_decode_ws.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_mla_decode_ws", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def skip_if_tileir_toolchain_unavailable() -> None:
    try:
        checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")


def enable_tileir_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    major, minor = torch.cuda.get_device_capability()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", "tileir")
    monkeypatch.setenv("TILELANG_TARGET", f"tileir -arch=sm_{major}{minor}")
    return torch


def prim_func_with_interleaved_scalar_param():
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "global")
    a_handle = tirx.Var("a_handle", pointer_type)
    scale = tirx.Var("scale", "float32")
    b_handle = tirx.Var("b_handle", pointer_type)
    a_buffer = tirx.decl_buffer((4,), "float32", name="A", data=a_handle)
    b_buffer = tirx.decl_buffer((4,), "float32", name="B", data=b_handle)
    return (
        tirx.PrimFunc(
            [a_handle, scale, b_handle],
            tirx.Evaluate(0),
            buffer_map={a_handle: a_buffer, b_handle: b_buffer},
        )
        .with_attr("global_symbol", "main")
        .with_attr("thread_extent", {"blockIdx.x": tirx.IntImm("int32", 1)})
    )


def prim_func_with_two_kernel_launches():
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "global")
    a_handle = tirx.Var("a_handle", pointer_type)
    scale = tirx.Var("scale", "float32")
    b_handle = tirx.Var("b_handle", pointer_type)
    a_buffer = tirx.decl_buffer((4,), "float32", name="A", data=a_handle)
    b_buffer = tirx.decl_buffer((4,), "float32", name="B", data=b_handle)

    def launch(name: str, body):
        block_x = tirx.IterVar(
            Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
            tirx.Var(name, "int32"),
            tirx.IterVar.ThreadIndex,
            "blockIdx.x",
        )
        return tirx.AttrStmt(block_x, "thread_extent", tirx.IntImm("int32", 1), body)

    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        tirx.SeqStmt(
            [
                launch("first", tirx.BufferStore(a_buffer, tirx.BufferLoad(a_buffer, [0]) * scale, [0])),
                launch("second", tirx.BufferStore(b_buffer, tirx.BufferLoad(a_buffer, [0]), [0])),
            ]
        ),
    )
    return tirx.PrimFunc(
        [a_handle, scale, b_handle],
        tirx.SBlockRealize([], tirx.const(True, "bool"), root),
        buffer_map={a_handle: a_buffer, b_handle: b_buffer},
    ).with_attr("global_symbol", "main")
