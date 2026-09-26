"""Target selection and dependency contracts for the TileIR JIT backend."""

from __future__ import annotations

from pathlib import Path
import types

import pytest

import tilelang
import tilelang.testing
from tilelang.tileir import checks
from tilelang.tileir.errors import TileIRLoweringNotImplementedError
from tilelang.backend import create_backend_context
from tilelang.backend.target import determine_target
from tvm.target import Target

from tileir_jit_test_utils import (
    cuda_target_for_test as _cuda_target_for_test,
    enable_tileir_runtime as _enable_tileir_runtime,
    load_mla_ws_example as _load_mla_ws_example,
    make_executable as _make_executable,
    skip_if_tileir_toolchain_unavailable as _skip_if_tileir_toolchain_unavailable,
)

_MOCK_TILEIRAS_PATH = Path("mock-tileiras")


def test_tileir_target_normalizes_to_cuda_with_tileir_key():
    target = determine_target("tileir -arch=sm_120", return_object=True)

    assert isinstance(target, Target)
    assert target.kind.name == "cuda"
    assert "tileir" in target.keys
    assert target.attrs["arch"] == "sm_120"


def test_tileir_target_malformed_cli_does_not_fall_back():
    with pytest.raises(AssertionError, match="TileIR target"):
        determine_target("tileir -arch='sm_120", return_object=True)


def test_tileir_target_uses_tileir_backend_for_auto_execution_backend():
    context = create_backend_context("tileir -arch=sm_120", "c", "auto")

    assert context.module.allowed_execution_backends(context.target) == ("tileir",)
    assert context.execution_backend.name == "tileir"


def test_cuda_target_allows_explicit_tileir_backend(monkeypatch):
    monkeypatch.setattr(checks, "is_tileir_available", lambda: True)
    target = _cuda_target_for_test()
    context = create_backend_context(target, "c", "tileir")

    assert "tileir" in context.module.allowed_execution_backends(context.target)
    assert context.execution_backend.name == "tileir"


def test_tileir_dependency_check_does_not_import_cutile_dsl(monkeypatch):
    fake_tileir = types.SimpleNamespace(**{symbol: object() for symbol in checks.CUDA_TILE_IR_REQUIRED_SYMBOLS})
    fake_runtime = types.SimpleNamespace(TileDispatcher=object(), launch=object())

    def fake_find_spec(module_name: str):
        if module_name == "cuda.tile":
            raise AssertionError("TileIR backend must not import the cuTile Python DSL")
        if module_name in {checks.CUDA_TILE_IR_MLIR_MODULE, checks.CUDA_TILE_RUNTIME_MODULE}:
            return object()
        return None

    def fake_import_module(module_name: str):
        if module_name == "cuda.tile":
            raise AssertionError("TileIR backend must not import the cuTile Python DSL")
        if module_name == checks.CUDA_TILE_IR_MLIR_MODULE:
            return fake_tileir
        if module_name == checks.CUDA_TILE_RUNTIME_MODULE:
            return fake_runtime
        raise ImportError(module_name)

    monkeypatch.setattr(checks.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(checks.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(checks, "_validated_cuda_tile_runtime_version", lambda: "1.5.0")
    monkeypatch.setattr(checks, "find_tileiras", lambda: _MOCK_TILEIRAS_PATH)
    monkeypatch.setattr(checks, "_tileiras_version", lambda _: "tileiras 13.4")

    toolchain = checks.check_tileir_available()

    assert toolchain.cuda_tile_ir_module == checks.CUDA_TILE_IR_MLIR_MODULE
    assert toolchain.tileiras_path == _MOCK_TILEIRAS_PATH


def test_tileir_dependency_error_points_to_cuda_tile_ir(monkeypatch):
    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda _: None)

    with pytest.raises(checks.TileIRDependencyError, match="NVIDIA/cuda-tile"):
        checks.check_tileir_available()


@pytest.mark.parametrize("missing_symbol", ["pack", "fpowf"])
def test_tileir_dependency_check_rejects_old_cuda_tile_ir_bindings(monkeypatch, missing_symbol):
    fake_tileir = types.SimpleNamespace(**{symbol: object() for symbol in checks.CUDA_TILE_IR_REQUIRED_SYMBOLS if symbol != missing_symbol})

    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(checks.importlib, "import_module", lambda _: fake_tileir)

    with pytest.raises(checks.TileIRDependencyError, match=f"older than 13.4.*{missing_symbol}"):
        checks.check_tileir_available()


def test_tileir_dependency_check_rejects_missing_native_dispatcher(monkeypatch):
    fake_tileir = types.SimpleNamespace(**{symbol: object() for symbol in checks.CUDA_TILE_IR_REQUIRED_SYMBOLS})

    def fake_find_spec(module_name: str):
        if module_name == checks.CUDA_TILE_IR_MLIR_MODULE:
            return object()
        if module_name == checks.CUDA_TILE_RUNTIME_MODULE:
            return None
        return None

    monkeypatch.setattr(checks.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(checks.importlib, "import_module", lambda _: fake_tileir)
    monkeypatch.setattr(checks, "find_tileiras", lambda: _MOCK_TILEIRAS_PATH)
    monkeypatch.setattr(checks, "_tileiras_version", lambda _: "tileiras 13.4")

    with pytest.raises(checks.TileIRDependencyError, match="native dispatcher"):
        checks.check_tileir_available()


def test_tileir_dependency_check_rejects_older_native_dispatcher_version(monkeypatch):
    fake_tileir = types.SimpleNamespace(**{symbol: object() for symbol in checks.CUDA_TILE_IR_REQUIRED_SYMBOLS})
    fake_runtime = types.SimpleNamespace(TileDispatcher=object(), launch=object())

    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(
        checks.importlib,
        "import_module",
        lambda module_name: fake_runtime if module_name == checks.CUDA_TILE_RUNTIME_MODULE else fake_tileir,
    )
    monkeypatch.setattr(checks, "_package_version", lambda package: "1.4.0" if package == "cuda-tile" else None)
    monkeypatch.setattr(checks, "find_tileiras", lambda: _MOCK_TILEIRAS_PATH)
    monkeypatch.setattr(checks, "_tileiras_version", lambda _: "tileiras 13.4")

    with pytest.raises(checks.TileIRDependencyError, match="requires cuda-tile 1.5"):
        checks.check_tileir_available()


def test_tileir_dependency_check_rejects_incompatible_tileiras(monkeypatch):
    fake_tileir = types.SimpleNamespace(**{symbol: object() for symbol in checks.CUDA_TILE_IR_REQUIRED_SYMBOLS})
    fake_runtime = types.SimpleNamespace(TileDispatcher=object(), launch=object())

    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(
        checks.importlib,
        "import_module",
        lambda module_name: fake_runtime if module_name == checks.CUDA_TILE_RUNTIME_MODULE else fake_tileir,
    )
    monkeypatch.setattr(checks, "_validated_cuda_tile_runtime_version", lambda: "1.5.0")
    monkeypatch.setattr(checks, "find_tileiras", lambda: _MOCK_TILEIRAS_PATH)
    monkeypatch.setattr(checks, "_tileiras_version", lambda _: "tileiras 13.3.36")

    with pytest.raises(checks.TileIRDependencyError, match="requires tileiras 13.4"):
        checks.check_tileir_available()


def test_tileir_dependency_check_rejects_unknown_tileiras_version(monkeypatch):
    fake_tileir = types.SimpleNamespace(**{symbol: object() for symbol in checks.CUDA_TILE_IR_REQUIRED_SYMBOLS})
    fake_runtime = types.SimpleNamespace(TileDispatcher=object(), launch=object())

    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(
        checks.importlib,
        "import_module",
        lambda module_name: fake_runtime if module_name == checks.CUDA_TILE_RUNTIME_MODULE else fake_tileir,
    )
    monkeypatch.setattr(checks, "_validated_cuda_tile_runtime_version", lambda: "1.5.0")
    monkeypatch.setattr(checks, "find_tileiras", lambda: _MOCK_TILEIRAS_PATH)
    monkeypatch.setattr(checks, "_tileiras_version", lambda _: "tileiras: NVIDIA CUDA Tile IR optimizing assembler")

    with pytest.raises(checks.TileIRDependencyError, match="could not determine.*version"):
        checks.check_tileir_available()


def test_tileir_dependency_check_accepts_explicit_tileiras_version_override(monkeypatch):
    fake_tileir = types.SimpleNamespace(**{symbol: object() for symbol in checks.CUDA_TILE_IR_REQUIRED_SYMBOLS})
    fake_runtime = types.SimpleNamespace(TileDispatcher=object(), launch=object())

    monkeypatch.setattr(checks.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(
        checks.importlib,
        "import_module",
        lambda module_name: fake_runtime if module_name == checks.CUDA_TILE_RUNTIME_MODULE else fake_tileir,
    )
    monkeypatch.setattr(checks, "_validated_cuda_tile_runtime_version", lambda: "1.5.0")
    monkeypatch.setattr(checks, "find_tileiras", lambda: _MOCK_TILEIRAS_PATH)
    monkeypatch.setattr(checks, "_tileiras_version", lambda _: "version unavailable")
    monkeypatch.setenv("TILELANG_TILEIRAS_VERSION", "13.4")

    toolchain = checks.check_tileir_available()

    assert toolchain.tileiras_version == "13.4"


def test_tileir_find_tileiras_prefers_explicit_override(monkeypatch, tmp_path):
    tileiras = tmp_path / "cuda" / "bin" / "tileiras"
    _make_executable(tileiras)
    monkeypatch.setenv("TILELANG_TILEIRAS", str(tileiras))

    assert checks.find_tileiras() == tileiras


def test_tileir_pip_tileiras_rejects_mismatched_nvjitlink(monkeypatch):
    versions = {
        "nvidia-cuda-tileiras": "13.4.59",
        "nvidia-cuda-nvcc": "13.4.59",
        "nvidia-nvvm": "13.4.59",
        "nvidia-nvjitlink": "13.2.78",
    }
    monkeypatch.setattr(checks, "_package_version", lambda package: versions[package])

    with pytest.raises(checks.TileIRDependencyError, match="nvidia-nvjitlink==13.2.78"):
        checks._find_pip_tileiras()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("num_split", [1, 2])
def test_tileir_rejects_deepseek_mla_ws_without_fallback(monkeypatch, num_split):
    _skip_if_tileir_toolchain_unavailable()
    _enable_tileir_runtime(monkeypatch)

    example = _load_mla_ws_example()
    batch, heads, kv_heads = 1, 64, 1
    kv_ctx = 128 if num_split == 1 else 256
    dim, pe_dim = 512, 64
    block_n, block_h = 64, 64
    softmax_scale = (dim + pe_dim) ** -0.5

    with pytest.raises(TileIRLoweringNotImplementedError, match="ptx_arrive_barrier|ptx_cp_async|wgmma_gemm"):
        example.flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_n, block_h, num_split, softmax_scale)
