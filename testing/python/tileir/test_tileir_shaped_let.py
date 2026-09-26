"""Regressions for participant-shaped implicit let bindings.

These tests deliberately use only structured CUDA Tile IR operations.  The
same index-classification path is used by TopK's shared histogram atomic, but
a GLOBAL gather isolates the frontend bug without requiring Native SIMT.
"""

from __future__ import annotations

import pytest
from tvm import tirx

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.tileir import checks

from tileir_test_utils import _tileir_source_for_test


def _shaped_let_gather_kernel(index_transform=lambda selected: selected):
    @tilelang.jit
    def kernel(source, indices, output):
        source: T.Tensor((128,), T.int32)
        indices: T.Tensor((128,), T.int32)
        output: T.Tensor((128,), T.int32)

        with T.Kernel(1, threads=128):
            for lane in T.Parallel(128):
                selected = indices[lane]
                output[lane] = source[index_transform(selected)]

    return kernel


@pytest.mark.skipif(
    not checks.has_cuda_tile_ir_bindings(),
    reason="CUDA Tile IR MLIR bindings unavailable",
)
@tilelang.testing.requires_cuda
def test_shaped_let_global_gather_stays_structured_without_native_simt():
    source = _tileir_source_for_test(_shaped_let_gather_kernel(), None, None, None)

    assert "load_ptr_tko" in source
    assert "tilelang_native_simt" not in source
    assert "preview$call" not in source


@pytest.mark.parametrize(
    "index_transform",
    [
        lambda selected: T.cast(T.cast(selected, "int16"), "int32"),
        lambda selected: tirx.Select(selected >= 0, selected, selected),
    ],
    ids=["cast", "select"],
)
@pytest.mark.skipif(
    not checks.has_cuda_tile_ir_bindings(),
    reason="CUDA Tile IR MLIR bindings unavailable",
)
@tilelang.testing.requires_cuda
def test_shaped_let_wrapped_global_gather_stays_structured_without_native_simt(index_transform):
    source = _tileir_source_for_test(_shaped_let_gather_kernel(index_transform), None, None, None)

    assert "load_ptr_tko" in source
    assert "tilelang_native_simt" not in source
    assert "preview$call" not in source


@pytest.mark.skipif(
    not checks.has_cuda_tile_ir_bindings(),
    reason="CUDA Tile IR MLIR bindings unavailable",
)
@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "index_transform",
    [
        lambda selected: T.cast(T.cast(selected, "int16"), "int32"),
        lambda selected: tirx.Select(selected >= 0, selected, selected),
    ],
    ids=["cast", "select"],
)
def test_shaped_let_wrapped_global_gather_numerical_without_native_simt(index_transform):
    import torch

    try:
        checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")

    compiled = tilelang.compile(
        _shaped_let_gather_kernel(index_transform).get_tir(None, None, None),
        execution_backend="tileir",
    )
    source = torch.arange(128, dtype=torch.int32, device="cuda") * 7 + 3
    indices = (torch.arange(128, dtype=torch.int32, device="cuda") * 37 + 11) % 128
    output = torch.empty_like(source)

    compiled(source, indices, output)

    torch.testing.assert_close(output, source[indices.long()], rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
