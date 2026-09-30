"""Regressions for dtype-changing shared-memory ``T.view`` aliases."""

import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.tileir import checks

from tileir_test_utils import _tileir_source_for_test


def _dtype_changing_shared_view_kernel():
    @tilelang.jit
    def kernel(source, output):
        source: T.Tensor((32,), T.float32)
        output: T.Tensor((32,), T.float32)

        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((64,), T.bfloat16)
            retyped = T.view(backing, shape=(32,), dtype=T.float32)
            for lane in T.Parallel(32):
                retyped[lane] = source[lane]
            for lane in T.Parallel(32):
                output[lane] = retyped[lane]

    return kernel


def _dtype_changing_shared_view_copy_kernel():
    @tilelang.jit
    def kernel(source, output):
        source: T.Tensor((32,), T.float32)
        output: T.Tensor((32,), T.float32)

        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((64,), T.bfloat16)
            retyped = T.view(backing, shape=(32,), dtype=T.float32)
            T.copy(source, retyped)
            T.copy(retyped, output)

    return kernel


@pytest.mark.skipif(
    not checks.has_cuda_tile_ir_bindings(),
    reason="CUDA Tile IR MLIR bindings unavailable",
)
@tilelang.testing.requires_cuda
def test_dtype_changing_shared_view_lowers_as_structured_reinterpret():
    source = _tileir_source_for_test(_dtype_changing_shared_view_kernel(), None, None)

    assert " = pack " in source
    assert " = unpack " in source
    assert "tilelang_native_simt" not in source
    assert "preview$call" not in source


@pytest.mark.skipif(
    not checks.has_cuda_tile_ir_bindings(),
    reason="CUDA Tile IR MLIR bindings unavailable",
)
@tilelang.testing.requires_cuda
def test_dtype_changing_shared_view_copy_lowers_as_structured_reinterpret():
    source = _tileir_source_for_test(_dtype_changing_shared_view_copy_kernel(), None, None)

    assert " = pack " in source
    assert " = unpack " in source
    assert "tilelang_native_simt" not in source
    assert "preview$call" not in source


@pytest.mark.skipif(
    not checks.has_cuda_tile_ir_bindings(),
    reason="CUDA Tile IR MLIR bindings unavailable",
)
@tilelang.testing.requires_cuda
def test_dtype_changing_shared_view_numerical_without_native_simt():
    import torch

    try:
        checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")

    compiled = tilelang.compile(
        _dtype_changing_shared_view_kernel().get_tir(None, None),
        execution_backend="tileir",
    )
    source = torch.linspace(-3.5, 7.25, 32, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)

    compiled(source, output)

    torch.testing.assert_close(output, source, rtol=0, atol=0)


@pytest.mark.skipif(
    not checks.has_cuda_tile_ir_bindings(),
    reason="CUDA Tile IR MLIR bindings unavailable",
)
@tilelang.testing.requires_cuda
def test_dtype_changing_shared_view_copy_numerical_without_native_simt():
    import torch

    try:
        checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")

    compiled = tilelang.compile(
        _dtype_changing_shared_view_copy_kernel().get_tir(None, None),
        execution_backend="tileir",
    )
    source = torch.linspace(-3.5, 7.25, 32, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)

    compiled(source, output)

    torch.testing.assert_close(output, source, rtol=0, atol=0)


def _tail_padded_view_kernel(size, write_alias, use_copy):
    source_dtype = "float32" if write_alias else "bfloat16"
    output_dtype = "bfloat16" if write_alias else "float32"
    source_size = size if write_alias else 2 * size
    output_size = 2 * size if write_alias else size

    @tilelang.jit
    def kernel(source, output):
        source: T.Tensor((source_size,), source_dtype)
        output: T.Tensor((output_size,), output_dtype)
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((2 * size,), T.bfloat16)
            alias = T.view(backing, shape=(size,), dtype=T.float32)
            if write_alias:
                if use_copy:
                    T.copy(source, alias)
                else:
                    for lane in T.Parallel(size):
                        alias[lane] = source[lane]
                T.copy(backing, output)
            else:
                T.copy(source, backing)
                if use_copy:
                    T.copy(alias, output)
                else:
                    for lane in T.Parallel(size):
                        output[lane] = alias[lane]

    return kernel


@pytest.mark.skipif(not checks.has_cuda_tile_ir_bindings(), reason="CUDA Tile IR MLIR bindings unavailable")
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("size", [24, 32])
@pytest.mark.parametrize("write_alias", [False, True])
@pytest.mark.parametrize("use_copy", [False, True])
def test_dtype_view_cross_storage_bits(size, write_alias, use_copy):
    """Observe both sides of the view, including its final valid element."""
    import torch

    try:
        checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")
    compiled = tilelang.compile(
        _tail_padded_view_kernel(size, write_alias, use_copy).get_tir(None, None),
        execution_backend="tileir",
    )
    values = torch.linspace(-3.5, 7.25, size, dtype=torch.float32, device="cuda")
    source = values if write_alias else values.view(torch.bfloat16)
    expected = values.view(torch.bfloat16) if write_alias else values
    output = torch.empty_like(expected)
    compiled(source, output)
    torch.testing.assert_close(output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@pytest.mark.parametrize(
    "base_shape,alias_shape,error",
    [
        ((48,), (24,), None),
        ((64,), (32,), None),
        ((3, 16), (3, 8), None),
        ((2, 3), (3,), "unsupported interior padding"),
        ((6,), (1, 3), None),
        ((48,), (32,), "changes logical storage capacity"),
    ],
)
def test_dtype_view_logical_capacity_and_padding(base_shape, alias_shape, error):
    from tilelang.tileir.errors import _UnsupportedTileIRNode
    from tilelang.tileir.lowering.sem_to_ir._base import _sem_buffer_to_tile_type, _validate_dtype_view_layout
    from tilelang.tileir.semantic import SemanticBuffer

    backing = SemanticBuffer(name="backing", shape=base_shape, dtype="bfloat16", scope="shared")
    alias = SemanticBuffer(name="alias", shape=alias_shape, dtype="float32", scope="shared")
    args = (backing, _sem_buffer_to_tile_type(backing), alias, _sem_buffer_to_tile_type(alias))
    if error:
        with pytest.raises(_UnsupportedTileIRNode, match=error):
            _validate_dtype_view_layout(*args)
    else:
        _validate_dtype_view_layout(*args)


@pytest.mark.skipif(not checks.has_cuda_tile_ir_bindings(), reason="CUDA Tile IR MLIR bindings unavailable")
@tilelang.testing.requires_cuda
def test_dtype_view_interior_padding_rejected_before_emission():
    from tilelang.tileir.errors import TileIRLoweringError

    @tilelang.jit
    def kernel(source, output):
        source: T.Tensor((3,), T.float32)
        output: T.Tensor((2, 3), T.bfloat16)
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((2, 3), T.bfloat16)
            alias = T.view(backing, shape=(3,), dtype=T.float32)
            T.copy(source, alias)
            T.copy(backing, output)

    with pytest.raises(TileIRLoweringError, match="unsupported interior padding"):
        _tileir_source_for_test(kernel, None, None)


@pytest.mark.skipif(not checks.has_cuda_tile_ir_bindings(), reason="CUDA Tile IR MLIR bindings unavailable")
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("write_alias", [False, True])
def test_dtype_view_outer_padding_cross_storage_bits(write_alias):
    import torch

    try:
        checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")
    source_shape = (3, 8) if write_alias else (3, 16)
    output_shape = (3, 16) if write_alias else (3, 8)
    source_dtype = "float32" if write_alias else "bfloat16"
    output_dtype = "bfloat16" if write_alias else "float32"

    @tilelang.jit
    def kernel(source, output):
        source: T.Tensor(source_shape, source_dtype)
        output: T.Tensor(output_shape, output_dtype)
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((3, 16), T.bfloat16)
            alias = T.view(backing, shape=(3, 8), dtype=T.float32)
            if write_alias:
                T.copy(source, alias)
                T.copy(backing, output)
            else:
                T.copy(source, backing)
                T.copy(alias, output)

    compiled = tilelang.compile(kernel.get_tir(None, None), execution_backend="tileir")
    values = torch.linspace(-3.5, 7.25, 24, dtype=torch.float32, device="cuda").reshape(3, 8)
    source = values if write_alias else values.view(torch.bfloat16)
    expected = values.view(torch.bfloat16) if write_alias else values
    output = torch.empty_like(expected)
    compiled(source, output)
    torch.testing.assert_close(output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
