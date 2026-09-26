"""Composed participant indices retain their tile shape and per-element values."""

import pytest
from tvm import tirx

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kind", ["direct", "cast", "select", "call_select", "bitwise", "add", "condition_select", "condition_call"])
def test_composed_gather_index_numerical(kind):
    try:
        check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(str(exc))
    import torch

    transforms = {
        "direct": lambda x: x,
        "cast": lambda x: T.cast(T.cast(x, "int16"), "int32"),
        "select": lambda x: tirx.Select(x >= 0, x, x),
        "call_select": lambda x: T.if_then_else(x >= 0, x, x),
        "bitwise": lambda x: T.bitwise_and(x, 127),
        "add": lambda x: x + 1,
        "condition_select": lambda x: tirx.Select(x >= 64, 1, 0),
        "condition_call": lambda x: T.if_then_else(x >= 64, 1, 0),
    }
    transform = transforms[kind]

    @tilelang.jit
    def kernel(source, indices, output):
        source: T.Tensor((128,), T.int32)
        indices: T.Tensor((128,), T.int32)
        output: T.Tensor((128,), T.int32)
        with T.Kernel(1, threads=128):
            for lane in T.Parallel(128):
                selected = indices[lane]
                output[lane] = source[transform(selected)]

    compiled = tilelang.compile(kernel.get_tir(None, None, None), execution_backend="tileir", target="cuda")
    assert "load_ptr_tko" in compiled.get_kernel_source()
    source = torch.arange(128, dtype=torch.int32, device="cuda") * 7 + 3
    indices = (torch.arange(128, dtype=torch.int32, device="cuda") * 37) % 127
    expected_indices = indices + 1 if kind == "add" else (indices >= 64).to(torch.int32) if kind.startswith("condition_") else indices
    output = torch.empty_like(indices)
    compiled(source, indices, output)
    torch.testing.assert_close(output, source[expected_indices.long()], rtol=0, atol=0)
