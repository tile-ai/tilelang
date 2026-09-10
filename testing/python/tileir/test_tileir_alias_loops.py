"""Views must observe the same loop-carried storage as their backing buffer."""

import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.tileir.errors import _UnsupportedTileIRNode

from tileir_test_utils import _skip_if_tileir_toolchain_unavailable, _tileir_source_for_test, skip_no_cuda_tile


def _alias_loop(backing_dtype, reverse, iterations):
    backing_n = 64 if backing_dtype == "bfloat16" else 32
    source_dtype, source_n = ("float32", 32) if reverse else (backing_dtype, backing_n)
    output_dtype, output_n = (backing_dtype, backing_n) if reverse else ("float32", 32)

    @tilelang.jit
    def kernel(source, output):
        source: T.Tensor((source_n,), source_dtype)
        output: T.Tensor((output_n,), output_dtype)
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((backing_n,), backing_dtype)
            alias = T.view(backing, shape=(32,), dtype=T.float32)
            if reverse:
                T.copy(source, alias)
            else:
                T.copy(source, backing)
            for _step in T.serial(iterations):
                for lane in T.Parallel(32):
                    alias[lane] = alias[lane] + 1
            if reverse:
                T.copy(backing, output)
            else:
                T.copy(alias, output)

    return kernel


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backing_dtype", ["float32", "int32", "bfloat16"])
@pytest.mark.parametrize("reverse", [False, True], ids=["backing-to-view", "view-to-backing"])
@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_alias_loop_updates(backing_dtype, reverse, iterations):
    import torch

    _skip_if_tileir_toolchain_unavailable()
    kernel = tilelang.compile(_alias_loop(backing_dtype, reverse, iterations).get_tir(None, None), execution_backend="tileir")
    dtype = getattr(torch, backing_dtype)
    x = torch.arange(32, device="cuda", dtype=torch.float32) / 8
    source = x if reverse else x.view(dtype)
    expected = (x + iterations).view(dtype) if reverse else x + iterations
    output = torch.empty_like(expected)
    kernel(source, output)
    # Compare bits: the view changes storage interpretation, not numerical dtype.
    torch.testing.assert_close(output.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_conditional_partial_alias_write_preserves_previous_iteration():
    import torch

    _skip_if_tileir_toolchain_unavailable()

    @tilelang.jit
    def kernel(source, output):
        source: T.Tensor((32,), T.int32)
        output: T.Tensor((2, 32), T.int32)
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((32,), T.int32)
            alias = T.view(backing, shape=(32,), dtype=T.float32)
            T.copy(source, backing)
            for step in T.serial(2):
                if step == 0:
                    alias[0] = 7.0
                T.copy(backing, output[step, :])

    compiled = tilelang.compile(kernel.get_tir(None, None), execution_backend="tileir")
    source = torch.arange(32, device="cuda", dtype=torch.float32) / 8
    expected = source.clone()
    expected[0] = 7.0
    output = torch.empty((2, 32), device="cuda", dtype=torch.int32)
    compiled(source.view(torch.int32), output)
    torch.testing.assert_close(output, expected.view(torch.int32).expand(2, 32), rtol=0, atol=0)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_alias_accumulator_cannot_bypass_break_loop_rejection():
    _skip_if_tileir_toolchain_unavailable()

    @tilelang.jit
    def kernel(output):
        output: T.Tensor((32,), T.float32)
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((32,), T.int32)
            alias = T.view(backing, shape=(32,), dtype=T.float32)
            T.fill(alias, 0)
            for step in T.serial(4):
                if step == 2:
                    T.loop_break()
                for lane in T.Parallel(32):
                    alias[lane] = alias[lane] + 1
            T.copy(alias, output)

    with pytest.raises(_UnsupportedTileIRNode, match="loop-carried"):
        _tileir_source_for_test(kernel, None)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_write_first_alias_scratch_in_break_loop():
    import torch

    _skip_if_tileir_toolchain_unavailable()

    @tilelang.jit
    def kernel(source, output):
        source: T.Tensor((32,), T.float32)
        output: T.Tensor((2, 32), T.float32)
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((32,), T.float32)
            alias = T.view(backing, shape=(32,), dtype=T.float32)
            T.fill(backing, 0)
            for step in T.serial(4):
                if step == 2:
                    T.loop_break()
                T.copy(source, backing)
                T.copy(alias, output[step, :])

    compiled = tilelang.compile(kernel.get_tir(None, None), execution_backend="tileir")
    source = torch.arange(32, device="cuda", dtype=torch.float32)
    output = torch.empty((2, 32), device="cuda", dtype=torch.float32)
    compiled(source, output)
    torch.testing.assert_close(output, source.expand(2, 32), rtol=0, atol=0)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("alias_dtype", ["float32", "int32"])
@pytest.mark.parametrize("use_tma", [False, True], ids=["copy", "tma-copy"])
def test_copy_to_alias_in_loop_preserves_load_token(alias_dtype, use_tma):
    """Post-loop backing reads must not depend on a token scoped inside the loop."""
    import torch

    _skip_if_tileir_toolchain_unavailable()

    @T.prim_func
    def main(Src: T.Tensor((2, 32), alias_dtype), Dst: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=128):
            backing = T.alloc_shared((32,), "float32")
            alias = T.view(backing, shape=(32,), dtype=alias_dtype)
            bar = T.alloc_barrier([128])
            T.fill(backing, -99)
            for step in T.serial(2):
                if use_tma:
                    T.tma_copy(Src[step, :], alias, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, step % 2)
                else:
                    T.copy(Src[step, :], alias)
            T.copy(backing, Dst)

    kernel = tilelang.compile(main, execution_backend="tileir")
    source = torch.arange(64, device="cuda", dtype=torch.float32).reshape(2, 32)
    output = torch.empty(32, device="cuda", dtype=torch.float32)
    kernel(source.view(getattr(torch, alias_dtype)), output)
    torch.testing.assert_close(output, source[1], rtol=0, atol=0)
