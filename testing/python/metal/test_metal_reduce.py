"""Metal lowering of portable fragment reductions."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tilelang.jit(target="metal", execution_backend="torch")
def reduce_kernel(rows, cols, threads, kind, axis=1, clear=True, dtype="float32", strided=False, batch=1):
    output_size = rows if axis == 1 else cols

    @T.prim_func
    def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((output_size,), dtype)):
        with T.Kernel(1, threads=threads):
            a = T.alloc_fragment((rows, cols), dtype)
            b = T.alloc_fragment((output_size,), dtype)
            if strided:
                T.annotate_layout(
                    {
                        a: T.Fragment((rows, cols), forward_fn=lambda i, j: (j * rows + i, 0)),
                        b: T.Fragment(
                            (rows,), forward_thread_fn=lambda i, rep: i + rep * rows, forward_index_fn=lambda i: 0, replicate=cols
                        ),
                    }
                )
            T.copy(A, a)
            if not clear:
                T.fill(b, 2)
            T.reduce(a, b, kind, dim=axis, clear=clear, batch=batch)
            T.copy(b, B)

    return main


@pytest.mark.parametrize("threads, rows, cols", [(32, 4, 8), (128, 1, 128)])
def test_reduce_codegen(threads, rows, cols):
    program = reduce_kernel.get_tir(rows, cols, threads, "sum")
    with tvm.target.Target("metal"):
        source = tilelang.lower(program, target="metal").kernel_source
    assert "simd_shuffle_xor" in source
    assert "__syncthreads" not in source


@tilelang.testing.requires_metal
@pytest.mark.parametrize("kind", ["sum", "max", "min", "abssum", "absmax"])
@pytest.mark.parametrize(
    "rows,cols,threads,axis,strided",
    [
        (4, 8, 32, 1, False),
        (4, 8, 32, 1, True),
        (4, 32, 128, 1, True),
        (8, 256, 128, 1, False),
        (128, 4, 128, 0, False),
    ],
)
@pytest.mark.parametrize("clear", [True, False])
def test_reduce_execution(kind, rows, cols, threads, axis, strided, clear):
    a = torch.randn(rows, cols)
    values = a.abs() if kind.startswith("abs") else a
    if kind in ("sum", "abssum"):
        expected = values.sum(axis)
        if not clear:
            expected += 2
    elif kind in ("max", "absmax"):
        expected = values.amax(axis)
        if not clear:
            expected = expected.clamp_min(2)
    else:
        expected = values.amin(axis)
        if not clear:
            expected = expected.clamp_max(2)
    b = torch.empty_like(expected, device="mps")
    reduce_kernel(rows, cols, threads, kind, axis, clear, strided=strided)(a.to("mps"), b)
    torch.testing.assert_close(b.cpu(), expected, atol=2e-5, rtol=2e-5)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", ["float16", "int32"])
def test_reduce_dtype(dtype):
    a = torch.arange(128).reshape(4, 32).remainder(7).to(getattr(torch, dtype))
    b = torch.empty(4, dtype=a.dtype, device="mps")
    reduce_kernel(4, 32, 128, "sum", dtype=dtype)(a.to("mps"), b)
    torch.testing.assert_close(b.cpu(), a.sum(1).to(a.dtype), atol=0, rtol=0)


@tilelang.testing.requires_metal
def test_reduce_batch():
    a = torch.randn(8, 128)
    b = torch.empty(8, device="mps")
    reduce_kernel(8, 128, 128, "sum", batch=2)(a.to("mps"), b)
    torch.testing.assert_close(b.cpu(), a.sum(1), atol=2e-5, rtol=2e-5)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("kind,op", [("bitand", torch.bitwise_and), ("bitor", torch.bitwise_or), ("bitxor", torch.bitwise_xor)])
@pytest.mark.parametrize("clear", [True, False])
def test_reduce_bitwise(kind, op, clear):
    a = torch.randint(-128, 128, (4, 128), dtype=torch.int32)
    expected = a[:, 0]
    for column in a[:, 1:].unbind(1):
        expected = op(expected, column)
    if not clear:
        expected = op(expected, 2)
    b = torch.empty(4, dtype=torch.int32, device="mps")
    reduce_kernel(4, 128, 128, kind, clear=clear, dtype="int32")(a.to("mps"), b)
    torch.testing.assert_close(b.cpu(), expected, atol=0, rtol=0)


def test_partial_cross_simd_reduce_rejected():
    program = reduce_kernel.get_tir(1, 64, 128, "sum", strided=True)
    with tvm.target.Target("metal"), pytest.raises(tvm.error.InternalError, match="require all block threads"):
        tilelang.lower(program, target="metal")
