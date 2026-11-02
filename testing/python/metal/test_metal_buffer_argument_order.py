"""The torch adapter must bind arguments to the buffers the kernel actually declares.

Metal device codegen is free to emit ``[[buffer(N)]]`` parameters in an order
other than the ``T.prim_func`` signature's, and to drop parameters the kernel
body never reads. Forwarding a call's arguments positionally then binds tensors
to the wrong slots, which fails silently: the kernel writes through an input and
reads through an output, and the call still returns normally.
"""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch

M, W = 8, 128


@T.prim_func
def window_from_pivots(
    pivots: T.Tensor((M, 4), "int32"),
    output: T.Tensor((M, W), "int32"),
):
    with T.Kernel(T.ceildiv(M, 4), T.ceildiv(W, 256), threads=128) as (bx, by):
        for i, j in T.Parallel(4, 256):
            row = bx * 4 + i
            col = by * 256 + j
            if row < M and col < W:
                output[row, col] = T.if_then_else(col < pivots[row, 1], pivots[row, 0] + col, -1)


@T.prim_func
def ignores_its_first_argument(
    unused: T.Tensor((M, 4), "int32"),
    output: T.Tensor((M, W), "int32"),
):
    with T.Kernel(T.ceildiv(M, 4), T.ceildiv(W, 256), threads=128) as (bx, by):
        for i, j in T.Parallel(4, 256):
            row = bx * 4 + i
            col = by * 256 + j
            if row < M and col < W:
                output[row, col] = col


@tilelang.testing.requires_metal
def test_arguments_follow_the_emitted_buffer_order():
    kernel = tilelang.compile(window_from_pivots, out_idx=[], target="metal", execution_backend="torch")

    pivots = torch.zeros(M, 4, dtype=torch.int32, device="mps")
    pivots[:, 0] = 100
    pivots[:, 1] = 5
    pivots_before = pivots.clone()
    output = torch.full((M, W), -99, dtype=torch.int32, device="mps")

    kernel(pivots, output)
    torch.mps.synchronize()

    expected = torch.full((M, W), -1, dtype=torch.int32)
    expected[:, :5] = 100 + torch.arange(5, dtype=torch.int32)
    torch.testing.assert_close(output.cpu(), expected)
    # Checked explicitly: the failure mode is writing through the wrong buffer,
    # which corrupts the input rather than raising.
    torch.testing.assert_close(pivots.cpu(), pivots_before.cpu())


@tilelang.testing.requires_metal
def test_argument_dropped_from_the_kernel_signature_does_not_shift_the_rest():
    kernel = tilelang.compile(ignores_its_first_argument, out_idx=[], target="metal", execution_backend="torch")

    unused = torch.full((M, 4), 777, dtype=torch.int32, device="mps")
    unused_before = unused.clone()
    output = torch.full((M, W), -99, dtype=torch.int32, device="mps")

    kernel(unused, output)
    torch.mps.synchronize()

    torch.testing.assert_close(output.cpu(), torch.arange(W, dtype=torch.int32).repeat(M, 1))
    torch.testing.assert_close(unused.cpu(), unused_before.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
