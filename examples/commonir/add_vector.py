# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch

dtype = "float32"
seq_len = 1024


def vec_add(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((N), dtype),
            B: T.Tensor((N), dtype),
            C: T.Tensor((N), dtype),
    ):
        with T.Kernel(n_num, 1) as (by, bx):
            start_y1 = by * block_N
            start_y = start_y1 + bx
            for (local_y) in T.Parallel(block_N):
                y = start_y + local_y
                C[y] = A[y] + B[y]

    return main


def test_vec_add():
    func = vec_add(seq_len, seq_len // 4)
    compiled_kernel = tilelang.compile(func)

    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()

    y_ref = v1 + v2
    compiled_kernel(v1, v2, v3)

    # print(y_ref)
    # print(v3)

    print(f'The maximum difference between torch and Tilellang is '
          f'{torch.max(torch.abs(y_ref - v3))}')

    torch.testing.assert_close(v3, y_ref, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test_vec_add()