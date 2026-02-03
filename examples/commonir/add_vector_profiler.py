# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T
from functools import partial

import torch
import torch_npu
import time
import numpy as np
from typing import Callable, Optional, Union, List


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

def ref_program(v1, v2):
    return v1 + v2

def test_vec_add():
    func = vec_add(seq_len, seq_len // 4)
    compiled_kernel = tilelang.compile(func, out_idx=[2])

    profiler = compiled_kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    latency = profiler.do_bench(ref_program, warmup=500)
    latency2 = profiler.do_bench(warmup=500)
    print(f"⏱ latency base is {latency}")
    print(f"⏱ latency is {latency2}")
 
if __name__ == "__main__":
    test_vec_add()