"""
Minimal reproducer for the missing vec_type arithmetic operators in common.h.

Without the fix, compiling a CPU kernel with `c` target fails:
  error: no match for 'operator-' (operand types are 'float4' and 'float4')

Requires: tilelang, torch
"""

import tilelang
import tilelang.language as T
import torch


@T.prim_func
def vec_add(
    A: T.Tensor((256,), "float32"),
    B: T.Tensor((256,), "float32"),
    C: T.Tensor((256,), "float32"),
):
    for i in T.Parallel(256):
        C[i] = A[i] + B[i]


def main():
    f = tilelang.compile(vec_add, target="c")
    A = torch.randn(256)
    B = torch.randn(256)
    C = torch.zeros(256)
    f(A, B, C)
    ref = A + B
    assert (C - ref).abs().max().item() < 1e-5
    print("OK: vec_add compiled and ran successfully")


if __name__ == "__main__":
    main()
