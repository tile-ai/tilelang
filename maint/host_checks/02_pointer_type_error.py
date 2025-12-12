"""Reproduce: Pointer-type argument expected but scalar provided.

We pass an integer for A; wrapper forwards it to the host where a pointer is expected.
Expected: error like "Expect buffer A_handle to be pointer or tensor" (exact name depends on kernel param).
"""

import torch
from common import build_matmul_kernel


def main():
    M = N = K = 256
    fn = build_matmul_kernel(M, N, K, target="cuda")

    # Wrong type for A (int instead of tensor)
    a = 1
    b = torch.empty((K, N), device="cuda", dtype=torch.float16)

    fn(a, b)


if __name__ == "__main__":
    main()
