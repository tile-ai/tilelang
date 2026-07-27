"""Reproduce: ndim (rank) mismatch for A."""

import torch
from common import build_matmul_kernel


def main():
    M = N = K = 128
    fn = build_matmul_kernel(M, N, K, target="cuda")

    # A has rank 3 instead of 2
    a = torch.empty((M, K, 1), device="cuda", dtype=torch.float16)
    b = torch.empty((K, N), device="cuda", dtype=torch.float16)

    fn(a, b)


if __name__ == "__main__":
    main()
