"""Reproduce: shape constant/symbol mismatch on A."""

import torch
from common import build_matmul_kernel


def main():
    M = N = K = 128
    fn = build_matmul_kernel(M, N, K, target="cuda")

    # A's second dimension is wrong (K+1 instead of K)
    a = torch.empty((M, K + 1), device="cuda", dtype=torch.float16)
    b = torch.empty((K, N), device="cuda", dtype=torch.float16)

    fn(a, b)


if __name__ == "__main__":
    main()
