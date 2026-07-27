"""Reproduce: strides check failure (non-contiguous A via transpose)."""

import torch
from common import build_matmul_kernel


def main():
    M = N = K = 128
    fn = build_matmul_kernel(M, N, K, target="cuda")

    a = torch.empty((M, K), device="cuda", dtype=torch.float16)
    a_nc = a.t()  # non-contiguous after transpose
    b = torch.empty((K, N), device="cuda", dtype=torch.float16)

    fn(a_nc, b)


if __name__ == "__main__":
    main()
