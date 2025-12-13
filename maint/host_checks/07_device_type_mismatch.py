"""Reproduce: device_type mismatch by passing CPU tensors to a CUDA kernel."""

import torch
from common import build_matmul_kernel


def main():
    M = N = K = 64
    fn = build_matmul_kernel(M, N, K, target="cuda")

    a = torch.empty((M, K), device="cpu", dtype=torch.float16)
    b = torch.empty((K, N), device="cpu", dtype=torch.float16)

    fn(a, b)


if __name__ == "__main__":
    main()
