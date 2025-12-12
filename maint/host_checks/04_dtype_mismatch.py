"""Reproduce: dtype mismatch for A (float32 vs expected float16)."""

import torch
from common import build_matmul_kernel


def main():
    M = N = K = 128
    fn = build_matmul_kernel(M, N, K, target="cuda")
    print(fn.get_host_source())

    a = torch.empty((M, K), device="cuda", dtype=torch.float32)  # should be float16
    b = torch.empty((K, N), device="cuda", dtype=torch.float16)

    fn(a, b)


if __name__ == "__main__":
    main()
