"""Reproduce: device_id mismatch (requires >=2 CUDA devices)."""

import torch
from common import build_matmul_kernel


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() < 2:
        print("[SKIP] Need at least 2 CUDA devices to reproduce device_id mismatch.")
        return

    M = N = K = 64
    fn = build_matmul_kernel(M, N, K, target="cuda")

    a = torch.empty((M, K), device="cuda:0", dtype=torch.float16)
    b = torch.empty((K, N), device="cuda:1", dtype=torch.float16)
    # Output device is derived by the adapter; mismatch occurs in host checks

    fn(a, b)


if __name__ == "__main__":
    main()
