"""Reproduce: Argument count mismatch.

Note: The adapter-level wrapper expects only inputs (A, B) because C is marked as output.
Calling with the wrong number of inputs raises a ValueError before host entry.
"""

import torch
from common import build_matmul_kernel


def main():
    M = N = K = 256
    fn = build_matmul_kernel(M, N, K, target="cuda")

    a = torch.empty((M, K), device="cuda", dtype=torch.float16)
    # Missing b
    # Expected: ValueError with message about expected vs. actual inputs
    fn(a)


if __name__ == "__main__":
    main()
