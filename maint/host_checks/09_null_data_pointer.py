"""Reproduce: NULL data pointer (advanced).

Passing None for a tensor argument will be forwarded through the adapter. Depending on
FFI handling, this commonly triggers a pointer-type assertion (e.g., "Expect buffer <name> to be pointer or tensor")
or a host-side non-NULL pointer check.

Note: Constructing a true DLTensor with NULL data in PyTorch is not typical; this script
demonstrates passing None, which still reproduces the intended class of failure.
"""

import torch
from common import build_matmul_kernel


def main():
    M = N = K = 64
    fn = build_matmul_kernel(M, N, K, target="cuda")

    a = None  # attempt to pass a null-like pointer
    b = torch.empty((K, N), device="cuda", dtype=torch.float16)

    fn(a, b)


if __name__ == "__main__":
    main()
