"""Ascend SIMT Philox RNG example: fill buffers with random values.

Exercises T.rng_init / T.rng_rand / T.rng_rand_float (uniform + normal) inside a
T.SimtVF(...) block. Each output element uses its own subsequence (seq = element
index) so values are independent and reproducible across runs.
"""

import tilelang
import tilelang.ascend.language as T

SEED = 42


def rng_fill(N, threads=256):
    NUM_BLOCKS = 1
    TILE = N  # single tile per block

    @T.prim_func
    def main(
        U: T.Tensor((N,), "float32"),  # uniform [0, 1)
        Nr: T.Tensor((N,), "float32"),  # standard normal
        I: T.Tensor((N,), "uint32"),  # raw uint32
    ):
        with T.Kernel(NUM_BLOCKS) as _bx:
            us = T.alloc_shared((TILE,), "float32")
            ns = T.alloc_shared((TILE,), "float32")
            iv = T.alloc_shared((TILE,), "uint32")
            with T.SimtVF(threads=threads):
                T.rng_init(SEED)
                for i in T.Parallel(TILE):
                    us[i] = T.rng_rand_float(dist="uniform")
                for i in T.Parallel(TILE):
                    ns[i] = T.rng_rand_float(dist="normal")
                for i in T.Parallel(TILE):
                    iv[i] = T.rng_rand()
            T.copy(us, U)
            T.copy(ns, Nr)
            T.copy(iv, I)

    return main


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["ascend"], default="ascend")
    args = parser.parse_args()

    N = 8192
    device = torch.device("npu")

    program = rng_fill(N)
    kernel = tilelang.compile(program, target=args.target)

    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    U = torch.empty(N, dtype=torch.float32, device=device)
    Nr = torch.empty(N, dtype=torch.float32, device=device)
    I = torch.empty(N, dtype=torch.uint32, device=device)
    kernel(U, Nr, I)
    torch.npu.synchronize()

    assert torch.isfinite(U).all(), "uniform contains non-finite values"
    assert torch.isfinite(Nr).all(), "normal contains non-finite values"
    assert (U >= 0).all() and (U < 1).all(), "uniform out of [0, 1)"
    assert U.unique().numel() > N // 2, "uniform values not independent enough"
    print(f"uniform mean={U.mean().item():.4f} (~0.5)  normal mean={Nr.mean().item():.4f} (~0)  std={Nr.std().item():.4f} (~1)")
    print("\nVerification passed!")
