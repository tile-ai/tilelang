import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [f"{sign}{bits}" for sign in ("int", "uint") for bits in (32, 64)])
@pytest.mark.parametrize("elements_per_thread", [1, 4])
def test_popcount(dtype, elements_per_thread):
    n = 32 * elements_per_thread

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n):
                B[i] = T.popcount(A[i])

    bits = torch.iinfo(getattr(torch, dtype)).bits
    mask = (1 << bits) - 1
    patterns = [0, 1, 7, mask, mask - 1, 1 << (bits - 1), mask >> 1, mask // 3]
    patterns *= n // len(patterns)
    values = [x - (1 << bits) if dtype.startswith("int") and x >= (1 << (bits - 1)) else x for x in patterns]
    a = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    assert kernel(a).cpu().tolist() == [x.bit_count() for x in patterns]


if __name__ == "__main__":
    tilelang.testing.main()
