"""Safe-memory zero else-branch on dynamically indexed uint8 copies (4, 8 and 16 byte lanes)."""

import pytest
import tilelang
import tilelang.rocm.language as T
import tilelang.testing
import torch


@tilelang.jit(target="hip")
def _copy_row_tile(cols, threads):
    @T.prim_func
    def main(
        A: T.Tensor((256, cols), T.uint8),
        idx: T.Tensor((1,), T.int32),
        out: T.Tensor((16, cols), T.uint8),
    ):
        with T.Kernel(1, threads=threads):
            tile = T.alloc_shared((16, cols), T.uint8)
            n = idx[0]
            T.copy(A[n * 16, 0], tile)
            for i, j in T.Parallel(16, cols):
                out[i, j] = tile[i, j]

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(("cols", "threads"), [(64, 256), (64, 128), (128, 128)])
def test_uint8_copy_with_safe_memory_else_branch(cols, threads):
    kernel = _copy_row_tile(cols, threads)
    A = torch.randint(1, 255, (256, cols), dtype=torch.uint8, device="cuda")
    out = torch.zeros(16, cols, dtype=torch.uint8, device="cuda")

    idx = torch.tensor([3], dtype=torch.int32, device="cuda")
    kernel(A, idx, out)
    assert torch.equal(out, A[48:64])

    # Past the end of A: the copy must read the safe value, which is zero.
    idx = torch.tensor([16], dtype=torch.int32, device="cuda")
    kernel(A, idx, out)
    assert torch.count_nonzero(out) == 0


if __name__ == "__main__":
    tilelang.testing.main()
