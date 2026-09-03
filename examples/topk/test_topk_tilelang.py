import tilelang.testing
import torch

import example_topk


@tilelang.testing.requires_cuda
def test_topk_tilelang():
    example_topk.main(argv=[])


@tilelang.testing.requires_rocm
def test_topk_tilelang_rocm():
    """Validate every generic Top-K thread candidate on ROCm."""
    torch.manual_seed(0)
    logits = torch.rand((64, 64), device="cuda", dtype=torch.float32)
    for threads in (128, 256, 512):
        example_topk.validate_topk(logits, topk=4, blk_m=64, threads=threads)


if __name__ == "__main__":
    tilelang.testing.main()
