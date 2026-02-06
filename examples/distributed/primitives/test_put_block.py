import tilelang.testing
import torch
import torch.multiprocessing

import example_put_block


@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_put_block():
    torch.multiprocessing.spawn(example_put_block.main, args=(2, None), nprocs=2)


if __name__ == "__main__":
    tilelang.testing.main()
