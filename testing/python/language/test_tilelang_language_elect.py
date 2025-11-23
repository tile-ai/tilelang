import torch

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def get_kernel():
    @T.prim_func
    def main(x: T.Tensor((1), 'int32')):
        with T.Kernel(1, threads=32):
            if T.elect_one_sync():
                x[0] += 1

    return main


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_elect_one_sync():
    a = torch.tensor([0], dtype=torch.int32, device='cuda')
    kernel = get_kernel()
    kernel(a)
    assert 'cute::elect_one_sync' in kernel.get_kernel_source()
    assert a[0] == 1


if __name__ == "__main__":
    tilelang.testing.main()