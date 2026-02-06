import torch

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def get_kernel():
    @T.prim_func
    def main(output: T.Tensor((6), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding(0)
            value = T.alloc_var("int32")
            result_any = T.alloc_var("int32")
            result_all = T.alloc_var("int32")
            value = 1
            result_any = T.warp_any(value)
            result_all = T.warp_all(value)
            if tx == 0:
                output[0] = result_any
                output[1] = result_all
            value = 0
            result_any = T.warp_any(value)
            result_all = T.warp_all(value)
            if tx == 0:
                output[2] = result_any
                output[3] = result_all
            value = tx % 2
            result_any = T.warp_any(value)
            result_all = T.warp_all(value)
            if tx == 0:
                output[4] = result_any
                output[5] = result_all

    return main


def test_vote():
    output = torch.tensor(6 * [-1], dtype=torch.int32, device="cuda")
    kernel = get_kernel()
    kernel(output)
    assert "__any_sync" and "__all_sync" in kernel.get_kernel_source()
    ref = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.int32, device="cuda")
    assert output.equal(ref)


if __name__ == "__main__":
    test_vote()
