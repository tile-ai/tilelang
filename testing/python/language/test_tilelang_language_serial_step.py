import tilelang
import tilelang.language as T
import torch
import tilelang.testing

tilelang.testing.set_random_seed()


def test_serial_with_step():

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def serial_with_step(A: T.Tensor((10,), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for i in range(0, 10, 2):
                A[i] = 1.0
            for i in range(1, 10, 2):
                A[i] = 2.0

    kernel = serial_with_step()
    data = kernel()
    ref = torch.tensor([1.0 if i % 2 == 0 else 2.0 for i in range(10)], dtype=torch.float32).cuda()
    torch.testing.assert_close(data, ref)


if __name__ == '__main__':
    tilelang.testing.main()
