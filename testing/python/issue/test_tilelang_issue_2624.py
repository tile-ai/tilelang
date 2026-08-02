import tilelang
import tilelang.language as T
import tilelang.testing
import torch

N = 4


def make_kernel(op, dt):
    @T.prim_func
    def main(A: T.Tensor((N,), dt), B: T.Tensor((1,), dt)):
        with T.Kernel(1, threads=N):
            As = T.alloc_fragment((N,), dt)
            Bs = T.alloc_fragment((1,), dt)
            T.copy(A, As)
            (T.reduce_absmax if op == "absmax" else T.reduce_max)(As, Bs, dim=0)
            T.copy(Bs, B)

    return main


def test_reduce_absmax():
    A = torch.tensor([100, 10, 20, 30], dtype=torch.uint32, device="cuda")
    uint32_max = tilelang.compile(make_kernel("max", "uint32"), out_idx=[1])(A)
    uint32_absmax = tilelang.compile(make_kernel("absmax", "uint32"), out_idx=[1])(A)
    torch.testing.assert_close(uint32_max, uint32_absmax, rtol=0, atol=0)


def test_reduce_abssum():
    A = torch.tensor([100, 10, 20, 30], dtype=torch.uint32, device="cuda")
    uint32_sum = tilelang.compile(make_kernel("sum", "uint32"), out_idx=[1])(A)
    uint32_abssum = tilelang.compile(make_kernel("abssum", "uint32"), out_idx=[1])(A)
    torch.testing.assert_close(uint32_sum, uint32_abssum, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
