import tilelang as tl
import tilelang.testing
import tilelang.language as T
import torch


@tilelang.testing.requires_cuda
def test_issue_1549_strange_var_vectorization():
    @tl.jit
    def get_wrong_kernel(M: int = 4096):
        dtype = "int32"
        num_threads = 64

        @T.prim_func
        def main(
            Data: T.Tensor((M,), dtype),
        ):
            with T.Kernel(1, threads=num_threads) as _:
                # Pre-allocated scalar variables (causes issue in 0.1.7.post1)
                idx = T.alloc_var(T.int32)
                for i in T.Parallel(M):
                    idx = i
                    Data[i] = idx

        return main

    kernel = get_wrong_kernel()
    M = 2048
    kernel = get_wrong_kernel(M)
    data = torch.randint(0, 100, (M,), dtype=torch.int32, device="cuda")
    kernel(data)
    code = kernel.get_kernel_source()
    print(code)
    assert (
        """for (int i = 0; i < 32; ++i) {
    idx = ((i * 64) + ((int)threadIdx.x));
    Data[((i * 64) + ((int)threadIdx.x))] = idx;
  }"""
        in code
    )


if __name__ == "__main__":
    tilelang.testing.main()
