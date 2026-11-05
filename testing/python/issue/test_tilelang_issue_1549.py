import re

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
    # Native definition renewal may rename the loop and pointer. The mutable
    # scalar must still be assigned inside its 32-iteration per-thread loop.
    loop = re.search(r"for \(int (\w+) = 0; \1 < 32; \+\+\1\) \{([^}]+)\}", code)
    assert loop is not None
    assert re.search(r"idx = .*threadIdx\.x", loop.group(2))
    assert re.search(r"Data(?:_ptr)?\[.*\] = idx;", loop.group(2))
    torch.testing.assert_close(data, torch.arange(M, device="cuda", dtype=torch.int32), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
