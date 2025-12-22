import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=-1)
def get_inf_kernel(dtype: str):
    @T.prim_func
    def main(A: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32):
            T.fill(A, T.infinity(dtype))

    return main


def _test_infinity(dtype: str):
    kernel = get_inf_kernel(dtype)
    output = kernel()

    assert torch.all(output == torch.inf), f"check failed for {dtype=}"


@tilelang.testing.requires_cuda
def test_infinity():
    _test_infinity(T.float16)
    _test_infinity(T.bfloat16)
    _test_infinity(T.float32)
    _test_infinity(T.float64)


if __name__ == "__main__":
    tilelang.testing.main()
