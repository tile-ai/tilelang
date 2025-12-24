import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import pytest

tilelang.testing.set_random_seed()


@tilelang.jit(out_idx=[1])
def parallel_elementwise_static(length=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length):
                B[i] = A[i] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def parallel_elementwise_dynamic(max_len=512, threads=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((max_len,), dtype),
        B: T.Tensor((max_len,), dtype),
        valid_len: T.int32,
    ):
        with T.Kernel(1, threads=threads) as _:
            for i in T.Parallel(max_len):
                B[i] = 0.0
            span = T.min(valid_len, max_len)
            for i in T.Parallel(span):
                B[i] = A[i] - 1.0

    return main


def _require_cuda_tensor(shape, dtype=torch.float32):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        return torch.randn(*shape, device="cuda", dtype=dtype)
    except RuntimeError as err:
        pytest.skip(f"CUDA runtime unavailable: {err}")


def test_parallel_static_extent():
    kernel = parallel_elementwise_static(length=256)
    data = _require_cuda_tensor((256,), torch.float32)
    result = kernel(data)
    torch.testing.assert_close(result, data + 1.0, atol=1e-5, rtol=1e-5)


def test_parallel_dynamic_extent():
    kernel = parallel_elementwise_dynamic(max_len=512, threads=256)
    data = _require_cuda_tensor((512,), torch.float32)
    for valid_len in [0, 13, 200, 600]:
        out = kernel(data, valid_len)
        reference = torch.zeros_like(data)
        clip = min(valid_len, data.shape[0])
        reference[:clip] = data[:clip] - 1.0
        torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
