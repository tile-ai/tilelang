import torch
import tilelang
import tilelang.testing

from tilelang.utils.sparse import compress_sm90, randn_semi_sparse


def _test_compress_sm90(M, K, block_k, dtype):
    A = randn_semi_sparse(M, K, dtype=dtype, device="cuda")
    A_sparse, E = compress_sm90(A, block_k, False)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_compress_sm90():
    _test_compress_sm90(1024, 1024, 128, torch.float16)
    _test_compress_sm90(1024, 1024, 64, torch.float16)
    _test_compress_sm90(1024, 1024, 32, torch.float16)

    _test_compress_sm90(1024, 1024, 128, torch.bfloat16)
    _test_compress_sm90(1024, 1024, 64, torch.bfloat16)
    _test_compress_sm90(1024, 1024, 32, torch.bfloat16)

    _test_compress_sm90(1024, 1024, 64, torch.float32)
    _test_compress_sm90(1024, 1024, 32, torch.float32)
    _test_compress_sm90(1024, 1024, 16, torch.float32)

    _test_compress_sm90(1024, 1024, 256, torch.float8_e4m3fn)
    _test_compress_sm90(1024, 1024, 128, torch.float8_e4m3fn)
    _test_compress_sm90(1024, 1024, 64, torch.float8_e4m3fn)

    _test_compress_sm90(1024, 1024, 256, torch.float8_e5m2)
    _test_compress_sm90(1024, 1024, 128, torch.float8_e5m2)
    _test_compress_sm90(1024, 1024, 64, torch.float8_e5m2)


if __name__ == "__main__":
    test_compress_sm90()
    print("All tests passed.")
