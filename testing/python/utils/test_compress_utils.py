import torch
import tilelang
import tilelang.testing

from tilelang.utils.sparse import compress, randn_semi_sparse


def _test_compress(M, K, dtype):
    A = randn_semi_sparse(M, K, dtype=dtype, device="cuda")
    A_sparse, E = compress(A)


@tilelang.testing.requires_cuda
def test_compress():
    _test_compress(1024, 1024, torch.float16)
    _test_compress(1024, 1024, torch.bfloat16)
    _test_compress(1024, 1024, torch.float32)

    for name in ("float8_e4m3fn", "float8_e5m2"):
        dt = getattr(torch, name, None)
        if dt is not None:
            _test_compress(1024, 1024, dt)


if __name__ == "__main__":
    test_compress()
    print("All tests passed.")
