import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def _nested_guard_cast_store(
    A: T.Tensor((16,), T.float8_e4m3fn),
    B: T.Tensor((16,), T.float8_e4m3fn),
):
    with T.Kernel(1, threads=32):
        a_local = T.alloc_local((16,), T.float32)
        T.copy(A, a_local)
        for i in T.vectorized(16):
            if i < 12:  # noqa: SIM102
                if i < 8:
                    B[i] = T.cast(a_local[i], T.float8_e4m3fn)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_nested_guard_cast_store_preserves_unwritten_lanes():
    a = (torch.arange(16, device="cuda", dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    b = torch.full((16,), 7.0, device="cuda").to(torch.float8_e4m3fn)

    _nested_guard_cast_store(a, b)

    expected = torch.full((16,), 7.0, device="cuda", dtype=torch.float32)
    expected[:8] = a[:8].to(torch.float32)
    torch.testing.assert_close(b.to(torch.float32), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
