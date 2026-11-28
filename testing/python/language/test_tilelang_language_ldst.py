"""Tests for global/shared store intrinsics and predicated forms."""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def _stg_combined_kernel():
    N = 128

    @T.prim_func
    def main(
        X: T.Tensor((N,), T.float32),
        Y0: T.Tensor((N,), T.float32),
        Y1: T.Tensor((N,), T.float32),
        Y2: T.Tensor((N,), T.float32),
        Y3: T.Tensor((N,), T.float32),
        Y4: T.Tensor((N,), T.float32),
        Y5: T.Tensor((N,), T.float32),
        Y6: T.Tensor((N,), T.float32),
        Y7: T.Tensor((N,), T.float32),
        Y8: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(N, threads=32) as pid:
            T.stg32(Y0[pid], T.reinterpret(X[pid], T.uint32))
        with T.Kernel(N // 2, threads=32) as pid:
            T.stg64(Y1[pid * 2 : pid * 2 + 2], T.ldg64(X[pid * 2 : pid * 2 + 2]))
        with T.Kernel(N // 4, threads=32) as pid:
            T.stg128(Y2[pid * 4 : pid * 4 + 4], T.ldg128(X[pid * 4 : pid * 4 + 4]))
        with T.Kernel(N, threads=32) as pid:
            shared = T.alloc_shared((N,), T.float32)
            T.sts32(shared[pid], T.reinterpret(X[pid], T.uint32))
            T.sync_threads()
            T.stg32(Y3[pid], T.lds32(shared[pid]))
        with T.Kernel(N // 2, threads=32) as pid:
            shared = T.alloc_shared((N,), T.float32)
            T.sts64(shared[pid * 2 : pid * 2 + 2], T.ldg64(X[pid * 2 : pid * 2 + 2]))
            T.sync_threads()
            T.stg64(Y4[pid * 2 : pid * 2 + 2], T.lds64(shared[pid * 2 : pid * 2 + 2]))
        with T.Kernel(N // 4, threads=32) as pid:
            shared = T.alloc_shared((N,), T.float32)
            T.sts128(shared[pid * 4 : pid * 4 + 4], T.ldg128(X[pid * 4 : pid * 4 + 4]))
            T.sync_threads()
            T.stg128(Y5[pid * 4 : pid * 4 + 4], T.lds128(shared[pid * 4 : pid * 4 + 4]))
        with T.Kernel(N, threads=32) as pid:
            T.stg32(Y6[pid], T.reinterpret(X[pid], T.uint32), pred=pid < N // 2)
        with T.Kernel(N // 2, threads=32) as pid:
            T.stg64(Y7[pid * 2 : pid * 2 + 2], T.ldg64(X[pid * 2 : pid * 2 + 2]), pred=pid < N // 4)
        with T.Kernel(N // 4, threads=32) as pid:
            T.stg128(Y8[pid * 4 : pid * 4 + 4], T.ldg128(X[pid * 4 : pid * 4 + 4]), pred=pid < N // 8)

    return main


@tilelang.testing.requires_cuda
def test_stg_codegen():
    kernel = tilelang.compile(_stg_combined_kernel(), target="cuda")
    src = kernel.get_kernel_source()
    for width in (32, 64, 128):
        assert f"store_global_{width}" in src
        assert f"store_global_{width}_conditional" in src
        assert f"store_shared_{width}" in src
        assert f"load_shared_{width}" in src

    x = torch.randn(128, dtype=torch.float32, device="cuda")
    outputs = [torch.zeros(128, dtype=torch.float32, device="cuda") for _ in range(9)]
    kernel(x, *outputs)
    for result in outputs[:6]:
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)
    for result in outputs[6:]:
        torch.testing.assert_close(result[:64], x[:64], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result[64:], torch.zeros_like(result[64:]), atol=0, rtol=0)


def _stg256_combined_kernel():
    N = 256

    @T.prim_func
    def main(
        X: T.Tensor((N,), T.float32),
        Y0: T.Tensor((N,), T.float32),
        Y1: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(N // 8, threads=32) as pid:
            T.stg256(Y0[pid * 8 : pid * 8 + 8], T.ldg256(X[pid * 8 : pid * 8 + 8]))
        with T.Kernel(N // 8, threads=32) as pid:
            T.stg256(Y1[pid * 8 : pid * 8 + 8], T.ldg256(X[pid * 8 : pid * 8 + 8]), pred=pid < N // 16)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_stg256_codegen():
    kernel = tilelang.compile(_stg256_combined_kernel(), target="cuda")
    src = kernel.get_kernel_source()
    assert "store_global_256" in src
    assert "store_global_256_conditional" in src

    x = torch.randn(256, dtype=torch.float32, device="cuda")
    outputs = [torch.zeros(256, dtype=torch.float32, device="cuda") for _ in range(2)]
    kernel(x, *outputs)
    torch.testing.assert_close(outputs[0], x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(outputs[1][:128], x[:128], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(outputs[1][128:], torch.zeros_like(outputs[1][128:]), atol=0, rtol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_lt(10, 0)
def test_implicit_256_bit_vector_store_fallback_on_pre_sm100():
    @T.prim_func
    def main(X: T.Tensor((1,), T.float32), Y: T.Tensor((8,), T.float32)):
        with T.Kernel(1, threads=1):
            Y[0:8] = T.Broadcast(X[0], 8)

    x = torch.tensor([3.5], dtype=torch.float32, device="cuda")
    y = torch.zeros(8, dtype=torch.float32, device="cuda")
    tilelang.compile(main, target="cuda")(x, y)
    torch.testing.assert_close(y, torch.full_like(y, 3.5), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
