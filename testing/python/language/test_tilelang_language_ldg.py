"""Tests for load_global_32/64/128/256 intrinsics and predicated forms."""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def _ldg_combined_kernel():
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
    ):
        with T.Kernel(N, threads=32) as pid:
            Y0[pid] = T.reinterpret(T.ldg32(X[pid]), T.float32)
        with T.Kernel(N // 2, threads=32) as pid:
            Y1[pid * 2 : pid * 2 + 2] = T.reinterpret(T.ldg64(X[pid * 2 : pid * 2 + 2]), T.float32x2)
        with T.Kernel(N // 4, threads=32) as pid:
            Y2[pid * 4 : pid * 4 + 4] = T.reinterpret(T.ldg128(X[pid * 4 : pid * 4 + 4]), T.float32x4)
        with T.Kernel(N, threads=32) as pid:
            Y3[pid] = T.reinterpret(T.ldg32(X[pid], pred=pid < N // 2), T.float32)
        with T.Kernel(N // 2, threads=32) as pid:
            Y4[pid * 2 : pid * 2 + 2] = T.reinterpret(T.ldg64(X[pid * 2 : pid * 2 + 2], pred=pid < N // 4), T.float32x2)
        with T.Kernel(N // 4, threads=32) as pid:
            Y5[pid * 4 : pid * 4 + 4] = T.reinterpret(T.ldg128(X[pid * 4 : pid * 4 + 4], pred=pid < N // 8), T.float32x4)

    return main


@tilelang.testing.requires_cuda
def test_ldg_codegen():
    kernel = tilelang.compile(_ldg_combined_kernel(), target="cuda")
    src = kernel.get_kernel_source()
    for width in (32, 64, 128):
        assert f"load_global_{width}" in src
        assert f"load_global_{width}_conditional" in src

    x = torch.randn(128, dtype=torch.float32, device="cuda")
    outputs = [torch.zeros(128, dtype=torch.float32, device="cuda") for _ in range(6)]
    kernel(x, *outputs)
    for result in outputs[:3]:
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)
    for result in outputs[3:]:
        torch.testing.assert_close(result[:64], x[:64], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result[64:], torch.zeros_like(result[64:]), atol=0, rtol=0)


def _ldg256_combined_kernel():
    N = 256

    @T.prim_func
    def main(
        X: T.Tensor((N,), T.float32),
        Y0: T.Tensor((N,), T.float32),
        Y1: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(N // 8, threads=32) as pid:
            Y0[pid * 8 : pid * 8 + 8] = T.reinterpret(T.ldg256(X[pid * 8 : pid * 8 + 8]), T.float32x8)
        with T.Kernel(N // 8, threads=32) as pid:
            Y1[pid * 8 : pid * 8 + 8] = T.reinterpret(
                T.ldg256(X[pid * 8 : pid * 8 + 8], pred=pid < N // 16),
                T.float32x8,
            )

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_ldg256_codegen():
    kernel = tilelang.compile(_ldg256_combined_kernel(), target="cuda")
    src = kernel.get_kernel_source()
    assert "load_global_256" in src
    assert "load_global_256_conditional" in src

    x = torch.randn(256, dtype=torch.float32, device="cuda")
    outputs = [torch.zeros(256, dtype=torch.float32, device="cuda") for _ in range(2)]
    kernel(x, *outputs)
    torch.testing.assert_close(outputs[0], x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(outputs[1][:128], x[:128], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(outputs[1][128:], torch.zeros_like(outputs[1][128:]), atol=0, rtol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_lt(10, 0)
def test_ldg256_stg256_fallback_on_pre_sm100():
    @T.prim_func
    def main(X: T.Tensor((256,), T.float32), Y: T.Tensor((256,), T.float32)):
        with T.Kernel(32, threads=32) as pid:
            value = T.ldg256(X[pid * 8 : pid * 8 + 8])
            T.stg256(Y[pid * 8 : pid * 8 + 8], value)

    x = torch.randn(256, dtype=torch.float32, device="cuda")
    y = torch.empty(256, dtype=torch.float32, device="cuda")
    tilelang.compile(main, target="cuda")(x, y)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
