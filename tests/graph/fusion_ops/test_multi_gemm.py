"""Standalone multi GEMM test for TileLang torch.compile backend."""

import torch
import torch._dynamo
import tilelang  # noqa: F401 — triggers backend registration
import tilelang.profiler as profiler


def test_multi_gemm():
    """Multi GEMM test compiled via TileLang.

    This pattern exercises fusion of two matmuls where the output
    of the first feeds directly into the second.  Rule-based
    FuseOps often cannot fuse this because the intermediate tensor
    has multiple consumers or complex access patterns.
    """

    def fn(x, w1, w2, w3, w4, w5):
        x1 = torch.matmul(x, w1)
        x2 = torch.matmul(x1, w2)
        x3 = torch.matmul(w3, w4)
        x4 = torch.matmul(x2, x3)
        x5 = torch.matmul(x4, w5)
        return x5

    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w1 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w2 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w3 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w4 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w5 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, w1, w2, w3, w4, w5)
    torch.testing.assert_close(out, fn(x, w1, w2, w3, w4, w5), atol=0.01, rtol=0.01)

    eager_ms = profiler.do_bench(lambda: fn(x, w1, w2, w3, w4, w5))
    fused_ms = profiler.do_bench(lambda: compiled(x, w1, w2, w3, w4, w5))
    print(f"Multi GEMM eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"Multi GEMM TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"Multi GEMM speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_multi_gemm()
    print("PASS: test_multi_gemm")
