"""Standalone double GEMM test for TileLang torch.compile backend."""

import torch
import torch._dynamo
import tilelang  # noqa: F401 — triggers backend registration
import tilelang.profiler as profiler


def test_double_gemm():
    """Double matmul compiled via TileLang.

    y = matmul(x, w1); z = matmul(y, w2)

    This pattern exercises fusion of two matmuls where the output
    of the first feeds directly into the second.  Rule-based
    FuseOps often cannot fuse this because the intermediate tensor
    has multiple consumers or complex access patterns.
    """

    def fn(x, w1, w2):
        y = torch.matmul(x, w1)
        z = torch.matmul(y, w2)
        return z

    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w1 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w2 = torch.randn(64, 64, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, w1, w2)
    torch.testing.assert_close(out, fn(x, w1, w2), atol=0.01, rtol=0.01)

    eager_ms = profiler.do_bench(lambda: fn(x, w1, w2))
    fused_ms = profiler.do_bench(lambda: compiled(x, w1, w2))
    print(f"Double GEMM eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"Double GEMM TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"Double GEMM speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_double_gemm()
    print("PASS: test_double_gemm")
