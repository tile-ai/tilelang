"""Standalone rms_norm test for TileLang torch.compile backend."""

import torch
import torch._dynamo
import tilelang  # noqa: F401 — triggers backend registration
import tilelang.profiler as profiler


def test_rms_norm():

    def fn(x, w1):
        x_squared = x.pow(2)
        mean_squared = x_squared.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + 1e-5)
        y = (x / rms) * w1
        return y

    x = torch.randn(2048, 4096, device="cuda", dtype=torch.float)
    w1 = torch.randn(4096, device="cuda", dtype=torch.float)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, w1)
    torch.testing.assert_close(out, fn(x, w1), atol=0.01, rtol=0.01)

    eager_ms = profiler.do_bench(lambda: fn(x, w1))
    fused_ms = profiler.do_bench(lambda: compiled(x, w1))
    print(f"RMSNorm eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"RMSNorm TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"RMSNorm speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_rms_norm()
    print("PASS: test_rms_norm")
