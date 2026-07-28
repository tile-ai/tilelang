import torch
import torch._dynamo
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_rms_norm():
    def fn(x, weight):
        x_squared = x.pow(2)
        mean_squared = x_squared.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + 1e-5)
        return (x / rms) * weight

    x = torch.randn(2048, 4096, device="cuda", dtype=torch.float32)
    weight = torch.randn(4096, device="cuda", dtype=torch.float32)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, weight)
    torch.testing.assert_close(out, fn(x, weight), atol=1e-2, rtol=1e-2)

    eager_ms = profiler.do_bench(lambda: fn(x, weight))
    fused_ms = profiler.do_bench(lambda: compiled(x, weight))
    print(f"RMSNorm eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"RMSNorm TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"RMSNorm speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_rms_norm()
    print("PASS: test_rms_norm")
