import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_conv():
    def fn(x, w1):
        return F.conv2d(x, w1, stride=1, padding=0)

    x = torch.randn(1, 8, 16, 16, device="cuda", dtype=torch.float16)
    w1 = torch.randn(8, 8, 3, 3, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, w1)
    torch.testing.assert_close(out, fn(x, w1), atol=0.02, rtol=0.02)

    inductor_compiled = torch.compile(fn, backend="inductor")

    eager_ms = profiler.do_bench(lambda: inductor_compiled(x, w1))
    fused_ms = profiler.do_bench(lambda: compiled(x, w1))
    print(f"ConvPipe Conv eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"ConvPipe Conv TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"ConvPipe Conv speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_conv()
    print("PASS: test_conv")
