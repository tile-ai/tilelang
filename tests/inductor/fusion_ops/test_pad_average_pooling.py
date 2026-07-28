import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_pad_average_pooling():
    def fn(x):
        x = F.pad(x, (1, 1, 1, 1))
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=0, count_include_pad=True)

    x = torch.randn(1, 8, 16, 16, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x)
    torch.testing.assert_close(out, fn(x), atol=0.01, rtol=0.01)

    eager_ms = profiler.do_bench(lambda: fn(x))
    fused_ms = profiler.do_bench(lambda: compiled(x))
    print(f"Pad + AvgPool2D eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"Pad + AvgPool2D TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"Pad + AvgPool2D speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_pad_average_pooling()
    print("PASS: test_pad_average_pooling")
