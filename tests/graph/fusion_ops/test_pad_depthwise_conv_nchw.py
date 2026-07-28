import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_pad_depthwise_conv_nchw():
    channels = 8

    def fn(x, weight):
        x = F.pad(x, (1, 1, 1, 1))
        return F.conv2d(x, weight, stride=1, padding=0, groups=channels)

    x = torch.randn(1, channels, 16, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(channels, 1, 3, 3, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, weight)
    torch.testing.assert_close(out, fn(x, weight), atol=0.02, rtol=0.02)

    eager_ms = profiler.do_bench(lambda: fn(x, weight))
    fused_ms = profiler.do_bench(lambda: compiled(x, weight))
    print(f"Pad + DepthwiseConv2D NCHW eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"Pad + DepthwiseConv2D NCHW TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"Pad + DepthwiseConv2D NCHW speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_pad_depthwise_conv_nchw()
    print("PASS: test_pad_depthwise_conv_nchw")
