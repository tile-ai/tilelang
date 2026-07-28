import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_unet_upconv():
    def fn(skip, x, up_weight, w1, w2):
        x = F.conv_transpose2d(x, up_weight, stride=2, padding=0)
        return x
        # x = torch.cat((skip, x), dim=1)
        # x = F.relu(F.conv2d(F.pad(x, (1, 1, 1, 1)), w1, stride=1, padding=0))
        # return F.relu(F.conv2d(F.pad(x, (1, 1, 1, 1)), w2, stride=1, padding=0))

    skip = torch.randn(1, 8, 16, 16, device="cuda", dtype=torch.float16)
    x = torch.randn(1, 16, 8, 8, device="cuda", dtype=torch.float16)
    up_weight = torch.randn(16, 8, 2, 2, device="cuda", dtype=torch.float16)
    w1 = torch.randn(8, 16, 3, 3, device="cuda", dtype=torch.float16)
    w2 = torch.randn(8, 8, 3, 3, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(skip, x, up_weight, w1, w2)
    torch.testing.assert_close(out, fn(skip, x, up_weight, w1, w2), atol=0.02, rtol=0.02)

    eager_ms = profiler.do_bench(lambda: fn(skip, x, up_weight, w1, w2))
    fused_ms = profiler.do_bench(lambda: compiled(skip, x, up_weight, w1, w2))
    print(f"UNet UpConv eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"UNet UpConv TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"UNet UpConv speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_unet_upconv()
    print("PASS: test_unet_upconv")
