import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_unet_upsample():
    def fn(skip, x, up_weight):
        x = F.conv_transpose2d(x, up_weight, stride=2, padding=0)
        return torch.cat((skip, x), dim=1)

    skip = torch.randn(1, 8, 16, 16, device="cuda", dtype=torch.float16)
    x = torch.randn(1, 16, 8, 8, device="cuda", dtype=torch.float16)
    up_weight = torch.randn(16, 8, 2, 2, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(skip, x, up_weight)
    torch.testing.assert_close(out, fn(skip, x, up_weight), atol=0.02, rtol=0.02)

    eager_ms = profiler.do_bench(lambda: fn(skip, x, up_weight))
    fused_ms = profiler.do_bench(lambda: compiled(skip, x, up_weight))
    print(f"UNet UpSample eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"UNet UpSample TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"UNet UpSample speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_unet_upsample()
    print("PASS: test_unet_upsample")
