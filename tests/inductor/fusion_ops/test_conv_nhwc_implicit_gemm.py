import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_conv_nhwc_implicit_gemm():
    n, h, w, c, out_c = 1, 14, 14, 16, 32
    kh = kw = 3

    def fn(x_nhwc, weight_hwcf):
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        weight = weight_hwcf.permute(3, 2, 0, 1).contiguous()
        x = F.pad(x, (1, 1, 1, 1))
        out = F.conv2d(x, weight, stride=1, padding=0)
        return out.permute(0, 2, 3, 1).contiguous()

    x = torch.randn(n, h, w, c, device="cuda", dtype=torch.float16)
    weight = torch.randn(kh, kw, c, out_c, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, weight)
    torch.testing.assert_close(out, fn(x, weight), atol=0.02, rtol=0.02)

    eager_ms = profiler.do_bench(lambda: fn(x, weight))
    fused_ms = profiler.do_bench(lambda: compiled(x, weight))
    print(f"NHWC Pad + Conv2D implicit GEMM eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"NHWC Pad + Conv2D implicit GEMM TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"NHWC Pad + Conv2D implicit GEMM speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_conv_nhwc_implicit_gemm()
    print("PASS: test_conv_nhwc_implicit_gemm")
