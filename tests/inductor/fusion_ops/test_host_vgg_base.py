import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_host_vgg_base():
    def fn(x, w1, w2, w3, w4, w5):
        x = F.relu(F.conv2d(F.pad(x, (1, 1, 1, 1)), w1, stride=1, padding=0))
        x = F.relu(F.conv2d(F.pad(x, (1, 1, 1, 1)), w2, stride=1, padding=0))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(F.conv2d(F.pad(x, (1, 1, 1, 1)), w3, stride=1, padding=0))
        x = F.relu(F.conv2d(F.pad(x, (1, 1, 1, 1)), w4, stride=1, padding=0))
        return F.relu(F.conv2d(F.pad(x, (1, 1, 1, 1)), w5, stride=1, padding=0))

    x = torch.randn(1, 4, 16, 16, device="cuda", dtype=torch.float16)
    w1 = torch.randn(4, 4, 3, 3, device="cuda", dtype=torch.float16)
    w2 = torch.randn(4, 4, 3, 3, device="cuda", dtype=torch.float16)
    w3 = torch.randn(8, 4, 3, 3, device="cuda", dtype=torch.float16)
    w4 = torch.randn(8, 8, 3, 3, device="cuda", dtype=torch.float16)
    w5 = torch.randn(8, 8, 3, 3, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, w1, w2, w3, w4, w5)
    torch.testing.assert_close(out, fn(x, w1, w2, w3, w4, w5), atol=0.03, rtol=0.03)

    eager_ms = profiler.do_bench(lambda: fn(x, w1, w2, w3, w4, w5))
    fused_ms = profiler.do_bench(lambda: compiled(x, w1, w2, w3, w4, w5))
    print(f"Host VGG16Base compact eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"Host VGG16Base compact TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"Host VGG16Base compact speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_host_vgg_base()
    print("PASS: test_host_vgg_base")
