import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_pad():
    def fn(x):
        return F.pad(x, (1, 1, 1, 1))

    x = torch.randn(1, 8, 16, 16, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x)
    torch.testing.assert_close(out, fn(x), atol=0.02, rtol=0.02)

    eager_ms = profiler.do_bench(lambda: fn(x))
    fused_ms = profiler.do_bench(lambda: compiled(x))
    print(f"Pad eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"Pad TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"Pad speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_pad()
    print("PASS: test_pad")
