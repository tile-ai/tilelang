import torch
import torch._dynamo
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_matmul_bias():
    def fn(x, weight, bias):
        return torch.matmul(x, weight) + bias

    x = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    bias = torch.randn(128, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, weight, bias)
    torch.testing.assert_close(out, fn(x, weight, bias), atol=0.01, rtol=0.01)

    eager_ms = profiler.do_bench(lambda: fn(x, weight, bias))
    fused_ms = profiler.do_bench(lambda: compiled(x, weight, bias))
    print(f"MatMul + BiasAdd eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"MatMul + BiasAdd TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"MatMul + BiasAdd speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_matmul_bias()
    print("PASS: test_matmul_bias")
