import torch
import torch._dynamo
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_double_gemm():
    def fn(x, w1, w2):
        x = torch.matmul(x, w1)
        return torch.matmul(x, w2)

    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w1 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    w2 = torch.randn(64, 64, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, w1, w2)
    torch.testing.assert_close(out, fn(x, w1, w2), atol=0.01, rtol=0.01)

    eager_ms = profiler.do_bench(lambda: fn(x, w1, w2))
    fused_ms = profiler.do_bench(lambda: compiled(x, w1, w2))
    print(f"Double GEMM eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"Double GEMM TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"Double GEMM speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_double_gemm()
    print("PASS: test_double_gemm")
