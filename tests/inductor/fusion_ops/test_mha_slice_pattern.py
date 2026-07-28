import math

import torch
import torch._dynamo
import torch.nn.functional as F
import tilelang  # noqa: F401
import tilelang.profiler as profiler


def test_mha_slice_pattern():
    batch, seq_len, heads, head_dim = 1, 8, 4, 8
    hidden = heads * head_dim

    def fn(qkv):
        q = qkv[:, :, 0:hidden]
        k = qkv[:, :, hidden : 2 * hidden]
        v = qkv[:, :, 2 * hidden : 3 * hidden]
        q = q.reshape(batch, seq_len, heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, heads, head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(batch, seq_len, hidden)

    qkv = torch.randn(batch, seq_len, 3 * hidden, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(qkv)
    torch.testing.assert_close(out, fn(qkv), atol=0.02, rtol=0.02)

    eager_ms = profiler.do_bench(lambda: fn(qkv))
    fused_ms = profiler.do_bench(lambda: compiled(qkv))
    print(f"MHA Slice Pattern eager:          {eager_ms:.4f} ms ({eager_ms * 1000:.1f} us)")
    print(f"MHA Slice Pattern TileLang fused: {fused_ms:.4f} ms ({fused_ms * 1000:.1f} us)")
    print(f"MHA Slice Pattern speedup:        {eager_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    test_mha_slice_pattern()
    print("PASS: test_mha_slice_pattern")
