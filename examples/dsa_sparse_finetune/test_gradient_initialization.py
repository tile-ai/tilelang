import pytest
import torch

from indexer_bwd import indexer_bwd_interface, ref_indexer_bwd
from sparse_mla_bwd import sparse_mla_bwd


@pytest.fixture(autouse=True)
def _allocation_backend(monkeypatch):
    # NVRTC allocates outputs through torch.empty; tvm_ffi may allocate natively.
    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", "nvrtc")


def _poison_empty(monkeypatch):
    # torch.empty makes no promise about prior contents. Use a nonzero value
    # deterministically instead of relying on the CUDA allocator to reuse it.
    original = torch.empty
    poisoned = []

    def empty(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.is_cuda and result.is_floating_point():
            result.fill_(7)
            poisoned.append((tuple(result.shape), result.dtype))
        return result

    monkeypatch.setattr(torch, "empty", empty)
    return poisoned


@pytest.mark.parametrize("seq_len", [64, 80])
def test_indexer_gradient_initialization(monkeypatch, seq_len):
    torch.manual_seed(42)
    heads, dim, topk = 16, 128, 32
    # Keep reference comparisons away from the ReLU discontinuity at zero.
    q = torch.rand(seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, heads, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(seq_len, dim, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(seq_len, 1)
    causal = torch.arange(topk, device="cuda")[None, :] <= torch.arange(seq_len, device="cuda")[:, None]
    attn = torch.randn(seq_len, topk, device="cuda").masked_fill(~causal, float("-inf")).softmax(-1)
    scores, ref_dq, ref_dw, ref_dk = ref_indexer_bwd(q, weights, k, indices, attn, offsets)
    args = (q.detach(), weights.detach(), k.detach(), attn.detach(), scores.detach(), indices, offsets)
    indexer_bwd_interface(*args)  # Compile before instrumenting allocations.
    poisoned = _poison_empty(monkeypatch)
    for _ in range(2):
        poisoned.clear()
        dq, dw, dk = indexer_bwd_interface(*args)
        assert (tuple(dk.shape), dk.dtype) in poisoned, "dIndexK allocation was not poisoned"
        torch.testing.assert_close(dk[topk:], torch.zeros_like(dk[topk:]), atol=0, rtol=0)
        for actual, expected in ((dq, ref_dq), (dw, ref_dw), (dk, ref_dk)):
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.05, rtol=0.05)


@pytest.mark.parametrize("seq_len", [64, 80])
def test_mla_gradient_initialization(monkeypatch, seq_len):
    torch.manual_seed(42)
    heads, dim, value_dim, topk = 16, 576, 512, 32
    q = torch.randn(seq_len, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(seq_len, 1, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(seq_len, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).view(1, 1, topk).repeat(seq_len, 1, 1)
    logits = torch.einsum("shd,kd->shk", q.float(), kv[:topk, 0].float()) * dim**-0.5
    causal = torch.arange(topk, device="cuda")[None, :] <= torch.arange(seq_len, device="cuda")[:, None]
    logits = logits.masked_fill(~causal[:, None, :], float("-inf"))
    lse = logits.logsumexp(-1).detach().contiguous()
    out = torch.einsum("shk,kd->shd", logits.softmax(-1), kv[:topk, 0, :value_dim].float())
    out.backward(do.float())
    args = (q.detach(), kv.detach(), out.detach().to(torch.bfloat16).contiguous(), do, indices, lse, offsets)
    sparse_mla_bwd(*args)  # Compile before instrumenting allocations.
    poisoned = _poison_empty(monkeypatch)
    for _ in range(2):
        poisoned.clear()
        dq, dkv = sparse_mla_bwd(*args)
        assert (tuple(kv.shape), torch.float32) in poisoned, "dKV accumulator allocation was not poisoned"
        torch.testing.assert_close(dkv[topk:], torch.zeros_like(dkv[topk:]), atol=0, rtol=0)
        torch.testing.assert_close(dq.float(), q.grad.float(), atol=0.05, rtol=0.05)
        torch.testing.assert_close(dkv.float(), kv.grad.float(), atol=0.05, rtol=0.05)
