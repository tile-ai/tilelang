"""NPU regressions for the 64-dimensional QK tail and persistent jobs."""

import pytest
import torch

from examples.ascend.hy4_mla.prefill import compile_prefill
from examples.ascend.hy4_mla.testing import check_outputs, make_inputs, sparse_mla_reference


@pytest.fixture
def npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_op_timeout_ms(30000)
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize(
    "nq,nk,topk,component,sink,pipeline",
    [
        (1, 79, 64, "rope", True, True),
        (33, 128, 128, "latent", False, True),
        (33, 257, 320, "both", True, True),
        (65, 193, 640, "rope", True, True),
        (33, 257, 704, "both", True, True),
        (1, 79, 2048, "both", True, True),
        (65, 257, 2048, "both", True, True),
        (1, 79, 64, "both", True, False),
    ],
)
def test_bf16_core(npu, nq, nk, topk, component, sink, pipeline):
    inputs = make_inputs(nq, nk, topk, component=component)
    expected = sparse_mla_reference(*inputs, scale=1 / 16, sink=sink)
    device_inputs = tuple(t.to(npu) for t in inputs)
    kernel = compile_prefill(nq, nk, topk, scale=1 / 16, sink=sink, variable_lengths=True, pipeline_stages=pipeline)
    for _ in range(2):
        actual = kernel(*device_inputs)
        torch.npu.synchronize()
        check_outputs(actual, expected)


def test_last_tile_rope_rescale_across_launches(npu):
    nq, nk, topk = 33, 128, 128
    q = torch.zeros((nq, 64, 576), dtype=torch.bfloat16)
    q[:, 32:, 512:] = 1
    kv = torch.zeros((nk, 576), dtype=torch.bfloat16)
    indices = torch.arange(topk, dtype=torch.int32).expand(nq, topk).contiguous()
    attn_sink = torch.zeros(64, dtype=torch.float32)
    lengths = torch.full((nq,), topk, dtype=torch.int32)
    kernel = compile_prefill(nq, nk, topk, scale=0.5, sink=False)
    for value in (1, 2, 1):
        kv[64:, :] = value
        inputs = q, kv, indices, attn_sink, lengths
        expected = sparse_mla_reference(*inputs, scale=0.5, sink=False)
        actual = kernel(*(t.to(npu) for t in inputs))
        torch.npu.synchronize()
        check_outputs(actual, expected)
