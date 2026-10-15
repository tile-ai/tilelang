"""Correctness tests for causal sparse attention on Ascend NPU."""

import pytest
import torch

pytest.importorskip("torch_npu")

from example_dsa import compile_kernel, make_inputs, reference_attention
from tilelang.carver.arch.driver import get_num_cube_cores

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an Ascend NPU")


@pytest.mark.parametrize(
    "batch_size,seq_len,kv_len,top_k,num_blocks",
    [
        pytest.param(1, 280, 512, 128, None, id="single-block-l2"),
        pytest.param(2, 280, 257, 4096, 2, id="multi-block-gm"),
    ],
)
def test_dsa(batch_size, seq_len, kv_len, top_k, num_blocks):
    if num_blocks is None:
        cores = get_num_cube_cores(torch.npu.current_device())
        seq_len = (seq_len + cores - 1) // cores * cores
    # Early queries have partial/empty blocks; later queries also have full blocks.
    # Two cores in the second case also exercise the output L2-bypass policy.
    kernel = compile_kernel(batch_size, seq_len, kv_len, top_k, num_blocks=num_blocks)
    inputs = make_inputs(batch_size, seq_len, kv_len, top_k)
    expected = reference_attention(*inputs)
    actual = kernel(*inputs)
    torch.npu.synchronize()
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)
