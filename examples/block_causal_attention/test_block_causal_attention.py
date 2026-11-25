import pytest

import tilelang.testing
from tilelang.utils.device import get_max_dynamic_shared_memory

import block_causal_attention
import block_causal_attention_varlen


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dllm_block", block_causal_attention._SUPPORTED_DLLM_BLOCKS)
def test_block_causal_attention_fixed(dllm_block):
    block_causal_attention._run_fixed_case(2, 256, 2, 64, dllm_block)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "lengths,dim,dllm_block,block_size",
    [
        pytest.param([128, 256, 384], 64, dllm_block, 64, id=f"dllm-block-{dllm_block}")
        for dllm_block in block_causal_attention._SUPPORTED_DLLM_BLOCKS
    ]
    + [
        pytest.param([128, 256], 64, 16, 32, id="tile-32"),
        pytest.param([512, 1024], 128, 64, 128, id="tile-128"),
    ],
)
def test_block_causal_attention_varlen(lengths, dim, dllm_block, block_size):
    # The largest tile needs ~192 KiB of shared memory. Match the example's
    # fallback on devices with smaller per-block limits.
    if block_size == 128 and get_max_dynamic_shared_memory() < 200 * 1024:
        block_size = 64
    block_causal_attention_varlen._run_varlen_case(lengths, heads=2, dim=dim, dllm_block=dllm_block, block_size=block_size)


if __name__ == "__main__":
    tilelang.testing.main()
