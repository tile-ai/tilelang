import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "block_causal_attention"
sys.path.insert(0, str(_EXAMPLE_DIR))

from block_causal_attention import _run_fixed_case  # noqa: E402
from block_causal_attention_varlen import _run_varlen_case  # noqa: E402


@tilelang.testing.requires_rocm
def test_block_causal_attention_fixed_forward_backward():
    _run_fixed_case(
        batch=1,
        seq_len=128,
        heads=1,
        dim=64,
        dllm_block=16,
        dtype=torch.float16,
    )


@tilelang.testing.requires_rocm
def test_block_causal_attention_varlen_forward_backward():
    _run_varlen_case(
        lengths=[128, 256],
        heads=1,
        dim=64,
        dllm_block=16,
        block_size=64,
        dtype=torch.float16,
    )
