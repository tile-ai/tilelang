import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepseek_v32"
sys.path.insert(0, str(_EXAMPLE_DIR))

from topk_selector import tl_topk  # noqa: E402


@tilelang.testing.requires_rocm
def test_deepseek_v32_topk_selector():
    torch.manual_seed(0)
    batch, seq_len, topk = 2, 4096, 256
    values = torch.randn(batch, seq_len, dtype=torch.float32, device="cuda")
    starts = torch.tensor([0, 257], dtype=torch.int32, device="cuda")
    ends = torch.tensor([seq_len, seq_len - 137], dtype=torch.int32, device="cuda")

    indices = tl_topk(values, starts, ends, topk)

    valid = torch.arange(seq_len, device=values.device).unsqueeze(0)
    valid = (valid >= starts.unsqueeze(1)) & (valid < ends.unsqueeze(1))
    reference = torch.topk(values.masked_fill(~valid, float("-inf")), topk, dim=-1).indices.to(torch.int32)

    torch.testing.assert_close(
        torch.sort(indices, dim=-1).values,
        torch.sort(reference, dim=-1).values,
        rtol=0,
        atol=0,
    )
