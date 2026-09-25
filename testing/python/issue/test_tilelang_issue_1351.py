import importlib.util
from pathlib import Path

import pytest
import torch

import tilelang.testing


_SOURCE = Path(__file__).resolve().parents[3] / "examples" / "deepseek_v32" / "topk_selector.py"
_SPEC = importlib.util.spec_from_file_location("topk_selector_issue_1351", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "distribution,seq_len,topk,partial",
    [
        ("low_bits", 32768, 2048, False),
        ("low_bits", 32768, 2048, True),
        ("negative_low_bits", 32768, 2048, False),
        ("equal", 4095, 256, False),
        ("equal", 4096, 256, False),
        ("equal", 4097, 256, False),
        ("equal", 32768, 2048, False),
        ("equal", 32768, 2048, True),
        ("low_bits", 8193, 1, False),
        ("low_bits", 4097, 4097, False),
        ("boundary", 32768, 2048, False),
        ("random", 32768, 2048, False),
        ("random", 8193, 256, True),
    ],
)
def test_topk_candidate_overflow(distribution, seq_len, topk, partial):
    torch.manual_seed(42)
    batch = 2
    if distribution in ("low_bits", "negative_low_bits"):
        bits = torch.randint(0, 1024, (batch, seq_len), dtype=torch.int32, device="cuda")
        values = (bits | 0x3F900000).view(torch.float32)
        if distribution == "negative_low_bits":
            values = -values
    elif distribution == "equal":
        values = torch.ones((batch, seq_len), device="cuda")
    elif distribution == "boundary":
        # Exactly k - 1 larger values, followed by an oversized tied bucket.
        values = torch.ones((batch, seq_len), device="cuda")
        values[:, : topk - 1] = 2.0
        values = values[:, torch.randperm(seq_len, device="cuda")].contiguous()
    else:
        values = torch.randn((batch, seq_len), device="cuda")
    starts = torch.tensor([0, 257] if partial else [0, 0], dtype=torch.int32, device="cuda")
    ends = torch.tensor([seq_len - 137, seq_len - 31] if partial else [seq_len, seq_len], dtype=torch.int32, device="cuda")

    indices = _MODULE.tl_topk(values, starts, ends, topk)
    torch.cuda.synchronize()
    assert indices.shape == (batch, topk)
    assert indices.dtype == torch.int32
    assert ((indices >= starts[:, None]) & (indices < ends[:, None])).all().item()
    sorted_indices = indices.sort(dim=-1).values
    assert (sorted_indices[:, 1:] != sorted_indices[:, :-1]).all().item()

    positions = torch.arange(seq_len, device="cuda")[None, :]
    valid = (positions >= starts[:, None]) & (positions < ends[:, None])
    reference = values.masked_fill(~valid, float("-inf")).topk(topk, dim=-1).values
    selected = values.gather(1, indices.long()).sort(dim=-1, descending=True).values
    # This operator selects existing FP32 values; no arithmetic tolerance is needed.
    torch.testing.assert_close(selected, reference, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
