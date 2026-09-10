"""pytest test for example_simtvf_rng.py — Ascend SIMT Philox RNG."""

import pytest
import torch
import tilelang

from example_simtvf_rng import rng_fill


@pytest.mark.parametrize(
    "target",
    [
        "ascend",
    ],
)
def test_simtvf_rng(target):
    N = 8192
    kernel = tilelang.compile(rng_fill(N), target=target)
    device = torch.device("npu")

    U = torch.empty(N, dtype=torch.float32, device=device)
    Nr = torch.empty(N, dtype=torch.float32, device=device)
    I = torch.empty(N, dtype=torch.uint32, device=device)
    kernel(U, Nr, I)
    torch.npu.synchronize()

    assert torch.isfinite(U).all()
    assert torch.isfinite(Nr).all()
    assert (U >= 0).all() and (U < 1).all()
    # Per-element seq -> independent streams, so values should be distinct.
    assert U.unique().numel() > N // 2
    # Sanity on distribution moments (loose bounds).
    assert abs(U.mean().item() - 0.5) < 0.1
    assert abs(Nr.mean().item()) < 0.15
    assert abs(Nr.std().item() - 1.0) < 0.2


if __name__ == "__main__":
    test_simtvf_rng("ascend")
    print("PASS: test_simtvf_rng (ascend)")
