import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepseek_mhc"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_mhc_post  # noqa: E402
import example_mhc_pre  # noqa: E402


@tilelang.testing.requires_rocm
def test_mhc_pre():
    """Validate the wave-aware BF16 mHC pre path on ROCm."""
    test_data = example_mhc_pre.generate_test_data(n=32, hc_mult=4, hidden_size=256)

    actual = example_mhc_pre.mhc_pre(**test_data)
    expected = example_mhc_pre.mhc_pre_ref(**test_data)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_rocm
def test_mhc_post():
    """Validate the backend-neutral mHC post path on ROCm."""
    test_data = example_mhc_post.generate_test_data(n=8, h=256, hc_mult=4)
    output = torch.empty_like(test_data["residual"])

    example_mhc_post.mhc_post_tilelang(
        test_data["comb_res_mix"],
        test_data["residual"],
        test_data["post_layer_mix"].squeeze(-1),
        test_data["x"],
        output,
        test_data["residual"].shape[-2],
        test_data["residual"].shape[-1],
    )
    expected = example_mhc_post.mhc_post_ref(**test_data)

    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
