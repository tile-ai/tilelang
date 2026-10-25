import re

import pytest
import torch

from tilelang.utils.tensor import torch_assert_close


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.float32], ids=["bool", "uint8", "float32"])
def test_torch_assert_close_accepts_identical_values(dtype):
    """Accept equal values without requiring subtraction in the input dtype."""
    expected = torch.tensor([True, False], dtype=dtype)

    assert torch_assert_close(expected.clone(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.float32], ids=["bool", "uint8", "float32"])
def test_torch_assert_close_accepts_allowed_mismatch(dtype):
    """Keep the mismatch budget unchanged for Boolean, unsigned, and float inputs."""
    expected = torch.tensor([True, False, False, True], dtype=dtype)
    actual = expected.clone()
    actual[0] = False

    assert torch_assert_close(actual, expected, rtol=0, atol=0, max_mismatched_ratio=0.25)


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.float32], ids=["bool", "uint8", "float32"])
def test_torch_assert_close_reports_rejected_mismatches(dtype):
    """Raise a comparison assertion with correct indices, counts, and differences."""
    expected = torch.tensor([True, False, False, True], dtype=dtype)
    actual = torch.tensor([False, False, True, True], dtype=dtype)

    with pytest.raises(AssertionError, match=r"First mismatch at index \[0\]") as error:
        torch_assert_close(actual, expected, rtol=0, atol=0, max_mismatched_ratio=0.25)

    message = str(error.value)
    assert "Too many mismatched elements: 2 > 1" in message
    assert re.search(r"Greatest absolute difference: 1(?:\.0)?,", message)
    assert f"\nLHS: {actual}" in message
    assert f"\nRHS: {expected}" in message


@pytest.mark.parametrize("actual, expected", [(0, 1), (0, 255), (255, 0)])
def test_torch_assert_close_uint8_differences_do_not_wrap(actual, expected):
    """Report unsigned differences without wraparound, including the full range."""
    with pytest.raises(AssertionError) as error:
        torch_assert_close(
            torch.tensor([actual], dtype=torch.uint8),
            torch.tensor([expected], dtype=torch.uint8),
            rtol=0,
            atol=0,
            max_mismatched_ratio=0,
        )

    message = str(error.value)
    difference = abs(actual - expected)
    assert f"Greatest absolute difference: {difference}," in message
    relative = re.search(r"Greatest relative difference: ([^\n]+)", message)
    assert relative is not None
    assert float(relative.group(1)) == pytest.approx(difference / (abs(expected) + 1e-12))


if __name__ == "__main__":
    pytest.main([__file__])
