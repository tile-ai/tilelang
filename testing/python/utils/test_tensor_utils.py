import pytest
import torch

from tilelang.utils.tensor import torch_assert_close


@pytest.mark.parametrize("transpose", [False, True], ids=["contiguous", "transposed"])
def test_torch_assert_close_allows_mismatch_within_budget(transpose):
    """Accept one mismatch within budget for contiguous and transposed tensors."""
    expected = torch.zeros((2, 3), device="cpu")
    actual = expected.clone()
    if transpose:
        actual, expected = actual.t(), expected.t()
    actual[1, 0] = 1

    assert torch_assert_close(actual, expected, rtol=0, atol=0, max_mismatched_ratio=0.2)


@pytest.mark.parametrize("transpose", [False, True], ids=["contiguous", "transposed"])
def test_torch_assert_close_reports_mismatch_beyond_budget(transpose):
    """Report the logical first index and count when mismatches exceed the budget."""
    expected = torch.zeros((2, 3), device="cpu")
    actual = expected.clone()
    if transpose:
        actual, expected = actual.t(), expected.t()
    actual[1, 0] = 1
    actual[-1, -1] = 2

    with pytest.raises(AssertionError, match=r"First mismatch at index \[1, 0\]") as error:
        torch_assert_close(actual, expected, rtol=0, atol=0, max_mismatched_ratio=0.2)
    assert "Too many mismatched elements: 2 > 1" in str(error.value)


if __name__ == "__main__":
    pytest.main([__file__])
