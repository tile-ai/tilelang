import pytest
import torch

from tilelang.profiler import Profiler, TensorSupplyType


@pytest.mark.parametrize("raises", [False, True], ids=["success", "callback-error"])
def test_manual_check_preserves_print_options(raises):
    """Respect caller formatting inside and after a successful or failed check."""
    output = torch.ones(1)
    profiler = Profiler([], [], TensorSupplyType.Auto, adapter=lambda: output)
    probe = torch.arange(10)
    observed = []

    # Restore the process-wide options even when the regression assertion fails.
    with torch._tensor_str.printoptions(threshold=2, edgeitems=1):
        expected_format = str(probe)

        def check(actual, expected):
            """Observe caller formatting and optionally raise after validation."""
            torch.testing.assert_close(actual[0], expected[0])
            observed.append(str(probe))
            if raises:
                raise RuntimeError("callback failed")

        if raises:
            with pytest.raises(RuntimeError, match="callback failed"):
                profiler.manual_assert_close(lambda: output, input_tensors=[], manual_check_prog=check)
        else:
            profiler.manual_assert_close(lambda: output, input_tensors=[], manual_check_prog=check)

        assert observed == [expected_format]
        assert str(probe) == expected_format
