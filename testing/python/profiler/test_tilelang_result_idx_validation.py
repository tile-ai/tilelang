import pytest

from tilelang.jit.adapter import BaseKernelAdapter
from tilelang.profiler import Profiler
from tilelang.utils.tensor import TensorSupplyType


class _TestAdapter(BaseKernelAdapter):
    def _convert_torch_func(self):
        return lambda: None


_PARAMS = [object(), object(), object()]


def _legalize_with_adapter(result_idx):
    return _TestAdapter(None, _PARAMS, result_idx).result_idx


def _legalize_with_profiler(result_idx):
    return Profiler(_PARAMS, result_idx, TensorSupplyType.Integer).result_idx


@pytest.mark.parametrize(
    "legalize",
    [_legalize_with_adapter, _legalize_with_profiler],
    ids=["adapter", "profiler"],
)
@pytest.mark.parametrize("as_list", [False, True], ids=["int", "list"])
@pytest.mark.parametrize(
    ("index", "expected"),
    [(-3, [0]), (-1, [2]), (0, [0]), (2, [2])],
)
def test_result_idx_accepts_valid_boundaries(legalize, as_list, index, expected):
    result_idx = [index] if as_list else index

    assert legalize(result_idx) == expected


@pytest.mark.parametrize(
    "legalize",
    [_legalize_with_adapter, _legalize_with_profiler],
    ids=["adapter", "profiler"],
)
@pytest.mark.parametrize("as_list", [False, True], ids=["int", "list"])
@pytest.mark.parametrize("index", [-4, 3])
def test_result_idx_rejects_out_of_range_boundaries(legalize, as_list, index):
    result_idx = [index] if as_list else index

    with pytest.raises(ValueError, match=r"between -3 and 2"):
        legalize(result_idx)


@pytest.mark.parametrize(
    "legalize",
    [_legalize_with_adapter, _legalize_with_profiler],
    ids=["adapter", "profiler"],
)
@pytest.mark.parametrize("index", [0.5, "0"])
def test_result_idx_rejects_non_integer_list_elements(legalize, index):
    with pytest.raises(ValueError, match="list of integers"):
        legalize([index])
