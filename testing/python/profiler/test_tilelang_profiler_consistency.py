import pytest
import torch

from tilelang.profiler import Profiler, TensorSupplyType


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2)])
def test_consistent_single_tensor_output(shape):
    instance = Profiler([], [], TensorSupplyType.Auto, lambda: torch.ones(shape))

    instance.assert_consistent(repeat=2)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("counts", [(0, 1), (1, 0), (1, 2), (2, 1)])
def test_inconsistent_output_count(container, counts):
    outputs = iter(container(torch.ones(2) for _ in range(count)) for count in counts)
    instance = Profiler([], [], TensorSupplyType.Auto, lambda: next(outputs))

    with pytest.raises(AssertionError, match="Number of kernel outputs"):
        instance.assert_consistent(repeat=1)


@pytest.mark.parametrize(
    "shapes, output_index",
    [(((2, 2), (1, 2)), 0), (((0, 2), (0, 3)), 0), (((2,), (2, 1)), 0), (((2, 2), (1, 2)), 1)],
)
def test_inconsistent_tensor_shape(shapes, output_index):
    # Equal values must not hide a changed shape through zip truncation or broadcasting.
    outputs = iter(torch.ones(shape) if output_index == 0 else [torch.ones(1), torch.ones(shape)] for shape in shapes)
    instance = Profiler([], [], TensorSupplyType.Auto, lambda: next(outputs))

    with pytest.raises(AssertionError, match=f"Shape of kernel output {output_index} is not consistent"):
        instance.assert_consistent(repeat=1)


def test_inconsistent_tensor_values():
    outputs = iter([torch.ones(2), torch.zeros(2)])
    instance = Profiler([], [], TensorSupplyType.Auto, lambda: next(outputs))

    with pytest.raises(AssertionError, match="result is not consistent"):
        instance.assert_consistent(repeat=1)


@pytest.mark.parametrize("container", [None, list, tuple])
def test_inconsistent_reused_output_storage(container):
    output = torch.zeros(2)

    def run():
        output.add_(1)
        return output if container is None else container([output])

    instance = Profiler([], [], TensorSupplyType.Auto, run)

    with pytest.raises(AssertionError, match="result is not consistent"):
        instance.assert_consistent(repeat=1)
