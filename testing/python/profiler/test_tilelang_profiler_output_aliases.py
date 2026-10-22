import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.profiler import Profiler, TensorSupplyType


def _make_mutating_kernel():
    @T.prim_func
    def main(A: T.Tensor((16,), T.float32), B: T.Tensor((16,), T.float32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(16):
                A[i] = A[i] + 1
                B[i] = A[i]

    return main


@pytest.fixture(scope="module")
def mutating_kernel():
    return tilelang.compile(_make_mutating_kernel(), out_idx=[1], target="cuda")


def _check(profiler, method, reference, inputs):
    if method == "assert_allclose":
        profiler.assert_allclose(reference, input_tensors=inputs, atol=0, rtol=0, max_mismatched_ratio=0)
    else:
        profiler.manual_assert_close(
            reference,
            input_tensors=inputs,
            manual_check_prog=lambda actual, expected: torch.testing.assert_close(actual, expected, atol=0, rtol=0),
        )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("method", ["assert_allclose", "manual_assert_close"])
@pytest.mark.parametrize("view", [False, True])
def test_checks_reject_mutated_reference_alias(mutating_kernel, method, view):
    inputs = [torch.zeros(16, device="cuda")]

    def reference(a):
        return a.view_as(a) if view else a

    with pytest.raises(AssertionError):
        _check(mutating_kernel.get_profiler(), method, reference, inputs)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("method", ["assert_allclose", "manual_assert_close"])
def test_checks_accept_correct_reference(mutating_kernel, method):
    inputs = [torch.zeros(16, device="cuda")]
    _check(mutating_kernel.get_profiler(), method, lambda a: a + 1, inputs)


@tilelang.testing.requires_cuda
def test_reference_snapshot_with_independent_streams(mutating_kernel):
    reference_stream = torch.cuda.Stream()
    kernel_stream = torch.cuda.Stream()
    inputs = [torch.zeros(16, device="cuda")]

    def reference(a):
        with torch.cuda.stream(reference_stream):
            a.fill_(3)
        return a.view_as(a)

    def run(a):
        with torch.cuda.stream(kernel_stream):
            return mutating_kernel(a)

    profiler = mutating_kernel.get_profiler().with_default_adapter(run)
    with pytest.raises(AssertionError):
        _check(profiler, "assert_allclose", reference, inputs)


@tilelang.testing.requires_cuda
def test_consistency_check_rejects_reused_kernel_output():
    kernel = tilelang.compile(_make_mutating_kernel(), target="cuda")
    a = torch.zeros(16, device="cuda")
    output = torch.empty_like(a)

    def run():
        kernel(a, output)
        return output

    profiler = Profiler([], [], TensorSupplyType.Auto, run)
    with pytest.raises(AssertionError, match="result is not consistent"):
        profiler.assert_consistent(repeat=2)


@pytest.mark.parametrize("container", [list, tuple])
def test_checks_preserve_optional_outputs(container):
    a = torch.zeros(16)

    def run(value):
        value.add_(1)
        return container((value, None))

    profiler = Profiler([], [], TensorSupplyType.Auto, run)
    with pytest.raises(AssertionError):
        profiler.assert_allclose(
            lambda value: container((value.view_as(value), None)),
            input_tensors=[a],
            atol=0,
            rtol=0,
            max_mismatched_ratio=0,
        )


def test_checks_accept_no_outputs():
    profiler = Profiler([], [], TensorSupplyType.Auto, lambda: None)
    profiler.assert_allclose(lambda: None, input_tensors=[])


@pytest.mark.parametrize("container", [list, tuple, dict])
@pytest.mark.parametrize("mutate", [False, True])
def test_manual_check_snapshots_nested_outputs(container, mutate):
    value = torch.zeros(4)

    def wrap(tensor):
        nested = ([tensor.view_as(tensor)], None)
        return {"nested": nested} if container is dict else container((nested,))

    def run(tensor):
        if mutate:
            tensor.add_(1)
        return wrap(tensor)

    def check(actual, expected):
        assert type(actual) is container
        assert type(expected) is container
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    profiler = Profiler([], [], TensorSupplyType.Auto, run)
    if mutate:
        with pytest.raises(AssertionError):
            profiler.manual_assert_close(wrap, input_tensors=[value], manual_check_prog=check)
    else:
        profiler.manual_assert_close(wrap, input_tensors=[value], manual_check_prog=check)


@pytest.mark.parametrize("record_type", ["namedtuple", "torch_return"])
@pytest.mark.parametrize("mutate", [False, True])
def test_manual_check_preserves_output_record(record_type, mutate):
    from collections import namedtuple

    record = namedtuple("Output", ["values", "indices"])
    value = torch.zeros(4)

    def wrap(tensor):
        fields = (tensor.view_as(tensor), None)
        return record(*fields) if record_type == "namedtuple" else torch.return_types.max(fields)

    def run(tensor):
        if mutate:
            tensor.add_(1)
        return wrap(tensor)

    def check(actual, expected):
        assert type(actual) is type(expected)
        torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
        assert expected.indices is None

    profiler = Profiler([], [], TensorSupplyType.Auto, run)
    if mutate:
        with pytest.raises(AssertionError, match="Mismatched elements"):
            profiler.manual_assert_close(wrap, input_tensors=[value], manual_check_prog=check)
    else:
        profiler.manual_assert_close(wrap, input_tensors=[value], manual_check_prog=check)
