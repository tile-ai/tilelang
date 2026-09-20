"""pytest tests for example_atomic.py through the AscendC backend."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T

from example_atomic import atomic_add_gm_float, atomic_max_gm_float, atomic_min_gm_float


SHAPES = [
    (8, 256),
    (4, 128),
]

ATOMIC_CASES = [
    pytest.param("atomic_add", atomic_add_gm_float, 0.0, id="add"),
    pytest.param("atomic_max", atomic_max_gm_float, -float("inf"), id="max"),
    pytest.param("atomic_min", atomic_min_gm_float, float("inf"), id="min"),
]


def atomic_add_return_gm_float():
    @T.prim_func
    def main(
        counter: T.Tensor((1,), "float32"),
        previous: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1), T.SimtVF(threads=1):
            previous[0] = T.atomic_add(counter[0], 3.0, return_prev=True)

    return main


def atomic_max_return_gm_float():
    @T.prim_func
    def main(
        counter: T.Tensor((1,), "float32"),
        previous: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1), T.SimtVF(threads=1):
            previous[0] = T.atomic_max(counter[0], 11.0, return_prev=True)

    return main


def atomic_min_return_gm_float():
    @T.prim_func
    def main(
        counter: T.Tensor((1,), "float32"),
        previous: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1), T.SimtVF(threads=1):
            previous[0] = T.atomic_min(counter[0], 3.0, return_prev=True)

    return main


ATOMIC_RETURN_CASES = [
    pytest.param("atomic_add", atomic_add_return_gm_float, 7.0, 10.0, id="add"),
    pytest.param("atomic_max", atomic_max_return_gm_float, 7.0, 11.0, id="max"),
    pytest.param("atomic_min", atomic_min_return_gm_float, 7.0, 3.0, id="min"),
]


def _assert_atomic_source(kernel, op_name, target):
    source = kernel.get_kernel_source()
    expected = f"pto.{op_name}(" if target == "pto" else f"asc_{op_name}("
    assert expected in source, f"missing {expected} in generated {target} source"


def _run_atomic_gm_float(op_name, program_factory, initial, num_blocks, threads, target):
    kernel = tilelang.compile(
        program_factory(1, num_blocks, threads),
        target=target,
    )
    _assert_atomic_source(kernel, op_name, target)
    device = torch.device("npu")
    counter = torch.full((1,), initial, dtype=torch.float32, device=device)
    kernel(counter)
    torch.npu.synchronize()

    num_updates = num_blocks * threads
    expected = {
        "atomic_add": float(num_updates),
        "atomic_max": float(num_updates - 1),
        "atomic_min": 0.0,
    }[op_name]
    actual = counter.item()
    assert abs(actual - expected) < 0.5, f"{op_name}: {actual} != {expected}"


def _run_atomic_return_gm_float(op_name, program_factory, initial, expected, target):
    kernel = tilelang.compile(
        program_factory(),
        target=target,
    )
    _assert_atomic_source(kernel, op_name, target)
    device = torch.device("npu")
    counter = torch.full((1,), initial, dtype=torch.float32, device=device)
    previous = torch.empty((1,), dtype=torch.float32, device=device)
    kernel(counter, previous)
    torch.npu.synchronize()

    actual_previous = previous.item()
    actual_counter = counter.item()
    assert actual_previous == initial, f"{op_name}: previous={actual_previous} != {initial}"
    assert actual_counter == expected, f"{op_name}: counter={actual_counter} != {expected}"


@pytest.mark.parametrize(
    ("num_blocks", "threads"),
    SHAPES,
)
@pytest.mark.parametrize(
    ("op_name", "program_factory", "initial"),
    ATOMIC_CASES,
)
@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_atomic_gm_float(op_name, program_factory, initial, num_blocks, threads, target):
    _run_atomic_gm_float(
        op_name,
        program_factory,
        initial,
        num_blocks,
        threads,
        target,
    )


@pytest.mark.parametrize(
    ("op_name", "program_factory", "initial", "expected"),
    ATOMIC_RETURN_CASES,
)
@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_atomic_return_gm_float(op_name, program_factory, initial, expected, target):
    _run_atomic_return_gm_float(
        op_name,
        program_factory,
        initial,
        expected,
        target,
    )


if __name__ == "__main__":
    for target in ("ascend", "pto"):
        for op_name, program_factory, initial in [
            ("atomic_add", atomic_add_gm_float, 0.0),
            ("atomic_max", atomic_max_gm_float, -float("inf")),
            ("atomic_min", atomic_min_gm_float, float("inf")),
        ]:
            for num_blocks, threads in SHAPES:
                test_atomic_gm_float(
                    op_name,
                    program_factory,
                    initial,
                    num_blocks,
                    threads,
                    target,
                )
                print(f"PASS: test_atomic_gm_float target={target} op={op_name} blocks={num_blocks} threads={threads}")
        for op_name, program_factory, initial, expected in [
            ("atomic_add", atomic_add_return_gm_float, 7.0, 10.0),
            ("atomic_max", atomic_max_return_gm_float, 7.0, 11.0),
            ("atomic_min", atomic_min_return_gm_float, 7.0, 3.0),
        ]:
            test_atomic_return_gm_float(
                op_name,
                program_factory,
                initial,
                expected,
                target,
            )
            print(f"PASS: test_atomic_return_gm_float target={target} op={op_name}")
