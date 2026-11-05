"""Run real aligned, misaligned, masked and partial-tail copies on CUDA."""

import hashlib
import json
import re

import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.testing
import tilelang.language as T


def unconditional_kernel():
    source_parameter = ir.Var("source_parameter", tirx.TensorType((32, 8), "float16", data_alignment=16))
    output_parameter = ir.Var("output_parameter", tirx.TensorType((32, 8), "float16", data_alignment=16))

    @T.prim_func
    def main(A: source_parameter, Out: output_parameter, valid_rows: T.int32):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((32, 8), T.float16)
            for row, column in T.Parallel(32, 8, prefer_async=True):
                shared[row, column] = A[row, column]
            for row, column in T.Parallel(32, 8):
                Out[row, column] = shared[row, column]

    return main


def uniform_flag_kernel():
    source_parameter = ir.Var("source_parameter", tirx.TensorType((32, 8), "float16", data_alignment=16))
    output_parameter = ir.Var("output_parameter", tirx.TensorType((32, 8), "float16", data_alignment=16))

    @T.prim_func
    def main(A: source_parameter, Out: output_parameter, enabled: T.int32):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((32, 8), T.float16)
            for row, column in T.Parallel(32, 8, prefer_async=True):
                shared[row, column] = T.if_then_else(enabled > 0, A[row, column], T.float16(0.0))
            for row, column in T.Parallel(32, 8):
                Out[row, column] = shared[row, column]

    return main


def staging_kernel(source_rows, alignment, fill):
    source_parameter = ir.Var("source_parameter", tirx.TensorType((source_rows, 8), "float16", data_alignment=alignment))
    output_parameter = ir.Var("output_parameter", tirx.TensorType((32, 8), "float16", data_alignment=16))

    @T.prim_func
    def main(A: source_parameter, Out: output_parameter, valid_rows: T.int32):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((32, 8), T.float16)
            for row, column in T.Parallel(32, 8, prefer_async=True):
                shared[row, column] = T.if_then_else(row < valid_rows, A[row, column], T.float16(fill))
            for row, column in T.Parallel(32, 8):
                Out[row, column] = shared[row, column]

    return main


def flat_tail_kernel():
    source_parameter = ir.Var("source_parameter", tirx.TensorType((257,), "float16", data_alignment=16))
    output_parameter = ir.Var("output_parameter", tirx.TensorType((288,), "float16", data_alignment=16))

    @T.prim_func
    def main(A: source_parameter, Out: output_parameter, valid_elements: T.int32):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((288,), T.float16)
            for index in T.Parallel(288, prefer_async=True):
                shared[index] = T.if_then_else(index < valid_elements, A[index], T.float16(0.0))
            for index in T.Parallel(288):
                Out[index] = shared[index]

    return main


def execute_copy(function, input_shape, output_shape, shift, limits, fill, disable_vectorize, expected_async, tmp_path):
    assert torch.cuda.is_available(), "The numerical suite requires its reserved CUDA device"
    device = torch.cuda.get_device_properties(0)
    arch = "sm_" + str(device.major) + str(device.minor)
    before = ir.save_json(function)
    (tmp_path / "input.json").write_text(before)
    kernel = tilelang.compile(
        function,
        target={"kind": "cuda", "arch": arch},
        target_host="c",
        execution_backend="tvm_ffi",
        pass_configs={"tirx.disable_vectorize": disable_vectorize},
    )
    source = kernel.get_kernel_source()
    (tmp_path / "kernel.cu").write_text(source)
    (tmp_path / "device.json").write_text(ir.save_json(kernel.adapter.device_mod))
    widths = [int(value) for value in re.findall(r"cp_async_gs(?:_conditional)?<\s*(\d+)\s*>", source)]
    assert all(width in (4, 8, 16) for width in widths), widths
    if expected_async is not None:
        assert bool(widths) is expected_async, source
    assert ir.save_json(function) == before

    input_count = 1
    output_count = 1
    for extent in input_shape:
        input_count *= extent
    for extent in output_shape:
        output_count *= extent
    storage = torch.arange(input_count + 32, device="cuda", dtype=torch.float32).to(torch.float16) * 0.25 + 1.5
    values = storage[shift : shift + input_count].reshape(input_shape)
    assert values.data_ptr() % 16 == (2 if shift else 0)
    source_before = storage.clone()
    trials = []
    for limit in limits:
        output_storage = torch.full((output_count + 32,), -71.0, device="cuda", dtype=torch.float16)
        output = output_storage[16 : 16 + output_count].reshape(output_shape)
        kernel(values, output, limit)
        torch.cuda.synchronize()
        expected = torch.full(output_shape, fill, device="cuda", dtype=torch.float16)
        expected[:limit] = values[:limit]
        assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8)), (limit, output, expected)
        assert torch.all(output_storage[:16] == -71.0) and torch.all(output_storage[-16:] == -71.0)
        assert torch.equal(storage.view(torch.uint8), source_before.view(torch.uint8))
        trials.append({"limit": limit, "all_output_bytes_equal": True, "source_unchanged": True, "output_guards_unchanged": True})
    exported = tmp_path / "executed-kernel.so"
    kernel.adapter.get_exportable_executable().export_library(str(exported), **kernel.adapter.get_export_kwargs())
    (tmp_path / "numeric.json").write_text(
        json.dumps(
            {
                "gpu": device.name,
                "compute_capability": [device.major, device.minor],
                "arch": arch,
                "declared_source_alignment": int(function.params[0].ty.data_alignment),
                "actual_source_address_mod_16": values.data_ptr() % 16,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "fill": repr(fill),
                "disable_vectorize": disable_vectorize,
                "cp_async_widths_bytes": widths,
                "trials": trials,
                "cuda_sdk_compilation": "Completed by the normal tilelang.compile tvm_ffi backend before successful execution",
                "exported_executable": {"path": str(exported), "sha256": hashlib.sha256(exported.read_bytes()).hexdigest()},
                "oracle": "Every output byte equals the selected input or exact fill bits; input and both output guard regions remain unchanged",
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "alignment,shift,source_rows,fill,disable_vectorize,expected_async",
    [
        # The actual parallel lowerer retains vec in `(tx * 8 + vec) % 256 // 8`.
        # That predicate has not proved lane independence at the injection point.
        (16, 0, 32, 0.0, False, False),
        (16, 0, 32, 0.0, True, False),
        (1, 0, 32, 0.0, False, False),
        (1, 1, 32, 0.0, False, False),
        (16, 0, 29, 0.0, False, False),
        (16, 0, 29, -0.0, False, False),
    ],
)
@tilelang.testing.requires_cuda
def test_cuda_staging_bytes(alignment, shift, source_rows, fill, disable_vectorize, expected_async, tmp_path):
    execute_copy(
        staging_kernel(source_rows, alignment, fill),
        (source_rows, 8),
        (32, 8),
        shift,
        [0, source_rows - 1, source_rows],
        fill,
        disable_vectorize,
        expected_async,
        tmp_path,
    )


@pytest.mark.parametrize("disable_vectorize", [False, True])
@tilelang.testing.requires_cuda
def test_cuda_partial_vector_tail(disable_vectorize, tmp_path):
    execute_copy(flat_tail_kernel(), (257,), (288,), 0, [0, 253, 257], 0.0, disable_vectorize, None, tmp_path)


@pytest.mark.parametrize("disable_vectorize", [False, True])
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_cuda_complete_aligned_async_copy(disable_vectorize, tmp_path):
    execute_copy(unconditional_kernel(), (32, 8), (32, 8), 0, [32], 0.0, disable_vectorize, True, tmp_path)


@pytest.mark.parametrize("disable_vectorize", [False, True])
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_cuda_uniform_conditional_async_zero_fill(disable_vectorize, tmp_path):
    execute_copy(uniform_flag_kernel(), (32, 8), (32, 8), 0, [0, 32], 0.0, disable_vectorize, True, tmp_path)
