"""Explicit copies guard their actual pointer alignment without stronger types."""

import pytest
from tvm_tirx import ir, tirx
import tilelang.testing
import tilelang.language as T
from testing.python.transform.test_async_width_cuda_execute import execute_copy


def unaligned_explicit_kernel(rows):
    source_parameter = ir.Var("source_parameter", tirx.TensorType((rows, 8), "float16", data_alignment=1))
    output_parameter = ir.Var("output_parameter", tirx.TensorType((32, 8), "float16", data_alignment=16))

    @T.prim_func
    def main(A: source_parameter, Out: output_parameter, valid_rows: T.int32):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((32, 8), T.float16)
            T.async_copy(A[0, 0], shared)
            T.ptx_wait_group(0)
            T.copy(shared, Out)

    return main


@pytest.mark.parametrize("rows,shift,disable_vectorize", [(32, 0, False), (32, 1, False), (32, 1, True), (29, 1, False), (29, 1, True)])
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_explicit_async_checks_actual_alignment_and_keeps_all_bytes(rows, shift, disable_vectorize, tmp_path):
    function = unaligned_explicit_kernel(rows)
    assert int(function.params[0].ty.data_alignment) == 1
    execute_copy(function, (rows, 8), (32, 8), shift, [rows], 0.0, disable_vectorize, True, tmp_path)
    assert int(function.params[0].ty.data_alignment) == 1
