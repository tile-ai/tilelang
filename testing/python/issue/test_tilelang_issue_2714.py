"""Regression tests for GitHub issue #2714.

T.all_of/T.any_of over a row of swizzled shared memory must scan logical row
elements, not the contiguous physical addresses after the first swizzled base.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import (
    Layout,
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
    make_swizzled_layout,
)

THREADS = 128

SHAPES = [
    pytest.param((8, 128), id="8x128"),
    pytest.param((16, 128), id="16x128"),
    pytest.param((24, 128), id="24x128"),
]
ELEMENT_TYPES = ["int8", "int8x2", "int16", "float16", "int32", "float32"]
LAYOUT_FNS = [
    pytest.param(make_swizzled_layout, id="auto"),
    pytest.param(make_quarter_bank_swizzled_layout, id="32B"),
    pytest.param(make_half_bank_swizzled_layout, id="64B"),
    pytest.param(make_full_bank_swizzled_layout, id="128B"),
]
SLICE_RANGES = [
    pytest.param((0, 128), id="full"),
    pytest.param((8, 40), id="slice8-40"),
]
# The CUDA and HIP backends share the lowering and emit the same
# tl::Logical[Vector]ReduceMap template, so both run the whole matrix.
TARGETS = [
    pytest.param("cuda", id="cuda", marks=list(tilelang.testing.requires_cuda.marks())),
    pytest.param("hip", id="hip", marks=list(tilelang.testing.requires_rocm.marks())),
]


def _xfail_packed_element_type_on_hip(request, target, element_type):
    """HIP cannot emit a vector access to a buffer of packed elements.

    Applied here rather than as a mark: the condition spans two parametrize
    axes.
    """
    if target == "hip" and tilelang.DataType(element_type).lanes > 1:
        request.applymarker(
            pytest.mark.xfail(
                reason="HIP emits a >4-lane Ramp for vector accesses to packed-element buffers",
                raises=tvm.error.InternalError,
                strict=True,
            )
        )


def _compile(func, target):
    cache_was_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    try:
        return tilelang.compile(func, out_idx=[1], target=target)
    finally:
        if cache_was_enabled:
            tilelang.enable_cache()


def _derive_vectorization(element_type, slice_range):
    dtype = tilelang.DataType(element_type)
    vector_elements = 1
    max_vector_elements = 128 // (dtype.bits * dtype.lanes)
    while vector_elements <= max_vector_elements // 2:
        vector_elements *= 2

    slice_begin, slice_end = slice_range
    extent = slice_end - slice_begin
    while vector_elements > 1 and (slice_begin % vector_elements != 0 or extent % vector_elements != 0):
        vector_elements //= 2

    vector_lanes = dtype.lanes * vector_elements
    vector_count = extent // vector_elements
    return vector_lanes, vector_count


def _make_random_input(shape, reduce_kind, slice_range, element_type):
    dtype = tilelang.DataType(element_type)
    lanes = dtype.lanes
    scalar_type = element_type.removesuffix(f"x{lanes}") if lanes > 1 else element_type
    data = torch.randint(-4, 5, shape, dtype=tilelang.DataType(scalar_type).as_torch(), device="cuda")
    selected_rows = torch.randperm(shape[0], device="cuda")[: shape[0] // 4]
    slice_begin, slice_end = slice_range
    scalar_begin = slice_begin * lanes
    scalar_end = slice_end * lanes
    data[selected_rows, scalar_begin:scalar_end] = 0 if reduce_kind == "any_of" else 1
    return data


def _torch_logical_reduce(data, reduce_kind, slice_range, lanes=1):
    slice_begin, slice_end = slice_range
    predicate = data[:, slice_begin * lanes : slice_end * lanes] != 0
    reduce_fn = torch.any if reduce_kind == "any_of" else torch.all
    return reduce_fn(predicate, dim=1).to(torch.int8)


def _logical_reduce_kernel(reduce_kind, shape, element_type, layout_fn, slice_range):
    assert reduce_kind in ("all_of", "any_of")
    use_any = reduce_kind == "any_of"
    slice_begin, slice_end = slice_range
    dtype = tilelang.DataType(element_type)
    lanes = dtype.lanes
    scalar_type = element_type.removesuffix(f"x{lanes}") if lanes > 1 else element_type
    input_shape = (shape[0], shape[1] * lanes)

    @T.macro
    def load_input(A, r, c):
        if lanes == 1:
            return A[r, c]
        else:
            return A[r, T.Ramp(c * lanes, 1, lanes)]

    @T.macro
    def logical_reduce(shared, r):
        if use_any:
            return T.any_of(shared[r, slice_begin:slice_end])
        else:
            return T.all_of(shared[r, slice_begin:slice_end])

    @T.prim_func
    def main(A: T.Tensor(input_shape, scalar_type), Out: T.Tensor((shape[0],), "int8")):
        with T.Kernel(1, threads=THREADS):
            shared = T.alloc_shared(shape, element_type)
            T.annotate_layout({shared: layout_fn(shared)})

            for r, c in T.Parallel(shape[0], shape[1]):
                shared[r, c] = load_input(A, r, c)
            T.sync_threads()

            for r in T.Parallel(shape[0]):
                Out[r] = T.cast(logical_reduce(shared, r), "int8")

    return main


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("reduce_kind", ["any_of", "all_of"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
@pytest.mark.parametrize("layout_fn", LAYOUT_FNS)
@pytest.mark.parametrize("slice_range", SLICE_RANGES)
def test_swizzled_shared_row_scans_logical_elements(
    request,
    target,
    reduce_kind,
    shape,
    element_type,
    layout_fn,
    slice_range,
):
    _xfail_packed_element_type_on_hip(request, target, element_type)
    kernel = _compile(
        _logical_reduce_kernel(
            reduce_kind,
            shape,
            element_type,
            layout_fn,
            slice_range,
        ),
        target,
    )
    lanes = tilelang.DataType(element_type).lanes
    input_shape = (shape[0], shape[1] * lanes)
    vector_lanes, vector_count = _derive_vectorization(element_type, slice_range)
    data = _make_random_input(input_shape, reduce_kind, slice_range, element_type)

    expected = _torch_logical_reduce(data, reduce_kind, slice_range, lanes)
    torch.testing.assert_close(kernel(data), expected)

    source = kernel.get_kernel_source()
    assert f"tl::{reduce_kind.removesuffix('_of').title()}(" not in source
    is_any = str(reduce_kind == "any_of").lower()
    assert re.search(
        rf"tl::LogicalVectorReduceMap<{is_any}, [^,]+, {vector_lanes}>\({vector_count}",
        source,
    )
    vector_words = tilelang.DataType(element_type).bits * vector_lanes // 32
    assert re.search(rf"return \*\([A-Za-z_][A-Za-z0-9_]*{vector_words}\*\)", source)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("reduce_kind", ["any_of", "all_of"])
def test_padded_layout_logical_reduce(target, reduce_kind):
    def padded_layout(_):
        return Layout((4, 17), lambda i, j: [i, j * 2])

    kernel = _compile(_logical_reduce_kernel(reduce_kind, (4, 17), "int8", padded_layout, (0, 17)), target)
    data = _make_random_input((4, 17), reduce_kind, (0, 17), "int8")
    expected = _torch_logical_reduce(data, reduce_kind, (0, 17))
    torch.testing.assert_close(kernel(data), expected)


if __name__ == "__main__":
    tilelang.testing.main()
