"""Compilation coverage for asynchronous Sunmmio RSRAM transpose."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.utils.target as target_utils
from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    find_async_op_lines,
    lower_sunmmio_kernel_to_device_tir,
    validate_sunmmio_codegen_with_npuir_opt,
)
from tilelang import tvm
from tilelang.layout import make_zn_layout, make_zz_layout


tilelang.env.disable_cache()


TRANSPOSE_CONFIGS = [
    pytest.param(64, 64, "bfloat16", "zz", id="bfloat16-zz-64x64"),
    pytest.param(64, 128, "bfloat16", "zn", id="bfloat16-zn-64x128"),
    pytest.param(64, 128, "float32", "zz", id="float32-zz-64x128"),
]


@pytest.fixture
def _sunmmio_region_validation_guard():
    old_value = target_utils.ENABLE_SUNMMIO_REGION_VALIDATION

    try:
        yield
    finally:
        target_utils.set_sunmmio_region_validation(old_value)


@target("Sunmmio")
def mesh_transpose_kernel(
    m,
    n,
    dtype,
    layout_family,
    control_flow="plain",
    expect_transposed=True,
):
    """Build a replicated transpose with matching DRAM and RSRAM layouts."""
    placement = T.placement.replicated()
    src_layout = make_zz_layout((m, n)) if layout_family == "zz" else make_zn_layout((m, n), [0, 1], (32, 32))
    transposed_layout = make_zz_layout((n, m)) if layout_family == "zz" else make_zn_layout((n, m), [0, 1], (32, 32))
    output_shape = (n, m) if expect_transposed else (m, n)
    output_layout = transposed_layout if expect_transposed else src_layout

    @T.prim_func
    def main(
        a: T.MeshTensor((m, n), placement, dtype, layout=src_layout),
        b: T.MeshTensor(output_shape, placement, dtype, layout=output_layout),
    ):
        with T.Kernel():
            src = T.alloc_shared((m, n), dtype, scope="shared.rsram")
            dst = T.alloc_shared((n, m), dtype, scope="shared.rsram")
            if layout_family == "zn":
                T.annotate_layout({src: make_zn_layout((m, n), [0, 1], (32, 32))})
            T.copy(a, src)

            if control_flow == "loop":
                # Each loop iteration performs a round trip. The odd case adds
                # one final transpose, while the even case leaves src unchanged.
                for _ in T.serial(2):
                    T.transpose(src, dst)
                    T.transpose(dst, src)
                if expect_transposed:
                    T.transpose(src, dst)
            else:
                T.transpose(src, dst)

            if expect_transposed:
                T.copy(dst, b)
            else:
                T.copy(src, b)

    return main


@target("Sunmmio")
def mesh_transpose_order_kernel(source_constraint_first):
    """Build the same transpose layout constraints in either source order."""
    size = 64
    dtype = "bfloat16"
    placement = T.placement.replicated()
    zn_layout = make_zn_layout((size, size), [0, 1], (32, 32))

    @T.prim_func
    def main(
        a: T.MeshTensor((size, size), placement, dtype, layout=zn_layout),
        b: T.MeshTensor((size, size), placement, dtype, layout=zn_layout),
    ):
        with T.Kernel():
            zn_src = T.alloc_shared((size, size), dtype, scope="shared.rsram")
            src = T.alloc_shared((size, size), dtype, scope="shared.rsram")
            dst = T.alloc_shared((size, size), dtype, scope="shared.rsram")
            T.annotate_layout({zn_src: zn_layout})
            T.copy(a, zn_src)

            if source_constraint_first:
                T.transpose(zn_src, src)
                T.transpose(src, dst)
            else:
                T.transpose(src, dst)
                T.transpose(zn_src, src)

            T.copy(dst, b)

    return main


@target("Sunmmio")
def mesh_transpose_scope_kernel(global_operand):
    """Build a transpose with one operand intentionally in global memory."""
    size = 64
    dtype = "bfloat16"
    placement = T.placement.replicated()
    layout = make_zz_layout((size, size))

    @T.prim_func
    def main(
        a: T.MeshTensor((size, size), placement, dtype, layout=layout),
        b: T.MeshTensor((size, size), placement, dtype, layout=layout),
    ):
        with T.Kernel():
            local = T.alloc_shared((size, size), dtype, scope="shared.rsram")
            if global_operand == "source":
                T.transpose(a, local)
                T.copy(local, b)
            else:
                T.copy(a, local)
                T.transpose(local, b)

    return main


@pytest.mark.parametrize("enable_region_validation", [True, False], ids=["region_validation_on", "region_validation_off"])
@pytest.mark.parametrize(("m", "n", "dtype", "layout_family"), TRANSPOSE_CONFIGS)
def test_transpose_codegen_matrix(
    tmp_path,
    m,
    n,
    dtype,
    layout_family,
    enable_region_validation,
    _sunmmio_region_validation_guard,
):
    target_utils.set_sunmmio_region_validation(enable_region_validation)

    src = validate_sunmmio_codegen_with_npuir_opt(
        mesh_transpose_kernel(m, n, dtype, layout_family),
        tmp_path,
        mlir_filename=(f"transpose_{dtype}_{layout_family}_{m}x{n}_rv_{'on' if enable_region_validation else 'off'}.mlir"),
        expected_tokens=(
            "suvm.copy_async",
            "suvm.transpose_async",
            "suvm.sync",
        ),
    )

    transpose_lines = find_async_op_lines(src, "suvm.transpose_async")
    assert len(transpose_lines) == 1
    assert "#suvm.unit<odma1>" in transpose_lines[0]


@pytest.mark.parametrize("enable_region_validation", [True, False], ids=["region_validation_on", "region_validation_off"])
def test_transpose_loop_codegen(
    tmp_path,
    enable_region_validation,
    _sunmmio_region_validation_guard,
):
    target_utils.set_sunmmio_region_validation(enable_region_validation)

    src = validate_sunmmio_codegen_with_npuir_opt(
        mesh_transpose_kernel(
            64,
            128,
            "float32",
            "zn",
            control_flow="loop",
            expect_transposed=False,
        ),
        tmp_path,
        mlir_filename=f"transpose_loop_rv_{'on' if enable_region_validation else 'off'}.mlir",
        expected_tokens=(
            "scf.for",
            "suvm.copy_async",
            "suvm.transpose_async",
            "suvm.sync",
        ),
    )

    transpose_lines = find_async_op_lines(src, "suvm.transpose_async")
    assert len(transpose_lines) == 2
    assert all("#suvm.unit<odma1>" in line for line in transpose_lines)


@pytest.mark.parametrize(
    "source_constraint_first",
    [False, True],
    ids=["dependent_first", "constraint_first"],
)
def test_transpose_layout_inference_is_order_independent(
    tmp_path,
    source_constraint_first,
):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mesh_transpose_order_kernel(source_constraint_first),
        tmp_path,
        mlir_filename=(f"transpose_order_{'constraint_first' if source_constraint_first else 'dependent_first'}.mlir"),
        expected_tokens=(
            "suvm.copy_async",
            "suvm.transpose_async",
            "suvm.sync",
        ),
    )

    transpose_lines = find_async_op_lines(src, "suvm.transpose_async")
    assert len(transpose_lines) == 2
    assert all("#suvm.unit<odma1>" in line for line in transpose_lines)


def test_transpose_rejects_vector_element_dtype():
    with pytest.raises(tvm.error.InternalError, match="requires scalar element types"):
        lower_sunmmio_kernel_to_device_tir(mesh_transpose_kernel(64, 64, "float32x2", "zz"))


def test_transpose_rejects_global_source():
    with pytest.raises(tvm.error.InternalError, match="source must use shared.rsram"):
        lower_sunmmio_kernel_to_device_tir(mesh_transpose_scope_kernel("source"))


def test_transpose_rejects_global_destination():
    with pytest.raises(tvm.error.InternalError, match="destination must use shared.rsram"):
        lower_sunmmio_kernel_to_device_tir(mesh_transpose_scope_kernel("destination"))
