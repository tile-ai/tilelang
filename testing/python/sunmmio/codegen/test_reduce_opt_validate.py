import os
import re
import warnings

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_aligned_row_major, make_row_major, make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)
from tilelang.tileview import make_tileview


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


REDUCE_IN_TILE_CASES = [
    ((32, 1024), 1, False),
    ((32, 1024), 0, False),
    ((1024, 32), 1, True),
    ((256, 128), 1, True),
    ((256, 128), 0, False),
    ((256, 128), 1, False),
    ((64, 256, 128), 0, True),
    ((32, 256, 128), 2, True),
    ((32, 256, 128), 1, False),
]

LOOSE_OPT_ARGS = ("--verify-each",)
STRICT_OPT_ARGS = ("--verify-each", "--suvm-to-llvm-pipeline")


def validate_sunmmio_codegen_loose(kernel, tmp_path, *, mlir_filename, expected_tokens=()):
    return validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=LOOSE_OPT_ARGS,
    )


def validate_sunmmio_codegen_strict(kernel, tmp_path, *, mlir_filename, expected_tokens=()):
    return validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=STRICT_OPT_ARGS,
    )


def _dram_input_layout(shape):
    if len(shape) >= 2:
        return make_zz_layout(shape, [len(shape) - 2, len(shape) - 1], (32, 32))
    return make_row_major(shape)


def _dram_reduce_output_layout(shape):
    return make_row_major(shape)


def _valid_extent(tile_index, block, total):
    start = tile_index * block
    remaining = total - start
    return T.min(block, remaining)


@target("Sunmmio")
def reduce_kernel_builder(
    shape,
    reduce_axis,
    dtype="bfloat16",
    clear=True,
    tile_size=None,
    out_dtype=None,
):
    shape = tuple(shape)
    out_dtype = out_dtype or dtype
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]
    out_shape = tuple(out_shape)

    shard_policy = T.placement.replicated()
    input_layout = _dram_input_layout(shape)
    output_layout = _dram_reduce_output_layout(out_shape)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=input_layout),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, out_dtype, layout=output_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, out_dtype, scope="shared.rsram")

            if tile_size is not None:
                T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, (-2, -1))})
            if len(shape) == 3:
                for bb in T.serial(shape[0]):
                    T.copy(A[bb, :, :], A_shared[bb, :, :])
            else:
                T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis, clear=clear)
            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def mixed_dtype_reduce_pipeline_builder():
    shape = (256, 128)
    out_shape = (256,)
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, "bfloat16", layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, "float32", layout=make_row_major(out_shape)),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, "bfloat16", scope="shared.rsram")
            Max_shared = T.alloc_shared(out_shape, "bfloat16", scope="shared.rsram")
            Exp_shared = T.alloc_shared(shape, "float32", scope="shared.rsram")
            Sum_shared = T.alloc_shared(out_shape, "float32", scope="shared.rsram")

            T.copy(A, A_shared)
            T.reduce_max(A_shared, Max_shared, dim=1, clear=True)
            for i, j in T.Tiles(shape):
                Exp_shared[i, j] = T.Cast("float32", A_shared[i, j] - Max_shared[i])
            T.reduce_sum(Exp_shared, Sum_shared, dim=1, clear=True)
            T.copy(Sum_shared, Out)

    return main


@target("Sunmmio")
def reduce_keepdim_kernel_builder(shape=(32, 128), reduce_axis=1, dtype="float32"):
    out_shape = list(shape)
    out_shape[reduce_axis] = 1
    out_shape = tuple(out_shape)
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            if len(shape) == 3:
                for bb in T.serial(shape[0]):
                    T.copy(A[bb, :, :], A_shared[bb, :, :])
            else:
                T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis)

    return main


@target("Sunmmio")
def reduce_dynamic_region_kernel_builder(max_k=128, dtype="float32"):
    shape = (32, max_k)
    out_shape = (32,)
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
        k: T.int32,
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared[:, 0:k], Out_shared, dim=1)
            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_manual_dst_tileview_kernel_builder(dst_tile_size):
    shape = (32, 128)
    out_shape = (32,)
    dtype = "float32"
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            T.annotate_tileview({Out_shared: make_tileview(Out_shared, (dst_tile_size,), (-1,))})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=1)
            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_manual_tileview_kernel_builder(src_tile_size=None, dst_tile_size=None):
    shape = (32, 128)
    out_shape = (32,)
    dtype = "float32"
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            if src_tile_size is not None:
                T.annotate_tileview({A_shared: make_tileview(A_shared, src_tile_size, (-2, -1))})
            if dst_tile_size is not None:
                T.annotate_tileview({Out_shared: make_tileview(Out_shared, (dst_tile_size,), (-1,))})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=1)
            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_row_major_covered_tail_kernel_builder(
    reduce_op,
    logical_extent=1000,
    tile_extent=1024,
    initialize_with_fill=False,
    clear=True,
):
    shape = (logical_extent,)
    out_shape = (1,)
    dtype = "bfloat16"
    shard_policy = T.placement.replicated()
    input_layout = make_aligned_row_major(shape, dtype, 64)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=input_layout),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            T.annotate_layout({A_shared: input_layout})
            T.annotate_tileview({A_shared: make_tileview(A_shared, (tile_extent,), (-1,))})
            if initialize_with_fill:
                T.fill(A_shared, 1.0)
            else:
                T.copy(A, A_shared)
            if not clear:
                T.fill(Out_shared, 1.0)
            if reduce_op == "sum":
                T.reduce_sum(A_shared, Out_shared, dim=0, clear=clear)
            elif reduce_op == "max":
                T.reduce_max(A_shared, Out_shared, dim=0, clear=clear)
            else:
                T.reduce_min(A_shared, Out_shared, dim=0, clear=clear)

    return main


@target("Sunmmio")
def reduce_clear_false_kernel_builder(reduce_op):
    shape = (32, 128)
    out_shape = (32,)
    dtype = "float32"
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, shard_policy, dtype, layout=_dram_input_layout(shape)),  # type: ignore
        Out: T.MeshTensor(out_shape, shard_policy, dtype, layout=make_row_major(out_shape)),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            T.copy(A, A_shared)
            T.fill(Out_shared, 1.0)
            if reduce_op == "sum":
                T.reduce_sum(A_shared, Out_shared, dim=1, clear=False)
            elif reduce_op == "max":
                T.reduce_max(A_shared, Out_shared, dim=1, clear=False)
            else:
                T.reduce_min(A_shared, Out_shared, dim=1, clear=False)
            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_tail_region_kernel(
    m=1000,
    n=2000,
    block_m=256,
    block_n=96,
    dtype="bfloat16",
):
    shard_policy = T.placement.replicated()
    input_shape = (block_m, block_n)
    output_shape = (block_m,)
    input_layout = make_zz_layout(input_shape, [0, 1], (32, 32))
    output_layout = make_row_major(output_shape)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.MeshTensor(input_shape, shard_policy, dtype, layout=input_layout),  # type: ignore
        Out: T.MeshTensor(output_shape, shard_policy, dtype, layout=output_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared((block_m,), dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            T.fill(Out_shared, -T.infinity(dtype))

            for by in T.serial(grid_m):
                for bx in T.serial(grid_n):
                    T.reduce_max(
                        A_shared[
                            0 : _valid_extent(by, block_m, m),
                            0 : _valid_extent(bx, block_n, n),
                        ],
                        Out_shared[0 : _valid_extent(by, block_m, m)],
                        dim=1,
                        clear=False,
                    )

            T.copy(Out_shared, Out)

    return main


@target("Sunmmio")
def reduce_tiled_test(
    b=64,
    m=512,
    n=1024,
    block_b=32,
    block_m=256,
    block_n=128,
    reduce_axis=1,
    dtype="bfloat16",
    clear=False,
):
    out_shape_full = (b, m) if reduce_axis == 2 else (b, n) if reduce_axis == 1 else (m, n)
    out_shape_block = (block_b, block_m) if reduce_axis == 2 else (block_b, block_n) if reduce_axis == 1 else (block_m, block_n)
    input_shape = (b, m, n)

    shard_policy = T.placement.replicated()
    input_layout = make_zz_layout(input_shape, [1, 2], (32, 32))
    output_layout = make_aligned_row_major(out_shape_full, dtype, align_bytes=1024)
    output_shared_layout = make_aligned_row_major(out_shape_block, dtype, align_bytes=1024)
    grid_b = T.ceildiv(b, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.MeshTensor(input_shape, shard_policy, dtype, layout=input_layout),  # type: ignore
        Out: T.MeshTensor(out_shape_full, shard_policy, dtype, layout=output_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape_block, dtype, scope="shared.rsram")
            T.annotate_layout({Out_shared: output_shared_layout})

            if reduce_axis == 2:
                for bz in T.serial(grid_b):
                    for by in T.serial(grid_m):
                        if clear:
                            T.fill(Out_shared, 0)
                        else:
                            T.copy(
                                Out[
                                    bz * block_b : (bz + 1) * block_b,
                                    by * block_m : (by + 1) * block_m,
                                ],
                                Out_shared,
                            )
                        for bx in T.serial(grid_n):
                            for bb in T.serial(block_b):
                                T.copy(
                                    A[
                                        bz * block_b + bb,
                                        by * block_m : (by + 1) * block_m,
                                        bx * block_n : (bx + 1) * block_n,
                                    ],
                                    A_shared[bb, :, :],
                                )
                            T.reduce_abssum(A_shared, Out_shared, dim=reduce_axis, clear=False)
                        T.copy(
                            Out_shared,
                            Out[
                                bz * block_b : (bz + 1) * block_b,
                                by * block_m : (by + 1) * block_m,
                            ],
                        )
            elif reduce_axis == 1:
                for bz in T.serial(grid_b):
                    for bx in T.serial(grid_n):
                        if clear:
                            T.fill(Out_shared, 0)
                        else:
                            T.copy(
                                Out[
                                    bz * block_b : (bz + 1) * block_b,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                Out_shared,
                            )
                        for by in T.serial(grid_m):
                            for bb in T.serial(block_b):
                                T.copy(
                                    A[
                                        bz * block_b + bb,
                                        by * block_m : (by + 1) * block_m,
                                        bx * block_n : (bx + 1) * block_n,
                                    ],
                                    A_shared[bb, :, :],
                                )
                            T.reduce_abssum(A_shared, Out_shared, dim=reduce_axis, clear=False)
                        T.copy(
                            Out_shared,
                            Out[
                                bz * block_b : (bz + 1) * block_b,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                        )
            else:
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        if clear:
                            T.fill(Out_shared, 0)
                        else:
                            T.copy(
                                Out[
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                Out_shared,
                            )
                        for bz in T.serial(grid_b):
                            for bb in T.serial(block_b):
                                T.copy(
                                    A[
                                        bz * block_b + bb,
                                        by * block_m : (by + 1) * block_m,
                                        bx * block_n : (bx + 1) * block_n,
                                    ],
                                    A_shared[bb, :, :],
                                )
                            T.reduce_abssum(A_shared, Out_shared, dim=reduce_axis, clear=False)
                        T.copy(
                            Out_shared,
                            Out[
                                by * block_m : (by + 1) * block_m,
                                bx * block_n : (bx + 1) * block_n,
                            ],
                        )

    return main


@pytest.mark.parametrize("shape,reduce_axis,clear", REDUCE_IN_TILE_CASES)
def test_reduce_generic_in_tile_codegen_generates_expected_ops(tmp_path, shape, reduce_axis, clear):
    shape_label = "x".join(str(dim) for dim in shape)
    expected_tokens = ("suvm.copy_async",)
    if reduce_axis != 0:
        expected_tokens = (*expected_tokens, "suvm.tile.reduce")
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder(shape, reduce_axis, clear=clear),
        tmp_path,
        mlir_filename=f"reduce_shape_{shape_label}_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=expected_tokens,
    )
    if reduce_axis == 0:
        assert_source_contains(src, ("suvm.tile.addf", "suvm.tile.store"))
    else:
        assert_source_contains(src, ("suvm.tile.reduce", "sum"))
    if shape == (32, 256, 128) and reduce_axis == 2:
        aligned_view_lines = [
            line
            for line in src.splitlines()
            if "suvm.get_partitioned_tile_view" in line
            and "!suvm.memtensor<32x256xbf16" in line
            and "tiled_dims = [1]" in line
            and "-> !suvm.tile_view<32xbf16>" in line
        ]
        assert aligned_view_lines
        assert any("indices = [%arg2," in line or "(%3, %arg2," in line for line in aligned_view_lines)


@pytest.mark.parametrize("reduce_axis,clear", [(1, False), (2, True)])
def test_reduce_tiled_in_tile_codegen_generates_expected_ops(tmp_path, reduce_axis, clear):
    shape_overrides = {"n": 128} if reduce_axis == 1 else {"m": 256}
    src = validate_sunmmio_codegen_loose(
        reduce_tiled_test(reduce_axis=reduce_axis, clear=clear, **shape_overrides),
        tmp_path,
        mlir_filename=f"reduce_tiled_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.reduce"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "sum"))
    if reduce_axis == 2:
        assert_source_contains(
            src,
            (
                "!suvm.tile<32x32xbf16>",
                "suvm.tile.squeeze",
                "suvm.tile.unsqueeze",
                "!suvm.tile<1x32xbf16>",
            ),
        )


def test_reduce_layout_bounded_zz_block_reaches_raw_suvm(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder((512, 128), 1, dtype="bfloat16", clear=True),
        tmp_path,
        mlir_filename="reduce_layout_bounded_zz_block_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "!suvm.tile<32x32xbf16>"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "!suvm.tile<32x32xbf16>"))


def test_reduce_register_aliases_distinguish_same_named_mixed_dtype_temps(tmp_path):
    src = validate_sunmmio_codegen_strict(
        mixed_dtype_reduce_pipeline_builder(),
        tmp_path,
        mlir_filename="reduce_mixed_dtype_register_alias_suvm.mlir",
        expected_tokens=("!suvm.tile<32x32xbf16>", "!suvm.tile<32x32xf32>"),
    )
    assert src.count("suvm.tile.reduce") == 2


@pytest.mark.parametrize("reduce_axis", [0, 1])
def test_reduce_bf16_source_uses_fp32_semantic_accumulator(tmp_path, reduce_axis):
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder(
            (32, 128),
            reduce_axis,
            dtype="bfloat16",
            out_dtype="float32",
        ),
        tmp_path,
        mlir_filename=f"reduce_bf16_to_fp32_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.tile.cast", "suvm.tile.reduce", "xf32>"),
    )
    cast = re.search(
        r"suvm\.tile\.cast .*!suvm\.tile<(\d+x\d+)xbf16> -> !suvm\.tile<\1xf32>",
        src,
    )
    assert cast, src
    assert re.search(
        rf"suvm\.tile\.reduce\s+sum, .*: !suvm\.tile<{cast.group(1)}xf32>",
        src,
    )


def test_reduce_keepdim_trailing_unit_axis_warns_before_lowering():
    with pytest.warns(
        UserWarning,
        match="may cause layout issues and prevent the kernel from compiling",
    ) as caught:
        kernel = reduce_keepdim_kernel_builder(reduce_axis=1)

    assert kernel is not None
    assert caught[0].filename == __file__


def test_reduce_keepdim_leading_unit_axis_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reduce_keepdim_kernel_builder(reduce_axis=0)

    assert not any("trailing unit-dimension output" in str(item.message) for item in caught)


def test_reduce_keepdim_leading_unit_destination_axis_lowers_strict(tmp_path):
    src = validate_sunmmio_codegen_strict(
        reduce_keepdim_kernel_builder(reduce_axis=0),
        tmp_path,
        mlir_filename="reduce_keepdim_axis_0_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "xf32>"),
    )
    assert "!suvm.memtensor<1x128xf32" in src
    assert re.search(
        r"suvm\.tile\.reduce\s+sum, .*\{axis = 0 : i64\}.*"
        r"-> !suvm\.tile<1x\d+xf32>",
        src,
    )
    assert re.search(
        r"get_partitioned_tile_view .* tiled_dims = \[1\].*"
        r"!suvm\.memtensor<1x128xf32",
        src,
    )


def test_reduce_dynamic_k_is_preserved_in_raw_suvm(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_dynamic_region_kernel_builder(),
        tmp_path,
        mlir_filename="reduce_dynamic_k_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "scf.for"),
    )
    assert re.search(r"func\.func .*\bi32\b", src)
    assert re.search(r"scf\.for .*%arg\d+", src)
    assert_source_contains(src, ("suvm.tile.select", "suvm.tile.reduce  sum"))
    assert re.search(r"suvm\.tile\.reduce\s+sum, .*: !suvm\.tile<\d+x32xf32>", src)


def test_reduce_manual_destination_tileview_is_reflected_in_raw_suvm(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_manual_dst_tileview_kernel_builder(16),
        tmp_path,
        mlir_filename="reduce_manual_dst_tile_16_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "!suvm.tile<16x1xf32>"),
    )
    assert_source_contains(src, ("!suvm.tile<16x32xf32>", "!suvm.tile<16x1xf32>"))


@pytest.mark.parametrize(
    "src_tile_size,dst_tile_size",
    [
        ((16, 32), None),
        (None, 16),
        ((16, 32), 16),
    ],
)
def test_reduce_manual_tileview_combinations_reach_raw_suvm(tmp_path, src_tile_size, dst_tile_size):
    label = f"src_{src_tile_size is not None}_dst_{dst_tile_size is not None}"
    src = validate_sunmmio_codegen_loose(
        reduce_manual_tileview_kernel_builder(src_tile_size, dst_tile_size),
        tmp_path,
        mlir_filename=f"reduce_manual_{label}_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "!suvm.tile<16x32xf32>"),
    )
    assert_source_contains(src, ("!suvm.tile<16x32xf32>", "!suvm.tile<16x1xf32>"))


@pytest.mark.parametrize(
    "reduce_op,identity",
    [
        ("sum", "0.000000e+00"),
        ("max", "0xFF80"),
        ("min", "0x7F80"),
    ],
)
def test_reduce_row_major_covered_tail_identity_precedes_raw_suvm_reduce(tmp_path, reduce_op, identity):
    src = validate_sunmmio_codegen_loose(
        reduce_row_major_covered_tail_kernel_builder(reduce_op),
        tmp_path,
        mlir_filename=f"reduce_row_major_covered_tail_{reduce_op}_suvm.mlir",
        expected_tokens=("suvm.tile.select", f"suvm.tile.reduce  {reduce_op}"),
    )
    assert_source_contains(src, ("!suvm.tile<1024xbf16>", identity))
    assert_source_contains(
        src,
        (
            "!suvm.tile<1x1024xbf16>",
            "!suvm.tile<1x1xbf16>",
            "suvm.tile.pick",
            "suvm.tile.set",
        ),
    )
    select_pos = src.index("suvm.tile.select")
    reduce_pos = src.index(f"suvm.tile.reduce  {reduce_op}")
    assert src.rfind(identity, 0, select_pos) >= 0
    assert select_pos < reduce_pos
    assert re.search(rf"suvm\.tile\.reduce\s+{reduce_op}, .*\{{axis = 1 : i64\}}", src)


@pytest.mark.parametrize("reduce_op", ["sum", "max", "min"])
def test_reduce_rank1_lowers_via_rank2_carrier(tmp_path, reduce_op):
    src = validate_sunmmio_codegen_strict(
        reduce_row_major_covered_tail_kernel_builder(
            reduce_op,
            logical_extent=32,
            tile_extent=32,
            initialize_with_fill=True,
        ),
        tmp_path,
        mlir_filename=f"reduce_rank1_{reduce_op}_strict.mlir",
        expected_tokens=("suvm.tile.reduce",),
    )
    assert "!suvm.tile<1x32x" in src
    assert "!suvm.tile<1x1x" in src
    assert re.search(rf"suvm\.tile\.reduce\s+{reduce_op}, .*\{{axis = 1 : i64\}}", src)
    assert "suvm.tile.pick" in src
    assert "suvm.tile.set" in src


@pytest.mark.parametrize(
    "reduce_op,combine_op",
    [
        ("sum", "suvm.tile.addf"),
        ("max", "suvm.tile.maxf"),
        ("min", "suvm.tile.minf"),
    ],
)
def test_reduce_rank1_clear_false_combines_and_stores_scalar(
    tmp_path,
    reduce_op,
    combine_op,
):
    src = validate_sunmmio_codegen_strict(
        reduce_row_major_covered_tail_kernel_builder(
            reduce_op,
            logical_extent=32,
            tile_extent=32,
            initialize_with_fill=True,
            clear=False,
        ),
        tmp_path,
        mlir_filename=f"reduce_rank1_{reduce_op}_clear_false_strict.mlir",
        expected_tokens=("suvm.tile.reduce", combine_op),
    )
    assert "!suvm.tile<1x32x" in src
    assert "!suvm.tile<1x1x" in src
    assert re.search(rf"suvm\.tile\.reduce\s+{reduce_op}, .*\{{axis = 1 : i64\}}", src)
    assert "suvm.tile.pick" in src
    assert "suvm.tile.set" in src


@pytest.mark.parametrize(
    "reduce_op,combine_op",
    [
        ("sum", "suvm.tile.addf"),
        ("max", "suvm.tile.maxf"),
        ("min", "suvm.tile.minf"),
    ],
)
def test_reduce_clear_false_combines_destination_once_in_raw_suvm(tmp_path, reduce_op, combine_op):
    src = validate_sunmmio_codegen_loose(
        reduce_clear_false_kernel_builder(reduce_op),
        tmp_path,
        mlir_filename=f"reduce_clear_false_{reduce_op}_suvm.mlir",
        expected_tokens=(f"suvm.tile.reduce  {reduce_op}", combine_op),
    )
    reduce_pos = src.index(f"suvm.tile.reduce  {reduce_op}")
    combine_positions = [match.start() for match in re.finditer(re.escape(combine_op), src)]
    assert len(combine_positions) == 2
    assert combine_positions[0] < reduce_pos < combine_positions[1]


@pytest.mark.parametrize("reduce_axis", [1, 2])
def test_reduce_rank3_keepdim_projects_raw_suvm_destination(tmp_path, reduce_axis):
    shape = (4, 64, 128)
    src = validate_sunmmio_codegen_loose(
        reduce_keepdim_kernel_builder(shape=shape, reduce_axis=reduce_axis),
        tmp_path,
        mlir_filename=f"reduce_rank3_keepdim_axis_{reduce_axis}_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "xf32>"),
    )
    out_shape = list(shape)
    out_shape[reduce_axis] = 1
    expected_out_shape = "x".join(str(extent) for extent in out_shape)
    expected_tile_axis = reduce_axis - 1
    assert f"!suvm.memtensor<{expected_out_shape}xf32" in src
    assert re.search(
        rf"suvm\.tile\.reduce\s+sum, .*\{{axis = {expected_tile_axis} : i64\}}",
        src,
    )


def test_reduce_small_1d_result_uses_aligned_store_bridge(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder((32, 64, 256), 2, clear=True, tile_size=(8, 32)),
        tmp_path,
        mlir_filename="reduce_small_1d_result_aligned_store_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "suvm.tile.insert_slice", "suvm.tile.store"),
    )
    assert_source_contains(src, ("suvm.tile.reduce", "!suvm.tile<8x1xbf16>", "!suvm.tile<32xbf16>"))
    assert "suvm.tile.unsqueeze" not in src
    assert "fake_tile_insert_slice" not in src
    assert "suvm.tile.store" in src
    assert "fake_tile_store" not in src
    assert "!suvm.tile_view<32x1xbf16>" not in src
    assert "fake_partitioned_tile_view" not in src


def test_reduce_dim0_multisegment_store_uses_partition_coordinate(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_kernel_builder((4, 128), 0, dtype="float32", clear=True),
        tmp_path,
        mlir_filename="reduce_dim0_multisegment_store_suvm.mlir",
        expected_tokens=("suvm.tile.reduce", "suvm.tile.store"),
    )

    definitions = {match.group(1): match.group(2) for line in src.splitlines() if (match := re.match(r"\s*(%[\w.]+)\s*=\s*(.+)", line))}
    source_views = [
        line
        for line in src.splitlines()
        if "suvm.get_partitioned_tile_view" in line
        and "!suvm.memtensor<4x128xf32" in line
        and "#suvm.memory_space<rsram>" in line
        and "-> !suvm.tile_view<4x32xf32>" in line
    ]
    destination_stores = [
        line for line in src.splitlines() if "suvm.tile.store" in line and "!suvm.tile<32xf32>, !suvm.tile_view<32xf32>" in line
    ]
    assert len(source_views) == 1, source_views
    assert len(destination_stores) == 1, destination_stores

    source_indices = re.search(r"indices = \[(%[\w.]+), (%[\w.]+)\]", source_views[0])
    assert source_indices, source_views[0]
    segment_index = source_indices.group(2)

    store_match = re.search(
        r"suvm\.tile\.store\s+%[\w.]+,\s+(%[\w.]+)\s+:",
        destination_stores[0],
    )
    assert store_match, destination_stores[0]
    destination_view = store_match.group(1)
    destination_view_definition = definitions.get(destination_view, "")
    assert "suvm.get_partitioned_tile_view" in destination_view_definition
    assert "!suvm.memtensor<128xf32" in destination_view_definition
    assert "-> !suvm.tile_view<32xf32>" in destination_view_definition

    index_match = re.search(r"indices = \[(%[\w.]+)\]", destination_view_definition)
    assert index_match, destination_view_definition
    destination_index = index_match.group(1)

    # Canonicalization may fold `(segment * 32) / 32` to `segment`.  Before
    # that fold, require the complete def-use chain instead of accepting an
    # unrelated `arith.divsi` result.
    if destination_index == segment_index:
        return

    div_match = re.fullmatch(
        r"arith\.divsi (%[\w.]+), (%[\w.]+) : index",
        definitions.get(destination_index, ""),
    )
    assert div_match, f"destination index is not defined by divsi: {destination_view_definition}"
    numerator, divisor = div_match.groups()
    assert definitions.get(divisor) == "arith.constant 32 : index"

    numerator_cast = re.fullmatch(
        r"arith\.index_cast (%[\w.]+) : i32 to index",
        definitions.get(numerator, ""),
    )
    assert numerator_cast, definitions.get(numerator)
    product = numerator_cast.group(1)

    multiply = re.fullmatch(
        r"arith\.muli (%[\w.]+), (%[\w.]+) : i32",
        definitions.get(product, ""),
    )
    assert multiply, definitions.get(product)
    lhs, rhs = multiply.groups()
    if definitions.get(lhs) == "arith.constant 32 : i32":
        factor, segment_i32 = lhs, rhs
    else:
        segment_i32, factor = lhs, rhs
    assert definitions.get(factor) == "arith.constant 32 : i32"
    assert definitions.get(segment_i32) == (f"arith.index_cast {segment_index} : index to i32"), (
        "The reduction destination must be indexed by "
        "(segment_index * 32) / 32.\n"
        f"source view: {source_views[0]}\n"
        f"destination view: {destination_view_definition}"
    )


def test_reduce_tail_region_codegen_uses_identity_select(tmp_path):
    src = validate_sunmmio_codegen_loose(
        reduce_tail_region_kernel(),
        tmp_path,
        mlir_filename="reduce_tail_region_suvm.mlir",
        expected_tokens=(
            "suvm.tile.reduce  max",
            "suvm.tile.select",
            "0xFF80",
            "suvm.tile.insert_slice",
            "suvm.tile.store",
        ),
    )
    assert "fake_missing" not in src
    assert_source_contains(src, ("!suvm.tile<8x32xbf16>", "!suvm.tile<8x1xbf16>", "!suvm.tile<32xbf16>"))


if __name__ == "__main__":
    tilelang.testing.main()
