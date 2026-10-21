import os
import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_aligned_row_major, make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)
STRICT_OPT_ARGS = ("--verify-each", "--suvm-to-llvm-pipeline")


@target("Sunmmio")
def predicated_store_mask_dtype_kernel(dtype, extent=512, valid_n=33):
    shard_policy = T.placement.replicated()
    tensor_shape = (extent,)
    tensor_layout = make_aligned_row_major(tensor_shape, dtype, align_bytes=1024)

    @T.prim_func
    def main(out: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout)):  # type: ignore
        with T.Kernel():
            out_shared = T.alloc_shared(tensor_shape, dtype)
            T.annotate_layout({out_shared: make_aligned_row_major(tensor_shape, dtype, align_bytes=1024)})

            T.clear(out_shared)
            for i in T.Tiles([valid_n], parallel=True):
                out_shared[i] = T.Cast(dtype, 7)
            T.copy(out_shared, out[0:extent])

    return main


@target("Sunmmio")
def bf16_select_with_i32_row_mask_kernel(
    m=128,
    n=128,
    block_m=32,
    block_n=32,
    valid_rows=17,
    valid_cols=19,
):
    dtype = T.bfloat16
    shard_policy = T.placement.full_shard(0, 1)
    tensor_shape = (m, n)
    tensor_layout = make_zz_layout(tensor_shape, [0, 1], (block_m, block_n))

    @T.prim_func
    def main(
        inp: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        out: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            inp_shared = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            out_shared = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            valid_rows_var = T.alloc_var(T.int32, init=valid_rows)
            valid_cols_var = T.alloc_var(T.int32, init=valid_cols)

            T.copy(inp[0:block_m, 0:block_n], inp_shared)
            for row, col in T.Tiles([block_m, block_n], parallel=True):
                out_shared[row, col] = T.if_then_else(
                    T.And(row < valid_rows_var, col < valid_cols_var),
                    inp_shared[row, col],
                    T.Cast(dtype, 0),
                )
            T.copy(out_shared, out[0:block_m, 0:block_n])

    return main


@target("Sunmmio")
def bf16_dynamic_rect_tail_mask_kernel(block_m=32, block_n=32, valid_rows=17, valid_cols=19):
    dtype = T.bfloat16
    shard_policy = T.placement.full_shard(0, 1)
    tensor_shape = (block_m, block_n)
    tensor_layout = make_zz_layout(tensor_shape, [0, 1], (block_m, block_n))

    @T.prim_func
    def main(out: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout)):  # type: ignore
        with T.Kernel():
            out_shared = T.alloc_shared(tensor_shape, dtype, scope="shared.rsram")
            valid_rows_var = T.alloc_var(T.int32, init=valid_rows)
            valid_cols_var = T.alloc_var(T.int32, init=valid_cols)

            T.clear(out_shared)
            for row, col in T.Tiles([valid_rows_var, valid_cols_var], parallel=True):
                out_shared[row, col] = T.Cast(dtype, 7)
            T.copy(out_shared, out[0:block_m, 0:block_n])

    return main


@pytest.mark.parametrize(
    "dtype,expected_index_dtype",
    [
        (T.bfloat16, "i16"),
        (T.int16, "i16"),
        (T.float32, "i32"),
        (T.int32, "i32"),
        (T.uint32, "i32"),
    ],
)
def test_predicated_store_mask_range_uses_value_dtype_width(dtype, expected_index_dtype, tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        predicated_store_mask_dtype_kernel(dtype),
        tmp_path,
        mlir_filename=f"predicated_store_mask_{expected_index_dtype}_suvm.mlir",
        expected_tokens=("suvm.tile.range", "suvm.tile.cmpi", "suvm.tile.select"),
        opt_args=LOOSE_OPT_ARGS,
    )
    assert re.search(rf"suvm\.tile\.range : !suvm\.tile<[^>]*x{expected_index_dtype}>", src)


def test_bf16_select_with_i32_row_mask_lowers_to_llvm(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        bf16_select_with_i32_row_mask_kernel(),
        tmp_path,
        mlir_filename="bf16_select_with_i32_row_mask_suvm.mlir",
        expected_tokens=("suvm.tile.range", "suvm.tile.cmpi", "suvm.tile.select"),
        opt_args=STRICT_OPT_ARGS,
    )
    assert re.search(r"suvm\.tile\.range : !suvm\.tile<[^>]*xi16>", src)
    assert re.search(r"suvm\.tile\.cmpi .* : !suvm\.tile<[^>]*xi16>", src)
    assert "suvm.tile.addi" not in src


def test_bf16_dynamic_rect_tail_mask_uses_unsigned_i16_compare(tmp_path, monkeypatch):
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")
    src = validate_sunmmio_codegen_with_npuir_opt(
        bf16_dynamic_rect_tail_mask_kernel(),
        tmp_path,
        mlir_filename="bf16_dynamic_rect_tail_mask_suvm.mlir",
        expected_tokens=("suvm.tile.range", "suvm.tile.cmpi", "suvm.tile.andi"),
        opt_args=STRICT_OPT_ARGS,
    )
    assert len(re.findall(r"suvm\.tile\.cmpi\s+ult, .* : !suvm\.tile<[^>]*xi16>", src)) >= 2
    assert not re.search(r"suvm\.tile\.cmpi\s+slt, .* : !suvm\.tile<[^>]*xi16>", src)


def test_float16_sunmmio_codegen_reports_unsupported_dtype(tmp_path):
    with pytest.raises(
        Exception,
        match="Unsupported SunMMIO SUVM dtype.*float16.*does not support float16.*Use bfloat16",
    ):
        validate_sunmmio_codegen_with_npuir_opt(
            predicated_store_mask_dtype_kernel(T.float16),
            tmp_path,
            mlir_filename="predicated_store_mask_float16_rejected_suvm.mlir",
        )


if __name__ == "__main__":
    tilelang.testing.main()
