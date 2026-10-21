import os
import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

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


@target("Sunmmio")
def tile_elementwise_ops_test(
    batch=2,
    m=256,
    n=128,
    block_b=2,
    block_m=256,
    block_n=128,
    dtype="bfloat16",
):
    shard_policy = T.placement.replicated()
    tensor_shape = (batch, m, n)
    tensor_layout = make_zz_layout(tensor_shape, [1, 2], (32, 32))
    grid_b = T.ceildiv(batch, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        B: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        C: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_b, block_m, block_n), dtype)

            for bz in T.serial(grid_b):
                for by in T.serial(grid_m):
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
                            T.copy(
                                B[
                                    bz * block_b + bb,
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                                B_shared[bb, :, :],
                            )

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            x = T.max(A_shared[b, i, j], B_shared[b, i, j])
                            y = T.min(x, T.exp(B_shared[b, i, j]))
                            z = T.log(T.abs(y) + T.float32(1.0))
                            C_shared[b, i, j] = T.if_then_else(
                                z > T.float32(0.5),
                                T.rsqrt(z + T.float32(1.0)),
                                T.ieee_frcp(z + T.float32(2.0)),
                            )

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            neg_a = T.float32(0.0) - T.Cast("float32", A_shared[b, i, j])
                            rem_b = T.fmod(T.Cast("float32", B_shared[b, i, j]), T.float32(3.0))
                            C_shared[b, i, j] = (
                                T.ceil(C_shared[b, i, j])
                                + T.floor(A_shared[b, i, j])
                                + T.round(B_shared[b, i, j])
                                + T.trunc(A_shared[b, i, j])
                                + neg_a
                                + rem_b
                            )

                        for bb in T.serial(block_b):
                            T.copy(
                                C_shared[bb, :, :],
                                C[
                                    bz * block_b + bb,
                                    by * block_m : (by + 1) * block_m,
                                    bx * block_n : (bx + 1) * block_n,
                                ],
                            )

    return main


@target("Sunmmio")
def tile_elementwise_ops_2d_test(
    m=256,
    n=1024,
    block_m=256,
    block_n=512,
    dtype="bfloat16",
):
    shard_policy = T.placement.replicated()
    tensor_shape = (m, n)
    tensor_layout = make_zz_layout(tensor_shape, [0, 1], (32, 32))
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        B: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        C: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_m, block_n), dtype)

            for by in T.serial(grid_m):
                for bx in T.serial(grid_n):
                    T.copy(
                        A[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                        A_shared,
                    )
                    T.copy(
                        B[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                        B_shared,
                    )

                    for i, j in T.Tiles(A_shared, parallel=True):
                        x = T.max(A_shared[i, j], B_shared[i, j])
                        y = T.min(x, T.exp(B_shared[i, j]))
                        z = T.log(T.abs(y) + T.float32(1.0))
                        C_shared[i, j] = T.if_then_else(
                            z > T.float32(0.5),
                            T.rsqrt(z + T.float32(1.0)),
                            T.ieee_frcp(z + T.float32(2.0)),
                        )

                    for i, j in T.Tiles(A_shared, parallel=True):
                        neg_a = T.float32(0.0) - T.Cast("float32", A_shared[i, j])
                        rem_b = T.fmod(T.Cast("float32", B_shared[i, j]), T.float32(3.0))
                        C_shared[i, j] = (
                            T.ceil(C_shared[i, j])
                            + T.floor(A_shared[i, j])
                            + T.round(B_shared[i, j])
                            + T.trunc(A_shared[i, j])
                            + neg_a
                            + rem_b
                        )

                    T.copy(
                        C_shared,
                        C[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                    )

    return main


@target("Sunmmio")
def fp32_select_then_bf16_cast_test(m=32, n=32):
    input_dtype = T.float32
    output_dtype = T.bfloat16
    shard_policy = T.placement.replicated()
    tensor_shape = (m, n)
    tensor_layout = make_zz_layout(tensor_shape, [0, 1], tensor_shape)

    @T.prim_func
    def main(
        A: T.MeshTensor(tensor_shape, shard_policy, input_dtype, layout=tensor_layout),  # type: ignore
        C: T.MeshTensor(tensor_shape, shard_policy, output_dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(tensor_shape, input_dtype)
            C_shared = T.alloc_shared(tensor_shape, output_dtype)

            T.copy(A, A_shared)
            for i, j in T.Tiles(A_shared, parallel=True):
                C_shared[i, j] = T.Cast(
                    output_dtype,
                    T.if_then_else(
                        A_shared[i, j] > T.float32(0),
                        A_shared[i, j],
                        T.float32(0),
                    ),
                )
            T.copy(C_shared, C)

    return main


@target("Sunmmio")
def tile_bitwise_ops_test(m=32, n=32, dtype="int32"):
    shard_policy = T.placement.replicated()
    tensor_shape = (m, n)
    tensor_layout = make_zz_layout(tensor_shape, [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        B: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        C: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(tensor_shape, dtype)
            B_shared = T.alloc_shared(tensor_shape, dtype)
            C_shared = T.alloc_shared(tensor_shape, dtype)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            for i, j in T.Tiles(A_shared, parallel=True):
                and_value = T.bitwise_and(A_shared[i, j], B_shared[i, j])
                or_value = T.bitwise_or(and_value, A_shared[i, j])
                C_shared[i, j] = T.bitwise_xor(or_value, B_shared[i, j])
            T.copy(C_shared, C)

    return main


@pytest.mark.parametrize("dtype", ["int32", "uint32"])
@pytest.mark.parametrize("m", [32, 40], ids=["full-tile", "1024-byte-tail"])
def test_tile_bitwise_ops_codegen_validates_with_npuir_opt(tmp_path, dtype, m):
    src = validate_sunmmio_codegen_with_npuir_opt(
        tile_bitwise_ops_test(m=m, dtype=dtype),
        tmp_path,
        mlir_filename=f"tile_bitwise_ops_{dtype}_{m}_suvm.mlir",
        expected_tokens=("suvm.tile.andi", "suvm.tile.ori", "suvm.tile.xori"),
        opt_args=STRICT_OPT_ARGS,
    )
    assert_source_contains(
        src,
        ("suvm.tile.andi", "suvm.tile.ori", "suvm.tile.xori"),
    )


@pytest.mark.parametrize("dtype", ["int16", "uint16", "int64", "uint64"])
def test_tile_bitwise_ops_rejects_unsupported_dtype(tmp_path, dtype):
    with pytest.raises(
        Exception,
        match=rf"SunMMIO tile bitwise operations require int32 or uint32 elements, got {dtype}",
    ):
        validate_sunmmio_codegen_with_npuir_opt(
            tile_bitwise_ops_test(dtype=dtype),
            tmp_path,
            mlir_filename=f"tile_bitwise_ops_{dtype}_rejected_suvm.mlir",
        )


def test_tile_elementwise_ops_2d_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        tile_elementwise_ops_2d_test(),
        tmp_path,
        mlir_filename="tile_elementwise_ops_2d_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.store"),
    )
    assert_source_contains(
        src,
        (
            "suvm.tile.abs",
            "suvm.tile.ceil",
            "suvm.tile.cmpf",
            "suvm.tile.floor",
            "suvm.tile.ln",
            "suvm.tile.maxf",
            "suvm.tile.minf",
            "suvm.tile.neg",
            "suvm.tile.recip",
            "suvm.tile.remf",
            "suvm.tile.round",
            "suvm.tile.rsqrt",
            "suvm.tile.select",
            "suvm.tile.trunc",
        ),
    )


def test_tile_elementwise_ops_codegen_validates_loose_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_loose(
        tile_elementwise_ops_test(),
        tmp_path,
        mlir_filename="tile_elementwise_ops_suvm.mlir",
        expected_tokens=("suvm.tile.store",),
    )
    assert_source_contains(
        src,
        (
            "suvm.tile.abs",
            "suvm.tile.ceil",
            "suvm.tile.cmpf",
            "suvm.tile.floor",
            "suvm.tile.ln",
            "suvm.tile.maxf",
            "suvm.tile.minf",
            "suvm.tile.neg",
            "suvm.tile.recip",
            "suvm.tile.remf",
            "suvm.tile.round",
            "suvm.tile.rsqrt",
            "suvm.tile.select",
            "suvm.tile.trunc",
        ),
    )


def test_fp32_select_is_evaluated_before_bf16_cast(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        fp32_select_then_bf16_cast_test(),
        tmp_path,
        mlir_filename="fp32_select_then_bf16_cast_suvm.mlir",
        expected_tokens=("suvm.tile.cmpf", "suvm.tile.select", "suvm.tile.cast"),
        opt_args=STRICT_OPT_ARGS,
    )

    select = re.search(
        r"(?P<result>%[\w.]+) = suvm\.tile\.select .*"
        r"!suvm\.tile<[^>]*xf32>, !suvm\.tile<[^>]*xf32>"
        r" -> !suvm\.tile<[^>]*xf32>",
        src,
    )
    assert select, src
    assert re.search(
        rf"suvm\.tile\.cast {re.escape(select.group('result'))} : "
        r"!suvm\.tile<[^>]*xf32> -> !suvm\.tile<[^>]*xbf16>",
        src,
    ), src


if __name__ == "__main__":
    tilelang.testing.main()
