import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_aligned_row_major, make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


def validate_sunmmio_codegen_loose(kernel, tmp_path, *, mlir_filename, expected_tokens=()):
    return validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=LOOSE_OPT_ARGS,
    )


@target("Sunmmio")
def dot_mul_tiled_parallel_3d(
    batch=64,
    m=512,
    n=1024,
    block_b=2,
    block_m=256,
    block_n=128,
    dtype="bfloat16",
    accum_dtype="bfloat16",
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
        C: T.MeshTensor(tensor_shape, shard_policy, accum_dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_b, block_m, block_n), accum_dtype)

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
                            A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                            B_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]
                            C_shared[b, i, j] = T.exp(A_shared[b, i, j]) + T.exp(B_shared[b, i, j])

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
def dot_mul_tiled_parallel_2d(
    m=512,
    n=1024,
    block_m=256,
    block_n=512,
    dtype="bfloat16",
    accum_dtype="bfloat16",
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
        C: T.MeshTensor(tensor_shape, shard_policy, accum_dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_m, block_n), accum_dtype)

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
                        A_shared[i, j] = A_shared[i, j] * T.float32(2.0)
                        B_shared[i, j] = A_shared[i, j] * B_shared[i, j]
                        C_shared[i, j] = T.exp(A_shared[i, j]) + T.exp(B_shared[i, j])

                    T.copy(
                        C_shared,
                        C[
                            by * block_m : (by + 1) * block_m,
                            bx * block_n : (bx + 1) * block_n,
                        ],
                    )

    return main


@target("Sunmmio")
def tiles_broadcast(
    batch=64,
    m=512,
    n=1024,
    block_b=2,
    block_m=512,
    block_n=128,
    dtype="bfloat16",
    accum_dtype="bfloat16",
):
    shard_policy = T.placement.replicated()
    tensor_shape = (batch, m, n)
    tensor_layout = make_zz_layout(tensor_shape, [1, 2], (32, 32))
    vector_shape = (m,)
    vector_layout = make_aligned_row_major(vector_shape, dtype, align_bytes=1024)
    vector_shared_layout = make_aligned_row_major((block_m,), dtype, align_bytes=1024)
    grid_b = T.ceildiv(batch, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        B: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        C: T.MeshTensor(tensor_shape, shard_policy, accum_dtype, layout=tensor_layout),  # type: ignore
        D: T.MeshTensor(vector_shape, shard_policy, dtype, layout=vector_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_b, block_m, block_n), accum_dtype)
            D_shared = T.alloc_shared((block_m,), dtype)
            T.annotate_layout({D_shared: vector_shared_layout})

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
                        T.copy(D[by * block_m : (by + 1) * block_m], D_shared)

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            A_shared[b, i, j] = A_shared[b, i, j] + D_shared[i]
                            A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                            C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            A_shared[b, i, j] = A_shared[b, i, j] + D_shared[j]
                            A_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
                            C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]

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
def tiles_broadcast_copy(
    batch=64,
    m=512,
    n=1024,
    block_b=2,
    block_m=512,
    block_n=128,
    dtype="bfloat16",
    accum_dtype="bfloat16",
):
    shard_policy = T.placement.replicated()
    tensor_shape = (batch, m, n)
    tensor_layout = make_zz_layout(tensor_shape, [1, 2], (32, 32))
    vector_shape = (m,)
    vector_layout = make_aligned_row_major(vector_shape, dtype, align_bytes=1024)
    vector_shared_layout = make_aligned_row_major((block_m,), dtype, align_bytes=1024)
    grid_b = T.ceildiv(batch, block_b)
    grid_m = T.ceildiv(m, block_m)
    grid_n = T.ceildiv(n, block_n)

    @T.prim_func
    def main(
        A: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        B: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        C: T.MeshTensor(tensor_shape, shard_policy, accum_dtype, layout=tensor_layout),  # type: ignore
        D: T.MeshTensor(vector_shape, shard_policy, dtype, layout=vector_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_b, block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_b, block_m, block_n), accum_dtype)
            D_shared = T.alloc_shared((block_m,), dtype)
            T.annotate_layout({D_shared: vector_shared_layout})

            for bz in T.serial(grid_b):
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        T.clear(C_shared)
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
                        T.copy(D[by * block_m : (by + 1) * block_m], D_shared)

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            C_shared[b, i, j] = D_shared[i]
                            C_shared[b, i, j] = C_shared[b, i, j] * T.float32(2.0)

                        for b, i, j in T.Tiles(A_shared, parallel=True):
                            C_shared[b, i, j] = T.if_then_else(
                                D_shared[j] >= 0,
                                T.float32(0.0),
                                -T.infinity(accum_dtype),
                            )
                            C_shared[b, i, j] = C_shared[b, i, j] * T.float32(2.0)

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
def tiles_1d(m=512, block_m=512, dtype="bfloat16", accum_dtype="bfloat16"):
    shard_policy = T.placement.replicated()
    tensor_shape = (m,)
    tensor_layout = make_aligned_row_major(tensor_shape, dtype, align_bytes=1024)
    shared_layout = make_aligned_row_major((block_m,), dtype, align_bytes=1024)
    accum_shared_layout = make_aligned_row_major((block_m,), accum_dtype, align_bytes=1024)
    grid_m = T.ceildiv(m, block_m)

    @T.prim_func
    def main(
        A: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        B: T.MeshTensor(tensor_shape, shard_policy, dtype, layout=tensor_layout),  # type: ignore
        C: T.MeshTensor(tensor_shape, shard_policy, accum_dtype, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_m,), dtype)
            B_shared = T.alloc_shared((block_m,), dtype)
            C_shared = T.alloc_shared((block_m,), accum_dtype)
            T.annotate_layout({A_shared: shared_layout, B_shared: shared_layout, C_shared: accum_shared_layout})

            for by in T.serial(grid_m):
                T.clear(C_shared)
                T.copy(A[by * block_m : (by + 1) * block_m], A_shared)
                T.copy(B[by * block_m : (by + 1) * block_m], B_shared)
                for i in T.Tiles(A_shared, parallel=True):
                    C_shared[i] = A_shared[i] * B_shared[i]
                T.copy(C_shared, C[by * block_m : (by + 1) * block_m])

    return main


@target("Sunmmio")
def tiles_dynamic_extent_zz_store():
    compute_block = 128
    tile_block = 32
    output_shape = (compute_block, tile_block)
    side_shape = (256,)
    shard_policy = T.placement.replicated()
    output_layout = make_zz_layout(output_shape, [0, 1], (tile_block, tile_block))
    side_layout = make_aligned_row_major(side_shape, T.int32, align_bytes=1024)

    @T.prim_func
    def main(
        Output: T.MeshTensor(output_shape, shard_policy, T.bfloat16, layout=output_layout),  # type: ignore
        q_lengths: T.MeshTensor(side_shape, shard_policy, T.int32, layout=side_layout),  # type: ignore
    ):
        with T.Kernel():
            q_lengths_shared = T.alloc_shared(side_shape, T.int32, scope="shared.rsram")
            source_tile = T.alloc_shared((tile_block, tile_block), T.bfloat16, scope="shared.rsram")
            output_staging = T.alloc_shared(output_shape, T.bfloat16, scope="shared.rsram")
            T.annotate_layout({q_lengths_shared: side_layout, output_staging: output_layout})
            q_len = T.alloc_var(T.int32, init=0)

            T.copy(q_lengths, q_lengths_shared)
            q_len = T.max(T.min(q_lengths_shared[0], compute_block), 0)
            T.fill(source_tile, 1)
            T.fill(output_staging, 0)

            for q_chunk in T.serial(T.ceildiv(q_len, tile_block)):
                for row, col in T.Tiles([tile_block, tile_block], parallel=True):
                    output_staging[q_chunk * tile_block + row, col] = source_tile[row, col]

            T.copy(output_staging, Output)

    return main


@target("Sunmmio")
def tiles_rank2_first_tile_partial(rows=4, cols=4, dtype="float32"):
    output_shape = (32, 32)
    output_layout = make_zz_layout(output_shape, [0, 1], (32, 32))
    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        output: T.MeshTensor(output_shape, shard_policy, dtype, layout=output_layout),  # type: ignore
    ):
        with T.Kernel():
            source = T.alloc_shared((rows, cols), dtype)
            matrix = T.alloc_shared((rows, cols), dtype)
            output_shared = T.alloc_shared(output_shape, dtype)

            T.fill(source, 1.0)
            T.fill(output_shared, 0.0)
            for i, j in T.Tiles(matrix, parallel=True):
                matrix[i, j] = source[i, j]
            for i, j in T.Tiles(matrix, parallel=True):
                output_shared[i, j] = matrix[i, j]
            T.copy(output_shared, output)

    return main


def test_dot_mul_tiled_parallel_2d_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        dot_mul_tiled_parallel_2d(),
        tmp_path,
        mlir_filename="dot_mul_tiled_parallel_2d_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.mulf", "suvm.tile.exp"),
    )
    assert_source_contains(src, ("suvm.tile.mulf", "suvm.tile.exp"))


def test_tiles_rank2_small_logical_buffer_uses_carrier_rmw(tmp_path):
    src = validate_sunmmio_codegen_loose(
        tiles_rank2_first_tile_partial(),
        tmp_path,
        mlir_filename="tiles_rank2_first_tile_partial_suvm.mlir",
        expected_tokens=("!suvm.tile_view<4x32xf32>", "suvm.tile.extract_slice", "suvm.tile.insert_slice", "suvm.tile.store"),
    )
    assert "suvm.tile.cmpi" not in src
    assert "suvm.tile.select" not in src


def test_dot_mul_tiled_parallel_3d_large_block_codegen_validates_loose_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_loose(
        dot_mul_tiled_parallel_3d(
            batch=64,
            m=512,
            n=1024,
            block_b=32,
            block_m=256,
            block_n=128,
            dtype="bfloat16",
            accum_dtype="bfloat16",
        ),
        tmp_path,
        mlir_filename="dot_mul_tiled_parallel_3d_large_block_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tile.mulf", "suvm.tile.exp"),
    )
    assert_source_contains(src, ("suvm.copy_async", "suvm.tile.mulf", "suvm.tile.exp"))


def test_tiles_dynamic_extent_zz_store_codegen_validates_loose_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_loose(
        tiles_dynamic_extent_zz_store(),
        tmp_path,
        mlir_filename="tiles_dynamic_extent_zz_store_suvm.mlir",
        expected_tokens=("scf.for", "suvm.tile.store"),
    )
    assert_source_contains(src, ("scf.for", "suvm.tile.store"))


@pytest.mark.parametrize(
    "case_name,kernel_factory,expected_tokens",
    [
        ("dot_mul_tiled_parallel_3d", dot_mul_tiled_parallel_3d, ("suvm.tile.mulf", "suvm.tile.exp")),
        ("tiles_broadcast", tiles_broadcast, ("suvm.tile.mulf",)),
        ("tiles_broadcast_copy", tiles_broadcast_copy, ("suvm.tile.mulf", "suvm.tile.select")),
        ("tiles_1d", tiles_1d, ("suvm.tile.mulf",)),
        (
            "dot_mul_tiled_parallel_3d_tail_canonical_shared",
            lambda: dot_mul_tiled_parallel_3d(
                batch=64,
                m=512,
                n=256,
                block_b=32,
                block_m=64,
                block_n=64,
                dtype="bfloat16",
                accum_dtype="bfloat16",
            ),
            ("suvm.tile.mulf", "suvm.tile.exp"),
        ),
    ],
)
def test_tiles_codegen_validates_loose_with_npuir_opt(tmp_path, case_name, kernel_factory, expected_tokens):
    src = validate_sunmmio_codegen_loose(
        kernel_factory(),
        tmp_path,
        mlir_filename=f"{case_name}_suvm.mlir",
        expected_tokens=expected_tokens,
    )
    assert_source_contains(src, expected_tokens)


if __name__ == "__main__":
    tilelang.testing.main()
