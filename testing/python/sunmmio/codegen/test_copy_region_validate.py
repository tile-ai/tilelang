import os

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_row_major, make_zz_layout

from testing.python.sunmmio.common.codegen_validation import (
    lower_sunmmio_kernel_to_device_tir,
    validate_sunmmio_codegen_with_npuir_opt,
)
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")

DTYPE = "bfloat16"
LOOSE_OPT_ARGS = ("--verify-each",)
_SUMMA_COPY_SHAPES = (
    (256, 128, 128),
    (128, 128, 128),
    (256, 256, 128),
    (256, 256, 256),
)


@target("Sunmmio")
def zz_major_sub_block_copy_kernel():
    shape = (128, 128)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: global_layout})
            T.copy(A[0:8, 0:32], A_shared[0:8, 0:32])

    return main


@target("Sunmmio")
def zz_non_major_sub_block_copy_kernel():
    shape = (128, 128)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: global_layout})
            T.copy(A[0:32, 0:8], A_shared[0:32, 0:8])

    return main


@target("Sunmmio")
def zz_both_dims_sub_block_copy_kernel():
    shape = (128, 128)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: global_layout})
            T.copy(A[0:8, 0:8], A_shared[0:8, 0:8])

    return main


@target("Sunmmio")
def zz_major_sub_block_multi_block_copy_kernel():
    shape = (128, 128)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: global_layout})
            T.copy(A[0:8, 0:64], A_shared[0:8, 0:64])

    return main


@target("Sunmmio")
def zz_fully_coalesced_full_block_copy_kernel():
    shape = (64, 32)
    region_shape = (32, 32)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    shared_layout = make_row_major(region_shape)

    @T.prim_func
    def main(
        B: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            B_shared = T.alloc_shared(region_shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({B_shared: shared_layout})
            for bk in T.serial(2):
                T.copy(B[bk * 32, 0], B_shared)

    return main


@target("Sunmmio")
def zz_fully_coalesced_non_major_sub_block_copy_kernel():
    shape = (64, 32)
    placement = T.placement.replicated()
    global_layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        B: T.MeshTensor(shape, placement, DTYPE, layout=global_layout),  # type: ignore
    ):
        with T.Kernel():
            B_shared = T.alloc_shared(shape, DTYPE, scope="shared.rsram")
            T.annotate_layout({B_shared: global_layout})
            T.copy(B[0:32, 0:8], B_shared[0:32, 0:8])

    return main


@target("Sunmmio")
def let_bound_mesh_local_shape_copy_kernel():
    global_shape = (256, 256)
    shard_policy = T.placement.full_shard(0, 1)
    tensor_layout = make_zz_layout(global_shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(global_shape, shard_policy, DTYPE, layout=tensor_layout),  # type: ignore
    ):
        with T.Kernel():
            local_m, local_n = A.local_shape
            A_shared = T.alloc_shared((local_m, local_n), DTYPE)
            T.copy(A, A_shared)

    return main


@target("Sunmmio")
def singleton_dimension_broadcast_kernel():
    src_shape = (1, 32, 32)
    dst_shape = (1, 32, 32)
    src_layout = make_zz_layout(src_shape, axes=[1, 2], block_shape=(32, 32))
    dst_layout = make_zz_layout(dst_shape, axes=[1, 2], block_shape=(32, 32))

    @T.prim_func
    def main():
        with T.Kernel():
            src_shared = T.alloc_shared(src_shape, DTYPE, scope="shared.rsram")
            dst_shared = T.alloc_shared(dst_shape, DTYPE, scope="shared.rsram")
            T.annotate_layout(
                {
                    src_shared: src_layout,
                    dst_shared: dst_layout,
                }
            )
            T.comm.broadcast(src_shared, dst_shared, (0, 0), direction="h")

    return main


@target("Sunmmio")
def summa_output_copy_kernel(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    dtype="float16",
    accum_dtype="float32",
):
    shard_policy = T.placement.full_shard(0, 1)
    A_shape = (M, K)
    B_shape = (K, N)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, axes=[0, 1], block_shape=(32, 32))
    B_layout = make_zz_layout(B_shape, axes=[0, 1], block_shape=(32, 32))
    C_layout = make_zz_layout(C_shape, axes=[0, 1], block_shape=(32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor(A_shape, shard_policy, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor(B_shape, shard_policy, dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor(C_shape, shard_policy, accum_dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel():
            sharded_M, _ = A.local_shape
            _, sharded_N = B.local_shape
            C_local = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(C_local, C[bx * block_M, by * block_N])

    return main


def _validate_copy_codegen(kernel, tmp_path, mlir_filename, tile_shape, element_type="bf16"):
    src = validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=("suvm.copy_async", "suvm.get_partitioned_tile_view"),
        opt_args=LOOSE_OPT_ARGS,
    )
    assert f"!suvm.tile_view<{tile_shape}x{element_type}>" in src
    return src


def test_zz_major_sub_block_copy_passes(tmp_path):
    _validate_copy_codegen(
        zz_major_sub_block_copy_kernel(),
        tmp_path,
        "zz_major_sub_block_copy.mlir",
        "8x32",
    )


def test_zz_non_major_sub_block_copy_lowers_in_compact_mode():
    lowered = lower_sunmmio_kernel_to_device_tir(zz_non_major_sub_block_copy_kernel())

    assert lowered.get_global_vars()


def test_zz_both_dims_sub_block_copy_lowers_in_compact_mode():
    lowered = lower_sunmmio_kernel_to_device_tir(zz_both_dims_sub_block_copy_kernel())

    assert lowered.get_global_vars()


def test_zz_major_sub_block_multi_block_copy_passes(tmp_path):
    _validate_copy_codegen(
        zz_major_sub_block_multi_block_copy_kernel(),
        tmp_path,
        "zz_major_sub_block_multi_block_copy.mlir",
        "8x64",
    )


def test_zz_fully_coalesced_full_block_copy_passes(tmp_path):
    _validate_copy_codegen(
        zz_fully_coalesced_full_block_copy_kernel(),
        tmp_path,
        "zz_fully_coalesced_full_block_copy.mlir",
        "32x32",
    )


def test_zz_fully_coalesced_non_major_sub_block_copy_lowers_in_compact_mode():
    lowered = lower_sunmmio_kernel_to_device_tir(zz_fully_coalesced_non_major_sub_block_copy_kernel())

    assert lowered.get_global_vars()


def test_let_bound_mesh_local_shape_copy_codegen_passes(tmp_path):
    src = _validate_copy_codegen(
        let_bound_mesh_local_shape_copy_kernel(),
        tmp_path,
        "let_bound_mesh_local_shape_copy.mlir",
        "64x64",
    )

    assert "sunmmio.fake" not in src
    assert src.count("suvm.copy_async") == 1
    assert src.count("suvm.get_partitioned_tile_view") == 2


def test_singleton_dimension_broadcast_codegen_passes(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        singleton_dimension_broadcast_kernel(),
        tmp_path,
        mlir_filename="singleton_dimension_broadcast.mlir",
        expected_tokens=(
            "suvm.get_partitioned_tile_view",
            "suvm.mcast_tok",
            "suvm.sync",
        ),
        opt_args=LOOSE_OPT_ARGS,
    )
    assert "!suvm.tile_view<32x32xbf16>" in src
    assert "tiled_dims = [1, 2]" in src
    assert src.count("suvm.mcast_tok") >= 1
    assert "sunmmio.fake" not in src


@pytest.mark.parametrize("M,N,K", _SUMMA_COPY_SHAPES)
def test_summa_output_copy_codegen_passes(tmp_path, M, N, K):
    src = _validate_copy_codegen(
        summa_output_copy_kernel(M=M, N=N, K=K),
        tmp_path,
        f"summa_output_copy_m{M}_n{N}_k{K}.mlir",
        "32x32",
        element_type="f32",
    )
    assert "sunmmio.fake" not in src
    assert src.count("suvm.copy_async") >= 1


if __name__ == "__main__":
    tilelang.testing.main()
