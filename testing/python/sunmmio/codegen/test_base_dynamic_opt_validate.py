import os

import tilelang
import tilelang.language as T
import tilelang.testing
import tvm_ffi
from tilelang.layout import (
    make_nzz_layout,
    make_zn_layout,
    make_zz_layout,
    make_zzz_layout,
)

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    lower_sunmmio_kernel_to_device_tir,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()

os.environ["SUNMMIO_TEST_PRINT"] = "0"


_layout_logical_shape = tvm_ffi.get_global_func("tl.CuteLayout_logical_shape")
_layout_mode_shape = tvm_ffi.get_global_func("tl.CuteLayout_mode_shape")
_layout_mode_stride = tvm_ffi.get_global_func("tl.CuteLayout_mode_stride")
_layout_dim_levels = tvm_ffi.get_global_func("tl.CuteLayout_dim_levels")
_layout_covered_shape = tvm_ffi.get_global_func("tl.CuteLayout_covered_shape")
_layout_storage_size = tvm_ffi.get_global_func("tl.CuteLayout_storage_size")


def _as_list(value):
    return [str(item) for item in value]


def _layout_obj(layout):
    return getattr(layout, "_inner", layout)


def get_layout_info(layout):
    layout = _layout_obj(layout)
    return {
        "logical_shape": _as_list(_layout_logical_shape(layout)),
        "mode_shape": _as_list(_layout_mode_shape(layout)),
        "mode_stride": _as_list(_layout_mode_stride(layout)),
        "dim_levels": _as_list(_layout_dim_levels(layout)),
        "covered_shape": _as_list(_layout_covered_shape(layout)),
        "storage_size": str(_layout_storage_size(layout)),
    }


def print_layout_info(name, layout):
    for key, value in get_layout_info(layout).items():
        print(f"{name}.{key} = {value}")


def print_kernel_layouts(name, kernel):
    print(f"========== {name} tensor layouts ==========")
    tensor_meta = kernel.attrs["tensor_meta"]
    for tensor_name in ("A", "B", "C"):
        meta = tensor_meta[tensor_name]
        print(f"-- {tensor_name} global_layout --")
        print_layout_info(f"{tensor_name}.global", meta["global_layout"])
        print(f"-- {tensor_name} sharded_layout --")
        print_layout_info(f"{tensor_name}.sharded", meta["sharded_layout"])


@target("Sunmmio")
def dynamic_allocate_copy_mma_kernel(
    block_M=32,
    block_N=32,
    block_K=32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    nrows, ncols = T.nrows(), T.ncols()

    M_sharded = T.dynamic("m")
    N_sharded = T.dynamic("n")
    K_sharded = T.dynamic("k")
    M = M_sharded * nrows
    N = N_sharded * ncols
    K = K_sharded * ncols
    # K = K_sharded
    # A_layout = make_zz_layout((M, K))
    # B_layout = make_zz_layout((K, N))
    # C_layout = make_zz_layout((M, N))

    shard_policy = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), shard_policy, dtype),  # type: ignore
        B: T.MeshTensor((K, N), shard_policy, dtype),  # type: ignore
        C: T.MeshTensor((M, N), shard_policy, accum_dtype),  # type: ignore
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            sharded_N = B.local_shape[1]
            A_shared_dist = T.alloc_shared((block_M, block_K * T.ncols()), dtype)
            B_shared_dist = T.alloc_shared((block_K * T.nrows(), block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for by in T.serial(T.ceildiv(sharded_M, block_M)):
                for bn in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for ko in T.serial(T.ceildiv(sharded_K, block_K)):
                        T.comm.all_gather(
                            A[
                                by * block_M : (by + 1) * block_M,
                                ko * block_K : (ko + 1) * block_K,
                            ],
                            A_shared_dist,
                            direction="horizontal",
                            axis=-1,
                        )
                        T.comm.all_gather(
                            B[
                                ko * block_K : (ko + 1) * block_K,
                                bn * block_N : (bn + 1) * block_N,
                            ],
                            B_shared_dist,
                            direction="vertical",
                            axis=0,
                        )
                        T.gemm(A_shared_dist, B_shared_dist, C_shared)
                    T.copy(C_shared, C[by * block_M, bn * block_N])

    return main


@target("Sunmmio")
def dynamic_zz_allocate_copy_mma_kernel(
    block_M=32,
    block_N=32,
    block_K=32,
    dtype=T.bfloat16,
):
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = T.dynamic("k")
    A_layout = make_zz_layout((M, K))
    B_layout = make_zz_layout((K, N))
    C_layout = make_zz_layout((M, N))

    shard_policy = T.placement.replicated()

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), shard_policy, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor((K, N), shard_policy, dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor((M, N), shard_policy, dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel() as _cid:
            sharded_M = A.local_shape[0]
            sharded_N = B.local_shape[1]
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for by in T.serial(T.ceildiv(sharded_M, block_M)):
                for bn in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(A[by * block_M, 0], A_shared)
                    T.copy(B[0, bn * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_shared)
                    # need all_gather or reduce to cover real-world scenarios
                    # T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def test_dynamic_named_layout_constructors_accept_symbolic_shapes():
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = T.dynamic("k")

    layouts = {
        "zz": make_zz_layout((M, K)),
        "zn": make_zn_layout((M, K), [0, 1], (32, 32)),
        "zzz": make_zzz_layout((M, N), [0, 1], (32, 32), (2, 2)),
        "nzz": make_nzz_layout((K, N), [0, 1], (32, 32), (2, 2)),
    }
    for name, layout in layouts.items():
        print_layout_info(name, layout)
    assert all(layout is not None for layout in layouts.values())


def test_print_dynamic_kernel_layouts():
    print_kernel_layouts("dynamic row-major", dynamic_allocate_copy_mma_kernel())
    print_kernel_layouts("dynamic zz", dynamic_zz_allocate_copy_mma_kernel())


def test_dynamic_allocate_copy_mma_lowers_to_device_tir():
    device_mod = lower_sunmmio_kernel_to_device_tir(
        dynamic_allocate_copy_mma_kernel(),
        print_ir=False,
    )
    script = device_mod.script()
    assert_source_contains(
        script,
        (
            "k: T.int32",
            "m: T.int32",
            "n: T.int32",
            "for by, bn in T.grid((m + 31) // 32, (n + 31) // 32)",
            "for ko in range((k + 31) // 32)",
            "T.dma_copy",
            "T.broadcast_",
            "T.mma_sunmmio",
        ),
    )


def test_dynamic_zz_allocate_copy_mma_codegen_validates_with_npuir_opt(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        dynamic_zz_allocate_copy_mma_kernel(),
        tmp_path,
        mlir_filename="dynamic_zz_allocate_copy_mma_suvm.mlir",
        expected_tokens=(
            "arith.addi",
            "arith.divsi",
            "suvm.bind_layout",
            "dynamic_shapes =",
            "dynamic_strides =",
            "suvm.copy_async",
            "suvm.tc.mma",
        ),
    )


def test_dynamic_allocate_copy_mma_codegen_validates_with_npuir_opt(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        dynamic_allocate_copy_mma_kernel(),
        tmp_path,
        mlir_filename="dynamic_allocate_copy_mma_suvm.mlir",
        expected_tokens=(
            "suvm.bind_layout",
            "suvm.alloc",
            "suvm.mcast_tok",
            "suvm.sync",
            "suvm.copy_async",
            "suvm.tc.mma",
        ),
    )


if __name__ == "__main__":
    tilelang.testing.main()
