import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    find_async_op_lines,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def comm_broadcast_kernel(M=128, N=128, block_M=32, block_N=32, dtype="bfloat16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            for bx in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[0, bx * block_N], A_shared)
                T.comm.broadcast(A_shared, B_shared, (0, 0), direction="h")
                T.copy(B_shared, B[0, bx * block_N])

    return main


@target("Sunmmio")
def comm_put_kernel(M=128, N=128, block_M=32, block_N=32, dtype="bfloat16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype)):
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            B_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            for bx in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[0, bx * block_N], A_shared)
                T.comm.put(A_shared, B_shared, (1, 2), (2, 3))

    return main


@target("Sunmmio")
def comm_all_gather_kernel(
    *,
    M=128,
    N=128,
    block_M=32,
    block_N=32,
    dtype="bfloat16",
    direction="all",
    axis=None,
):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype)):
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")
            if axis == 0:
                R_shared = T.alloc_shared((16 * block_M, block_N), dtype, scope="shared.rsram")
            elif axis == -1:
                extent = 4 * block_N if direction in ("h", "horizontal") else 16 * block_N
                R_shared = T.alloc_shared((block_M, extent), dtype, scope="shared.rsram")
            else:
                R_shared = T.alloc_shared((16, block_M, block_N), dtype, scope="shared.rsram")

            for bx in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[0, bx * block_N], A_shared)
                if axis is None:
                    T.comm.all_gather(A_shared, R_shared, direction=direction)
                else:
                    T.comm.all_gather(A_shared, R_shared, direction=direction, axis=axis)

    return main


@target("Sunmmio")
def sync_simple_copy_kernel(M=128, N=128, block_M=32, block_N=32, dtype="bfloat16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            for bx in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[0, bx * block_N], A_shared)
                T.copy(A_shared, B[0, bx * block_N])

    return main


@target("Sunmmio")
def sync_mma_kernel(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype="bfloat16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
                T.copy(C_shared, C[0, bx * block_N])

    return main


@target("Sunmmio")
def sync_if_broadcast_kernel(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype="bfloat16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            D_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

                if bx == 0:
                    T.comm.broadcast(C_shared, D_shared, (0, 0), direction="h")

    return main


@target("Sunmmio")
def sync_loop_broadcast_kernel(M=128, N=128, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def main(C: T.Tensor((M, N), dtype)):
        with T.Kernel():
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            D_shared = T.alloc_shared((block_M, block_N), dtype)

            for bx in T.serial(T.ceildiv(N, block_N)):
                T.copy(C[0, bx * block_N], D_shared)
                for _i in T.serial(2):
                    T.comm.broadcast(C_shared, D_shared, (0, 0), direction="h")
                    T.comm.broadcast(D_shared, C_shared, (0, 0), direction="h")

    return main


def _validate_kernel(kernel, tmp_path, *, mlir_filename, expected_tokens=(), opt_args=LOOSE_OPT_ARGS):
    src = validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=opt_args,
    )
    assert "sunmmio.fake" not in src
    return src


def test_comm_broadcast_kernel_codegen_validates_with_npuir_opt(tmp_path):
    src = _validate_kernel(
        comm_broadcast_kernel(),
        tmp_path,
        mlir_filename="comm_broadcast_kernel_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.mcast_tok",
            "suvm.sync",
            "suvm.barrier.arrive_and_wait",
        ),
    )

    mcast_lines = find_async_op_lines(src, "suvm.mcast_tok")
    assert all("#suvm.unit<odma1>" in line for line in mcast_lines)
    lines = src.splitlines()
    barrier_indices = [i for i, line in enumerate(lines) if "suvm.barrier.arrive_and_wait" in line]
    mcast_index = next(i for i, line in enumerate(lines) if "suvm.mcast_tok" in line)
    link_sync_index = next(i for i, line in enumerate(lines) if "suvm.sync" in line and "hlink" in line)
    consumer_index = next(i for i, line in enumerate(lines) if i > mcast_index and "suvm.copy_async" in line)
    assert len(barrier_indices) >= 2
    assert barrier_indices[0] < mcast_index < link_sync_index < barrier_indices[1] < consumer_index


def test_comm_put_kernel_codegen_validates_with_npuir_opt(tmp_path):
    src = _validate_kernel(
        comm_put_kernel(),
        tmp_path,
        mlir_filename="comm_put_kernel_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.mcast_tok", "suvm.sync", "suvm.barrier.arrive_and_wait"),
    )

    assert src.count("suvm.mcast_tok") >= 2


@pytest.mark.parametrize(
    "case_name,direction,axis,min_mcast_count",
    [
        ("all_gather_axis0_all", "all", 0, 2),
        ("all_gather_axis_last_horizontal", "horizontal", -1, 1),
        ("all_gather_axis_last_all", "all", -1, 2),
    ],
)
def test_comm_all_gather_kernel_codegen_validates_with_npuir_opt(
    tmp_path,
    case_name,
    direction,
    axis,
    min_mcast_count,
):
    src = _validate_kernel(
        comm_all_gather_kernel(direction=direction, axis=axis),
        tmp_path,
        mlir_filename=f"comm_{case_name}_kernel_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.mcast_tok", "suvm.sync"),
    )

    mcast_lines = find_async_op_lines(src, "suvm.mcast_tok")
    assert len(mcast_lines) >= min_mcast_count
    for line in mcast_lines:
        direction = line.split("direction =", 1)[1].lstrip()
        expected_unit = "odma1" if direction.startswith("row") else "odma0"
        assert f"#suvm.unit<{expected_unit}>" in line


def test_sync_simple_copy_kernel_codegen_validates_with_npuir_opt(tmp_path):
    src = _validate_kernel(
        sync_simple_copy_kernel(),
        tmp_path,
        mlir_filename="sync_simple_copy_kernel_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.sync"),
    )

    copy_lines = find_async_op_lines(src, "suvm.copy_async")
    assert len(copy_lines) >= 2
    assert all("#suvm.unit<odma0>" in line for line in copy_lines)
    assert src.count("suvm.sync") == 2
    final_sync = next(line.strip() for line in reversed(src.splitlines()) if "suvm.sync" in line)
    assert "odma0" in final_sync
    assert all(unit not in final_sync for unit in ("odma1", "tc", "vector", "rsram"))


def test_sync_mma_kernel_codegen_validates_with_npuir_opt(tmp_path):
    src = _validate_kernel(
        sync_mma_kernel(),
        tmp_path,
        mlir_filename="sync_mma_kernel_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma", "suvm.sync"),
    )

    copy_lines = find_async_op_lines(src, "suvm.copy_async")
    assert len(copy_lines) >= 3
    assert {unit for unit in ("odma0", "odma1") if any(unit in line for line in copy_lines)} == {
        "odma0",
        "odma1",
    }
    assert len(find_async_op_lines(src, "suvm.tc.mma")) == 2
    final_sync = next(line.strip() for line in reversed(src.splitlines()) if "suvm.sync" in line)
    assert "odma0" in final_sync
    assert all(unit not in final_sync for unit in ("odma1", "tc", "vector", "rsram"))


def test_sync_if_broadcast_kernel_codegen_validates_with_npuir_opt(tmp_path):
    src = _validate_kernel(
        sync_if_broadcast_kernel(),
        tmp_path,
        mlir_filename="sync_if_broadcast_kernel_suvm.mlir",
        expected_tokens=(
            "scf.if",
            "suvm.copy_async",
            "suvm.tc.mma",
            "suvm.mcast_tok",
            "suvm.sync",
            "suvm.barrier.arrive_and_wait",
        ),
    )

    assert src.count("suvm.mcast_tok") >= 1


def test_sync_loop_broadcast_kernel_codegen_validates_with_npuir_opt(tmp_path):
    src = _validate_kernel(
        sync_loop_broadcast_kernel(),
        tmp_path,
        mlir_filename="sync_loop_broadcast_kernel_suvm.mlir",
        expected_tokens=("scf.for", "suvm.copy_async", "suvm.mcast_tok", "suvm.sync", "suvm.barrier.arrive_and_wait"),
    )

    assert src.count("suvm.mcast_tok") >= 2


if __name__ == "__main__":
    tilelang.testing.main()
