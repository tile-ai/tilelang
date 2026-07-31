import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue


def _descriptor_base_byte_offsets(func, arch):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
    with target, tvm.transform.PassContext(config=pass_configs):
        mod = CUDAPassPipelineBodyPrologue(tvm.IRModule.from_expr(func), target)

    create_tiled = tvm.ir.Op.get("tl.create_tma_descriptor")
    create_im2col = tvm.ir.Op.get("tl.create_tma_im2col_descriptor")
    handle_add = tvm.ir.Op.get("tirx.handle_add_byte_offset")
    offsets = []

    def collect(expr):
        if not isinstance(expr, tvm.tirx.Call):
            return
        if not (expr.op.same_as(create_tiled) or expr.op.same_as(create_im2col)):
            return
        base = expr.args[2]
        assert isinstance(base, tvm.tirx.Call)
        assert base.op.same_as(handle_add)
        offset = tvm.arith.Analyzer().simplify(base.args[1])
        assert isinstance(offset, tvm.tirx.IntImm)
        offsets.append(int(offset.value))

    for global_var in mod.get_global_vars():
        tvm.tirx.stmt_functor.post_order_visit(mod[global_var].body, collect)
    return sorted(offsets)


def _bulk_copy_program():
    @T.prim_func
    def main(src_handle: T.handle, dst_handle: T.handle):
        src = T.match_buffer(src_handle, (16, 80), T.float16, elem_offset=16)
        dst = T.match_buffer(dst_handle, (16, 80), T.float16, elem_offset=32)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((8, 64), T.float16)
            mbar = T.alloc_barrier(1)
            T.tma_copy(src[0:8, 0:64], shared, barrier=mbar)
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(shared, dst[0:8, 0:64])

    return main


def _gather_scatter_program():
    @T.prim_func
    def main(src_handle: T.handle, rows: T.Tensor((4,), T.int32), dst_handle: T.handle):
        src = T.match_buffer(src_handle, (64, 64), T.float16, elem_offset=16)
        dst = T.match_buffer(dst_handle, (64, 64), T.float16, elem_offset=32)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((4, 64), T.float16)
            mbar = T.alloc_barrier(1)
            row_indices = [rows[0], rows[1], rows[2], rows[3]]
            T.tma_gather4(src, shared, 0, row_indices, barrier=mbar)
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.tma_scatter4(shared, dst, 0, row_indices)

    return main


def _im2col_program():
    @T.prim_func
    def main(src_handle: T.handle):
        src = T.match_buffer(src_handle, (1, 8, 8, 64), T.float16, elem_offset=16)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 32), T.float16)
            T.im2col(src, shared, 0, 0, 3, 1, 1, 1)

    return main


def _tma_atomic_program():
    @T.prim_func
    def main(dst_handle: T.handle):
        dst = T.match_buffer(dst_handle, (16, 16), T.float32, elem_offset=8)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 16), T.float32)
            T.atomic_add(dst, shared, use_tma=True)

    return main


@pytest.mark.parametrize(
    ("program", "arch", "expected_offsets"),
    [
        (_bulk_copy_program, "sm_90", [32, 64]),
        (_gather_scatter_program, "sm_100a", [32, 64]),
        (_im2col_program, "sm_90", [32]),
        (_tma_atomic_program, "sm_90", [32]),
    ],
)
def test_tma_descriptor_base_includes_buffer_elem_offset(program, arch, expected_offsets):
    assert _descriptor_base_byte_offsets(program(), arch) == expected_offsets
