"""TMEM allocation must follow the inferred physical fragment shape."""

import pytest

from tilelang import tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.cuda.intrinsics.layout.mma_sm100_layout import (
    TCGEN05Meta,
    make_tmem_frg_c,
)


TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100"})


def _lower(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    mod = tl.transform.LayoutInference()(mod)
    # LowerTileOp materializes the inferred TMEM layout (physical buffer
    # shape + physical access coordinates) before allocation sizing.
    mod = tl.transform.LowerTileOp()(mod)
    return tl.cuda.transform.LowerSharedTmem()(mod)


def _collect_calls(stmt, op_name):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", None) == op_name:
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, visitor)
    return calls


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    ("dtype", "expected_num_b32_cols"),
    [(T.float32, 128), (T.bfloat16, 64)],
)
def test_tmem_outer_m_tiles_allocate_physical_columns(dtype, expected_num_b32_cols):
    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            tmem = T.alloc_tmem((256, 64), dtype)
            T.annotate_layout(
                {
                    tmem: T.Layout(
                        (256, 64),
                        lambda i, j: [i % 128, (i // 128) * 64 + j],
                    )
                }
            )
            T.evaluate(tmem[0, 0])

    body = _lower(func)["main"].body
    alloc = _collect_calls(body, "tl.ptx_init_tensor_memory")
    dealloc = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")
    assert len(alloc) == len(dealloc) == 1
    assert alloc[0].args[1].value == expected_num_b32_cols
    assert dealloc[0].args[1].value == expected_num_b32_cols


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16, T.float32])
def test_tmem_frg_type_c_allocates_int32_storage_columns(dtype):
    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            c_tmem = T.alloc_tmem((128, 64), dtype)
            fragment = make_tmem_frg_c(c_tmem, TCGEN05Meta(128, 64, 16, False, False), is_ts=True)
            T.annotate_layout({c_tmem: fragment.to_tilelang()})
            T.evaluate(c_tmem[0, 0])

    body = _lower(func)["main"].body
    alloc = _collect_calls(body, "tl.ptx_init_tensor_memory")
    dealloc = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")
    assert len(alloc) == len(dealloc) == 1
    assert alloc[0].args[1].value == 64
    assert dealloc[0].args[1].value == 64


if __name__ == "__main__":
    pytest.main([__file__])
