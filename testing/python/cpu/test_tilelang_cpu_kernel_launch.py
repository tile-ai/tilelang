"""CPU launch validation and metadata lowering."""

import pytest

import tilelang
import tilelang.cpu.language as T
from tilelang import tvm
from tvm import tirx


def _nodes(func, kind):
    nodes = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: nodes.append(node) if isinstance(node, kind) else None)
    return nodes


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "4"])
def test_cpu_num_threads_validation(count):
    with pytest.raises(ValueError, match="cpu_num_threads must be a positive integer"):

        @T.prim_func
        def main(A: T.Tensor((2,), T.int32)):
            with T.Kernel(2, cpu_num_threads=count) as bx:
                A[bx] = 1


@pytest.mark.parametrize("parallel", [False, True])
def test_cpu_launch_thread_count_is_per_kernel(parallel):
    @T.prim_func
    def main(A: T.Tensor((8,), T.int32)):
        with T.Kernel(2, 2, cpu_num_threads=3) as (bx, by):
            A[bx * 2 + by] = 1
        with T.Kernel(2, 2, cpu_num_threads=5) as (bx, by):
            A[4 + bx * 2 + by] = 2

    blocks = [block for block in _nodes(main, tirx.SBlock) if "tl.cpu_num_threads" in block.annotations]
    assert sorted(int(block.annotations["tl.cpu_num_threads"]) for block in blocks) == [3, 5]
    assert all("tl.cpu_num_threads" not in loop.annotations for loop in _nodes(main, tirx.For))

    mod = tvm.IRModule.from_expr(main)
    with tvm.transform.PassContext(config={"tl.cpu_parallel": parallel}):
        prepared = tilelang.cpu.transform.LowerCPUKernelLaunch()(mod)
        materialized = tilelang.transform.MaterializeKernelLaunch(lower_thread_binding=False)(prepared)
        lowered = tilelang.transform.LowerOpaqueBlock()(materialized)
    result = prepared["main"]
    loops = _nodes(result, tirx.For)
    assert all(loop.kind == tirx.ForKind.THREAD_BINDING for loop in loops)
    outer = [loop for loop in loops if str(loop.thread_binding.thread_tag) == "blockIdx.x"]
    inner = [loop for loop in loops if str(loop.thread_binding.thread_tag) == "blockIdx.y"]
    assert sorted(int(loop.annotations["tl.cpu_num_threads"]) for loop in outer) == [3, 5]
    assert all("tl.cpu_num_threads" not in loop.annotations for loop in inner)
    assert all("tl.cpu_num_threads" not in block.annotations for block in _nodes(result, tirx.SBlock))
    if parallel:
        assert all(int(loop.annotations["tl.cpu_grid_dim"]) == 0 for loop in outer)
        assert all(int(loop.annotations["tl.cpu_grid_dim"]) == 1 for loop in inner)
    else:
        assert all("tl.cpu_grid_dim" not in loop.annotations for loop in loops)

    for original, serial in zip(loops, _nodes(lowered["main"], tirx.For), strict=True):
        assert serial.kind == tirx.ForKind.SERIAL
        assert serial.loop_var.same_as(original.loop_var)
        tvm.ir.assert_structural_equal(original.annotations, serial.annotations)


def test_cpu_launch_marks_only_grid_loops():
    @T.prim_func
    def main(A: T.Tensor((48,), T.int32)):
        with T.Kernel(2, 3, 4) as (bx, by, bz):
            for i in T.serial(2):
                A[((bx * 3 + by) * 4 + bz) * 2 + i] = 1

    with tvm.transform.PassContext(config={"tl.cpu_parallel": True}):
        prepared = tilelang.cpu.transform.LowerCPUKernelLaunch()(tvm.IRModule.from_expr(main))
        materialized = tilelang.transform.MaterializeKernelLaunch(lower_thread_binding=False)(prepared)
    loops = _nodes(prepared["main"], tirx.For)
    grid = {str(loop.thread_binding.thread_tag): loop for loop in loops if loop.thread_binding is not None}
    assert [int(grid[f"blockIdx.{axis}"].annotations["tl.cpu_grid_dim"]) for axis in "xyz"] == [0, 1, 2]
    assert all("tl.cpu_grid_dim" not in loop.annotations for loop in loops if loop.thread_binding is None)
    serial = _nodes(materialized["main"], tirx.For)
    assert sorted(int(loop.annotations["tl.cpu_grid_dim"]) for loop in serial if "tl.cpu_grid_dim" in loop.annotations) == [0, 1, 2]
    assert sum("tl.cpu_grid_dim" not in loop.annotations for loop in serial) == 1


def test_cpu_launch_without_thread_count_is_unchanged():
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        with T.Kernel(2) as bx:
            A[bx] = 1

    result = tilelang.cpu.transform.LowerCPUKernelLaunch()(tvm.IRModule.from_expr(main))["main"]
    tvm.ir.assert_structural_equal(main, result)
