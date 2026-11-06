"""CPU launch validation and metadata lowering."""

import pytest
import tvm_ffi

import tilelang
import tilelang.cpu.language as T
import tvm_tirx
from tvm_tirx import ir
from tilelang.testing.ir import collect_nodes


def _nodes(func, kind):
    return collect_nodes(func, kind)


def _launches(func):
    return [region for region in _nodes(func, ir.RegionStmt) if region.op.name == "tl.launch_kernel"]


def _launch_hints(region):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[4], region.attrs["hint_schema"])


def _kernel_headers(func):
    headers = []
    for region in _nodes(func, ir.RegionStmt):
        if region.op.name == "tl.attr":
            fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"])
            if fields["key"] == "tl.kernel":
                headers.append(fields["value"])
    return headers


def _materialize(mod):
    # Native CPU grids are ordinary For nodes before the thread-count pass.
    return tilelang.transform.MaterializeKernelLaunch(lower_grid_binding=False, lower_thread_binding=False)(mod)


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

    launches = _launches(main)
    assert len(launches) == 2
    assert sorted(int(_launch_hints(region)["tl.cpu_num_threads"]) for region in launches) == [3, 5]
    assert all("tl.cpu_num_threads" not in loop.annotations for loop in _nodes(main, ir.For))

    mod = tvm_tirx.IRModule.from_expr(main)
    with tvm_tirx.transform.PassContext(config={"tl.cpu_parallel": parallel}):
        materialized = _materialize(mod)
        prepared = tilelang.cpu.transform.LowerCPUKernelLaunch()(materialized)
        lowered = tilelang.transform.LowerSourceAnnotations()(prepared)
    result = prepared["main"]
    loops = _nodes(result, ir.For)
    assert len(loops) == 4 and all(loop.kind == ir.ForKind.DEFAULT for loop in loops)
    outer = [loop for loop in loops if any(loop.var.same_as(region.body_params[0]) for region in launches)]
    inner = [loop for loop in loops if any(loop.var.same_as(region.body_params[1]) for region in launches)]
    assert len(outer) == len(inner) == 2
    assert sorted(int(loop.annotations["tl.cpu_num_threads"]) for loop in outer) == [3, 5]
    assert all("tl.cpu_num_threads" not in loop.annotations for loop in inner)
    assert all("tl.cpu_num_threads" not in header for header in _kernel_headers(result))
    if parallel:
        assert all(int(loop.annotations["tl.cpu_grid_dim"]) == 0 for loop in outer)
        assert all(int(loop.annotations["tl.cpu_grid_dim"]) == 1 for loop in inner)
    else:
        assert all("tl.cpu_grid_dim" not in loop.annotations for loop in loops)

    for original, serial in zip(loops, _nodes(lowered["main"], ir.For), strict=True):
        assert serial.kind == ir.ForKind.DEFAULT
        assert serial.var.same_as(original.var)
        tvm_tirx.ir.assert_structural_equal(original.annotations, serial.annotations)


def test_cpu_launch_marks_only_grid_loops():
    @T.prim_func
    def main(A: T.Tensor((48,), T.int32)):
        with T.Kernel(2, 3, 4) as (bx, by, bz):
            for i in T.serial(2):
                A[((bx * 3 + by) * 4 + bz) * 2 + i] = 1

    (launch,) = _launches(main)
    with tvm_tirx.transform.PassContext(config={"tl.cpu_parallel": True}):
        materialized = _materialize(tvm_tirx.IRModule.from_expr(main))
        prepared = tilelang.cpu.transform.LowerCPUKernelLaunch()(materialized)
    loops = _nodes(prepared["main"], ir.For)
    grid = [next(loop for loop in loops if loop.var.same_as(var)) for var in launch.body_params[:3]]
    assert [int(loop.annotations["tl.cpu_grid_dim"]) for loop in grid] == [0, 1, 2]
    assert all("tl.cpu_grid_dim" not in loop.annotations for loop in loops if all(not loop.same_as(axis) for axis in grid))
    serial = _nodes(materialized["main"], ir.For)
    assert sorted(int(loop.annotations["tl.cpu_grid_dim"]) for loop in serial if "tl.cpu_grid_dim" in loop.annotations) == [0, 1, 2]
    assert sum("tl.cpu_grid_dim" not in loop.annotations for loop in serial) == 1


def test_cpu_launch_without_thread_count_is_unchanged():
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        with T.Kernel(2) as bx:
            A[bx] = 1

    mod = _materialize(tvm_tirx.IRModule.from_expr(main))
    original = mod["main"].body[0]
    result = tilelang.cpu.transform.LowerCPUKernelLaunch()(mod)["main"]
    (serial,) = _nodes(result, ir.For)
    assert serial.var.same_as(original.var)
    assert "tl.cpu_num_threads" not in serial.annotations
    # Disabling parallelism consumes the intermediate grid marker. The loop
    # range, body and variable identity retain the original serial behavior.
    expected = ir.For(original.var, original.begin, original.end, original.step, original.kind, original.body)
    tvm_tirx.ir.assert_structural_equal(expected, serial)
