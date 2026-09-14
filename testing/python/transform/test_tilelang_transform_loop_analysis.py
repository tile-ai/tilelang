"""Loop rewrites must not reuse the original loop body's analyzer bindings."""

import pytest
import torch

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


_MODES = ["tensor", "dependent", "bind_chain", "single", "local", "static"]


def _nested_kernel(n, block, mode):
    @T.prim_func
    def main(bounds: T.Tensor((n, 4), T.int32), out: T.Tensor((n, 4, 4), T.int32)):
        with T.Kernel(T.ceildiv(n, block), threads=32) as bx:
            for p in T.Parallel(block):
                q = bx * block + p
                if q < n:
                    if mode == "tensor":
                        for y in T.serial(bounds[q, 0], bounds[q, 1]):
                            for x in T.serial(bounds[q, 2], bounds[q, 3]):
                                out[q, y, x] = q * 100 + y * 10 + x
                    elif mode == "dependent":
                        for y in T.serial(bounds[q, 0], bounds[q, 1]):
                            for x in T.serial(bounds[q, 2], T.min(bounds[q, 3] + y, 4)):
                                out[q, y, x] = q * 100 + y * 10 + x
                    elif mode == "bind_chain":
                        height = q % 4 + 1
                        for y in T.serial(height):
                            width = T.min(height, y + 1)
                            for x in T.serial(width):
                                out[q, y, x] = q * 100 + y * 10 + x
                    elif mode == "single":
                        for y in T.serial(bounds[q, 0], bounds[q, 1]):
                            out[q, y, 0] = q * 100 + y * 10
                    elif mode == "local":
                        height = T.alloc_var(T.int32, init=bounds[q, 1])
                        width = T.alloc_var(T.int32, init=bounds[q, 3])
                        for y in T.serial(height):
                            for x in T.serial(width):
                                out[q, y, x] = q * 100 + y * 10 + x
                    else:
                        for y in T.serial(4):
                            for x in T.serial(4):
                                out[q, y, x] = q * 100 + y * 10 + x

    return main


def _lower(func, arch="sm_80"):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        return tl.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_90", "sm_120"])
@pytest.mark.parametrize("n,block", [(32, 32), (35, 64), (97, 64)])
@pytest.mark.parametrize("mode", _MODES)
def test_nested_serial_after_thread_partition(mode, n, block, arch):
    # Includes nonzero minima, a dependent inner range, and more logical
    # iterations than threads. This path used to rebind y from bounds[p] to
    # bounds[tx], or to a residual-loop/thread expression.
    _lower(_nested_kernel(n, block, mode), arch)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("mode", _MODES)
def test_nested_serial_after_thread_partition_execution(mode):
    n = 97
    data = [[i % 3, i % 5, (i + 1) % 3, (i * 3) % 5] for i in range(n)]
    bounds = torch.tensor(data, device="cuda", dtype=torch.int32)
    out = torch.full((n, 4, 4), -1, device="cuda", dtype=torch.int32)
    expected = torch.full((n, 4, 4), -1, dtype=torch.int32)
    for q, (y_start, y_stop, x_start, x_stop) in enumerate(data):
        if mode in ("bind_chain", "local", "static"):
            y_start, x_start = 0, 0
        if mode == "bind_chain":
            y_stop = q % 4 + 1
        elif mode == "static":
            y_stop, x_stop = 4, 4
        elif mode == "single":
            x_start, x_stop = 0, 1
        for y in range(y_start, y_stop):
            stop = min(x_stop + y, 4) if mode == "dependent" else x_stop
            if mode == "bind_chain":
                stop = min(y_stop, y + 1)
            for x in range(x_start, stop):
                expected[q, y, x] = q * 100 + y * 10 + x

    kernel = tl.compile(_nested_kernel(n, 64, mode))
    kernel(bounds, out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def _sibling_kernel():
    @T.prim_func
    def main(bounds: T.Tensor((2, 35, 2), T.int32), out: T.Tensor((2, 35, 4, 4), T.int32)):
        with T.Kernel(1, threads=32):
            for batch in T.serial(2):
                for p in T.Parallel(35):
                    if p % 2 == 0:
                        for y in T.serial(bounds[batch, p, 0]):
                            for x in T.serial(T.min(bounds[batch, p, 1] + y, 4)):
                                out[batch, p, y, x] = batch * 10000 + p * 100 + y * 10 + x
                    else:
                        for y in T.serial(bounds[batch, p, 0]):
                            for x in T.serial(bounds[batch, p, 1]):
                                out[batch, p, y, x] = batch * 10000 + p * 100 + y * 10 + x
                for p in T.Parallel(35):
                    for y in T.serial(bounds[batch, p, 0]):
                        for x in T.serial(bounds[batch, p, 1]):
                            out[batch, p, y, x] += 1

    return main


@tilelang.testing.requires_cuda
def test_sibling_regions_under_outer_loop_and_branches():
    _lower(_sibling_kernel())


@tilelang.testing.requires_cuda
def test_sibling_regions_under_outer_loop_and_branches_execution():
    data = [[[((p + b) % 5), ((p * 3 + b) % 5)] for p in range(35)] for b in range(2)]
    bounds = torch.tensor(data, device="cuda", dtype=torch.int32)
    out = torch.full((2, 35, 4, 4), -1, device="cuda", dtype=torch.int32)
    expected = torch.full((2, 35, 4, 4), -1, dtype=torch.int32)
    for b in range(2):
        for p, (height, width) in enumerate(data[b]):
            for y in range(height):
                for x in range(min(width + y, 4) if p % 2 == 0 else width):
                    expected[b, p, y, x] = b * 10000 + p * 100 + y * 10 + x
                for x in range(width):
                    expected[b, p, y, x] += 1
    tl.compile(_sibling_kernel())(bounds, out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("aligned_by", ["binding", "constraint"])
def test_outer_binding_and_constraint_preserve_vectorization(aligned_by):
    @T.prim_func
    def main(A: T.Tensor((1024,), T.float32), B: T.Tensor((1024,), T.float32), shift: T.int32):
        with T.Kernel(1, threads=32):
            offset = shift * 4 if aligned_by == "binding" else shift
            if shift >= 0 and shift < 16 and shift % 4 == 0:
                for p in T.Parallel(128):
                    B[offset + p] = A[offset + p]

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    with target:
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
    extents = []

    def collect(node):
        if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.VECTORIZED:
            extents.append(int(node.extent))

    post_order_visit(mod["main"].body, collect)
    assert 4 in extents, mod.script()
    defined = [*mod["main"].params, *(buffer.data for buffer in mod["main"].buffer_map.values())]
    assert not tvm.tirx.analysis.undefined_vars(mod["main"].body, defined)


def test_conflicting_analyzer_bindings_still_raise():
    analyzer = tvm.arith.Analyzer()
    p = tvm.tirx.Var("p", "int32")
    tx = tvm.tirx.Var("tx", "int32")
    y = tvm.tirx.Var("y", "int32")
    bounds = tvm.tirx.decl_buffer((32,), "int32", name="bounds")
    analyzer.bind(y, tvm.ir.Range.from_min_extent(0, bounds[p]))
    with pytest.raises(tvm.error.InternalError, match="different maximum value"):
        analyzer.bind(y, tvm.ir.Range.from_min_extent(0, bounds[tx]))


if __name__ == "__main__":
    tilelang.testing.main()
