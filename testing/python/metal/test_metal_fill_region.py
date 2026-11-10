"""Direct SIMD-group partial fills must preserve unselected matrix tiles."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_FILL_MATRIX_OP = tvm.ir.Op.get("tirx.make_filled_simdgroup_matrix")
_CASES = [
    pytest.param(8, 16, 0, 8, [2], id="row-min"),
    pytest.param(0, 16, 0, 8, [0, 2], id="row-stride"),
    pytest.param(0, 8, 8, 16, [1], id="col-min"),
]


def _make_fill_kernel(row_begin, row_end, col_begin, col_end, threads=32):
    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=threads):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.fill(buf, T.float32(-7.0))
            T.fill(buf[row_begin:row_end, col_begin:col_end], T.float32(3.0))
            T.copy(buf, output)

    return kernel


def _lower(func):
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        return tilelang.lower(func, target="metal")


def _fill_calls(artifact):
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and node.op.same_as(_FILL_MATRIX_OP):
            calls.append(node)

    for func in artifact.device_mod.functions.values():
        if isinstance(func, tvm.tirx.PrimFunc):
            tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _fill_value(call):
    value = call.args[2]
    while isinstance(value, tvm.tirx.Cast):
        value = value.value
    assert isinstance(value, tvm.tirx.FloatImm)
    return value.value


@tilelang.testing.requires_metal
@pytest.mark.parametrize("row_begin,row_end,col_begin,col_end,expected", _CASES)
def test_partial_fill_indices(row_begin, row_end, col_begin, col_end, expected):
    artifact = _lower(_make_fill_kernel(row_begin, row_end, col_begin, col_end))
    calls = _fill_calls(artifact)
    initializers = [call for call in calls if _fill_value(call) == -7.0]
    updates = [call for call in calls if _fill_value(call) == 3.0]
    assert initializers
    buffer_var = initializers[0].args[0]
    assert all(call.args[0].same_as(buffer_var) for call in calls)
    assert all(isinstance(call.args[1], tvm.tirx.IntImm) for call in updates)
    assert [call.args[1].value for call in updates] == expected


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "region,threads,reason",
    [
        pytest.param((8, 16, 0, 8), 128, "exactly one simdgroup", id="multi-group"),
        pytest.param((1, 9, 0, 8), 32, "8-aligned", id="unaligned"),
    ],
)
def test_partial_fill_diagnostic(region, threads, reason):
    # The C++ user-facing CHECK raises a typed ValueError through TVM FFI.
    with pytest.raises(ValueError, match=reason):
        _lower(_make_fill_kernel(*region, threads=threads))


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "row_begin,row_end,col_begin,col_end,expected",
    [*_CASES, pytest.param(0, 16, 0, 16, None, id="full-fill")],
)
def test_partial_fill_preserves_outside(row_begin, row_end, col_begin, col_end, expected):
    func = _make_fill_kernel(row_begin, row_end, col_begin, col_end)
    compiled = tilelang.compile(
        func,
        out_idx=[],
        target="metal",
        target_host="c",
        execution_backend="tvm_ffi",
    )
    output = torch.full((16, 16), 5.0, dtype=torch.float32, device="mps")
    compiled(output)
    torch.mps.synchronize()

    reference = torch.full((16, 16), -7.0, dtype=torch.float32)
    reference[row_begin:row_end, col_begin:col_end] = 3.0
    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "row_begin,row_end,col_begin,col_end,expected",
    [*_CASES, pytest.param(8, 16, 8, 16, [3], id="both-mins")],
)
@pytest.mark.parametrize("shared", [False, True], ids=["global", "shared"])
def test_partial_fill_copy_region(row_begin, row_end, col_begin, col_end, expected, shared):
    rows, cols = row_end - row_begin, col_end - col_begin

    @T.prim_func
    def kernel(output: T.Tensor((24, 24), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.fill(buf, T.float32(-7))
            T.fill(buf[row_begin:row_end, col_begin:col_end], T.float32(3))
            if shared:
                staging = T.alloc_shared((24, 24), T.float32)
                T.fill(staging, T.float32(5))
                T.copy(buf[row_begin:row_end, col_begin:col_end], staging[8 : 8 + rows, 8 : 8 + cols])
                T.copy(staging, output)
            else:
                T.copy(buf[row_begin:row_end, col_begin:col_end], output[8 : 8 + rows, 8 : 8 + cols])

    compiled = tilelang.compile(kernel, out_idx=[], target="metal", target_host="c", execution_backend="tvm_ffi")
    output = torch.full((24, 24), 5.0, dtype=torch.float32, device="mps")
    compiled(output)
    torch.mps.synchronize()
    reference = torch.full((24, 24), 5.0, dtype=torch.float32)
    reference[8 : 8 + rows, 8 : 8 + cols] = 3
    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "threads,active_threads,row_begin,row_end,reason",
    [
        pytest.param(64, 64, 0, 16, "exactly one simdgroup", id="multi-group"),
        pytest.param(64, 32, 0, 16, "exactly one simdgroup", id="predicated-group"),
        pytest.param((32, 2), 32, 0, 16, "exactly one simdgroup", id="2d-launch"),
        pytest.param(32, 16, 0, 16, "complete simdgroup execution range", id="incomplete-group"),
        pytest.param(32, 32, 1, 9, "8-aligned", id="unaligned-min"),
        pytest.param(32, 32, 0, 7, "8-aligned", id="unaligned-extent"),
    ],
)
def test_partial_source_copy_diagnostic(threads, active_threads, row_begin, row_end, reason):
    @T.prim_func
    def kernel(output: T.Tensor((16, 8), T.float32)):
        with T.Kernel(1, threads=threads):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.fill(buf, T.float32(3))
            tx = T.get_thread_binding()
            if tx < active_threads:
                T.copy(buf[row_begin:row_end, 0:8], output[0 : row_end - row_begin, 0:8])

    with pytest.raises(ValueError, match="Metal partial-source copy:.*" + reason):
        _lower(kernel)


def _make_predicated_fill_kernel(threads, begin, end):
    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=threads):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.fill(buf, T.float32(-7.0))
            tx = T.get_thread_binding()
            if begin <= tx and tx < end:
                T.fill(buf[8:16, 0:8], T.float32(3.0))
            T.copy(buf, output)

    return kernel


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "threads,begin,end",
    [
        pytest.param(64, 0, 32, id="first-group"),
        pytest.param(64, 16, 48, id="cross-group"),
        pytest.param(64, 32, 64, id="second-group"),
        pytest.param((32, 2), 0, 32, id="two-dimensional-launch"),
    ],
)
def test_partial_fill_checks_launch_size_before_predicate(threads, begin, end):
    with pytest.raises(ValueError, match="exactly one simdgroup"):
        _lower(_make_predicated_fill_kernel(threads, begin, end))


@tilelang.testing.requires_metal
def test_partial_fill_rejects_incomplete_execution_range():
    with pytest.raises(ValueError, match="complete simdgroup execution range"):
        _lower(_make_predicated_fill_kernel(32, 0, 16))


@tilelang.testing.requires_metal
def test_partial_fill_accepts_complete_execution_range():
    calls = _fill_calls(_lower(_make_predicated_fill_kernel(32, 0, 32)))
    assert [call.args[1].value for call in calls if _fill_value(call) == 3.0] == [2]


def _make_accumulator_kernel(partial):
    @T.prim_func
    def kernel(
        a: T.Tensor((32, 32), T.float16),
        b: T.Tensor((32, 32), T.float16),
        output: T.Tensor((32, 32), T.float32),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((32, 32), T.float16)
            b_shared = T.alloc_shared((32, 32), T.float16)
            acc = T.alloc_fragment((32, 32), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.clear(acc)
            T.gemm(a_shared, b_shared, acc)
            if partial:
                T.fill(acc[8:24, 8:24], T.float32(2.0))
            T.copy(acc, output)

    return kernel


@tilelang.testing.requires_metal
def test_multigroup_accumulator_partial_fill_is_unsupported():
    with pytest.raises(ValueError, match="Metal partial fill:.*exactly one simdgroup"):
        _lower(_make_accumulator_kernel(True))
    # The same accumulator with only full clear retains its supported lowering.
    _lower(_make_accumulator_kernel(False))


@tilelang.testing.requires_metal
def test_partial_fill_launch_size_does_not_leak_between_kernels():
    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=64):
            first = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.clear(first)
            T.copy(first, output)
        with T.Kernel(1, threads=32):
            second = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.fill(second, T.float32(-7.0))
            T.fill(second[8:16, 0:8], T.float32(3.0))
            T.copy(second, output)

    calls = _fill_calls(_lower(kernel))
    assert [call.args[1].value for call in calls if _fill_value(call) == 3.0] == [2]


@tilelang.testing.requires_metal
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("scope", ["metal.simdgroup", "metal.cooperative_tensor"])
@pytest.mark.parametrize("begin,end", [(0, 1), (1, 2), (0, 2)], ids=["prefix", "offset", "whole"])
def test_high_dimensional_fill_diagnostic(ndim, scope, begin, end):
    shape = (2,) * (ndim - 1) + (256,)

    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            buf = T.alloc_local(shape, T.float32, scope=scope)
            if ndim == 3:
                T.fill(buf[begin:end, :, :], T.float32(3))
            else:
                T.fill(buf[begin:end, :, :, :], T.float32(3))

    if (begin, end) == (0, 2) or (scope == "metal.cooperative_tensor" and ndim == 3 and (begin, end) == (0, 1)):
        _lower(kernel)
    else:
        with pytest.raises(ValueError, match="Metal partial fill:.*3 or more dimensions"):
            _lower(kernel)


def _make_1d_fill_kernel(begin, end, threads=32, active_threads=32):
    @T.prim_func
    def kernel(output: T.Tensor((192,), T.float32)):
        with T.Kernel(1, threads=threads):
            buf = T.alloc_local((192,), T.float32, scope="metal.simdgroup")
            T.fill(buf, T.float32(-7))
            tx = T.get_thread_binding()
            if tx < active_threads:
                T.fill(buf[begin:end], T.float32(3))
            # T.copy currently requires a 2D simdgroup source. Store each matrix
            # directly so the readback does not depend on partial-copy indexing.
            for i in T.unroll(3):
                T.evaluate(tvm.tirx.simdgroup_store(buf.data, i, T.address_of(output[i * 64]), 8))

    return kernel


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "begin,end",
    [(0, 64), (64, 128), (64, 192), (0, 192)],
    ids=["prefix", "middle", "suffix", "whole"],
)
def test_1d_fill_preserves_outside(begin, end):
    func = _make_1d_fill_kernel(begin, end)
    compiled = tilelang.compile(func, out_idx=[], target="metal", target_host="c", execution_backend="tvm_ffi")
    output = torch.empty((192,), dtype=torch.float32, device="mps")
    compiled(output)
    torch.mps.synchronize()
    reference = torch.full((192,), -7.0, dtype=torch.float32)
    reference[begin:end] = 3
    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "begin,end,threads,active_threads,reason",
    [
        pytest.param(1, 65, 32, 32, "64-aligned", id="unaligned-min"),
        pytest.param(0, 63, 32, 32, "64-aligned", id="unaligned-extent"),
        pytest.param(64, 128, 64, 32, "exactly one simdgroup", id="predicated-group"),
        pytest.param(64, 128, 32, 16, "complete simdgroup execution range", id="incomplete-group"),
    ],
)
def test_1d_partial_fill_diagnostic(begin, end, threads, active_threads, reason):
    with pytest.raises(ValueError, match="Metal partial fill:.*" + reason):
        _lower(_make_1d_fill_kernel(begin, end, threads, active_threads))


@tilelang.testing.requires_metal
@pytest.mark.parametrize("shape", [(2, 2, 256), (2, 1, 2, 256)])
def test_cooperative_tensor_prefix_fill(shape):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            buf = T.alloc_local(shape, T.float32, scope="metal.cooperative_tensor")
            if len(shape) == 3:
                T.fill(buf[0:1, :, :], T.float32(3))
            else:
                T.fill(buf[0:1, :, :, :], T.float32(3))

    artifact = _lower(kernel)
    calls = []
    fill_op = tvm.ir.Op.get("tl.cooperative_tensor_fill")
    for func in artifact.device_mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: calls.append(node) if isinstance(node, tvm.tirx.Call) and node.op.same_as(fill_op) else None,
        )
    assert len(calls) == 1
    assert calls[0].args[1].value == 0
    assert _fill_value(calls[0]) == 3.0


@tilelang.testing.requires_metal
@pytest.mark.parametrize("use_else", [False, True])
def test_partial_fill_rejects_divergent_lanes(use_else):
    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.clear(buf)
            tx = T.get_thread_binding()
            if use_else:
                if tx == 16:
                    T.evaluate(0)
                else:
                    T.fill(buf[8:16, 0:8], T.float32(3))
            else:
                if tx != 16:
                    T.fill(buf[8:16, 0:8], T.float32(3))
            T.copy(buf, output)

    with pytest.raises(ValueError, match="thread-uniform control flow"):
        _lower(kernel)


@tilelang.testing.requires_metal
def test_cooperative_tensor_rejects_noncontiguous_prefix():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((2, 4, 256), T.float32, scope="metal.cooperative_tensor")
            # 512 elements at zero minima, but with gaps between rows.
            T.fill(buf[0:1, 0:4, 0:128], T.float32(3))

    with pytest.raises(ValueError, match="Metal partial fill:.*3 or more dimensions"):
        _lower(kernel)


@tilelang.testing.requires_metal
def test_partial_fill_accepts_uniform_branch_and_restores_state():
    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(2, threads=32) as bx:
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.clear(buf)
            tx = T.get_thread_binding()
            if tx != 16:
                output[tx // 16, tx % 16] = T.float32(0)
            if bx == 0:
                T.fill(buf[8:16, 0:8], T.float32(3))
            T.copy(buf, output)

    calls = _fill_calls(_lower(kernel))
    assert [call.args[1].value for call in calls if _fill_value(call) == 3.0] == [2]
