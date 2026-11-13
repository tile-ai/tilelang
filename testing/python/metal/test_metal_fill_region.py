"""Direct SIMD-group partial fills must preserve unselected matrix tiles."""

import pytest
import torch
from tilelang.metal.language import cooperative_tensor_store

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
@pytest.mark.parametrize("shape", [(1024,), (32, 32), (4, 256), (2, 2, 256), (2, 1, 2, 256)])
@pytest.mark.parametrize("whole", [False, True])
def test_cooperative_tensor_prefix_fill(shape, whole):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            buf = T.alloc_local(shape, T.float32, scope="metal.cooperative_tensor")
            if whole:
                T.fill(buf, T.float32(3))
            elif len(shape) == 1:
                T.fill(buf[0:512], T.float32(3))
            elif len(shape) == 2:
                T.fill(buf[0 : 512 // shape[1], :], T.float32(3))
            elif len(shape) == 3:
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
    assert [call.args[1].value for call in calls] == ([0, 1] if whole else [0])
    assert all(_fill_value(call) == 3.0 for call in calls)


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


@tilelang.testing.requires_metal
@pytest.mark.parametrize("kind", ["direct", "arithmetic", "alias", "uniform"])
@pytest.mark.parametrize("operation", ["fill", "copy"])
def test_partial_region_buffer_bound_branch(kind, operation):
    @T.prim_func
    def kernel(flags: T.Tensor((32,), T.int32), output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((16, 16), T.float32, scope="metal.simdgroup")
            T.clear(buf)
            tx = T.get_thread_binding()
            index = 0 if kind == "uniform" else tx
            flag = flags[index] + (1 if kind == "arithmetic" else 0)
            alias = flag * 2 if kind == "alias" else flag
            if alias != 0:
                if operation == "fill":
                    T.fill(buf[8:16, 0:8], T.float32(3))
                else:
                    T.copy(buf[8:16, 0:8], output[8:16, 0:8])
            # A preceding branch must not taint later unconditional operations.
            T.fill(buf[0:8, 0:8], T.float32(4))

    if kind == "uniform":
        _lower(kernel)
    else:
        with pytest.raises(ValueError, match="thread-uniform control flow"):
            _lower(kernel)


def _make_cooperative_region_kernel(shape, begin, end, dtype=T.float32, threads=32, initialize=True):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=threads):
            buf = T.alloc_local(shape, dtype, scope="metal.cooperative_tensor")
            if initialize:
                T.fill(buf, T.cast(-7, dtype))
            if len(shape) == 1:
                T.fill(buf[begin[0] : end[0]], T.cast(3, dtype))
            else:
                T.fill(buf[begin[0] : end[0], begin[1] : end[1]], T.cast(3, dtype))

    return kernel


def _interpret_cooperative_fills(artifact, slots):
    """Execute lowered fill predicates for each lane, leaving stores untouched."""
    tir = tvm.tirx
    state = [[None] * slots for _ in range(32)]
    analyzer = tvm.arith.Analyzer()

    def evaluate(expr, env):
        result = analyzer.simplify(tir.stmt_functor.substitute(expr, env))
        assert isinstance(result, (tir.IntImm, tir.FloatImm)), result
        return result.value

    def walk(stmt, env, lane):
        if isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                walk(child, env, lane)
        elif isinstance(stmt, tir.AttrStmt):
            if stmt.attr_key == "thread_extent":
                env = {**env, stmt.node.var: lane if stmt.node.thread_tag == "threadIdx.x" else 0}
            walk(stmt.body, env, lane)
        elif isinstance(stmt, tir.IfThenElse):
            branch = stmt.then_case if evaluate(stmt.condition, env) else stmt.else_case
            if branch is not None:
                walk(branch, env, lane)
        elif isinstance(stmt, tir.For):
            start = evaluate(stmt.min, env)
            for i in range(start, start + evaluate(stmt.extent, env)):
                walk(stmt.body, {**env, stmt.loop_var: i}, lane)
        elif isinstance(stmt, tir.SBlockRealize):
            walk(stmt.block, env, lane)
        elif isinstance(stmt, tir.SBlock):
            walk(stmt.body, env, lane)
        elif isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            call = stmt.value
            name = call.op.name
            if name == "tl.cooperative_tensor_fill":
                tile = evaluate(call.args[1], env)
                for element in range(16):
                    state[lane][tile * 16 + element] = evaluate(call.args[2], env)
            elif name == "tl.cooperative_tensor_set_element":
                tile = evaluate(call.args[1], env)
                element = evaluate(call.args[2], env)
                state[lane][tile * 16 + element] = evaluate(call.args[3], env)
        elif isinstance(stmt, tir.Bind):
            env[stmt.var] = evaluate(stmt.value, env)
        elif hasattr(stmt, "body"):
            walk(stmt.body, env, lane)

    for func in artifact.device_mod.functions.values():
        for lane in range(32):
            walk(func.body, {}, lane)
    return state


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", [T.float16, T.float32])
@pytest.mark.parametrize(
    "shape,begin,end",
    [
        ((1024,), (512,), (1024,)),
        ((1024,), (7,), (519,)),
        ((32, 64), (0, 32), (16, 64)),
        ((32, 64), (0, 0), (16, 32)),
        ((32, 64), (0, 0), (8, 64)),
        ((32, 64), (7, 13), (25, 49)),
        ((32, 64), (31, 63), (32, 64)),
    ],
)
def test_cooperative_tensor_arbitrary_region(shape, begin, end, dtype):
    artifact = _lower(_make_cooperative_region_kernel(shape, begin, end, dtype))
    rows, cols = (shape[0] // 32, 32) if len(shape) == 1 else shape
    state = _interpret_cooperative_fills(artifact, rows * cols // 32)
    # Inverse of the existing cooperative-tensor store mapping: inspect every
    # logical coordinate, including values outside the selected rectangle.
    for row in range(rows):
        for col in range(cols):
            lane = (col % 16 // 4 % 2) + (row % 4) * 2 + (col % 16 // 8) * 8 + (row % 8 // 4) * 16
            slot = (row // 16) * (cols // 16) * 8 + (col // 16) * 8 + (row % 16 // 8) * 4 + col % 4
            selected = begin[0] <= row * cols + col < end[0] if len(shape) == 1 else begin[0] <= row < end[0] and begin[1] <= col < end[1]
            assert state[lane][slot] == (3 if selected else -7), (row, col, lane, slot)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "shape,begin,end,threads,reason",
    [
        ((16, 48), (0, 0), (1, 1), 32, "16x32-aligned buffer shape"),
        ((32, 64), (31, 0), (33, 1), 32, "exceeds"),
        ((1024,), (1024,), (1025,), 32, "exceeds"),
        ((1024,), (7,), (519,), 64, "exactly one simdgroup"),
    ],
)
def test_cooperative_tensor_partial_region_diagnostic(shape, begin, end, threads, reason):
    with pytest.raises(ValueError, match=reason):
        _lower(_make_cooperative_region_kernel(shape, begin, end, threads=threads, initialize=False))


@tilelang.testing.requires_metal
def test_cooperative_tensor_partial_fill_rejects_bound_branch():
    @T.prim_func
    def kernel(flags: T.Tensor((32,), T.int32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((1024,), T.float32, scope="metal.cooperative_tensor")
            T.clear(buf)
            flag = flags[T.get_thread_binding()]
            if flag != 0:
                T.fill(buf[7:519], T.float32(3))

    with pytest.raises(ValueError, match="thread-uniform control flow"):
        _lower(kernel)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", [T.float16, T.float32])
@pytest.mark.parametrize("begin,end", [((0, 32), (16, 64)), ((0, 0), (8, 64)), ((7, 13), (25, 49)), ((31, 63), (32, 64))])
def test_cooperative_tensor_partial_fill_preserves_outside(dtype, begin, end):
    @T.prim_func
    def kernel(output: T.Tensor((32, 64), dtype)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((32, 64), dtype, scope="metal.cooperative_tensor")
            T.fill(buf, T.cast(-7, dtype))
            T.fill(buf[begin[0] : end[0], begin[1] : end[1]], T.cast(3, dtype))
            T.copy(buf, output)

    compiled = tilelang.compile(kernel, target="metal", target_host="c", execution_backend="tvm_ffi")
    output = torch.empty((32, 64), dtype=dtype.as_torch(), device="mps")
    compiled(output)
    torch.mps.synchronize()
    expected = torch.full((32, 64), -7, dtype=dtype.as_torch())
    expected[begin[0] : end[0], begin[1] : end[1]] = 3
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", [T.float16, T.float32])
def test_cooperative_tensor_1d_unaligned_fill_preserves_outside(dtype):
    @T.prim_func
    def kernel(output: T.Tensor((1024,), dtype)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((1024,), dtype, scope="metal.cooperative_tensor")
            T.fill(buf, T.cast(-7, dtype))
            T.fill(buf[7:519], T.cast(3, dtype))
            # The 1D storage is a sequence of 16x32 destination tiles.
            for i in T.unroll(2):
                cooperative_tensor_store(buf.data, i, T.address_of(output[i * 512]), 32, 16, 32, False, 16, 32, 16, 2)

    compiled = tilelang.compile(kernel, target="metal", target_host="c", execution_backend="tvm_ffi")
    output = torch.empty((1024,), dtype=dtype.as_torch(), device="mps")
    compiled(output)
    torch.mps.synchronize()
    expected = torch.full((1024,), -7, dtype=dtype.as_torch())
    expected[7:519] = 3
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_metal
def test_cooperative_tensor_partial_fill_after_gemm_codegen():
    @T.prim_func
    def kernel(A: T.Tensor((32, 16), T.float16), B: T.Tensor((16, 64), T.float16), output: T.Tensor((32, 64), T.float32)):
        with T.Kernel(1, threads=32):
            buf = T.alloc_local((32, 64), T.float32, scope="metal.cooperative_tensor")
            T.gemm(A, B, buf, clear_accum=True)
            T.fill(buf[31:32, 63:64], T.float32(3))
            T.copy(buf, output)

    target = tvm.target.Target({"kind": "metal", "keys": ["metal", "gpu", "metal4"]})
    with target:
        artifact = tilelang.lower(kernel, target=target)
    # Matmul destinations retain native tensor objects; the one selected
    # coordinate belongs to the final lane-local element of destination tile 3.
    assert "__pct_c3[15] = " in artifact.kernel_source
    assert "mpp::tensor_ops::matmul2d" in artifact.kernel_source
