"""CPU grid access independence, allocation privacy, and serial fallback."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T
from tilelang import tvm
from tvm import tirx


def _compile_serial(func, out_idx=-1):
    kernel = tilelang.compile(
        func,
        target="c",
        execution_backend="cython",
        out_idx=out_idx,
        pass_configs={"tl.cpu_parallel": True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()
    return kernel


def _parallel_loop_count(func, target):
    with tvm.target.Target(target), tvm.transform.PassContext(config={"tl.cpu_parallel": True}):
        lowered = tilelang.lower(func, target=target)
    loops = []
    for mod in (lowered.host_mod, lowered.device_mod):
        for f in mod.functions.values():
            tirx.stmt_functor.post_order_visit(
                f.body, lambda node: loops.append(node) if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL else None
            )
    return len(loops)


def _apply(func, target="c"):
    mod = tvm.IRModule.from_expr(func.with_attr({"target": tvm.target.Target(target), "global_symbol": "main"}))
    result = tilelang.cpu.transform.MaterializeCPUParallelGrid()(mod)["main"]
    kinds = []
    tirx.stmt_functor.post_order_visit(
        result.body, lambda n: kinds.append(n.kind) if isinstance(n, tirx.For) and n.loop_var.name == "bx" else None
    )
    assert len(kinds) == 1
    return kinds[0]


def test_while_reset_stays_serial(capfd):
    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        s = T.alloc_buffer((1,), "int32", scope="local")
        with T.Kernel(32, cpu_num_threads=4) as bx:
            k = T.alloc_buffer((1,), "int32", scope="local")
            k[0] = bx
            while k[0] == 0:
                s[0] = 0
                k[0] = 1
            s[0] = s[0] + A[bx]
            B[bx] = s[0]

    kernel = _compile_serial(main)
    torch.testing.assert_close(kernel(torch.ones(32, dtype=torch.int32)), torch.arange(1, 33, dtype=torch.int32))
    assert "stays serial" in capfd.readouterr().err


def test_read_before_covering_loop_finishes_stays_serial():
    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        s = T.alloc_buffer((2,), "int32", scope="local")
        with T.Kernel(16, cpu_num_threads=4) as bx:
            if bx == 0:
                s[0] = 0
                s[1] = 0
            for j in T.serial(2):
                s[j] = A[bx * 2 + j]
                B[bx * 2 + j] = s[(j + 1) % 2]

    kernel = _compile_serial(main)
    torch.testing.assert_close(kernel(torch.arange(1, 33, dtype=torch.int32)), torch.arange(32, dtype=torch.int32))


@pytest.mark.parametrize("reset_kind", ["empty_outer", "predicate", "block_predicate", "complete"])
def test_direct_reset_control_flow(reset_kind):
    # Complete initialization must not escape an empty or predicated region.
    bx = tirx.Var("bx", "int32")
    j = tirx.Var("j", "int32")
    t = tirx.Var("t", "int32")
    B = tirx.decl_buffer((16,), "int32", name="B")
    s = tirx.decl_buffer((4,), "int32", name="s", scope="local")
    zero = tirx.IntImm("int32", 0)

    def fill(value, predicate=None):
        return tirx.For(j, 0, 4, tirx.ForKind.SERIAL, tirx.BufferStore(s, value, [j], predicate=predicate))

    reset = fill(zero)
    if reset_kind == "empty_outer":
        reset = tirx.For(t, 0, bx, tirx.ForKind.SERIAL, reset)
    elif reset_kind == "predicate":
        reset = fill(zero, bx == 0)
    elif reset_kind == "block_predicate":
        reset = tirx.SBlockRealize([], bx == 0, tirx.SBlock([], [], [], "reset", reset))
    grid = tirx.For(
        bx,
        0,
        16,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.IfThenElse(bx == 0, fill(zero), None),
                reset,
                tirx.BufferStore(B, tirx.BufferLoad(s, [1]), [bx]),
                fill(bx + 1),
            ]
        ),
        annotations={"tl.cpu_grid_dim": 0},
    )
    func = tirx.PrimFunc([B.data], tirx.SeqStmt([tirx.AllocBuffer(s), grid]), buffer_map={B.data: B})
    expected = tirx.ForKind.PARALLEL if reset_kind == "complete" else tirx.ForKind.SERIAL
    assert _apply(func) == expected


def test_stateful_extern_without_pointer_stays_serial(capfd):
    @T.prim_func
    def main(A: T.Tensor((1024,), "int32"), B: T.Tensor((1024,), "int32")):
        with T.Kernel(
            1024,
            cpu_num_threads=4,
            prelude='extern "C" int cpu_grid_counter() { static int value = 0; return value++; }\n',
        ) as bx:
            B[bx] = T.call_extern("int32", "cpu_grid_counter") + A[bx]

    kernel = _compile_serial(main)
    torch.testing.assert_close(kernel(torch.zeros(1024, dtype=torch.int32)), torch.arange(1024, dtype=torch.int32))
    assert "effects that cannot be proven safe" in capfd.readouterr().err


@pytest.mark.parametrize("pure", [False, True])
def test_direct_extern_effects(pure, capfd):
    bx = tirx.Var("bx", "int32")
    B = tirx.decl_buffer((16,), "int32", name="B")
    call = tirx.call_pure_extern("int32", "pure_scalar", bx) if pure else tirx.call_extern("int32", "counter")
    grid = tirx.For(bx, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(B, call, [bx]), annotations={"tl.cpu_grid_dim": 0})
    func = tirx.PrimFunc([B.data], grid, buffer_map={B.data: B})
    assert _apply(func, "llvm") == (tirx.ForKind.PARALLEL if pure else tirx.ForKind.SERIAL)
    if not pure:
        assert "effects that cannot be proven safe" in capfd.readouterr().err


@pytest.mark.parametrize(
    "target,cast_dtype,extent,parallel",
    [
        ("c", "int8", 512, False),
        ("llvm", "int8", 512, False),
        ("c", "int8", 128, True),
        ("c", "int8", None, False),
        ("llvm", "int64", 512, True),
    ],
)
def test_cast_range_proof(target, cast_dtype, extent, parallel):
    bx = tirx.Var("bx", "int32")
    B = tirx.decl_buffer((512,), "int32", name="B")
    n = tirx.Var("n", "int32")
    index = tirx.Cast(cast_dtype, bx)
    body = tirx.IfThenElse(index >= 0, tirx.BufferStore(B, bx, [index]), None)
    grid = tirx.For(bx, 0, n if extent is None else extent, tirx.ForKind.SERIAL, body, annotations={"tl.cpu_grid_dim": 0})
    func = tirx.PrimFunc([B.data, n] if extent is None else [B.data], grid, buffer_map={B.data: B})
    assert _apply(func, target) == (tirx.ForKind.PARALLEL if parallel else tirx.ForKind.SERIAL)


@pytest.mark.parametrize("mixed_casts", [False, True])
def test_casts_preserve_iterator_identity(mixed_casts):
    bx = tirx.Var("bx", "int32")
    B = tirx.decl_buffer((16,), "int32", name="B")
    cast_bx = tirx.Cast("int16", bx)
    if mixed_casts:
        # Distinct cast dtypes of one iterator must not become independent.
        index = tirx.Cast("int32", cast_bx) - tirx.Cast("int32", tirx.Cast("int8", bx))
    else:
        # Retaining raw bx as a free parameter would hide this collision.
        index = tirx.Cast("int32", cast_bx) - bx
    grid = tirx.For(bx, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(B, bx, [index]), annotations={"tl.cpu_grid_dim": 0})
    func = tirx.PrimFunc([B.data], grid, buffer_map={B.data: B})
    assert _apply(func) == tirx.ForKind.SERIAL


@pytest.mark.parametrize("alias_contract", ["non_restrict", "no_noalias"])
def test_overlapping_parameter_slices_stay_serial(alias_contract, capfd):
    n = 256

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32")):
        with T.Kernel(n, cpu_num_threads=4) as bx:
            if alias_contract == "non_restrict":
                T.annotate_restrict_buffers(A, B)
            B[bx] = A[bx] + 1

    if alias_contract == "no_noalias":
        main = main.with_attr("tirx.noalias", False)
    kernel = _compile_serial(main, out_idx=None)
    storage = torch.zeros(n + 1, dtype=torch.int32)
    kernel(storage[:-1], storage[1:])
    torch.testing.assert_close(storage, torch.arange(n + 1, dtype=torch.int32))
    assert "may alias" in capfd.readouterr().err


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("alias_contract", ["noalias", "missing", "disabled", "both_non_restrict", "one_non_restrict", "read_only"])
def test_parameter_alias_contract(target, alias_contract):
    bx = tirx.Var("bx", "int32")
    A = tirx.decl_buffer((16,), "int32", name="A")
    B = tirx.decl_buffer((16,), "int32", name="B")
    body = tirx.BufferStore(B, A[bx] + 1, [bx])
    if alias_contract == "read_only":
        body = tirx.Evaluate(A[bx] + B[bx])
    grid = tirx.For(bx, 0, 16, tirx.ForKind.SERIAL, body, annotations={"tl.cpu_grid_dim": 0})
    func = tirx.PrimFunc([A.data, B.data], grid, buffer_map={A.data: A, B.data: B})
    if alias_contract not in ("missing", "read_only"):
        func = func.with_attr("tirx.noalias", alias_contract != "disabled")
    if alias_contract == "both_non_restrict":
        func = func.with_attr("tl.non_restrict_params", [A.data, B.data])
    elif alias_contract == "one_non_restrict":
        func = func.with_attr("tl.non_restrict_params", [A.data])
    parallel = alias_contract in ("noalias", "missing", "one_non_restrict", "read_only")
    assert _apply(func, target) == (tirx.ForKind.PARALLEL if parallel else tirx.ForKind.SERIAL)


@pytest.mark.parametrize(
    "target",
    ["c", pytest.param("llvm", marks=pytest.mark.skipif(not tvm.runtime.enabled("llvm"), reason="LLVM support is not built"))],
)
@pytest.mark.parametrize("view_dtype", ["uint8", "uint32"])
def test_parameter_view_access_width(target, view_dtype, capfd):
    n = 256
    width = 4 if view_dtype == "uint8" else 1

    @T.prim_func
    def main(B: T.Tensor((n,), "int32")):
        with T.Kernel(n, cpu_num_threads=4) as bx:
            B[bx] = bx + 100
            T.view(B, (n * width,), dtype=view_dtype)[bx] = T.Cast(view_dtype, 255)

    kernel = tilelang.compile(
        main, target=target, execution_backend="cython" if target == "c" else "tvm_ffi", pass_configs={"tl.cpu_parallel": True}
    )
    parallel = view_dtype == "uint32"
    if target == "c":
        assert ("#pragma omp" in kernel.get_kernel_source()) == parallel
    else:
        assert (_parallel_loop_count(main, target) > 0) == parallel
    output = torch.zeros(n, dtype=torch.int32)
    kernel(output)
    expected = torch.full_like(output, 255) if parallel else torch.arange(n, dtype=torch.int32) + 100
    if not parallel:
        expected[: n // 4] = -1
    torch.testing.assert_close(output, expected)
    if not parallel:
        assert "different element widths" in capfd.readouterr().err


@pytest.mark.parametrize("write", [False, True])
def test_mixed_width_parameter_reads(write):
    bx = tirx.Var("bx", "int32")
    B = tirx.decl_buffer((16,), "int32", name="B")
    view = tirx.decl_buffer((64,), "uint8", data=B.data)
    value = tirx.Cast("int32", view[bx]) + B[bx]
    body = tirx.BufferStore(B, value, [bx]) if write else tirx.Evaluate(value)
    grid = tirx.For(bx, 0, 16, tirx.ForKind.SERIAL, body, annotations={"tl.cpu_grid_dim": 0})
    func = tirx.PrimFunc([B.data], grid, buffer_map={B.data: B})
    assert _apply(func) == (tirx.ForKind.SERIAL if write else tirx.ForKind.PARALLEL)


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize(
    "reset_kind", ["subview", "offset_subview", "strided_view", "byte_subview", "padded_vector_view", "full_view", "byte_full_view"]
)
def test_local_view_reset_covers_allocation(target, reset_kind):
    bx = tirx.Var("bx", "int32")
    j = tirx.Var("j", "int32")
    output = tirx.decl_buffer((16,), "int32", name="output")
    padded_vector = reset_kind == "padded_vector_view"
    state = tirx.decl_buffer((2,), "int32x3" if padded_vector else "int32", name="state", scope="local")
    byte_view = reset_kind.startswith("byte_") or padded_vector
    complete = reset_kind in ("full_view", "byte_full_view")
    extent = (8 if complete else 4) if byte_view else (2 if complete or reset_kind == "strided_view" else 1)
    if padded_vector:
        # A three-lane storage element may be padded by the target ABI;
        # its logical bit count cannot prove the allocation's capacity.
        extent = 24
    view_dtype = "uint8" if byte_view else "int32"
    offset = 1 if reset_kind == "offset_subview" else 0
    carried_index = 0 if offset else 1
    view = tirx.decl_buffer(
        (extent,),
        view_dtype,
        data=state.data,
        elem_offset=offset,
        strides=[0] if reset_kind == "strided_view" else None,
        scope="local",
        name="view",
    )
    reset = tirx.For(j, 0, extent, tirx.ForKind.SERIAL, tirx.BufferStore(view, tirx.IntImm(view_dtype, 0), [j]))
    initial = tirx.Broadcast(tirx.IntImm("int32", 0), 3) if padded_vector else tirx.IntImm("int32", 0)
    carried = tirx.Shuffle([state[carried_index]], [2]) if padded_vector else state[carried_index]
    update = tirx.Broadcast(bx + 1, 3) if padded_vector else bx + 1
    grid = tirx.For(
        bx,
        0,
        16,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.IfThenElse(bx == 0, tirx.BufferStore(state, initial, [carried_index]), None),
                reset,
                tirx.BufferStore(output, carried, [bx]),
                tirx.BufferStore(state, update, [carried_index]),
            ]
        ),
        annotations={"tl.cpu_grid_dim": 0},
    )
    func = tirx.PrimFunc([output.data], tirx.SeqStmt([tirx.AllocBuffer(state), grid]), buffer_map={output.data: output})
    mod = tvm.IRModule.from_expr(func.with_attr({"target": tvm.target.Target(target), "global_symbol": "main"}))
    result = tilelang.cpu.transform.MaterializeCPUParallelGrid()(mod)["main"]
    grids = []
    tirx.stmt_functor.post_order_visit(
        result.body, lambda node: grids.append(node) if isinstance(node, tirx.For) and node.loop_var.same_as(bx) else None
    )
    assert len(grids) == 1
    assert grids[0].kind == (tirx.ForKind.PARALLEL if complete else tirx.ForKind.SERIAL)
    private_allocations = []
    tirx.stmt_functor.post_order_visit(
        grids[0].body,
        lambda node: (
            private_allocations.append(node) if isinstance(node, tirx.AllocBuffer) and node.buffer.data.same_as(state.data) else None
        ),
    )
    assert bool(private_allocations) == complete


@pytest.mark.parametrize(
    "target",
    ["c", pytest.param("llvm", marks=pytest.mark.skipif(not tvm.runtime.enabled("llvm"), reason="LLVM support is not built"))],
)
@pytest.mark.parametrize("complete_reset", [False, True])
def test_local_view_preserves_iteration_state(target, complete_reset):
    n = 32

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32")):
        state = T.alloc_buffer((2,), "int32", scope="local")
        with T.Kernel(n, cpu_num_threads=4) as bx:
            if complete_reset:
                for j in T.serial(8):
                    T.view(state, (8,), dtype="uint8")[j] = T.Cast("uint8", 0)
            else:
                if bx == 0:
                    state[1] = 0
                T.Tensor((1,), "int32", state.data)[0] = bx
            state[1] = state[1] + A[bx]
            B[bx] = state[1]

    kernel = tilelang.compile(
        main, target=target, execution_backend="cython" if target == "c" else "tvm_ffi", out_idx=-1, pass_configs={"tl.cpu_parallel": True}
    )
    if target == "c":
        assert ("#pragma omp" in kernel.get_kernel_source()) == complete_reset
    else:
        assert (_parallel_loop_count(main, target) > 0) == complete_reset
    expected = torch.ones(n, dtype=torch.int32) if complete_reset else torch.arange(1, n + 1, dtype=torch.int32)
    torch.testing.assert_close(kernel(torch.ones(n, dtype=torch.int32)), expected)
