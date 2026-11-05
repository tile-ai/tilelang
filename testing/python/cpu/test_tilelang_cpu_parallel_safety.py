"""CPU grid access independence, allocation privacy, and serial fallback."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.testing.ir import collect_nodes


@pytest.fixture
def fresh_compilation(monkeypatch):
    # Serial fallback diagnostics are emitted during lowering, not on cache hits.
    monkeypatch.setattr(tilelang.env, "TILELANG_DISABLE_CACHE", "1")


def _compile_serial(func, out_idx=-1):
    kernel = tilelang.compile(func, target="c", out_idx=out_idx, pass_configs={"tl.cpu_parallel": True})
    assert "#pragma omp" not in kernel.get_kernel_source()
    return kernel


def _parallel_loop_count(func, target):
    with tvm_tirx.target.Target(target), tvm_tirx.transform.PassContext(config={"tl.cpu_parallel": True}):
        lowered = tilelang.lower(func, target=target)
    return sum(loop.kind == ir.ForKind.PARALLEL for mod in (lowered.host_mod, lowered.device_mod) for loop in collect_nodes(mod, ir.For))


def _lower_grid(func, target="c"):
    mod = tvm_tirx.IRModule.from_expr(func.with_attr({"target": tvm_tirx.target.Target(target), "global_symbol": "main"}))
    result = tilelang.cpu.transform.MaterializeCPUParallelGrid()(mod)["main"]
    grids = [loop for loop in collect_nodes(result, ir.For) if loop.var.name == "bx"]
    assert len(grids) == 1
    return grids[0]


def _tensor(shape, dtype, *, name, scope="global"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout="default"))


def _allocation(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _view(backing, shape, dtype, *, name="view", offset=0, stride=1):
    layout = tirx.TileLayout(tirx.layout.S[shape : (stride,)] + offset)
    tensor = ir.Var(name, tirx.TensorType(shape, dtype, scope=backing.ty.scope, layout=layout))
    declaration = ir.Bind(tensor, ir.Call("tirx.decl_tensor", [tirx.tensor_data_ptr(backing)], ty_args=[tensor.ty]))
    return tensor, declaration


@pytest.mark.usefixtures("fresh_compilation")
def test_while_reset_stays_serial(capfd):
    @T.prim_func
    def main(A: T.Tensor((32,), T.int32), B: T.Tensor((32,), T.int32)):
        s = T.alloc_buffer((1,), T.int32, scope="local")
        with T.Kernel(32, cpu_num_threads=4) as bx:
            k = T.alloc_buffer((1,), T.int32, scope="local")
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
    def main(A: T.Tensor((32,), T.int32), B: T.Tensor((32,), T.int32)):
        s = T.alloc_buffer((2,), T.int32, scope="local")
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
    bx = tirx.Var("bx", T.int32)
    j = tirx.Var("j", T.int32)
    t = tirx.Var("t", T.int32)
    B = _tensor((16,), T.int32, name="B")
    s = _tensor((4,), T.int32, name="s", scope="local")
    zero = T.int32(0)

    def fill(value, predicate=None):
        store = ir.TensorStore(s, [j], value)
        if predicate is not None:
            store = ir.If(predicate, store, None)
        return ir.For(j, 0, 4, 1, ir.ForKind.DEFAULT, store)

    reset = fill(zero)
    if reset_kind == "empty_outer":
        reset = ir.For(t, 0, bx, 1, ir.ForKind.DEFAULT, reset)
    elif reset_kind == "predicate":
        reset = fill(zero, bx == 0)
    elif reset_kind == "block_predicate":
        reset = ir.If(bx == 0, ir.RegionStmt("tl.lexical_scope", [], [], {}, reset), None)
    grid = ir.For(
        bx,
        0,
        16,
        1,
        ir.ForKind.DEFAULT,
        ir.SeqStmt([ir.If(bx == 0, fill(zero), None), reset, ir.TensorStore(B, [bx], ir.TensorLoad(s, [1])), fill(bx + 1)]),
        annotations={"tl.cpu_grid_dim": 0},
    )
    func = tirx.Function([B], ir.SeqStmt([_allocation(s), grid]), ret_type=ir.PrimType("void"))
    expected = ir.ForKind.PARALLEL if reset_kind == "complete" else ir.ForKind.DEFAULT
    assert _lower_grid(func).kind == expected


@pytest.mark.usefixtures("fresh_compilation")
def test_stateful_extern_without_pointer_stays_serial(capfd):
    @T.prim_func
    def main(A: T.Tensor((1024,), T.int32), B: T.Tensor((1024,), T.int32)):
        with T.Kernel(
            1024,
            cpu_num_threads=4,
            prelude='extern "C" int cpu_grid_counter() { static int value = 0; return value++; }\n',
        ) as bx:
            B[bx] = T.call_extern(T.int32, "cpu_grid_counter") + A[bx]

    kernel = _compile_serial(main)
    torch.testing.assert_close(kernel(torch.zeros(1024, dtype=torch.int32)), torch.arange(1024, dtype=torch.int32))
    assert "effects that cannot be proven safe" in capfd.readouterr().err


@pytest.mark.parametrize("pure", [False, True])
def test_direct_extern_effects(pure, capfd):
    bx = tirx.Var("bx", T.int32)
    B = _tensor((16,), T.int32, name="B")
    call = tirx.call_pure_extern("pure_scalar", bx, ty="int32") if pure else tirx.call_extern("counter", ty="int32")
    grid = ir.For(bx, 0, 16, 1, ir.ForKind.DEFAULT, ir.TensorStore(B, [bx], call), annotations={"tl.cpu_grid_dim": 0})
    func = tirx.Function([B], grid, ret_type=ir.PrimType("void"))
    assert _lower_grid(func, "llvm").kind == (ir.ForKind.PARALLEL if pure else ir.ForKind.DEFAULT)
    if not pure:
        assert "effects that cannot be proven safe" in capfd.readouterr().err


@pytest.mark.parametrize(
    "target,cast_dtype,extent,parallel",
    [
        ("c", T.int8, 512, False),
        ("llvm", T.int8, 512, False),
        ("c", T.int8, 128, True),
        ("c", T.int8, None, False),
        ("llvm", T.int64, 512, True),
    ],
    ids=str,
)
def test_cast_range_proof(target, cast_dtype, extent, parallel):
    bx = tirx.Var("bx", T.int32)
    B = _tensor((512,), T.int32, name="B")
    n = tirx.Var("n", T.int32)
    index = tirx.Cast(cast_dtype, bx)
    body = ir.If(index >= 0, ir.TensorStore(B, [index], bx), None)
    grid = ir.For(bx, 0, n if extent is None else extent, 1, ir.ForKind.DEFAULT, body, annotations={"tl.cpu_grid_dim": 0})
    func = tirx.Function([B, n] if extent is None else [B], grid, ret_type=ir.PrimType("void"))
    assert _lower_grid(func, target).kind == (ir.ForKind.PARALLEL if parallel else ir.ForKind.DEFAULT)


@pytest.mark.parametrize("mixed_casts", [False, True])
def test_casts_preserve_iterator_identity(mixed_casts):
    bx = tirx.Var("bx", T.int32)
    B = _tensor((16,), T.int32, name="B")
    cast_bx = tirx.Cast(T.int16, bx)
    if mixed_casts:
        # Distinct cast dtypes of one iterator must not become independent.
        index = tirx.Cast(T.int32, cast_bx) - tirx.Cast(T.int32, tirx.Cast(T.int8, bx))
    else:
        # Retaining raw bx as a free parameter would hide this collision.
        index = tirx.Cast(T.int32, cast_bx) - bx
    grid = ir.For(bx, 0, 16, 1, ir.ForKind.DEFAULT, ir.TensorStore(B, [index], bx), annotations={"tl.cpu_grid_dim": 0})
    func = tirx.Function([B], grid, ret_type=ir.PrimType("void"))
    assert _lower_grid(func).kind == ir.ForKind.DEFAULT


@pytest.mark.usefixtures("fresh_compilation")
@pytest.mark.parametrize("alias_contract", ["non_restrict", "no_noalias"])
def test_overlapping_parameter_slices_stay_serial(alias_contract, capfd):
    n = 256

    @T.prim_func
    def main(A: T.Tensor((n,), T.int32), B: T.Tensor((n,), T.int32)):
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
    bx = tirx.Var("bx", T.int32)
    A = _tensor((16,), T.int32, name="A")
    B = _tensor((16,), T.int32, name="B")
    body = ir.TensorStore(B, [bx], A[bx] + 1)
    if alias_contract == "read_only":
        body = ir.Evaluate(A[bx] + B[bx])
    grid = ir.For(bx, 0, 16, 1, ir.ForKind.DEFAULT, body, annotations={"tl.cpu_grid_dim": 0})
    func = tirx.Function([A, B], grid, ret_type=ir.PrimType("void"))
    if alias_contract not in ("missing", "read_only"):
        func = func.with_attr("tirx.noalias", alias_contract != "disabled")
    if alias_contract == "both_non_restrict":
        func = func.with_attr("tl.non_restrict_params", [A, B])
    elif alias_contract == "one_non_restrict":
        func = func.with_attr("tl.non_restrict_params", [A])
    parallel = alias_contract in ("noalias", "missing", "one_non_restrict", "read_only")
    assert _lower_grid(func, target).kind == (ir.ForKind.PARALLEL if parallel else ir.ForKind.DEFAULT)


@pytest.mark.usefixtures("fresh_compilation")
@pytest.mark.parametrize(
    "target",
    ["c", pytest.param("llvm", marks=pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM support is not built"))],
)
@pytest.mark.parametrize("view_dtype", [T.uint8, T.uint32], ids=str)
def test_parameter_view_access_width(target, view_dtype, capfd):
    n = 256
    width = 4 if view_dtype == T.uint8 else 1

    @T.prim_func
    def main(B: T.Tensor((n,), T.int32)):
        with T.Kernel(n, cpu_num_threads=4) as bx:
            B[bx] = bx + 100
            T.view(B, (n * width,), dtype=view_dtype)[bx] = T.Cast(view_dtype, 255)

    kernel = tilelang.compile(main, target=target, pass_configs={"tl.cpu_parallel": True})
    parallel = view_dtype == T.uint32
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
    bx = tirx.Var("bx", T.int32)
    B = _tensor((16,), T.int32, name="B")
    view, view_definition = _view(B, (64,), T.uint8)
    value = tirx.Cast(T.int32, view[bx]) + B[bx]
    body = ir.TensorStore(B, [bx], value) if write else ir.Evaluate(value)
    grid = ir.For(bx, 0, 16, 1, ir.ForKind.DEFAULT, body, annotations={"tl.cpu_grid_dim": 0})
    func = tirx.Function([B], [view_definition, grid], ret_type=ir.PrimType("void"))
    assert _lower_grid(func).kind == (ir.ForKind.DEFAULT if write else ir.ForKind.PARALLEL)


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize(
    "reset_kind", ["subview", "offset_subview", "strided_view", "byte_subview", "padded_vector_view", "full_view", "byte_full_view"]
)
def test_local_view_reset_covers_allocation(target, reset_kind):
    bx = tirx.Var("bx", T.int32)
    j = tirx.Var("j", T.int32)
    output = _tensor((16,), T.int32, name="output")
    padded_vector = reset_kind == "padded_vector_view"
    state = _tensor((2,), T.int32.with_lanes(3) if padded_vector else T.int32, name="state", scope="local")
    byte_view = reset_kind.startswith("byte_") or padded_vector
    complete = reset_kind in ("full_view", "byte_full_view")
    extent = (8 if complete else 4) if byte_view else (2 if complete or reset_kind == "strided_view" else 1)
    if padded_vector:
        # A three-lane storage element may be padded by the target ABI;
        # its logical bit count cannot prove the allocation's capacity.
        extent = 24
    view_dtype = T.uint8 if byte_view else T.int32
    offset = 1 if reset_kind == "offset_subview" else 0
    carried_index = 0 if offset else 1
    view, view_definition = _view(state, (extent,), view_dtype, offset=offset, stride=0 if reset_kind == "strided_view" else 1)
    reset = ir.For(j, 0, extent, 1, ir.ForKind.DEFAULT, ir.TensorStore(view, [j], view_dtype(0)))
    initial = ir.prim.Broadcast(T.int32(0), 3) if padded_vector else T.int32(0)
    carried = ir.prim.Shuffle([state[carried_index]], [2]) if padded_vector else state[carried_index]
    update = ir.prim.Broadcast(bx + 1, 3) if padded_vector else bx + 1
    grid = ir.For(
        bx,
        0,
        16,
        1,
        ir.ForKind.DEFAULT,
        ir.SeqStmt(
            [
                ir.If(bx == 0, ir.TensorStore(state, [carried_index], initial), None),
                reset,
                ir.TensorStore(output, [bx], carried),
                ir.TensorStore(state, [carried_index], update),
            ]
        ),
        annotations={"tl.cpu_grid_dim": 0},
    )
    func = tirx.Function([output], ir.SeqStmt([_allocation(state), view_definition, grid]), ret_type=ir.PrimType("void"))
    grid = _lower_grid(func, target)
    assert grid.var.same_as(bx)
    assert grid.kind == (ir.ForKind.PARALLEL if complete else ir.ForKind.DEFAULT)
    private_allocations = [
        binding
        for binding in collect_nodes(grid.body, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor" and binding.var.same_as(state)
    ]
    assert bool(private_allocations) == complete


@pytest.mark.parametrize(
    "target",
    ["c", pytest.param("llvm", marks=pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM support is not built"))],
)
@pytest.mark.parametrize("complete_reset", [False, True])
def test_local_view_preserves_iteration_state(target, complete_reset):
    n = 32

    @T.prim_func
    def main(A: T.Tensor((n,), T.int32), B: T.Tensor((n,), T.int32)):
        state = T.alloc_buffer((2,), T.int32, scope="local")
        with T.Kernel(n, cpu_num_threads=4) as bx:
            if complete_reset:
                for j in T.serial(8):
                    T.view(state, (8,), dtype=T.uint8)[j] = T.uint8(0)
            else:
                if bx == 0:
                    state[1] = 0
                T.Tensor((1,), T.int32, state.data)[0] = bx
            state[1] = state[1] + A[bx]
            B[bx] = state[1]

    kernel = tilelang.compile(main, target=target, out_idx=-1, pass_configs={"tl.cpu_parallel": True})
    if target == "c":
        assert ("#pragma omp" in kernel.get_kernel_source()) == complete_reset
    else:
        assert (_parallel_loop_count(main, target) > 0) == complete_reset
    expected = torch.ones(n, dtype=torch.int32) if complete_reset else torch.arange(1, n + 1, dtype=torch.int32)
    torch.testing.assert_close(kernel(torch.ones(n, dtype=torch.int32)), expected)


def test_cpu_parallel_strided_ramp_reset_stays_serial():
    # A stride-zero ramp rewrites three elements, leaving s[9] carried.
    bx = tirx.Var("bx", T.int32)
    t = tirx.Var("t", T.int32)
    A = _tensor((256,), T.float32, name="A")
    B = _tensor((256,), T.float32, name="B")
    s = _tensor((12,), T.float32, name="s", scope="local")
    one = ir.prim.Broadcast(T.float32(1.0), 4)
    fake_reset = ir.For(t, 0, 3, 1, ir.ForKind.DEFAULT, ir.TensorStore(s, [ir.prim.Ramp(4 * t, 0, 4)], one))
    read_carried = ir.TensorStore(B, [bx], ir.TensorLoad(s, [9]) + ir.TensorLoad(s, [0]))
    write9 = ir.TensorStore(s, [9], ir.TensorLoad(A, [bx]))
    grid = ir.For(bx, 0, 256, 1, ir.ForKind.DEFAULT, ir.SeqStmt([fake_reset, read_carried, write9]), annotations={"tl.cpu_grid_dim": 0})
    func = tirx.Function([A, B], ir.SeqStmt([_allocation(s), grid]), ret_type=ir.PrimType("void"))
    assert _lower_grid(func).kind == ir.ForKind.DEFAULT


def test_cpu_parallel_strided_ramp_overlap_stays_serial():
    # Stride-two ramps from adjacent grid iterations overlap.
    bx = tirx.Var("bx", T.int32)
    B = _tensor((4 * 256 + 4,), T.float32, name="B")
    one = ir.prim.Broadcast(T.float32(1.0), 4)
    grid = ir.For(
        bx, 0, 256, 1, ir.ForKind.DEFAULT, ir.TensorStore(B, [ir.prim.Ramp(bx * 4, 2, 4)], one), annotations={"tl.cpu_grid_dim": 0}
    )
    func = tirx.Function([B], grid, ret_type=ir.PrimType("void"))
    assert _lower_grid(func).kind == ir.ForKind.DEFAULT
