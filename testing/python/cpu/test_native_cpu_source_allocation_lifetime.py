"""Preserve source allocation lifetimes before proving CPU grid privacy."""

from tilelang.language import _annotation_codec

# These regressions require distinct nested IR regions and conditional scopes.
# ruff: noqa: SIM102, SIM117

import pytest
import torch
import tvm_ffi
from tvm_ffi.serialization import to_json_graph_str
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target
from tvm_tirx.tirx.layout import Iter

import tilelang
import tilelang.cpu.language as T
import tilelang.cpu.transform

from tilelang.transform import PassConfigKey


def _config(mode):
    if mode == "default_off":
        return {}
    config = {PassConfigKey.TL_CPU_PARALLEL: True}
    if mode == "min_trip":
        config[PassConfigKey.TL_CPU_PARALLEL_MIN_TRIP] = 1024
    return config


def _prepare(function, target="c"):
    module = ir.IRModule.from_expr(function)
    module = tilelang.transform.NormalizeAliasAnnotations()(module)
    module = tilelang.transform.LowerTensorSpec()(module)
    module = tirx.transform.BindTarget(Target(target))(module)
    module = tilelang.transform.MaterializeKernelLaunch(False, False)(module)
    module = tilelang.cpu.transform.LowerCPUKernelLaunch()(module)
    assert tirx.analysis.check_well_formed(module)
    return module


def _allocations(value):
    """Return declarations with their actual lexical ancestors."""
    rows = []

    def visit(stmt, ancestors):
        if isinstance(stmt, ir.SeqStmt):
            for element in stmt.seq:
                visit(element, ancestors)
        elif isinstance(stmt, ir.Bind):
            if isinstance(stmt.value, ir.Call) and stmt.value.op.name == "tirx.alloc_tensor":
                rows.append((stmt, ancestors))
        elif isinstance(stmt, ir.For | ir.RegionStmt | ir.While):
            visit(stmt.body, (*ancestors, stmt))
        elif isinstance(stmt, ir.If):
            visit(stmt.then_case, (*ancestors, stmt))
            if stmt.else_case is not None:
                visit(stmt.else_case, (*ancestors, stmt))

    for function in value.functions.values():
        visit(function.body, ())
    return rows


def _dynamic_partial():
    n = T.dynamic("n")

    @T.prim_func
    def partial(A: T.Tensor((n,), T.int32), B: T.Tensor((n,), T.int32)):
        with T.Kernel(T.ceildiv(n, 4)) as bx:
            state = T.alloc_buffer((2,), T.int32, scope="local")
            state[0] = 7
            if bx == 0:
                state[1] = 0
            for i in T.serial(4):
                if bx * 4 + i < n:
                    state[1] = state[1] + A[bx * 4 + i]
                    B[bx * 4 + i] = state[1]

    return partial


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("mode", ["parallel", "default_off", "min_trip"])
def test_dynamic_source_allocation_carries_state_with_or_without_parallel_option(target, mode):
    function = _dynamic_partial()
    ((allocation, parents),) = _allocations(_prepare(function, target))
    assert not parents
    assert "tl.source_block_alloc" not in allocation.value.attrs
    kernel = tilelang.compile(function, target=target, out_idx=1, pass_configs=_config(mode))
    source = kernel.get_kernel_source()
    if target == "c":
        assert "#pragma omp" not in source
    else:
        assert "__TVMBackendParallelLaunch" not in source
    data = torch.arange(1, 20, dtype=torch.int32)
    torch.testing.assert_close(kernel(data), data.cumsum(0).to(torch.int32), rtol=0, atol=0)


def _nested_serial():
    @T.prim_func
    def nested(A: T.Tensor((8,), T.int32), B: T.Tensor((8,), T.int32)):
        for row in T.serial(2):
            with T.Kernel(4) as bx:
                state = T.alloc_buffer((2,), T.int32, scope="local")
                state[0] = 1
                if row == 0:
                    if bx == 0:
                        state[1] = 0
                state[1] = state[1] + A[row * 4 + bx]
                B[row * 4 + bx] = state[1]

    return nested


def _nested_kernel():
    @T.prim_func
    def nested(A: T.Tensor((8,), T.int32), B: T.Tensor((8,), T.int32)):
        with T.Kernel(2) as row:
            with T.Kernel(4) as bx:
                state = T.alloc_buffer((2,), T.int32, scope="local")
                state[0] = 1
                if row == 0:
                    if bx == 0:
                        state[1] = 0
                state[1] = state[1] + A[row * 4 + bx]
                B[row * 4 + bx] = state[1]

    return nested


def _nested_sblock():
    @T.prim_func
    def nested(A: T.Tensor((8,), T.int32), B: T.Tensor((8,), T.int32)):
        with T.Kernel(8) as bx:
            with T.sblock("state_pool"):
                state = T.alloc_buffer((2,), T.int32, scope="local")
                state[0] = 1
                if bx == 0:
                    state[1] = 0
                state[1] = state[1] + A[bx]
                B[bx] = state[1]

    return nested


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("factory", [_nested_serial, _nested_kernel, _nested_sblock])
def test_serial_control_flow_and_sblocks_share_the_source_pool(target, factory):
    function = factory()
    ((allocation, parents),) = _allocations(_prepare(function, target))
    assert allocation.var.name == "state"
    assert not parents
    kernel = tilelang.compile(function, target=target, out_idx=1, pass_configs=_config("parallel"))
    if target == "c":
        assert "#pragma omp" not in kernel.get_kernel_source()
    data = torch.arange(1, 9, dtype=torch.int32)
    torch.testing.assert_close(kernel(data), data.cumsum(0).to(torch.int32), rtol=0, atol=0)


def _outer_parallel():
    @T.prim_func
    def nested(A: T.Tensor((8,), T.int32), B: T.Tensor((8,), T.int32)):
        for row in T.parallel(2):
            with T.Kernel(4) as bx:
                state = T.alloc_buffer((2,), T.int32, scope="local")
                state[0] = 1
                if bx == 0:
                    state[1] = 0
                state[1] = state[1] + A[row * 4 + bx]
                B[row * 4 + bx] = state[1]

    return nested


def _outer_lexical():
    @T.prim_func
    def nested(A: T.Tensor((8,), T.int32), B: T.Tensor((8,), T.int32)):
        for row in T.serial(2):
            with T.attr(None, "lexical_alloc_scope", 1):
                with T.Kernel(4) as bx:
                    state = T.alloc_buffer((2,), T.int32, scope="local")
                    state[0] = 1
                    if bx == 0:
                        state[1] = 0
                    state[1] = state[1] + A[row * 4 + bx]
                    B[row * 4 + bx] = state[1]

    return nested


def _annotated_sblock():
    @T.prim_func
    def nested(A: T.Tensor((8,), T.int32), B: T.Tensor((8,), T.int32)):
        for row in T.serial(2):
            with T.sblock("row_pool"):
                T.sblock_attr({"lexical_alloc_scope": 1})
                with T.Kernel(4) as bx:
                    state = T.alloc_buffer((2,), T.int32, scope="local")
                    state[0] = 1
                    if bx == 0:
                        state[1] = 0
                    state[1] = state[1] + A[row * 4 + bx]
                    B[row * 4 + bx] = state[1]

    return nested


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_sblock_annotation_owns_the_source_pool_in_its_lexical_suffix(target):
    function = _annotated_sblock()
    ((allocation, parents),) = _allocations(_prepare(function, target))
    assert len(parents) == 2 and isinstance(parents[0], ir.For)
    assert parents[0].var.name == "row" and parents[1].op.name == "tl.attr"
    sequence = parents[1].body.seq
    assert isinstance(sequence[0], ir.Evaluate) and sequence[0].value.op.name == "tl.annotation"
    assert sequence[1].same_as(allocation)
    assert "tl.source_block_alloc" not in allocation.value.attrs
    kernel = tilelang.compile(function, target=target, out_idx=1, pass_configs=_config("parallel"))
    data = torch.arange(1, 9, dtype=torch.int32)
    expected = data.reshape(2, 4).cumsum(1).reshape(-1).to(torch.int32)
    torch.testing.assert_close(kernel(data), expected, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("factory", [_outer_parallel, _outer_lexical])
def test_explicit_parallel_and_lexical_boundaries_own_source_local_storage(target, factory):
    function = factory()
    ((_, parents),) = _allocations(_prepare(function, target))
    assert parents
    assert not any(isinstance(parent, ir.For) and parent.var.name.startswith("block_") for parent in parents)
    if factory is _outer_parallel:
        assert len(parents) == 1 and parents[0].kind == ir.ForKind.PARALLEL
    else:
        assert isinstance(parents[-1], ir.RegionStmt)
        assert parents[-1].op.name == "tl.attr"
    kernel = tilelang.compile(function, target=target, out_idx=1, pass_configs=_config("parallel"))
    data = torch.arange(1, 9, dtype=torch.int32)
    expected = data.reshape(2, 4).cumsum(1).reshape(-1).to(torch.int32)
    torch.testing.assert_close(kernel(data), expected, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_complete_reset_is_still_safe_to_privatize(target):
    @T.prim_func
    def reset(A: T.Tensor((32,), T.int32), B: T.Tensor((32,), T.int32)):
        with T.Kernel(8) as bx:
            scratch = T.alloc_buffer((4,), T.int32, scope="local")
            for i in T.serial(4):
                scratch[i] = A[bx * 4 + i] + 3
            for i in T.serial(4):
                B[bx * 4 + i] = scratch[i] * 2

    ((_, parents),) = _allocations(_prepare(reset, target))
    assert not parents
    kernel = tilelang.compile(reset, target=target, out_idx=1, pass_configs=_config("parallel"))
    source = kernel.get_kernel_source()
    if target == "c":
        assert "#pragma omp parallel for" in source
        assert source.index("int32_t scratch") > source.index("for (int32_t block_0")
    else:
        assert "__TVMBackendParallelLaunch" in source
    data = torch.arange(32, dtype=torch.int32)
    torch.testing.assert_close(kernel(data), (data + 3) * 2, rtol=0, atol=0)


def _tensor(name, shape):
    return ir.Var(name, tirx.TensorType(shape, "int32", scope="local", layout="default"))


def _allocate(tensor, *, source):
    annotations = {"tl.source_block_alloc": ir.const(True, "bool")} if source else {}
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], attrs=annotations, ty_args=[tensor.ty]))


def _function(body, params=()):
    return tirx.Function(list(params), body, ir.PrimType("void")).with_attr("global_symbol", "main")


def test_native_allocation_retains_lexical_lifetime_and_source_marker_is_consumed():
    bx = ir.Var("bx", "int32")
    source, native = _tensor("source", (2,)), _tensor("native", (2,))
    function = _function(ir.For(bx, 0, 4, 1, ir.ForKind.DEFAULT, [_allocate(source, source=True), _allocate(native, source=False)]))
    before = to_json_graph_str(function)
    module = _prepare(function)
    (source_bind, source_parents), (native_bind, native_parents) = _allocations(module)
    assert source_bind.var.same_as(source) and not source_parents
    assert native_bind.var.same_as(native) and len(native_parents) == 1
    assert isinstance(native_parents[0], ir.For)
    assert "tl.source_block_alloc" not in source_bind.value.attrs
    assert to_json_graph_str(function) == before
    again = tilelang.cpu.transform.LowerCPUKernelLaunch()(module)
    assert tvm_ffi.structural_equal(module, again)


@pytest.mark.parametrize("dependency", ["loop", "bind", "layout", "read"])
def test_source_allocation_never_leaves_its_type_dependencies(dependency):
    bx, extent = ir.Var("bx", "int32"), ir.Var("extent", "int32")
    parameter = ir.Var("sizes", tirx.TensorType((4,), "int32", layout="default"))
    if dependency == "read":
        tensor = _tensor("scratch", (parameter[bx],))
    elif dependency == "loop":
        tensor = _tensor("scratch", (bx + 1,))
    elif dependency == "bind":
        tensor = _tensor("scratch", (extent,))
    else:
        layout = tirx.TileLayout.from_iters(shard=[Iter(4, 1, "m")], replica=[], offset={"m": bx})
        tensor = ir.Var("scratch", tirx.TensorType((4,), "int32", scope="local", layout=layout))
    body = [_allocate(tensor, source=True)]
    if dependency == "bind":
        body.insert(0, ir.Bind(extent, bx + 1))
    function = _function(ir.For(bx, 0, 4, 1, ir.ForKind.DEFAULT, body), [parameter])
    module = _prepare(function)
    ((allocation, parents),) = _allocations(module)
    assert len(parents) == 1 and parents[0].var.same_as(bx)
    assert allocation.var.same_as(tensor)
    if dependency == "bind":
        assert parents[0].body.seq[0].var.same_as(extent)
        assert parents[0].body.seq[1].same_as(allocation)


@pytest.mark.parametrize("scope", ["local", "shared", "global"])
def test_explicit_source_lexical_scope_only_restricts_local_like_storage(scope):
    bx = ir.Var("bx", "int32")
    tensor = ir.Var("scratch", tirx.TensorType((4,), "int32", scope=scope, layout="default"))
    schema, payload = _annotation_codec.encode_annotations({"node": None, "key": "lexical_alloc_scope", "value": 1})
    region = ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, _allocate(tensor, source=True))
    module = _prepare(_function(ir.For(bx, 0, 4, 1, ir.ForKind.DEFAULT, region)))
    ((_, parents),) = _allocations(module)
    if scope == "local":
        assert len(parents) == 2 and isinstance(parents[-1], ir.RegionStmt)
    else:
        assert not parents


def test_explicit_native_lexical_scope_remains_a_boundary_for_source_allocation():
    bx = ir.Var("bx", "int32")
    tensor = _tensor("scratch", (4,))
    region = ir.RegionStmt("tl.lexical_scope", [], [], {"explicit_alloc_scope": True}, _allocate(tensor, source=True))
    module = _prepare(_function(ir.For(bx, 0, 4, 1, ir.ForKind.DEFAULT, region)))
    ((_, parents),) = _allocations(module)
    assert len(parents) == 2 and parents[-1].op.name == "tl.lexical_scope"


def _aliased_partial():
    @T.prim_func
    def aliased(A: T.Tensor((8,), T.int32), B: T.Tensor((8,), T.int32)):
        with T.Kernel(8) as bx:
            state = T.alloc_buffer((2,), T.int32, scope="local")
            view = T.decl_buffer((2,), T.int32, data=state.data)
            view[0] = 1
            if bx == 0:
                state[1] = 0
            view[1] = view[1] + A[bx]
            B[bx] = state[1]

    return aliased


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_view_of_source_allocation_keeps_the_same_persistent_storage(target):
    function = _aliased_partial()
    ((_, parents),) = _allocations(_prepare(function, target))
    assert not parents
    kernel = tilelang.compile(function, target=target, out_idx=1, pass_configs=_config("parallel"))
    if target == "c":
        assert "#pragma omp" not in kernel.get_kernel_source()
    data = torch.arange(1, 9, dtype=torch.int32)
    torch.testing.assert_close(kernel(data), data.cumsum(0).to(torch.int32), rtol=0, atol=0)


def _source_attr(key, body):
    schema, payload = _annotation_codec.encode_annotations({"node": None, "key": key, "value": 1})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def test_outermost_explicit_parallel_scope_owns_the_source_pool():
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    tensor = _tensor("scratch", (4,))
    body = ir.For(row, 0, 2, 1, ir.ForKind.PARALLEL, ir.For(column, 0, 4, 1, ir.ForKind.PARALLEL, _allocate(tensor, source=True)))
    ((_, parents),) = _allocations(_prepare(_function(body)))
    assert len(parents) == 1 and parents[0].var.same_as(row)


@pytest.mark.parametrize("scope", ["local", "shared", "global"])
def test_source_lexical_scope_overrides_parallel_attachment_only_for_local_storage(scope):
    row = ir.Var("row", "int32")
    tensor = ir.Var("scratch", tirx.TensorType((4,), "int32", scope=scope, layout="default"))
    body = _source_attr("lexical_alloc_scope", ir.For(row, 0, 2, 1, ir.ForKind.PARALLEL, _allocate(tensor, source=True)))
    ((_, parents),) = _allocations(_prepare(_function(body)))
    assert isinstance(parents[0], ir.RegionStmt)
    if scope == "local":
        assert len(parents) == 1
    else:
        assert len(parents) == 2 and parents[1].var.same_as(row)


def test_source_pragma_scope_owns_allocations_inside_serial_control_flow():
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    tensor = _tensor("scratch", (4,))
    body = ir.For(
        row,
        0,
        2,
        1,
        ir.ForKind.DEFAULT,
        _source_attr("pragma_auto_unroll_max_step", ir.For(column, 0, 4, 1, ir.ForKind.DEFAULT, _allocate(tensor, source=True))),
    )
    ((_, parents),) = _allocations(_prepare(_function(body)))
    assert len(parents) == 2 and parents[0].var.same_as(row) and isinstance(parents[1], ir.RegionStmt)


@pytest.mark.parametrize(
    "kind,target",
    [
        ("cuda", {"kind": "cuda", "arch": "sm_100a"}),
        ("hip", {"kind": "hip", "mcpu": "gfx942"}),
        ("metal", {"kind": "metal"}),
        ("webgpu", {"kind": "webgpu"}),
        ("ascend", {"kind": "ascend"}),
    ],
)
def test_non_cpu_finalizer_consumes_only_source_origin(kind, target):
    tensor = _tensor("scratch", (4,))
    location = ir.SourceLoc(ir.SourceName("source_pool.py"), 17, 3, 17, 35)
    annotations = {"tl.source_block_alloc": ir.const(True, "bool"), "volatile": ir.const(True, "bool"), "test.keep": 37}
    call = ir.Call("tirx.alloc_tensor", [], attrs=annotations, ty_args=[tensor.ty], loc=location)
    function = _function(ir.Bind(tensor, call)).with_attr("target", Target(target))
    before = to_json_graph_str(function)
    module = tilelang.transform.LowerSourceAllocations()(ir.IRModule.from_expr(function))
    ((allocation, parents),) = _allocations(module)
    assert not parents and allocation.var.same_as(tensor)
    result = allocation.value
    assert set(result.attrs.keys()) == {"volatile", "test.keep"}
    assert int(result.attrs["test.keep"]) == 37
    assert tvm_ffi.structural_equal(result.attrs["volatile"], annotations["volatile"])
    assert result.ty.same_as(call.ty) and result.ty_args[0].same_as(tensor.ty)
    assert result.loc.same_as(location) and len(result.args) == 0
    assert to_json_graph_str(function) == before
    assert tirx.analysis.check_well_formed(module)


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_cpu_finalization_rejects_unplanned_source_allocation(target):
    tensor = _tensor("scratch", (4,))
    function = _function(_allocate(tensor, source=True)).with_attr("target", Target(target))
    module = ir.IRModule.from_expr(function)
    before = to_json_graph_str(function)
    with pytest.raises(ValueError, match="LowerCPUKernelLaunch.*before LowerSourceAllocations"):
        tilelang.transform.LowerSourceAllocations()(module)
    assert to_json_graph_str(function) == before
    module = tilelang.cpu.transform.LowerCPUKernelLaunch()(module)
    module = tilelang.transform.LowerSourceAllocations()(module)
    ((allocation, _),) = _allocations(module)
    assert "tl.source_block_alloc" not in allocation.value.attrs


def test_finalizing_source_allocation_requires_a_bound_target():
    tensor = _tensor("scratch", (4,))
    module = ir.IRModule.from_expr(_function(_allocate(tensor, source=True)))
    with pytest.raises(ValueError, match="requires a bound target for source allocation origins"):
        tilelang.transform.LowerSourceAllocations()(module)


def test_finalizer_keeps_unmarked_native_allocation_without_a_target():
    tensor = _tensor("scratch", (4,))
    module = ir.IRModule.from_expr(_function(_allocate(tensor, source=False)))
    result = tilelang.transform.LowerSourceAllocations()(module)
    assert tvm_ffi.structural_equal(result, module)


def test_consuming_source_origin_does_not_relax_shared_arena_annotation_checks():
    allocations = []
    for name in ["first", "second"]:
        tensor = ir.Var(name, tirx.TensorType((4,), "int32", scope="shared.dyn", layout="default"))
        allocations.append(
            ir.Bind(
                tensor,
                ir.Call(
                    "tirx.alloc_tensor",
                    [],
                    ty_args=[tensor.ty],
                    attrs={"tl.source_block_alloc": ir.const(True, "bool"), "test.unconsumed": 1},
                ),
            )
        )
    function = _function(allocations).with_attr("target", Target({"kind": "cuda", "arch": "sm_100a"}))
    module = tilelang.transform.LowerSourceAllocations()(ir.IRModule.from_expr(function))
    with pytest.raises(ValueError, match="allocation annotations to be consumed first"):
        tilelang.transform.MergeSharedMemoryAllocations()(module)


def test_non_cpu_finalizer_rejects_an_invalid_source_origin_marker():
    tensor = _tensor("scratch", (4,))
    call = ir.Call("tirx.alloc_tensor", [], attrs={"tl.source_block_alloc": ir.const(False, "bool")}, ty_args=[tensor.ty])
    function = _function(ir.Bind(tensor, call)).with_attr("target", Target({"kind": "cuda", "arch": "sm_100a"}))
    with pytest.raises(ValueError, match="must be the source allocation marker"):
        tilelang.transform.LowerSourceAllocations()(ir.IRModule.from_expr(function))
