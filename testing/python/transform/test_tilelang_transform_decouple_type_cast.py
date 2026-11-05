import numpy as np
import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import tvm_ffi
import tvm_tirx
from tilelang.language._native import encode_annotations
from tilelang.testing.ir import collect_nodes
from tilelang.testing.packed import compile_packed_function
from tilelang.transform import DecoupleTypeCast

from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.tirx.layout import Iter


def _canonicalize_for_comparison(mod):
    def allocation_attrs(node, mutator):
        node = mutator.default_mutate(node)
        if isinstance(node.op, ir.Op) and node.op.name == "tirx.alloc_tensor" and node.attrs is None:
            # The operator accepts either None or an empty DictAttrs for no
            # allocation annotations. Nonempty attributes remain exact.
            return ir.Call(node.op, node.args, attrs={}, ty_args=node.ty_args, ty=node.ty, loc=node.loc)
        return node

    def redundant_scope(node, mutator):
        node = mutator.default_mutate(node)
        if (
            node.op.name == "tl.lexical_scope"
            and not node.args
            and not node.body_params
            and not node.result_vars
            and not node.attrs.dict
            and len(node.body) == 1
            and isinstance(node.body[0], ir.RegionStmt)
            and node.body[0].op.name == "tl.lexical_scope"
        ):
            # There are no sibling definitions in this outer scope, and the
            # inner scope keeps every allocation's lifetime and provenance.
            return node.body[0]
        return node

    # These comparisons retain CUDA's lexical allocation placement. Supply
    # the matching target only while consuming source-origin metadata, then
    # restore previously unbound functions for the exact comparison below.
    mod = mod.clone()
    unbound = []
    for symbol, function in list(mod.functions.items()):
        if "target" not in function.attrs:
            unbound.append(symbol)
            mod[symbol] = function.with_attr("target", tvm_tirx.target.Target("cuda"))
    mod = tilelang.transform.FinalizeAnnotations("cuda")(mod)
    for symbol in unbound:
        mod[symbol] = mod[symbol].without_attr("target")
    # DecoupleTypeCast creates an explicit row-major scratch layout. Materialize
    # implicit row-major layouts on both sides with the production pass; retain
    # every explicit stride, offset, replica and non-memory ownership axis.
    mod = tilelang.transform.PrepareTensorLayouts()(mod)
    mod = tvm_ffi.structural_map(mod, (tirx.TileLayout, lambda layout: layout.canonicalize()))
    mod = tvm_ffi.structural_mutate(mod, [(ir.RegionStmt, redundant_scope), (ir.Call, allocation_attrs)])
    assert tirx.analysis.check_well_formed(mod)
    return mod


def _check(original, transformed):
    """Apply DecoupleTypeCast pass and check IR matches expected output."""
    mod = tvm_tirx.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    transformed = tvm_tirx.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))

    # The facade's named sblock and the pass's allocation marker describe the
    # same lexical scope. Compare their real native lowering, including the
    # allocation-scope provenance, rather than the temporary annotation codec.
    # Buffer annotations carry an explicit zero memory offset; native scratch
    # TensorTypes omit it. Use the target's layout canonicalizer on both sides
    # so this representation detail does not hide the remaining exact check.
    mod = _canonicalize_for_comparison(mod)
    transformed = _canonicalize_for_comparison(transformed)

    tvm_tirx.ir.assert_structural_equal(mod["main"], transformed["main"], True)


@pytest.mark.parametrize("offset", [0, 3])
def test_comparison_layout_canonicalization_keeps_addresses(offset):
    layout = tirx.TileLayout.from_iters([Iter(4, 4, "m"), Iter(4, 1, "m")], [], {"m": offset})
    tensor = ir.Var("a", tirx.TensorType((4, 4), "float32", layout=layout, align=4))
    function = tirx.Function([tensor], ir.Evaluate(ir.const(0)), ret_type=ir.PrimType("void"))
    normalized = _canonicalize_for_comparison(tvm_tirx.IRModule.from_expr(function))["main"]
    analyzer = tvm_tirx.sym.Analyzer()
    for index in range(16):
        before = analyzer.simplify(layout.apply(index)["m"])
        after = analyzer.simplify(normalized.params[0].ty.layout.apply(index)["m"])
        assert isinstance(before, ir.IntImm) and before.value == index + offset
        assert isinstance(after, ir.IntImm) and after.value == index + offset


@pytest.mark.parametrize("boundary", ["empty", "binding", "following_store", "explicit_marker"])
def test_comparison_keeps_scope_bindings_and_provenance(boundary):
    tensor = ir.Var("a", tirx.TensorType((4,), "float32", layout="default"))
    scalar = ir.Var("value", ir.PrimType("float32"))
    inner_value = scalar if boundary == "binding" else ir.const(1, "float32")
    inner = ir.RegionStmt(
        "tl.lexical_scope",
        [],
        [],
        {"explicit_alloc_scope": True},
        [ir.TensorStore(tensor, [0], inner_value)],
    )
    body = [inner]
    if boundary == "binding":
        body.insert(0, ir.Bind(scalar, tensor[1]))
    elif boundary == "following_store":
        body.append(ir.TensorStore(tensor, [2], ir.const(2, "float32")))
    outer_attrs = {"explicit_alloc_scope": True} if boundary == "explicit_marker" else {}
    outer = ir.RegionStmt("tl.lexical_scope", [], [], outer_attrs, body)
    function = tirx.Function([tensor], outer, ret_type=ir.PrimType("void"))
    normalized = _canonicalize_for_comparison(tvm_tirx.IRModule.from_expr(function))["main"]
    scopes = collect_nodes(normalized, ir.RegionStmt)
    assert len(scopes) == (1 if boundary == "empty" else 2)
    assert sum(bool(scope.attrs.dict.get("explicit_alloc_scope", False)) for scope in scopes) == (2 if boundary == "explicit_marker" else 1)
    if boundary != "empty":
        tvm_tirx.ir.assert_structural_equal(normalized, function, True)


@pytest.mark.parametrize("difference", ["memory_offset", "store_index", "allocation_annotation"])
def test_comparison_still_rejects_memory_and_annotation_changes(difference):
    def function(changed):
        offset = 3 if changed and difference == "memory_offset" else 0
        layout = tirx.TileLayout.from_iters([Iter(4, 1, "m")], [], {"m": offset})
        tensor = ir.Var("a", tirx.TensorType((4,), "float32", layout=layout, align=4))
        local = ir.Var("scratch", tirx.TensorType((4,), "float32", scope="local", layout="default"))
        attrs = {"alignment": 128} if changed and difference == "allocation_annotation" else {}
        allocation = ir.Bind(local, ir.Call("tirx.alloc_tensor", [], attrs=attrs, ty_args=[local.ty]))
        index = 1 if changed and difference == "store_index" else 0
        return tirx.Function(
            [tensor],
            [allocation, ir.TensorStore(tensor, [index], ir.const(1, "float32"))],
            ret_type=ir.PrimType("void"),
        )

    before = _canonicalize_for_comparison(tvm_tirx.IRModule.from_expr(function(False)))
    after = _canonicalize_for_comparison(tvm_tirx.IRModule.from_expr(function(True)))
    assert not tvm_ffi.structural_equal(before, after, map_free_vars=True)


def test_local_to_memory():
    """Test local → memory: compute to cast buffer, then copy to memory."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float4_e2m1fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = T.cast(b_frag[i], T.float4_e2m1fn)
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_memory_to_local():
    """Test memory → local: copy from memory to cast buffer, then compute."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float4_e2m1fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = b_frag[i]
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_local_to_memory_nonzero_min():
    """Staging stores and copy loads use offsets within the loop extent."""

    @T.prim_func
    def before(b: T.Tensor[(8,), T.float16]):
        b_frag = T.alloc_local((8,), T.float32)
        for i in T.vectorized(2, 6):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(8,), T.float16]):
        b_frag = T.alloc_local((8,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((4,), T.float16, scope="local")
            for i in T.vectorized(2, 6):
                b_local_cast[i - 2] = T.cast(b_frag[i], T.float16)
            for i_copy in T.vectorized(2, 6):
                b[i_copy] = b_local_cast[i_copy - 2]

    _check(before, after)


def test_memory_to_local_nonzero_min():
    """Staging copy stores and compute loads use offsets within the extent."""

    @T.prim_func
    def before(b: T.Tensor[(8,), T.float16]):
        b_frag = T.alloc_local((8,), T.float32)
        for i in T.vectorized(2, 6):
            b_frag[i] = b[i]

    @T.prim_func
    def after(b: T.Tensor[(8,), T.float16]):
        b_frag = T.alloc_local((8,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((4,), T.float16, scope="local")
            for i_copy in T.vectorized(2, 6):
                b_local_cast[i_copy - 2] = b[i_copy]
            for i in T.vectorized(2, 6):
                b_frag[i] = T.cast(b_local_cast[i - 2], T.float32)

    _check(before, after)


@pytest.mark.parametrize("loop_min", [2, "symbolic"])
def test_decouple_preserves_logical_loop_domain(loop_min):
    """RMW staging uses ordinals while all three loops retain their domain."""
    extent = 4
    if loop_min == "symbolic":
        loop_min = ir.Var("start", ir.PrimType("int32"))
        scalar_params = [loop_min]
    else:
        loop_min = ir.const(loop_min, "int32")
        scalar_params = []
    a = ir.Var("a", tirx.TensorType((16,), "float16", layout="default"))
    i = ir.Var("i", ir.PrimType("int32"))
    value = prim.Cast(ir.PrimType("float16"), prim.Cast(ir.PrimType("float32"), a[i]) + ir.const(1, "float32"))
    before = tirx.Function(
        [a, *scalar_params],
        ir.For(
            i,
            loop_min,
            loop_min + extent,
            ir.const(1, "int32"),
            ir.ForKind.VECTORIZED,
            ir.TensorStore(a, [i], value),
        ),
        ret_type=ir.PrimType("void"),
    ).with_attr("global_symbol", "main")

    # Keep logical memory indices and use ordinals only for the scratch tensor.
    # The RMW load and store must share the same four-element staging tensor.
    cast_buf = ir.Var("a_local_cast", tirx.TensorType((4,), "float16", scope="local", layout="default"))
    copy_from_var = ir.Var("copy_from", ir.PrimType("int32"))
    copy_to_var = ir.Var("copy_to", ir.PrimType("int32"))
    copy_from = ir.For(
        copy_from_var,
        loop_min,
        loop_min + extent,
        ir.const(1, "int32"),
        ir.ForKind.VECTORIZED,
        ir.TensorStore(cast_buf, [copy_from_var - loop_min], a[copy_from_var]),
    )
    ordinal = i - loop_min
    compute_value = prim.Cast(
        ir.PrimType("float16"),
        prim.Cast(ir.PrimType("float32"), cast_buf[ordinal]) + ir.const(1, "float32"),
    )
    compute = ir.For(
        i,
        loop_min,
        loop_min + extent,
        ir.const(1, "int32"),
        ir.ForKind.VECTORIZED,
        ir.TensorStore(cast_buf, [ordinal], compute_value),
    )
    copy_to = ir.For(
        copy_to_var,
        loop_min,
        loop_min + extent,
        ir.const(1, "int32"),
        ir.ForKind.VECTORIZED,
        ir.TensorStore(a, [copy_to_var], cast_buf[copy_to_var - loop_min]),
    )
    schema, payload = encode_annotations({"node": None, "key": "lexical_alloc_scope", "value": 1})
    scope = ir.RegionStmt(
        "tl.attr",
        [payload],
        [],
        {"schema_version": 1, "annotation_schema": schema},
        [
            ir.Bind(cast_buf, ir.Call("tirx.alloc_tensor", [], ty_args=[cast_buf.ty])),
            copy_from,
            compute,
            copy_to,
        ],
    )
    expected = before.with_body(scope)

    actual_mod = DecoupleTypeCast()(tvm_tirx.IRModule.from_expr(before))
    tvm_tirx.ir.assert_structural_equal(actual_mod["main"], expected, True)


@pytest.mark.parametrize("decouple", [False, True])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize(
    "start, extent, values",
    [(2, 4, [2, 3, 4, 5]), (-3, 4, [-3, -2, -1, 0]), (2, 0, []), (2, 1, [2])],
)
def test_nonzero_min_loop_execution(decouple, index_dtype, start, extent, values):
    """Legalization preserves the number of iterations and their logical values."""
    a = ir.Var("a", tirx.TensorType((8,), "float32", layout="default"))
    i = ir.Var("i", ir.PrimType(index_dtype))
    index = i - start
    value = prim.Cast(
        ir.PrimType("float32"),
        prim.Cast(ir.PrimType("float64"), a[index]) + prim.Cast(ir.PrimType("float64"), i),
    )
    loop = ir.For(
        i,
        ir.const(start, index_dtype),
        ir.const(start + extent, index_dtype),
        ir.const(1, index_dtype),
        ir.ForKind.VECTORIZED,
        ir.TensorStore(a, [index], value),
    )
    func = tirx.Function([a], loop, ret_type=ir.PrimType("void")).with_attr("global_symbol", "main")
    mod = tvm_tirx.IRModule.from_expr(func)
    with tvm_tirx.target.Target("c"):
        if decouple:
            mod = DecoupleTypeCast()(mod)
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
    # Execute the legalized domains with scalar C codegen.
    with tvm_tirx.ir.transform.PassContext(config={"tirx.disable_vectorize": True}):
        executable = compile_packed_function(mod["main"])
    output = tvm_tirx.runtime.tensor(np.full(8, -1, dtype="float32"))
    executable["main"](output)
    expected = np.full(8, -1, dtype="float32")
    expected[: len(values)] += values
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("step", [2, None], ids=["nonunit", "dynamic"])
def test_nonunit_step_remains_scalar(step):
    """The unit-step transformations leave other loop steps unchanged."""
    a = ir.Var("a", tirx.TensorType((8,), "float32", layout="default"))
    i = ir.Var("i", ir.PrimType("int32"))
    params = [a]
    if step is None:
        step = ir.Var("step", ir.PrimType("int32"))
        params.append(step)
    else:
        step = ir.const(step, "int32")
    value = prim.Cast(ir.PrimType("float32"), prim.Cast(ir.PrimType("float64"), a[i]) + ir.const(1, "float64"))
    body = ir.TensorStore(a, [i], value)
    begin, end = ir.const(2, "int32"), ir.const(6, "int32")
    before = tirx.Function(
        params,
        ir.For(i, begin, end, step, ir.ForKind.VECTORIZED, body),
        ret_type=ir.PrimType("void"),
    )
    expected = before.with_body(ir.For(i, begin, end, step, ir.ForKind.DEFAULT, body))
    with tvm_tirx.target.Target("c"):
        mod = DecoupleTypeCast()(tvm_tirx.IRModule.from_expr(before))
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected, True)


def test_no_transform_same_dtype():
    """Test no transformation when dtypes are the same."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float32]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float32]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    _check(before, after)


def test_no_transform_local_to_local():
    """Test no transformation for local → local (both are local buffers)."""

    @T.prim_func
    def before():
        a_frag = T.alloc_local((16,), T.float32)
        b_frag = T.alloc_local((16,), T.float4_e2m1fn)
        for i in T.vectorized(16):
            b_frag[i] = a_frag[i]

    @T.prim_func
    def after():
        a_frag = T.alloc_local((16,), T.float32)
        b_frag = T.alloc_local((16,), T.float4_e2m1fn)
        for i in T.vectorized(16):
            b_frag[i] = T.cast(a_frag[i], T.float4_e2m1fn)

    _check(before, after)


def test_no_transform_if_then_else_condition():
    """Test no transformation when different dtype is only in if_then_else condition.

    The condition part of if_then_else doesn't participate in type casting,
    so a global/shared buffer load with different dtype in condition should
    not trigger cast buffer insertion.
    """

    @T.prim_func
    def before(cond_buf: T.Tensor[(1,), T.int32]):
        acc = T.alloc_local((8,), T.float32)
        for i in T.vectorized(8):
            # cond_buf is int32, acc is float32, but cond_buf is only in condition
            acc[i] = T.if_then_else(cond_buf[0] > 0, acc[i] * 2.0, acc[i])

    @T.prim_func
    def after(cond_buf: T.Tensor[(1,), T.int32]):
        acc = T.alloc_local((8,), T.float32)
        for i in T.vectorized(8):
            # Should remain unchanged - no cast buffer needed
            acc[i] = T.if_then_else(cond_buf[0] > 0, acc[i] * T.float32(2), acc[i])

    _check(before, after)


def test_rmw_same_buffer_different_indices():
    """RMW with different indices into the same buffer: a[i] = a[i] + a[i+32].

    Both loads and the store target the same buffer but at different index
    expressions. Each unique (buffer, indices) pair should get its own cast
    buffer, and the RMW load `a[i]` should read from the same cast buffer the
    store writes to (so the read-side copy-from and the write-side copy-to
    share that buffer).
    """

    @T.prim_func
    def before(a: T.Tensor[(64,), T.float8_e4m3fn]):
        for i in T.vectorized(32):
            a[i] = T.cast(
                T.cast(a[i], T.float32) + T.cast(a[i + 32], T.float32),
                T.float8_e4m3fn,
            )

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    # Sanity checks: pass ran, two distinct cast buffers were created, and the
    # RMW load site no longer references `a` directly in the compute body.
    text = mod["main"].script()
    assert "a_local_cast" in text, "Expected cast buffer for store-side of a[i]"
    assert "a_local_cast_1" in text, "Expected second cast buffer for a[i+32]"


def test_local_to_memory_with_let_stmt():
    """Test local → memory transform still triggers through LetStmt-bound loads."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            factor = scale[i]
            b[i] = a_frag[i] * factor

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = T.cast(a_frag[i] * scale[i], T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_local_to_memory_with_bind_chain():
    """Test flat Bind chains are inlined according to sequential dominance."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            factor = scale[i]
            scaled = a_frag[i] * (factor + T.float32(1))
            b[i] = scaled

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = T.cast(a_frag[i] * (scale[i] + T.float32(1)), T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_local_to_memory_with_branch_local_bind():
    """Test Bind definitions inside an IfThenElse branch do not escape."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 8:
                factor = scale[i]
                b[i] = a_frag[i] * factor

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                if i < 8:
                    b_local_cast[i] = T.cast(a_frag[i] * scale[i], T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                if i_copy < 8:
                    b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_local_to_memory_with_nested_guards():
    """Nested guards must be conjoined on the generated copy-to loop."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 12:  # noqa: SIM102
                if i < 8:
                    b[i] = a_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i in T.vectorized(16):
                if i < 12:  # noqa: SIM102
                    if i < 8:
                        b_local_cast[i] = T.cast(a_frag[i], T.float8_e4m3fn)
            for i_copy in T.vectorized(16):
                if T.if_then_else(i_copy < 12, i_copy < 8, False):
                    b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_memory_to_local_with_nested_guards():
    """Nested guards must be conjoined on the generated copy-from loop."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 12:  # noqa: SIM102
                if i < 8:
                    a_frag[i] = b[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
            for i_copy in T.vectorized(16):
                if T.if_then_else(i_copy < 12, i_copy < 8, False):
                    b_local_cast[i_copy] = b[i_copy]
            for i in T.vectorized(16):
                if i < 12:  # noqa: SIM102
                    if i < 8:
                        a_frag[i] = T.cast(b_local_cast[i], T.float32)

    _check(before, after)


def test_if_else_control_flow_skips_decoupling():
    """Copy loops cannot represent branch-specific guards, so keep the source."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 8:
                b[i] = a_frag[i]
            else:
                a_frag[i] = 0.0

    _check(before, before)


def test_cast_buffers_wrapped_in_lexical_alloc_scope():
    """The pass must wrap cast buffers in a block annotated with
    lexical_alloc_scope, so StorageRewrite keeps them scoped to the use site."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    annotated_blocks = [0]
    allocs_inside = [0]

    def _visit(node):
        if isinstance(node, ir.RegionStmt) and node.op.name == "tl.attr":
            fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[0], node.attrs["annotation_schema"])
            if fields["key"] != "lexical_alloc_scope":
                return
            annotated_blocks[0] += 1
            allocs_inside[0] += sum(
                isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.alloc_tensor" for bind in collect_nodes(node.body, ir.Bind)
            )

    for node in collect_nodes(mod["main"], ir.RegionStmt):
        _visit(node)
    assert annotated_blocks[0] == 1, f"Expected 1 lexical_alloc_scope block, got {annotated_blocks[0]}"
    assert allocs_inside[0] == 1, f"Expected the cast buffer alloc inside the scope, got {allocs_inside[0]}"


# =============================================================================
# CUDA Codegen Tests
# =============================================================================


@tilelang.testing.requires_cuda
def test_codegen_local_to_memory():
    """Test CUDA codegen for local → memory with vectorized copy."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float4_e2m1fn")
        with T.Kernel(1, threads=32):
            b_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                b[i] = b_frag[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"
    # Should have vectorized copy (fp4_e2_16_t is 16 fp4 elements = 64 bits)
    assert "fp4_e2_16_t" in source, "Expected vectorized fp4 copy in generated code"


@tilelang.testing.requires_cuda
def test_codegen_memory_to_local():
    """Test CUDA codegen for memory → local with vectorized copy."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float4_e2m1fn")
        with T.Kernel(1, threads=32):
            a_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                a_frag[i] = b[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"


@tilelang.testing.requires_cuda
def test_codegen_fp8_local_to_memory():
    """Test CUDA codegen for fp8 local → memory."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float8_e4m3fn")
        with T.Kernel(1, threads=32):
            b_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                b[i] = b_frag[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"
    # Should have fp8 conversion (uses __nv_cvt for fp8)
    assert "fp8" in source and "cvt" in source, "Expected fp8 conversion"


@tilelang.testing.requires_cuda
def test_codegen_no_cast_buffer_same_dtype():
    """Test no cast buffer when dtypes are the same."""

    @tilelang.jit
    def kernel_fn():
        @T.prim_func
        def kernel(b: T.Tensor[(16,), T.float32]):
            with T.Kernel(1, threads=32):
                b_frag = T.alloc_local((16,), T.float32)
                for i in T.vectorized(16):
                    b[i] = b_frag[i]

        return kernel

    kernel = kernel_fn()
    source = kernel.get_kernel_source()

    # Should NOT have local cast buffer when dtypes match
    assert "local_cast" not in source, "Should not have cast buffer when dtypes match"


# =============================================================================
# End-to-end correctness + vectorization tests for DecoupleTypeCast
# =============================================================================


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10)
def test_e2e_bf16_global_to_frag():
    """bf16 global -> float32 frag -> bf16 global: roundtrip should be lossless.

    With 1024 bf16 elements and 64 threads, each thread handles 16 bf16 = 256 bits,
    so aligned inputs should retain the 256-bit global load/store fast path.
    The same kernel must also preserve values for misaligned input views.
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.bfloat16),
            B: T.Tensor((1024,), dtype=T.bfloat16),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()

    # The global 32-byte helper keeps the 256-bit aligned fast path and a safe
    # fallback for offsets. The local temporaries use Global256=false and must
    # not satisfy this global-memory width check.
    source = kernel.get_kernel_source()
    assert "load_global_256" in source or "tl::load_vector<ulonglong4, 32, true>" in source, "Expected 256-bit global load"
    assert "store_global_256" in source or "tl::store_vector<ulonglong4, 32, true>" in source, "Expected 256-bit global store"

    # Correctness
    a = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=0, atol=0)

    # Keep the same kernel and exact numerical oracle for misaligned views.
    # BF16 offsets of 1, 15, and 17 elements exercise the byte fallback, not an
    # unproved alignment increase on the public Tensor parameters.
    storage = torch.randn(1024 + 31, device="cuda", dtype=torch.bfloat16)
    assert storage.data_ptr() % 32 == 0
    for offset in (1, 15, 17):
        view = storage[offset : offset + 1024]
        assert view.data_ptr() % 32 != 0
        torch.testing.assert_close(kernel(view), view, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8)
def test_e2e_bf16_global_shared_frag():
    """bf16 global -> shared -> float32 frag -> bf16 global: roundtrip should be lossless.

    Shared memory path uses TMA for global->shared, then 128-bit for shared->local.
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.bfloat16),
            B: T.Tensor((1024,), dtype=T.bfloat16),
        ):
            with T.Kernel(1, threads=64):
                a_shared = T.alloc_shared((1024,), dtype=T.bfloat16)
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_shared)
                T.copy(a_shared, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()

    # Check: shared path should NOT use 256-bit (shared doesn't support it)
    source = kernel.get_kernel_source()
    assert "uint4" in source, f"Expected uint4 store in {source}"

    # Correctness
    a = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_fp8_global_to_frag():
    """fp8 global -> float32 frag -> fp8 global: roundtrip should be lossless.

    Verifies that cast constraints do not pollute the memory access layout.
    With 1024 fp8 elements and 64 threads, each thread handles 16 fp8 = 128 bits,
    so the kernel should use fp8_e4_16_t (128-bit) loads/stores.
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.float8_e4m3fn),
            B: T.Tensor((1024,), dtype=T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()
    source = kernel.get_kernel_source()
    assert "fp8_e4_16_t" in source, (
        "Expected fp8_e4_16_t (128-bit) loads/stores for N=1024. Cast constraints may be polluting layout decisions."
    )

    a = (torch.randn(1024, device="cuda", dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    b = kernel(a)
    torch.testing.assert_close(
        b.to(torch.float32),
        a.to(torch.float32),
        rtol=0,
        atol=0,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_fp8_manual_decouple():
    """fp8 with manually decoupled copy stages: same result as auto-decoupled.

    Tests: fp8 global -> fp8 frag -> float32 frag -> fp8 frag -> fp8 global
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.float8_e4m3fn),
            B: T.Tensor((1024,), dtype=T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float8_e4m3fn)
                b_frag = T.alloc_fragment((1024,), dtype=T.float32)
                c_frag = T.alloc_fragment((1024,), dtype=T.float8_e4m3fn)
                T.copy(A, a_frag)
                T.copy(a_frag, b_frag)
                T.copy(b_frag, c_frag)
                T.copy(c_frag, B)

        return main

    kernel = kernel_fn()

    # Check vectorization
    source = kernel.get_kernel_source()
    assert "fp8_e4_16_t" in source, "Expected fp8_e4_16_t in kernel source"

    # Correctness
    a = (torch.randn(1024, device="cuda", dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    b = kernel(a)
    torch.testing.assert_close(
        b.to(torch.float32),
        a.to(torch.float32),
        rtol=0,
        atol=0,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_scalar_load_no_cast_buffer():
    """Test that scalar memory load (b[0]) is not decoupled into a cast buffer.

    When a vectorized loop stores to global with a scalar memory load in the
    expression (e.g. c[i] = a_local[i] * b[0]), the scalar load's index does
    not depend on the loop variable. It should remain in the compute loop as
    a broadcast, not be extracted into a local cast buffer.

    Previously this caused float32x32 codegen errors because both
    VectorizePlanner and DecoupleTypeCast treated b[0] as a vector memory
    access.
    """

    @tilelang.jit
    def kernel_fn():
        @T.prim_func
        def main(
            a: T.Tensor[(32,), T.float8_e4m3fn],
            b: T.Tensor[(1,), T.float32],
            c: T.Tensor[(32,), T.float8_e4m3fn],
        ):
            with T.Kernel(1, threads=32):
                a_local = T.alloc_local((32,), T.float8_e4m3fn)
                T.copy(a, a_local)

                for i in T.vectorized(32):
                    c[i] = a_local[i] * b[0]

        return main

    kernel = kernel_fn()
    source = kernel.get_kernel_source()

    assert "c_local_cast" in source, "Expected c_local_cast for store-side decoupling"
    assert "b_local_cast" not in source, "Scalar load b[0] should not get a cast buffer"


def test_no_transform_evaluate_root_opaque_intrinsic():
    """Test that a vectorized loop whose body is an opaque intrinsic is skipped.

    Decoupling splits the value edge of a BufferStore to insert a staging
    buffer. An Evaluate discards its result and stores nothing, so that edge
    does not exist. Here the int64 index casts inside the ptx_cp_async
    predicate previously made the pass treat the statement as mixed-precision
    compute and rewrite both address operands to local staging buffers.

    Regression test for issue #2497.
    """

    @T.prim_func
    def before(a: T.Tensor[(256,), T.bfloat16], b: T.Tensor[(256,), T.bfloat16]):
        for i in T.vectorized(8):
            T.ptx_cp_async(
                T.address_of(b[i]),
                T.address_of(a[i]),
                1,
                T.cast(i, "int64") < T.int64(128),
            )

    @T.prim_func
    def after(a: T.Tensor[(256,), T.bfloat16], b: T.Tensor[(256,), T.bfloat16]):
        for i in T.vectorized(8):
            T.ptx_cp_async(
                T.address_of(b[i]),
                T.address_of(a[i]),
                1,
                T.cast(i, "int64") < T.int64(128),
            )

    _check(before, after)


@tilelang.testing.requires_cuda
def test_codegen_pipelined_int64_index_cp_async_operands():
    """Test that pipelined global→shared copies keep global/shared cp_async operands.

    With int64 index arithmetic, DecoupleTypeCast used to rewrite the address
    operands of ptx_cp_async to kernel-local register arrays, emitting
    cp_async_gs_conditional(local, local). That violates the instruction's
    address-space constraints (dst must be shared, src must be global) and
    fails at launch with CUDA_ERROR_INVALID_ADDRESS_SPACE.

    Regression test for issue #2497.
    """
    block_size, head_dim, num_stages = 128, 128, 2

    @tilelang.jit
    def kernel_fn():
        total_len = T.dynamic("total_len", "int64")
        num_blocks = T.dynamic("num_blocks")

        @T.prim_func
        def main(
            Input: T.Tensor[(total_len, head_dim), T.bfloat16],
            Output: T.Tensor[(total_len, head_dim), T.float32],
            Offsets: T.Tensor[(num_blocks,), T.int64],
        ):
            with T.Kernel(num_blocks, threads=256) as bx:
                shared_buf = T.alloc_shared((block_size, head_dim), T.bfloat16)
                accum = T.alloc_fragment((block_size, head_dim), T.float32)

                offset = Offsets[bx]
                trip_count = T.ceildiv(T.cast(total_len - offset, "int32"), block_size)
                for i in T.Pipelined(trip_count, num_stages=num_stages):
                    start = T.cast(i, "int64") * block_size
                    T.copy(
                        Input[offset + start : offset + start + block_size, :],
                        shared_buf,
                        disable_tma=True,
                    )
                    for r, c in T.Parallel(block_size, head_dim):
                        accum[r, c] = T.cast(shared_buf[r, c], T.float32)
                        Output[offset + start + T.cast(r, "int64"), c] = accum[r, c]

        return main

    source = kernel_fn().get_kernel_source()

    # Filter for the data-movement calls specifically: cp_async_commit/cp_async_wait
    # also match a bare "cp_async" substring, so they alone would satisfy a laxer check
    # while leaving the operand assertion below with nothing to inspect.
    gs_lines = [line for line in source.splitlines() if "cp_async_gs" in line]
    assert gs_lines, "Expected cp_async_gs (global->shared) calls for disable_tma copy"
    for line in gs_lines:
        assert "_local_cast" not in line, f"cp_async operand points at local memory: {line.strip()}"


def test_memory_dependent_guard_skips_decoupling():
    @T.prim_func
    def before(b: T.Tensor((16,), T.float16)):
        local = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i < 8:  # noqa: SIM102
                if b[i] > T.float16(0):
                    b[i] = local[i]

    _check(before, before)


def test_effectful_guard_skips_decoupling():
    @T.prim_func
    def before(b: T.Tensor((16,), T.float16)):
        local = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if T.call_extern("bool", "guard", i):
                b[i] = local[i]

    _check(before, before)


def test_nested_guard_preserves_conditional_evaluation():
    @T.prim_func
    def before(b: T.Tensor((16,), T.float16)):
        local = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            if i != 0:  # noqa: SIM102
                if 16 // i > 2:
                    b[i] = local[i]

    mod = DecoupleTypeCast()(tvm_tirx.IRModule.from_expr(before))
    copy_conditions = []

    def collect(node):
        if (
            isinstance(node, ir.If)
            and len(node.then_case) == 1
            and isinstance(node.then_case[0], ir.TensorStore)
            and node.then_case[0].dest.name == "b"
        ):
            copy_conditions.append(node.condition)

    for node in collect_nodes(mod["before"], ir.If):
        collect(node)
    assert len(copy_conditions) == 1
    condition = copy_conditions[0]
    assert isinstance(condition, ir.Call)
    assert condition.op.same_as(tvm_tirx.ir.Op.get("prim.if_then_else"))
    assert int(condition.args[2]) == 0


if __name__ == "__main__":
    tilelang.testing.main()
