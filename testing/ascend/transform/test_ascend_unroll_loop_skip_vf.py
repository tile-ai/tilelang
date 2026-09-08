import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx


def _loop(name, kind, body, explicit=None, extent=2, step=None):
    loop_var = tirx.Var(name, "int32")
    annotations = {}
    if explicit is not None:
        annotations["pragma_unroll_explicit"] = explicit
    if callable(body):
        body = body(loop_var)
    return tirx.For(
        loop_var,
        0,
        extent,
        kind,
        body,
        annotations=annotations,
        step=step,
    )


def _apply(body, params=None, buffer_map=None):
    func = tirx.PrimFunc(
        params or [],
        body,
        buffer_map=buffer_map or {},
    ).with_attr("global_symbol", "main")
    return transform.UnrollLoopSkipVF()(tvm.IRModule.from_expr(func))["main"]


def test_unroll_loop_skip_vf_only_expands_requested_loops_outside_vf():
    explicit_outside = _loop(
        "explicit_outside",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.Evaluate(var),
        explicit=True,
    )
    non_explicit_outside = _loop(
        "non_explicit_outside",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.Evaluate(var),
        explicit=False,
    )
    unannotated_unroll = _loop(
        "unannotated_unroll",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.Evaluate(var),
    )
    annotated_serial = _loop(
        "annotated_serial",
        tirx.ForKind.SERIAL,
        lambda var: tirx.Evaluate(var),
        explicit=True,
    )

    nested_explicit = _loop(
        "nested_explicit",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.Evaluate(var),
        explicit=True,
    )
    non_explicit_outer = _loop(
        "non_explicit_outer",
        tirx.ForKind.UNROLLED,
        nested_explicit,
        explicit=False,
    )

    ordinary_explicit = _loop(
        "ordinary_explicit",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.Evaluate(var),
        explicit=True,
    )
    ordinary_block = tirx.SBlock([], [], [], "ordinary", ordinary_explicit)

    simd_explicit = _loop(
        "simd_explicit",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.Evaluate(var),
        explicit=True,
    )
    simd_block = tirx.SBlock([], [], [], "SIMD_VF", simd_explicit)

    simt_explicit = _loop(
        "simt_explicit",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.Evaluate(var),
        explicit=True,
    )
    simt_block = tirx.SBlock([], [], [], "SIMT_VF", simt_explicit)

    after = _apply(
        tirx.SeqStmt(
            [
                explicit_outside,
                non_explicit_outside,
                unannotated_unroll,
                annotated_serial,
                non_explicit_outer,
                ordinary_block,
                simd_block,
                simt_block,
            ]
        )
    )

    loops = {}

    def collect(node):
        if isinstance(node, tirx.For):
            loops[node.loop_var.name] = node

    tirx.stmt_functor.post_order_visit(after.body, collect)

    assert set(loops) == {
        "non_explicit_outside",
        "unannotated_unroll",
        "annotated_serial",
        "non_explicit_outer",
        "simd_explicit",
        "simt_explicit",
    }
    assert loops["non_explicit_outside"].kind == tirx.ForKind.UNROLLED
    assert loops["non_explicit_outside"].annotations["pragma_unroll_explicit"] is False
    assert loops["unannotated_unroll"].kind == tirx.ForKind.UNROLLED
    assert loops["annotated_serial"].kind == tirx.ForKind.SERIAL
    assert loops["simd_explicit"].body.same_as(simd_explicit.body)
    assert loops["simt_explicit"].body.same_as(simt_explicit.body)


def test_unroll_loop_skip_vf_preserves_non_unit_step_values():
    output = tirx.decl_buffer((8,), "int32", name="output")
    explicit_loop = _loop(
        "explicit_step",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.BufferStore(output, var, [var]),
        explicit=True,
        extent=4,
        step=tirx.IntImm("int32", 2),
    )

    after = _apply(
        explicit_loop,
        params=[output.data],
        buffer_map={output.data: output},
    )

    stores = []
    tirx.stmt_functor.post_order_visit(
        after.body,
        lambda node: stores.append(node) if isinstance(node, tirx.BufferStore) else None,
    )
    analyzer = tvm.arith.Analyzer()
    values = [int(analyzer.simplify(store.value).value) for store in stores]
    indices = [int(analyzer.simplify(store.indices[0]).value) for store in stores]

    assert values == [0, 2, 4, 6]
    assert indices == [0, 2, 4, 6]


def test_unroll_loop_skip_vf_expands_loop_wrapping_vf_blocks():
    inner_explicit = _loop(
        "inner_explicit",
        tirx.ForKind.UNROLLED,
        lambda var: tirx.Evaluate(var),
        explicit=True,
    )
    simd_block = tirx.SBlock([], [], [], "SIMD_VF", inner_explicit)
    outer_explicit = _loop(
        "outer_explicit",
        tirx.ForKind.UNROLLED,
        simd_block,
        explicit=True,
    )

    after = _apply(outer_explicit)
    loop_names = []
    vf_blocks = []

    def collect(node):
        if isinstance(node, tirx.For):
            loop_names.append(node.loop_var.name)
        elif isinstance(node, tirx.SBlock) and node.name_hint == "SIMD_VF":
            vf_blocks.append(node)

    tirx.stmt_functor.post_order_visit(after.body, collect)

    assert loop_names == ["inner_explicit", "inner_explicit"]
    assert len(vf_blocks) == 2


if __name__ == "__main__":
    tilelang.testing.main()
