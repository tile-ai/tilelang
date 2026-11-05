"""GPU reducer planning: concrete native IR, no device execution claim."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from test_native_reducer_contracts import (
    integer,
    tensor,
    allocate,
    region,
    tileop,
    annotation,
    decoded,
    annotations,
    update,
    loop,
    function,
    run_pass,
    objects,
)


def gpu_sum(dtype="float32", local_complete=False, seed=None, rows=8, threads=128):
    shape = [rows, 16] if local_complete else [rows]
    output_shape = [rows] if local_complete else [1]
    a, b = tensor("A", shape, dtype, "global"), tensor("B", output_shape, dtype, "global")
    acc, dst = tensor("acc", output_shape, dtype, "local.reducer"), tensor("dst", output_shape, dtype, "local.fragment")
    thread, i, k, publish = (ir.Var(name, "int32") for name in ("threadIdx_x", "i", "k", "publish"))
    args = [region(acc)]
    if seed is not None:
        args.append(ir.FloatImm(dtype, seed))
    updates = update(acc, [i] if local_complete else [0], ir.TensorLoad(a, [i, k] if local_complete else [i]))
    if local_complete:
        updates = loop(k, 16, [updates], parallel=False)
    body = [
        allocate(acc),
        allocate(dst),
        annotation({"reducer_info_v2": {acc: {"op": "sum"}}}),
        tileop("reducer_init", *args),
        loop(i, rows, [updates]),
        tileop("finalize_reducer_v2", region(acc), region(dst)),
        loop(publish, output_shape[0], [ir.TensorStore(b, [publish], ir.TensorLoad(dst, [publish]))]),
    ]
    return function(
        [a, b], [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(threads)], [thread], {}, body)], target="cuda"
    )


@pytest.mark.parametrize("force_baseline", [False, True])
def test_gpu_padded_scalar_uses_full_participant_plan(force_baseline):
    from tilelang.layout import PartialFragment

    value = gpu_sum()
    with tvm_tirx.ir.transform.PassContext(config={"tl.reducer_force_baseline": force_baseline}):
        value = run_pass("VerifyReducerEpoch", value)
        inferred = run_pass("LayoutInference", value)
        layouts = [
            layout
            for values in annotations(inferred)
            for _, layout in values.get("layout_map", {}).items()
            if isinstance(layout, PartialFragment)
        ]
        assert layouts
        assert all(int(layout.combine_size) == 128 for layout in layouts)
        # Eight active lanes do not prove full 128-thread coverage.  This
        # case must use the wide plan even without force_baseline.
        materialized = run_pass("ReducerPlanAndMaterialize", inferred)
    finalizers = [c for c in objects(materialized, ir.Call) if isinstance(c.op, ir.Op) and c.op.name == "tl.tileop.finalize_reducer"]
    assert len(finalizers) == 1
    args, metadata = decoded(finalizers[0])
    assert "plan" not in metadata
    assert len(args) == 2
    markers = []
    for stmt in objects(materialized, ir.RegionStmt):
        if stmt.op.name == "tl.attr":
            fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(stmt.args[0], stmt.attrs["annotation_schema"])
            if fields["key"] == "tl.parallel_multiplicity":
                markers.append(fields)
    assert len(markers) == 1
    assert tirx.analysis.check_well_formed(materialized)
    assert not tirx.analysis.undefined_vars(materialized)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_gpu_local_complete_and_packed_lanes_keep_physical_plan(dtype):
    value = run_pass("VerifyReducerEpoch", gpu_sum(dtype, local_complete=True, rows=128))
    inferred = run_pass("LayoutInference", value)
    materialized = run_pass("ReducerPlanAndMaterialize", inferred)
    assert not any(c.op.name == "tl.tileop.finalize_reducer" for c in objects(materialized, ir.Call) if isinstance(c.op, ir.Op))
    packed = [node.var for node in objects(materialized, ir.Bind) if node.var.name.endswith("_pk")]
    assert len(packed) == (0 if dtype == "float32" else 1)
    if packed:
        assert [int(x.value) for x in packed[0].ty.shape] == [128, 2]
        stores = [node for node in objects(materialized, ir.TensorStore) if node.dest.same_as(packed[0])]
        assert len(stores) == 1
        # The lane is the inner serial induction variable modulo two. Keeping
        # it an ordinary arithmetic expression prevents an opaque layout map
        # from silently changing a contribution's logical destination.
        lane = stores[0].indices[-1]
        variables = objects(lane, ir.Var)
        assert any(v.name == "k" for v in variables)
        k = next(v for v in variables if v.name == "k")
        for number in range(16):
            analyzer = tvm_tirx.sym.Analyzer()
            constraints = tvm_tirx.sym.ConstraintContext()
            constraints.append(tvm_tirx.sym.BindingConstraint(k, ir.IntImm(k.ty.dtype, number)))
            analyzer.sync_constraint_context(constraints)
            assert int(analyzer.simplify(lane).value) == number % 2


@pytest.mark.parametrize("extra_use", ["none", "load", "region", "alias"])
def test_finalize_layout_override_accounts_for_native_tensor_regions_and_aliases(extra_use):
    from tilelang.layout import Fragment, PartialFragment

    a, b = tensor("A", [2], scope="global"), tensor("B", [2], scope="global")
    acc, dst = tensor("acc", [2], scope="local.reducer"), tensor("dst", [2], scope="local.fragment")
    partial = PartialFragment([2], forward_thread_fn=lambda i, rep: i * 4 + rep, forward_index_fn=lambda i: 0, replicate=4, combine=2)
    original_dst = Fragment([2], forward_fn=lambda i: (i, 0))
    thread = ir.Var("threadIdx_x", "int32")
    body = [
        allocate(acc),
        allocate(dst),
        annotation({"reducer_info_v2": {acc: {"op": "sum"}}, "layout_map": {acc: partial, dst: original_dst}}),
        tileop("reducer_init", region(acc)),
        tileop("finalize_reducer_v2", region(acc), region(dst)),
        tileop("copy", region(dst), region(b)),
    ]
    if extra_use == "load":
        body.append(ir.TensorStore(b, [0], ir.TensorLoad(dst, [0])))
    elif extra_use == "region":
        body.append(tileop("fill", region(dst), ir.FloatImm("float32", 0)))
    elif extra_use == "alias":
        body.append(ir.Bind(tensor("alias", [2], scope="local.fragment"), dst))
    value = function(
        [a, b], [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(8)], [thread], {}, body)], target="cuda"
    )
    materialized = run_pass("ReducerPlanAndMaterialize", value)
    assigned = [
        layout for metadata in annotations(materialized) for key, layout in metadata.get("layout_map", {}).items() if key.same_as(dst)
    ]
    assert assigned
    # A plain copy chain is free to take the reducer's induced layout. Any
    # additional load/region/pointer use must preserve the existing layout.
    assert all(layout.is_equal(original_dst) == (extra_use != "none") for layout in assigned)


@pytest.mark.parametrize("force_baseline", [False, True])
@pytest.mark.parametrize("fragment_input", [False, True])
def test_gpu_full_domain_narrow_reduces_only_reduction_axis(force_baseline, fragment_input):
    from tilelang.layout import PartialFragment

    a, b = tensor("A", [8, 16], scope="global"), tensor("B", [8], scope="global")
    acc, dst = tensor("acc", [8], scope="local.reducer"), tensor("dst", [8], scope="local.fragment")
    staged = tensor("staged", [8, 16], scope="local.fragment")
    inputs = [allocate(staged), tileop("copy", region(a), region(staged))] if fragment_input else []
    source = staged if fragment_input else a
    i, k, thread = (ir.Var(name, "int32") for name in ("i", "k", "threadIdx_x"))
    value = function(
        [a, b],
        [
            ir.RegionStmt(
                "tirx.launch_thread",
                [ir.StringImm("threadIdx.x"), integer(128)],
                [thread],
                {},
                inputs
                + [
                    allocate(acc),
                    allocate(dst),
                    annotation({"reducer_info_v2": {acc: {"op": "sum"}}}),
                    tileop("reducer_init", region(acc)),
                    loop(i, 8, [loop(k, 16, [update(acc, [i], ir.TensorLoad(source, [i, k]))])]),
                    tileop("finalize_reducer_v2", region(acc), region(dst)),
                    tileop("copy", region(dst), region(b)),
                ],
            )
        ],
        target="cuda",
    )
    with tvm_tirx.ir.transform.PassContext(config={"tl.reducer_force_baseline": force_baseline}):
        inferred = run_pass("LayoutInference", run_pass("VerifyReducerEpoch", value))
        layouts = [
            layout
            for values in annotations(inferred)
            for _, layout in values.get("layout_map", {}).items()
            if isinstance(layout, PartialFragment)
        ]
        assert layouts
        assert all(int(layout.combine_size) == (16 if fragment_input and not force_baseline else 128) for layout in layouts)
        # A fragment access preserves the original nest. The baseline fuser
        # is allowed to fuse direct-global loops, whose derived output index
        # deliberately falls back to the full participant plan.
        materialized = run_pass("ReducerPlanAndMaterialize", inferred)
    finalizers = [c for c in objects(materialized, ir.Call) if isinstance(c.op, ir.Op) and c.op.name == "tl.tileop.finalize_reducer"]
    assert len(finalizers) == 1
    args, metadata = decoded(finalizers[0])
    if force_baseline or not fragment_input:
        assert "plan" not in metadata
        assert len(args) == 2
    else:
        assert metadata["plan"] == 1
        assert [int(x.value) for x in args[2:]] == [16, 1]
    assert tirx.analysis.check_well_formed(materialized)
    assert not tirx.analysis.undefined_vars(materialized)
