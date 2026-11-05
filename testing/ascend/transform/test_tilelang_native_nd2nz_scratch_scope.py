"""ND-to-NZ scratch storage retains its nearest source-block lifetime."""

import pytest
import tvm_ffi
from tilelang.ascend import transform
from tvm_tirx import ir, tirx

from testing.ascend._ir import (
    attribute,
    attribute_key,
    calls,
    copy,
    kernel,
    nodes,
    sblocks,
    statements,
    tensor,
)


def block(name, body):
    return attribute(None, "tl.sblock", {"name": name, "no_realize": True}, body)


def kernel_carrier(module, carrier):
    if carrier == "kernel":
        (root,) = sblocks(module, "tilelang_root")
        return ir.IRModule({"main": module["main"].with_body([attribute(None, "tl.kernel", {}, root.body)])})
    return module


def owner_block(module, name, carrier):
    if name == "tilelang_root" and carrier == "kernel":
        (owner,) = [item for item in nodes(module, ir.RegionStmt) if attribute_key(item) == "tl.kernel"]
        return owner
    (owner,) = sblocks(module, name)
    return owner


def rewrite(before):
    assert tirx.analysis.check_well_formed(before)
    serialized = ir.save_json(before)
    after = transform.InsertNd2Nz()(before)
    assert ir.save_json(before) == serialized
    assert tirx.analysis.check_well_formed(after)
    ir.assert_structural_equal(transform.InsertNd2Nz()(after), after)
    return after


@pytest.mark.parametrize("carrier", ["sblock", "kernel"])
@pytest.mark.parametrize("scope", ["root", "conditional-block", "iteration-block"])
def test_scratch_definition_stays_at_the_nearest_block_boundary(scope, carrier):
    src = tensor((4, 16, 64), "bfloat16", name="src", scope="shared.dyn", layout=None)
    dst = tensor((16, 64), "bfloat16", name="dst", scope="shared.l1", layout=None)
    flag = ir.Var("flag", "bool")
    i = ir.Var("i", "int32")
    operation = copy(src, dst, src_indices=[i, 0, 0], src_shape=[1, 16, 64])
    conditional = ir.If(flag, [operation], None)
    if scope == "conditional-block":
        loop = ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [operation])
        body = ir.If(flag, [block("inner", [loop])], None)
    elif scope == "iteration-block":
        body = ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [block("inner", [conditional])])
    else:
        body = ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [conditional])
    layout = tvm_ffi.get_global_func("tl.make_ascend_nz_layout")(dst)
    before = kernel(body, buffers=[src, dst], params=[flag], annotations={"layout_map": {dst: layout}})
    after = rewrite(kernel_carrier(before, carrier))
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    (post,) = calls(after, "tl.ascend_nd2nz_post_copy")
    scratch = scatter.args[1].args[0].source
    assert post.args[1].args[0].source.same_as(scratch)
    assert tuple(int(extent) for extent in scratch.ty.shape) == (17 * 64,)
    (binding,) = [item for item in nodes(after, ir.Bind) if item.var.same_as(scratch)]
    owner = owner_block(after, "tilelang_root" if scope == "root" else "inner", carrier)
    assert any(item.same_as(binding) for item in owner.body.seq)
    parents = next(parents for item, parents in statements(after) if item.same_as(binding))
    assert sum(isinstance(parent, ir.If) for parent in parents) == (scope == "conditional-block")
    assert sum(isinstance(parent, ir.For) for parent in parents) == (scope == "iteration-block")
    (branch,) = nodes(after, ir.If)
    (loop,) = nodes(after, ir.For)
    assert branch.condition.same_as(flag)
    assert loop.var.same_as(i)
    assert scatter.args[0].args[0].source.same_as(src)
    assert scatter.args[0].args[0].indices[0].same_as(i)
    assert post.args[0].args[0].source.same_as(dst)
    scatter_parents = next(parents for item, parents in statements(after) if isinstance(item, ir.Evaluate) and item.value.same_as(scatter))
    assert any(parent.same_as(branch) for parent in scatter_parents)
    assert any(parent.same_as(loop) for parent in scatter_parents)


@pytest.mark.parametrize("carrier", ["sblock", "kernel"])
@pytest.mark.parametrize("nested_blocks", [False, True])
def test_sibling_branches_keep_distinct_scratch_allocations(nested_blocks, carrier):
    src = tensor((16, 64), "bfloat16", name="src", scope="shared.dyn", layout=None)
    dst = tensor((16, 64), "bfloat16", name="dst", scope="shared.l1", layout=None)
    flag = ir.Var("flag", "bool")
    branches = [copy(src, dst), copy(src, dst)]
    if nested_blocks:
        branches = [block(name, [body]) for name, body in zip(("left", "right"), branches)]
    layout = tvm_ffi.get_global_func("tl.make_ascend_nz_layout")(dst)
    after = rewrite(
        kernel_carrier(
            kernel(
                ir.If(flag, [branches[0]], [branches[1]]),
                buffers=[src, dst],
                params=[flag],
                annotations={"layout_map": {dst: layout}},
            ),
            carrier,
        )
    )
    scatters = calls(after, "tl.ascend_nd2nz_scatter")
    posts = calls(after, "tl.ascend_nd2nz_post_copy")
    assert len(scatters) == len(posts) == 2
    scratch = [call.args[1].args[0].source for call in scatters]
    assert not scratch[0].same_as(scratch[1])
    assert all(call.args[1].args[0].source.same_as(storage) for call, storage in zip(posts, scratch))
    for index, storage in enumerate(scratch):
        binding, parents = next(
            (item, parents) for item, parents in statements(after) if isinstance(item, ir.Bind) and item.var.same_as(storage)
        )
        owner = next(parent for parent in reversed(parents) if attribute_key(parent) in ("tl.sblock", "tl.kernel"))
        expected = ("left", "right")[index] if nested_blocks else "tilelang_root"
        assert owner.same_as(owner_block(after, expected, carrier))
        assert any(item.same_as(binding) for item in owner.body.seq)
        assert any(isinstance(parent, ir.If) for parent in parents) == nested_blocks


@pytest.mark.parametrize("scope", ["conditional", "iteration"])
def test_function_without_an_owner_keeps_its_local_scratch_lifetime(scope):
    src = tensor((4, 16, 64), "bfloat16", name="src", scope="shared.dyn", layout=None)
    dst = tensor((16, 64), "bfloat16", name="dst", scope="shared.l1", layout=None)
    flag = ir.Var("flag", "bool")
    i = ir.Var("i", "int32")
    index = i if scope == "iteration" else ir.IntImm("int32", 0)
    operation = copy(src, dst, src_indices=[index, 0, 0], src_shape=[1, 16, 64])
    body = ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [operation]) if scope == "iteration" else ir.If(flag, [operation], None)
    layout = tvm_ffi.get_global_func("tl.make_ascend_nz_layout")(dst)
    module = kernel(body, buffers=[src, dst], params=[flag], annotations={"layout_map": {dst: layout}})
    (root,) = sblocks(module, "tilelang_root")
    before = ir.IRModule({"main": module["main"].with_body(root.body)})
    after = rewrite(before)
    assert not nodes(after, ir.RegionStmt)
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    (post,) = calls(after, "tl.ascend_nd2nz_post_copy")
    scratch = scatter.args[1].args[0].source
    assert post.args[1].args[0].source.same_as(scratch)
    assert tuple(int(extent) for extent in scratch.ty.shape) == (17 * 64,)
    binding, parents = next(
        (item, parents) for item, parents in statements(after) if isinstance(item, ir.Bind) and item.var.same_as(scratch)
    )
    control_type = ir.For if scope == "iteration" else ir.If
    (control,) = nodes(after, control_type)
    assert any(parent.same_as(control) for parent in parents)
    local_body = control.body if scope == "iteration" else control.then_case
    assert local_body.seq[0].same_as(binding)
    assert scatter.args[0].args[0].source.same_as(src)
    assert post.args[0].args[0].source.same_as(dst)
    if scope == "iteration":
        assert control.var.same_as(i)
        assert scatter.args[0].args[0].indices[0].same_as(i)
    else:
        assert control.condition.same_as(flag)
