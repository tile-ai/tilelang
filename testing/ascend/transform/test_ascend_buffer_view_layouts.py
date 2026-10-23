"""Layouts belong to buffer views; pointers and allocations retain that split."""

import pytest
import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import Fragment, Layout
from tvm import tirx

from testing.ascend._ir import calls, kernel


def _views():
    owner = tirx.decl_buffer((8, 16), "float32", name="owner", scope="shared")
    view = tirx.decl_buffer(owner.shape, owner.dtype, name="view", data=owner.data)
    return owner, view, Layout(owner.shape, lambda i, j: (i, j)), Layout(view.shape, lambda i, j: (j, i))


@pytest.mark.parametrize("region", [None, "SIMT_VF", "SIMD_VF"])
def test_explicit_layouts_remain_attached_to_each_view(region):
    owner, view, first, second = _views()
    body = tirx.SeqStmt([tirx.Evaluate(owner[3, 5]), tirx.Evaluate(view[3, 5])])
    annotations = {"layout_map": {owner: first, view: second}}
    if region:
        if region == "SIMT_VF":
            tx = tirx.IterVar(tvm.ir.Range(0, 32), tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
            body = tirx.AttrStmt(tx, "thread_extent", 32, body)
        body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], region, body, annotations=annotations))
        annotations = {}
    mod = kernel(body, buffers=[owner], annotations=annotations)
    with tvm.target.Target("ascend"):
        inferred = transform.AscendLayoutInference()(mod)
    layouts = inferred["main"].body.block.annotations["layout_map"]
    assert layouts[owner].is_equal(first)
    assert layouts[view].is_equal(second)


def test_unconstrained_reshape_inherits_the_unique_layout():
    owner, _, layout, _ = _views()
    view = tirx.decl_buffer((4, 32), owner.dtype, name="reshaped", data=owner.data)
    mod = kernel(tirx.Evaluate(view[1, 7]), buffers=[owner], annotations={"layout_map": {owner: layout}})
    with tvm.target.Target("ascend"):
        inferred = transform.AscendLayoutInference()(mod)
    layouts = inferred["main"].body.block.annotations["layout_map"]
    assert layouts[view].is_equal(layout.reshape(view.shape))


def test_unconstrained_access_cannot_choose_between_distinct_layouts():
    owner, view, first, second = _views()
    unknown = tirx.decl_buffer(owner.shape, owner.dtype, name="unknown", data=owner.data)
    body = tirx.SeqStmt([tirx.Evaluate(buf[0, 0]) for buf in (owner, view, unknown)])
    mod = kernel(body, buffers=[owner], annotations={"layout_map": {owner: first, view: second}})
    with tvm.target.Target("ascend"), pytest.raises(tvm.error.InternalError, match="Ambiguous layout for buffer view unknown"):
        transform.AscendLayoutInference()(mod)


def test_unused_storage_owner_does_not_need_a_view_layout():
    owner, first_view, _, _ = _views()
    second_view = tirx.decl_buffer(owner.shape, owner.dtype, name="second", data=owner.data)
    first = Layout(first_view.shape, lambda i, j: (i, j))
    second = Layout(second_view.shape, lambda i, j: (j, i))
    body = tirx.SeqStmt([tirx.Evaluate(buf[0, 0]) for buf in (first_view, second_view)])
    mod = kernel(body, buffers=[owner], annotations={"layout_map": {first_view: first, second_view: second}})
    with tvm.target.Target("ascend"):
        inferred = transform.AscendLayoutInference()(mod)
    layouts = inferred["main"].body.block.annotations["layout_map"]
    assert layouts[first_view].is_equal(first)
    assert layouts[second_view].is_equal(second)


def test_fragment_aliases_reject_conflicting_thread_ownership():
    owner = tirx.decl_buffer((256,), "float32", name="owner", scope="local.fragment")
    view = tirx.decl_buffer(owner.shape, owner.dtype, name="view", data=owner.data)
    first = Fragment((256,), forward_thread_fn=lambda i: i % 32, forward_index_fn=lambda i: i // 32)
    second = Fragment((256,), forward_thread_fn=lambda i: (i + 1) % 32, forward_index_fn=lambda i: i // 32)
    mod = kernel(tirx.Evaluate(view[0]), buffers=[owner], annotations={"layout_map": {owner: first, view: second}})
    with tvm.target.Target("ascend"), pytest.raises(tvm.error.InternalError, match="Conflicting fragment layouts for aliases"):
        transform.AscendLayoutInference()(mod)


def test_view_layouts_must_preserve_the_storage_footprint():
    owner, view, first, _ = _views()
    second = Layout(view.shape, lambda i, j: (i, j * 2))
    body = tirx.SeqStmt([tirx.Evaluate(buf[0, 0]) for buf in (owner, view)])
    mod = kernel(body, buffers=[owner], annotations={"layout_map": {owner: first, view: second}})
    with tvm.target.Target("ascend"), pytest.raises(tvm.error.InternalError, match="equal physical sizes"):
        transform.NormalizeAscendFractalStorage()(transform.AscendLayoutInference()(mod))


def test_typed_pointers_use_each_view_layout_and_one_storage():
    owner, view, first, second = _views()
    body = tirx.SeqStmt([tirx.Evaluate(T.access_ptr(buf[3, 5], "r", extent=1)) for buf in (owner, view)])
    mod = kernel(body, buffers=[owner], annotations={"layout_map": {owner: first, view: second}})
    with tvm.target.Target("ascend"):
        mod = transform.AscendLowerTileOp()(mod)
        mod = tilelang.transform.LowerAccessPtr()(mod)
    pointers = calls(mod, "tirx.tvm_access_ptr")
    assert len(pointers) == 2
    assert pointers[0].args[1].same_as(pointers[1].args[1])
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(pointers[0].args[2], 3 * 16 + 5)
    assert analyzer.can_prove_equal(pointers[1].args[2], 5 * 8 + 3)


@tilelang.testing.requires_ascend
def test_ub_view_reads_the_same_bytes_through_its_own_layout():
    import torch

    @T.prim_func
    def transpose(A: T.Tensor((8, 16), "float32"), B: T.Tensor((16, 8), "float32")):
        with T.Kernel(1):
            storage = T.alloc_shared((8, 16), "float32")
            view = T.view(storage, (16, 8))
            T.annotate_layout({storage: Layout((8, 16), lambda i, j: (i, j)), view: Layout((16, 8), lambda i, j: (j, i))})
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(8, 16):
                    storage[i, j] = A[i, j]
                for i, j in T.Parallel(16, 8):
                    B[i, j] = view[i, j]

    compiled = tilelang.compile(transpose, target="ascend", out_idx=-1)
    source = torch.arange(128, dtype=torch.float32, device="npu").reshape(8, 16)
    torch.testing.assert_close(compiled(source), source.T, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("annotate_owner", [True, False], ids=["owner", "view"])
def test_fragment_reshape_preserves_annotated_thread_ownership(annotate_owner):
    import torch

    layout = Fragment(
        (16, 16),
        forward_thread_fn=lambda i, j: (j * 16 + i) % 32,
        forward_index_fn=lambda i, j: (j * 16 + i) // 32,
    )

    @T.prim_func
    def program(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((256,), "float32")
            dst = T.alloc_shared((256,), "float32")
            T.copy(A, src)
            with T.SimtVF(threads=32):
                owner = T.alloc_fragment((16, 16), "float32")
                view = T.reshape(owner, (256,))
                if annotate_owner:
                    T.annotate_layout({owner: layout})
                else:
                    T.annotate_layout({view: layout.reshape((256,))})
                for i in T.Parallel(256):
                    view[i] = src[i] + T.cast(T.get_thread_binding(), "float32")
                for i in T.Parallel(256):
                    dst[i] = view[i]
            T.copy(dst, B)

    # Observe scalar lane ownership without relying on vectorized casts.
    compiled = tilelang.compile(program, target="ascend", out_idx=-1, pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True})
    source = torch.arange(256, dtype=torch.float32, device="npu")
    logical = torch.arange(256, dtype=torch.int64, device="npu")
    lanes = ((logical % 16) * 16 + logical // 16) % 32
    torch.testing.assert_close(compiled(source), source + lanes.float(), rtol=0, atol=0)
