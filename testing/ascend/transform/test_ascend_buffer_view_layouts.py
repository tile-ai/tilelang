"""Layouts belong to buffer views; pointers and allocations retain that split."""

import pytest
import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import Fragment, Layout, make_ascend_major_k_layout, make_ascend_major_mn_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel


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


@pytest.mark.parametrize("annotate_owner", [True, False], ids=["owner", "unused-view"])
def test_unconstrained_reshape_inherits_the_unique_layout(annotate_owner):
    owner, _, layout, _ = _views()
    view = tirx.decl_buffer((4, 32), owner.dtype, name="reshaped", data=owner.data)
    annotations = {owner: layout} if annotate_owner else {view: layout.reshape(view.shape)}
    accessed = view if annotate_owner else owner
    mod = kernel(tirx.Evaluate(accessed[1, 7]), buffers=[owner], annotations={"layout_map": annotations})
    with tvm.target.Target("ascend"):
        inferred = transform.AscendLayoutInference()(mod)
    layouts = inferred["main"].body.block.annotations["layout_map"]
    assert layouts[owner].is_equal(layout)
    assert layouts[view].is_equal(layout.reshape(view.shape))


@pytest.mark.parametrize("access_annotated_view", [True, False], ids=["accessed-view", "unused-view"])
def test_unconstrained_access_cannot_choose_between_distinct_layouts(access_annotated_view):
    owner, view, first, second = _views()
    unknown = tirx.decl_buffer(owner.shape, owner.dtype, name="unknown", data=owner.data)
    accessed = (owner, view, unknown) if access_annotated_view else (owner, unknown)
    body = tirx.SeqStmt([tirx.Evaluate(buf[0, 0]) for buf in accessed])
    mod = kernel(body, buffers=[owner], annotations={"layout_map": {owner: first, view: second}})
    with tvm.target.Target("ascend"), pytest.raises(tvm.error.InternalError, match="Ambiguous layout for buffer view unknown"):
        transform.AscendLayoutInference()(mod)


@pytest.mark.parametrize(
    "src_scope,dst_scope,alias_on_source",
    [
        ("global", "shared.l1", False),
        ("shared.l1", "shared.l0a", True),
        ("shared.l1", "shared.l0b", True),
        ("shared.l1", "shared.l0a", False),
        ("shared.l1", "shared.l0b", False),
    ],
)
def test_copy_defaults_do_not_override_the_unique_alias_layout(src_scope, dst_scope, alias_on_source):
    owner = tirx.decl_buffer((32, 32), "bfloat16", name="owner", scope=src_scope if alias_on_source else dst_scope)
    view = tirx.decl_buffer(owner.shape, owner.dtype, name="view", data=owner.data)
    other = tirx.decl_buffer(owner.shape, owner.dtype, name="other", scope=dst_scope if alias_on_source else src_scope)
    layout = make_ascend_major_mn_layout(owner)
    body = copy(view, other) if alias_on_source else copy(other, view)
    mod = kernel(
        body,
        buffers=[owner] if other.scope() == "global" else [owner, other],
        params=[other] if other.scope() == "global" else [],
        annotations={"layout_map": {owner: layout}},
    )
    with tvm.target.Target("ascend"):
        inferred = transform.AscendLayoutInference()(mod)
    layouts = inferred["main"].body.block.annotations["layout_map"]
    assert layouts[view].is_equal(layout)


def test_copy_default_cannot_hide_ambiguous_alias_layouts():
    owner = tirx.decl_buffer((32, 32), "bfloat16", name="owner", scope="shared.l1")
    second = tirx.decl_buffer(owner.shape, owner.dtype, name="second", data=owner.data)
    unknown = tirx.decl_buffer(owner.shape, owner.dtype, name="unknown", data=owner.data)
    dst = tirx.decl_buffer(owner.shape, owner.dtype, name="dst", scope="shared.l0a")
    body = tirx.SeqStmt([tirx.Evaluate(second[0, 0]), copy(unknown, dst)])
    mod = kernel(
        body,
        buffers=[owner, dst],
        annotations={"layout_map": {owner: make_ascend_major_mn_layout(owner), second: make_ascend_major_k_layout(second)}},
    )
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
@pytest.mark.parametrize("annotate_owner", [True, False], ids=["owner", "unused-view"])
def test_l1_reshape_inherits_major_mn_before_copy_defaults(annotate_owner):
    import torch

    @T.prim_func
    def program(A: T.Tensor((32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            owner = T.alloc_l1((32, 32), "bfloat16")
            view = T.reshape(owner, (32, 32))
            b = T.alloc_l1((32, 32), "bfloat16")
            a0 = T.alloc_l0a((32, 32), "bfloat16")
            b0 = T.alloc_l0b((32, 32), "bfloat16")
            accum = T.alloc_l0c((32, 32), "float32")
            if annotate_owner:
                T.annotate_layout({owner: make_ascend_major_mn_layout(owner)})
            else:
                # An annotation-only view still constrains its storage aliases.
                T.annotate_layout({view: make_ascend_major_mn_layout(view)})
            # Write the physical MajorMN representation before the L1->L0 read.
            T.copy(A, owner, transpose=True)
            T.copy(B, b)
            if annotate_owner:
                T.copy(view, a0)
            else:
                T.copy(owner, a0)
            T.copy(b, b0)
            T.gemm(a0, b0, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)

    compiled = tilelang.compile(program, target="ascend", out_idx=-1)
    a = ((torch.arange(1024).reshape(32, 32) * 7 + 3) % 17 - 8).to(device="npu", dtype=torch.bfloat16)
    b = ((torch.arange(1024).reshape(32, 32) * 5 + 1) % 13 - 6).to(device="npu", dtype=torch.bfloat16)
    torch.testing.assert_close(compiled(a, b).cpu(), a.cpu().float() @ b.cpu().float().T, rtol=0, atol=0)


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
