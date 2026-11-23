"""Tests for Layout, Fragment, and PartialFragment FFI serialization / pickle."""

import multiprocessing
import pickle

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import _ffi_api, tvm
from tilelang.layout import (
    Fragment,
    Layout,
    PartialFragment,
    make_ascend_nz_layout,
    make_linear_layout,
    make_swizzled_layout,
    try_extract_fractal_layout,
)
from tvm import tirx
from tvm.target import Target


def _roundtrip_pickle_twice(obj):
    return pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(obj))))


def _lower_in_worker(func):
    target = Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        artifact = tilelang.lower(func, target=target)
    return artifact.kernel_source


def test_layout_pickle_and_json_roundtrip():
    layout_1d = Layout([32], lambda i: i)
    for restored in (
        _roundtrip_pickle_twice(layout_1d),
        tvm.ir.load_json(tvm.ir.save_json(layout_1d)),
    ):
        assert list(restored.map_forward_index([7])) == [7]
        assert list(restored.get_output_shape()) == [32]
        assert list(restored.inverse().map_forward_index([7])) == [7]
        assert restored.is_equal(layout_1d)
        assert tvm.ir.structural_equal(restored, layout_1d)

    layout_2d = Layout([16, 8], lambda i, j: [j, i])
    restored_2d = _roundtrip_pickle_twice(layout_2d)
    assert list(restored_2d.map_forward_index([5, 3])) == [3, 5]
    assert list(restored_2d.get_output_shape()) == [8, 16]
    assert restored_2d.is_equal(layout_2d)


def test_ascend_fractal_layout_pickle_roundtrip():
    buf = tirx.decl_buffer((32, 64), "float16", name="A_l1")
    nz = make_ascend_nz_layout(buf)
    restored = _roundtrip_pickle_twice(nz)
    assert restored.is_equal(nz)
    assert list(restored.map_forward_index([19, 35])) == list(nz.map_forward_index([19, 35]))
    expected_info = try_extract_fractal_layout(nz, buf)
    restored_info = try_extract_fractal_layout(restored, buf)
    assert restored_info is not None and expected_info is not None
    assert (restored_info.kind, restored_info.c0, restored_info.row_frac, restored_info.c0_axis, restored_info.row16_axis) == (
        expected_info.kind,
        expected_info.c0,
        expected_info.row_frac,
        expected_info.c0_axis,
        expected_info.row16_axis,
    )
    assert tvm.ir.structural_equal(restored_info.outer0, expected_info.outer0)
    assert tvm.ir.structural_equal(restored_info.outer1, expected_info.outer1)


def test_fragment_and_partial_fragment_pickle_roundtrip():
    frag = Fragment(
        [16, 4],
        forward_thread_fn=lambda i, j, rep: (i % 8) * 4 + j + rep * 32,
        forward_index_fn=lambda i, j: i // 8,
        replicate=2,
    )
    restored_frag = _roundtrip_pickle_twice(frag)
    assert restored_frag.is_equal(frag)
    assert tvm.ir.structural_equal(restored_frag, frag)
    assert tvm.ir.structural_hash(restored_frag) == tvm.ir.structural_hash(frag)
    assert int(restored_frag.get_thread_size()) == int(frag.get_thread_size())
    assert list(restored_frag.map_forward_thread([1, 11, 3])) == [47]
    assert list(restored_frag.map_forward_index([1, 11, 3])) == [1]
    assert restored_frag.condense_rep_var().is_equal(frag.condense_rep_var())

    partial = PartialFragment([1], forward_thread_fn=lambda i, rep: rep, replicate=128, combine=8)
    restored_partial = _roundtrip_pickle_twice(partial)
    assert restored_partial.is_equal(partial)
    assert tvm.ir.structural_equal(restored_partial, partial)
    expected_steps = [tuple(map(int, s)) for s in _ffi_api.PartialFragment_combine_steps(partial)]
    actual_steps = [tuple(map(int, s)) for s in _ffi_api.PartialFragment_combine_steps(restored_partial)]
    assert actual_steps == expected_steps


def test_layout_pickle_preserves_outer_symbolic_vars_with_placeholder_names():
    # Outer TIR variables intentionally share names with Layout's internal
    # `_i` and `_rep` placeholders; serialization must rebind only the layout's
    # own placeholders while preserving pointer identity for the outer Vars.
    outer_i = tirx.Var("_i", "int32")
    outer_rep = tirx.Var("_rep", "int32")
    layout = Layout([outer_i], lambda x: x + outer_i)
    frag = Fragment(
        [outer_i],
        forward_thread_fn=lambda x, r: r + outer_rep,
        forward_index_fn=lambda x: x,
        replicate=4,
    )

    # Pack into one TVM container (matching PrimFunc serialization) so all nodes
    # share a single ToJSONGraph / FromJSONGraph object graph.
    u_i, u_rep, u_layout, u_frag = pickle.loads(pickle.dumps(tvm.runtime.convert([outer_i, outer_rep, layout, frag])))
    assert u_layout.get_input_shape()[0].same_as(u_i)
    mapped_idx = u_layout.map_forward_index([7])[0]
    assert tvm.ir.structural_equal(mapped_idx, u_i + tirx.IntImm("int32", 7))
    assert mapped_idx.a.same_as(u_i)

    # For replicate > 1, get_forward_vars() orders [_rep, _i].
    mapped_thread = u_frag.map_forward_thread([3, 5])[0]
    assert tvm.ir.structural_equal(mapped_thread, u_rep + tirx.IntImm("int32", 3))
    assert mapped_thread.a.same_as(u_rep)


def test_layout_owned_vars_no_cross_node_aliasing_and_high_rank():
    # Two distinct layouts/fragments in the same container must own distinct
    # bound Vars so container-level StructuralEqual succeeds under alpha-renaming.
    frag_a = Fragment([8, 4], forward_thread_fn=lambda x, y, r: x * 4 + y + r * 32, replicate=2)
    frag_b = frag_a.replicate(2)
    clone_a = Fragment([8, 4], forward_thread_fn=lambda m, n, k: m * 4 + n + k * 32, replicate=2)
    clone_b = Fragment([8, 4], forward_thread_fn=lambda m, n, k: m * 4 + n + k * 32, replicate=2).replicate(2)
    assert not frag_a.forward_vars[0].same_as(frag_b.forward_vars[0])
    assert not frag_a.replicate_var.same_as(frag_b.replicate_var)
    assert tvm.ir.structural_equal(tvm.runtime.convert([frag_a, frag_b]), tvm.runtime.convert([clone_a, clone_b]))

    # Layouts with > 16 dimensions are no longer capped by static placeholders.
    ndim = 20
    shape = [2] * ndim
    high_rank = Layout(shape, lambda *xs: list(reversed(xs)))
    restored_high_rank = _roundtrip_pickle_twice(high_rank)
    coords = [i % 2 for i in range(ndim)]
    assert list(restored_high_rank.map_forward_index(coords)) == list(reversed(coords))
    assert list(restored_high_rank.inverse().map_forward_index(list(reversed(coords)))) == coords
    assert tvm.ir.structural_equal(restored_high_rank, high_rank)

    high_rank_linear = make_linear_layout(tirx.decl_buffer(shape, "float16"))
    expected_linear = sum(c * (1 << (ndim - 1 - idx)) for idx, c in enumerate(coords))
    assert [int(x) for x in high_rank_linear.map_forward_index(coords)] == [expected_linear]

    # Fragment forward_index must depend only on logical input coordinates, not replicate_var.
    with pytest.raises(ValueError, match="forward_index must not depend on replication variable"):
        Fragment([8], forward_fn=lambda i, rep: (i * 2 + rep, rep), replicate=2)


@tilelang.testing.requires_cuda
def test_annotated_layouts_lower_across_spawn_process():
    @T.prim_func
    def identity_shared(A: T.Tensor((32,), T.float16), B: T.Tensor((32,), T.float16)):
        with T.Kernel(1, threads=32) as _:
            s = T.alloc_shared((32,), T.float16)
            T.annotate_layout({s: T.Layout((32,), lambda i: i)})
            for i in T.Parallel(32):
                s[i] = A[i]
            for i in T.Parallel(32):
                B[i] = s[i]

    @T.prim_func
    def swizzled_shared(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            s = T.alloc_shared((64, 64), T.float16)
            T.annotate_layout({s: make_swizzled_layout(s)})
            T.copy(A, s)
            T.copy(s, B)

    @T.prim_func
    def explicit_fragment(A: T.Tensor((32,), T.float32), B: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32) as _:
            frag = T.alloc_fragment((32,), T.float32)
            T.annotate_layout(
                {
                    frag: T.Fragment(
                        (32,),
                        forward_thread_fn=lambda i: i,
                        forward_index_fn=lambda i: 0,
                    )
                }
            )
            for i in T.Parallel(32):
                frag[i] = A[i] + 1.0
            for i in T.Parallel(32):
                B[i] = frag[i]

    funcs = [identity_shared, swizzled_shared, explicit_fragment]
    expected_sources = [_lower_in_worker(f) for f in funcs]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        worker_sources = pool.map(_lower_in_worker, funcs)

    assert worker_sources == expected_sources


if __name__ == "__main__":
    tilelang.testing.main()
