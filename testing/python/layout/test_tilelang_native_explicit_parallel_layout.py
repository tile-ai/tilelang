"""Explicit native layouts retain their logical axes through loop fusion."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.analysis._visitor import tilelang_metadata


def _layout(rank):
    if rank == 1:
        return T.Fragment((128,), forward_fn=lambda k: (k % 32, k // 32))
    if rank == 2:
        return T.Fragment((8, 16), forward_fn=lambda j, k: ((j * 16 + k) % 32, (j * 16 + k) // 32))
    return T.Fragment(
        (2, 4, 16),
        forward_fn=lambda i, j, k: (((i * 4 + j) * 16 + k) % 32, ((i * 4 + j) * 16 + k) // 32),
    )


def _copy(layout=None, annotations=None):
    @T.prim_func
    def main(A: T.Tensor((2, 4, 16), "float32"), B: T.Tensor((2, 4, 16), "float32")):
        with T.Kernel(1, threads=32):
            for i, j, k in T.Parallel(2, 4, 16, loop_layout=layout, annotations=annotations):
                B[i, j, k] = A[i, j, k]

    return main


def _infer(function):
    before = ir.save_json(function)
    target = Target({"kind": "cuda", "arch": "sm_80"})
    module = ir.IRModule({"main": function.with_attr("target", target)})
    module = tilelang.transform.MaterializeKernelLaunch()(module)
    try:
        with target:
            result = tilelang.transform.LayoutInference()(module)
    finally:
        assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result["main"])
    return result


def _loops_and_layouts(module):
    loops, layouts = [], []
    tvm_ffi.structural_walk(module["main"].body, (ir.For, loops.append), order="pre")

    def collect(node):
        metadata = tilelang_metadata(node)
        if metadata is not None and metadata.get("key") == "tl.loop.annotations":
            metadata = metadata["value"]
        if metadata is not None and "parallel_loop_layout" in metadata:
            layouts.append(metadata["parallel_loop_layout"])

    tvm_ffi.structural_walk(module["main"].body, ((ir.Call, ir.RegionStmt), collect), order="pre")
    return loops, layouts


def test_explicit_three_axis_layout_keeps_the_original_parallel_domain():
    loops, layouts = _loops_and_layouts(_infer(_copy(_layout(3))))
    assert [int(loop.end) for loop in loops] == [2, 4, 16]
    assert all(loop.kind == ir.ForKind.PARALLEL for loop in loops)
    assert layouts
    assert all([int(dim) for dim in layout.get_input_shape()] == [2, 4, 16] for layout in layouts)


@pytest.mark.parametrize("hints", [None, {"coalesced_width": 2}, {"coalesced_width": ir.IntImm("int32", 2)}])
def test_unannotated_layout_still_fuses_with_static_and_operand_backed_hints(hints):
    loops, layouts = _loops_and_layouts(_infer(_copy(annotations=hints)))
    assert len(loops) == 1
    assert int(loops[0].end) == 128
    assert layouts
    assert all([int(dim) for dim in layout.get_input_shape()] == [128] for layout in layouts)


@pytest.mark.parametrize("rank", [1, 2])
def test_explicit_layout_rank_must_match_the_original_nest(rank):
    with pytest.raises(ValueError, match="InputDim must match the complete loop nest"):
        _infer(_copy(_layout(rank)))


def test_explicit_layout_does_not_disable_fusion_for_a_sibling_nest():
    layout = _layout(3)

    @T.prim_func
    def main(A: T.Tensor((2, 4, 16), "float32"), B: T.Tensor((2, 4, 16), "float32"), C: T.Tensor((2, 4, 16), "float32")):
        with T.Kernel(1, threads=32):
            for i, j, k in T.Parallel(2, 4, 16, loop_layout=layout):
                B[i, j, k] = A[i, j, k]
            for i, j, k in T.Parallel(2, 4, 16):
                C[i, j, k] = B[i, j, k]

    loops, layouts = _loops_and_layouts(_infer(main))
    assert [int(loop.end) for loop in loops] == [2, 4, 16, 128]
    assert sorted({len(layout.get_input_shape()) for layout in layouts}) == [1, 3]


def test_inner_explicit_layout_cannot_redefine_the_outer_layout_domain():
    outer = _layout(3)
    inner = T.Fragment((4, 16), forward_fn=lambda j, k: ((j * 16 + k) % 32, (j * 16 + k) // 32))

    @T.prim_func
    def main(A: T.Tensor((2, 4, 16), "float32"), B: T.Tensor((2, 4, 16), "float32")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(2, loop_layout=outer):
                for j, k in T.Parallel(4, 16, loop_layout=inner):
                    B[i, j, k] = A[i, j, k]

    with pytest.raises(ValueError, match="InputDim must match the complete loop nest"):
        _infer(main)
