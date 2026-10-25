"""SimtVF thread queries refer to the current frame's actual IR bindings."""

import pytest
import tilelang.ascend.language as T
from tilelang.ascend.language import kernel as K
from tilelang import tvm
from tvm_tirx import ir
from testing.ascend._ir import nodes


@pytest.mark.parametrize("threads, expected", [(128, [128, 1, 1]), ([64, 2, 1], [64, 2, 1])], ids=["one-dimensional", "two-dimensional"])
def test_thread_queries_match_frame_bindings(threads, expected):
    queries = {}
    with tvm.target.Target("ascend"):

        @T.prim_func
        def program(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
            with T.Kernel(1), T.SimtVF(threads=threads):
                queries["bindings"] = K.get_thread_bindings()
                queries["binding_x"] = K.get_thread_binding(0)
                queries["extents"] = K.get_thread_extents()
                queries["extent_x"] = K.get_thread_extent(0)
                for i in T.Parallel(128):
                    B[i] = A[i] + 1.0

    bindings = queries["bindings"]
    assert len(bindings) == 3
    assert queries["binding_x"].same_as(bindings[0])
    assert queries["extents"] == expected
    assert queries["extent_x"] == expected[0]
    (vf,) = [region for region in nodes(program, ir.RegionStmt) if region.attrs.get("name_hint") == "SIMT_VF"]
    attributes = [node for node in nodes(vf, ir.RegionStmt) if node.op.name == "tirx.launch_thread"]
    by_tag = {str(node.args[0].value): node for node in attributes}
    assert "threadIdx.x" in by_tag
    for axis, tag in enumerate(("threadIdx.x", "threadIdx.y", "threadIdx.z")):
        if tag in by_tag:
            assert by_tag[tag].body_params[0].same_as(bindings[axis])
            assert int(by_tag[tag].args[1]) == expected[axis]


def test_sequential_vf_regions_have_independent_bindings():
    queries = {}
    with tvm.target.Target("ascend"):

        @T.prim_func
        def program(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32"), C: T.Tensor((256,), "float32")):
            with T.Kernel(1):
                with T.SimtVF(threads=128):
                    queries["first"] = K.get_thread_binding(0)
                    queries["first_extent"] = K.get_thread_extent(0)
                    for i in T.Parallel(256):
                        B[i] = A[i] + 1.0
                with T.SimtVF(threads=256):
                    queries["second"] = K.get_thread_binding(0)
                    queries["second_extent"] = K.get_thread_extent(0)
                    for i in T.Parallel(256):
                        C[i] = A[i] + 2.0

    assert queries["first_extent"] == 128 and queries["second_extent"] == 256
    assert not queries["first"].same_as(queries["second"])
