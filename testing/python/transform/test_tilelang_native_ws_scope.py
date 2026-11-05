"""Native WS slices keep scalar definitions and evaluation guards together."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.testing.ir import collect_calls, collect_nodes


def _materialize(trips, shift):
    extent = T.dynamic("extent") if trips is None else trips

    @T.prim_func
    def kernel(A: T.Tensor((extent, 8), T.float32)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((8,), T.float32)
            c = T.alloc_shared((8,), T.float32)
            fragment = T.alloc_fragment((8,), T.float32)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[T.WSRole("P", warps_lo=0, warps_hi=1), T.WSRole("C", warps_lo=4, warps_hi=8)],
                    pipelines=[T.WSPipeline("a", [a], depth=2), T.WSPipeline("c", [c], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop",
                            {
                                "P": [
                                    "bind",
                                    T.WSSync.producer_acquire("a", stage=0),
                                    "a_write",
                                    T.WSSync.producer_commit("a", stage=0),
                                    T.WSSync.producer_acquire("c", stage=shift),
                                    "c_write",
                                    T.WSSync.producer_commit("c", stage=shift),
                                ],
                                "C": [
                                    T.WSSync.consumer_wait("a"),
                                    "a_read",
                                    T.WSSync.consumer_release("a"),
                                    T.WSSync.consumer_wait("c"),
                                    "c_read",
                                    T.WSSync.consumer_release("c"),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"P": ["loop"], "C": ["loop"]}),
                    ],
                )
            )
            for i in T.Pipelined(extent, num_stages=2, annotations={T.WSID: "loop"}):
                with T.ws_op("bind"):
                    value = A[i, 0]
                if value > 0.0:
                    T.copy(A[i, 0], a, annotations={T.WSID: "a_write"})
                T.fill(c, 0.0, annotations={T.WSID: "c_write"})
                T.copy(a, fragment, annotations={T.WSID: "a_read"})
                T.copy(c, fragment, annotations={T.WSID: "c_read"})

    module = ir.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
    module = tirx.transform.BindTarget(Target({"kind": "cuda", "arch": "sm_100a"}))(module)
    module = tilelang.transform.MaterializeKernelLaunch()(module)
    return tilelang.cuda.transform.MaterializeWSSchedule()(module)["main"]


@pytest.mark.parametrize("trips", [None, 0, 1, 4])
@pytest.mark.parametrize("shift", [1, 2])
def test_static_and_dynamic_slices_have_no_escaping_definitions(trips, shift):
    function = _materialize(trips, shift)
    assert tirx.analysis.check_well_formed(function)
    renewed = tirx.renew_def(function)
    assert tirx.analysis.check_well_formed(renewed)
    assert tvm_ffi.structural_equal(function, renewed)
    definitions = [bind for bind in collect_nodes(function, ir.Bind) if isinstance(bind.value, ir.TensorLoad)]
    assert len({bind.var for bind in definitions}) == len(definitions)
    if trips == 0:
        assert not definitions
        simplified = tilelang.transform.Simplify()(ir.IRModule.from_expr(function))["main"]
        assert not collect_nodes(simplified, ir.For)
        assert not collect_calls(simplified, op="tl.tileop.copy")
        assert not collect_calls(simplified, op="tl.tileop.fill")
    else:
        assert definitions
        if trips is None:
            # Every prologue scalar load is dominated by a trip-count guard.
            # The same branch also contains its source guard and use, which
            # check_well_formed verifies against the actual native definitions.
            guarded = [
                branch
                for branch in collect_nodes(function, ir.If)
                if any(var.name == "extent" for var in collect_nodes(branch.condition, ir.Var))
            ]
            assert len(guarded) >= shift
            assert any(collect_nodes(branch.then_case, ir.Bind) for branch in guarded)
