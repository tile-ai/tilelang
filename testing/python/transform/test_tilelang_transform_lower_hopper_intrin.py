import tvm_tirx
import tilelang as tl
from tilelang.cuda import transform as cuda_transform
from tilelang.backend.target import determine_target
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls, collect_extern_calls, collect_nodes
from tvm_tirx import ir, tirx
import tvm_ffi

# Direct pass tests must register the backend that engine.lower normally owns.
tl.cuda.register_backend()
auto_target = tvm_tirx.target.Target(determine_target("auto"))


def _allocations(body):
    return [
        binding
        for binding in collect_nodes(body, ir.Bind)
        if isinstance(binding.var.ty, tirx.TensorType)
        and isinstance(binding.value, ir.Call)
        and binding.value.op.name == "tirx.alloc_tensor"
    ]


def _assert_shared_barrier_init(func):
    body = func.body
    assert not tirx.analysis.undefined_vars(body, func.params)
    assert_call_count(body, op="tl.ptx_init_barrier_thread_count", count=4)
    assert_call_count(body, op="tl.ptx_fence_barrier_init", count=1)
    assert_call_count(body, op="tirx.gpu_storage_sync", count=1)
    assert_call_count(body, op="tl.tl_shuffle_elect", count=1)

    (alloc,) = _allocations(body)
    barrier = alloc.var
    assert barrier.ty.scope.name == "shared.barrier"
    assert str(barrier.ty.dtype) == "uint64"
    tvm_tirx.ir.assert_structural_equal(barrier.ty.shape, tvm_ffi.convert([ir.IntImm("int32", 4)]))
    assert alloc.value.attrs["tl.barrier_initialized"]

    (guard,) = collect_nodes(body, ir.If)
    # Native alloc_tensor binds the TensorVar directly; its loads must refer to
    # that exact definition. There is no separate Buffer/data-pointer decl.
    # Direct siblings establish execution order; traversal order does not.
    (seq,) = [seq for seq in collect_nodes(body, ir.SeqStmt) if any(stmt.same_as(alloc) for stmt in seq.seq)]
    assert seq.seq[0].same_as(alloc)
    assert seq.seq[1].same_as(guard)
    for call in collect_calls(guard, op="tl.ptx_init_barrier_thread_count"):
        assert isinstance(call.args[0], ir.TensorLoad)
        assert call.args[0].source.same_as(barrier)
    expected = ir.SeqStmt(
        [
            ir.If(
                ir.Call("tl.tl_shuffle_elect", [0], ty="bool"),
                ir.SeqStmt(
                    [
                        ir.Evaluate(ir.Call("tl.ptx_init_barrier_thread_count", [ir.TensorLoad(barrier, [i]), 128], ty="void"))
                        for i in range(4)
                    ]
                ),
                None,
            ),
            ir.Evaluate(ir.Call("tl.ptx_fence_barrier_init", [], ty="void")),
            ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")], ty="void")),
        ]
    )
    tvm_tirx.ir.assert_structural_equal(ir.SeqStmt(seq.seq[1:4]), expected)


def _assert_tma_descriptor_init_after_alloc(func, original_call):
    body = func.body
    assert not tirx.analysis.undefined_vars(body, func.params)
    assert_call_count(body, op="tl.create_tma_descriptor", count=0)
    assert_call_count(body, op="tl.prefetch_tma_descriptor", count=1)
    assert_call_count(body, op="tirx.call_packed", count=1)
    assert not collect_extern_calls(body, symbol="tl::prefetch_tma_descriptor")

    (alloc,) = _allocations(body)
    assert alloc.var.ty.scope.name == "global" and str(alloc.var.ty.dtype) == "float16"
    tvm_tirx.ir.assert_structural_equal(alloc.var.ty.shape, tvm_ffi.convert([ir.IntImm("int32", 32)]))
    original_pointer = original_call.args[2]
    assert isinstance(original_pointer, ir.Call) and original_pointer.op.name == "tirx.tensor_data_ptr"
    assert original_pointer.args[0].same_as(alloc.var)
    (bind,) = [
        binding
        for binding in collect_nodes(body, ir.Bind)
        if isinstance(binding.var.ty, ir.PtrType) and isinstance(binding.var.ty.element_type, tirx.TensorMapType)
    ]
    descriptor = bind.var
    # TensorMapType now carries the descriptor-storage contract used for
    # by-value grid-constant kernel arguments; it replaces the old scope tag.
    init_args = tvm_ffi.convert([ir.StringImm("__tvm_tensormap_create_tiled"), descriptor, *original_call.args])
    (init_call,) = collect_calls(body, op="tirx.call_packed")
    tvm_tirx.ir.assert_structural_equal(init_call.args, init_args)
    assert init_call.args[1].same_as(descriptor)
    assert init_call.args[4].same_as(original_pointer)
    assert_call_count(body, op="tirx.stack_alloca", count=1)

    (seq,) = [seq for seq in collect_nodes(body, ir.SeqStmt) if any(stmt.same_as(alloc) for stmt in seq.seq)]
    assert seq.seq[0].same_as(alloc)
    expected_init = ir.SeqStmt(
        [
            ir.Bind(descriptor, ir.Call("tirx.stack_alloca", [ir.StringImm("tensormap"), 1], ty=descriptor.ty)),
            ir.Evaluate(ir.Call("tirx.call_packed", init_args, ty="int32")),
        ]
    )
    tvm_tirx.ir.assert_structural_equal(ir.SeqStmt(seq.seq[1:3]), expected_init)
    # The launch follows initialization in the same scope; the prefetch is
    # guarded inside that launch and uses the very same descriptor Var.
    launch = seq.seq[3]
    assert isinstance(launch, ir.RegionStmt) and launch.op.name == "tirx.launch_thread"
    assert launch.args[0].value == "threadIdx.x"
    assert len(launch.body_params) == 1 and len(launch.result_vars) == 0
    expected_body = ir.SeqStmt(
        [
            ir.If(
                ir.Call("tl.tl_shuffle_elect", [0], ty="bool"),
                ir.Evaluate(ir.Call("tl.prefetch_tma_descriptor", [descriptor], ty="void")),
                None,
            ),
            ir.Evaluate(descriptor),
        ]
    )
    tvm_tirx.ir.assert_structural_equal(launch.body, expected_body)
    (metadata_descriptor,) = list(func.attrs["tma_descriptor_args"])
    assert metadata_descriptor.same_as(descriptor)
    tvm_tirx.ir.assert_structural_equal(func.attrs["tma_descriptor_args"][descriptor], init_args)


@tilelang.testing.requires_cuda
def test_lower_shared_barrier():
    """Test that LowerSharedBarrier converts shared.barrier buffers + barrier_init
    annotations into ptx_init_barrier_thread_count calls.

    This replaces the old test_lower_hopper_intrin_barrier which tested the
    removed tl.create_list_of_mbarrier intrinsic.
    """

    @T.prim_func
    def before():
        with T.Kernel(8):
            _ = T.launch_thread("threadIdx.x", 128)
            mbarrier = T.alloc_barrier([128, 128, 128, 128])  # noqa: F841

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm_tirx.tirx.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    mod = cuda_transform.LowerSharedBarrier()(mod)
    mod = tl.transform.LowerSourceAnnotations()(mod)

    _assert_shared_barrier_init(mod["main"])


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_descriptor_init_after_alloc_global():
    @T.prim_func
    def before():
        T.func_attr({"tirx.is_entry_func": True, "tl.has_tma": T.bool(True)})
        Output_partial = T.alloc_buffer((32,), "float16")
        with T.launch_thread("threadIdx.x", 1):
            T.evaluate(
                T.create_tma_descriptor(
                    6,
                    4,
                    Output_partial.data,
                    8,
                    2,
                    2,
                    1,
                    2,
                    16,
                    32,
                    64,
                    8,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    2,
                    0,
                )
            )

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm_tirx.tirx.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    (original_call,) = collect_calls(mod["main"].body, op="tl.create_tma_descriptor")
    mod = cuda_transform.LowerHopperIntrin()(mod)
    _assert_tma_descriptor_init_after_alloc(mod["main"], original_call)


if __name__ == "__main__":
    tilelang.testing.main()
