from tilelang import tvm as tvm
import tilelang as tl
from tilelang.cuda import transform as cuda_transform
from tilelang.backend.target import determine_target
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls, collect_extern_calls, collect_nodes
from tvm_tirx import ir, tirx
from tvm_ffi import convert

auto_target = tvm.target.Target(determine_target("auto"))


def _allocations(body):
    return [
        binding
        for binding in collect_nodes(body, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor"
    ]


def _assert_shared_barrier_init(func):
    body = func.body
    assert_call_count(body, op="tl.mbarrier_init", count=4)
    assert_call_count(body, op="tl.ptx_fence_barrier_init", count=1)
    assert_call_count(body, op="tirx.gpu_storage_sync", count=1)
    assert_call_count(body, op="tl.tl_shuffle_elect", count=1)
    (allocation,) = _allocations(body)
    assert allocation.var.ty.scope.name == "shared"
    assert allocation.var.ty.dtype == "uint64"
    assert [int(extent) for extent in allocation.var.ty.shape] == [4]
    assert "barrier_init" not in allocation.value.attrs
    (guard,) = collect_nodes(body, ir.If)
    assert guard.condition.op.name == "tl.tl_shuffle_elect"
    initializers = collect_calls(guard.then_case, op="tl.mbarrier_init")
    assert len(initializers) == 4
    for index, call in enumerate(initializers):
        assert call.args[0].source.same_as(allocation.var)
        assert [int(coordinate) for coordinate in call.args[0].indices] == [index]
        assert int(call.args[1]) == 128
    (sequence,) = [
        sequence for sequence in collect_nodes(body, ir.SeqStmt) if any(statement.same_as(allocation) for statement in sequence.seq)
    ]
    assert sequence.seq[0].same_as(allocation)
    assert sequence.seq[1].same_as(guard)
    assert sequence.seq[2].value.op.name == "tl.ptx_fence_barrier_init"
    assert sequence.seq[3].value.op.name == "tirx.gpu_storage_sync"
    assert sequence.seq[3].value.args[0].value == "shared"
    assert tirx.analysis.check_well_formed(func)


def _assert_tma_descriptor_init_after_alloc(func, original_call):
    body = func.body
    assert not tirx.analysis.undefined_vars(body, func.params)
    assert_call_count(body, op="tl.create_tma_descriptor", count=0)
    assert_call_count(body, op="tl.prefetch_tma_descriptor", count=1)
    assert_call_count(body, op="tirx.cuda.tensormap_encode_tiled", count=1)
    assert not collect_extern_calls(body, symbol="tl::prefetch_tma_descriptor")
    (allocation,) = _allocations(body)
    assert allocation.var.ty.scope.name == "global"
    assert allocation.var.ty.dtype == "float16"
    assert [int(extent) for extent in allocation.var.ty.shape] == [32]
    assert original_call.args[2].args[0].same_as(allocation.var)
    (binding,) = [binding for binding in collect_nodes(body, ir.Bind) if isinstance(binding.var.ty, ir.PtrType)]
    descriptor = binding.var
    assert isinstance(descriptor.ty.element_type, tirx.TensorMapType)
    assert binding.value.op.name == "tirx.stack_alloca"
    assert binding.value.args[0].value == "tensormap"
    assert int(binding.value.args[1]) == 1
    (initializer,) = collect_calls(body, op="tirx.cuda.tensormap_encode_tiled")
    rank = int(original_call.args[1])
    expected_args = [
        descriptor,
        original_call.args[2],
        *original_call.args[3 : 3 + rank],
        *original_call.args[4 + rank : 3 + 2 * rank],
        *original_call.args[3 + 2 * rank : 3 + 4 * rank],
    ]
    tvm.ir.assert_structural_equal(initializer.args, convert(expected_args))
    assert initializer.attrs.rank == rank
    for offset, name in enumerate(("interleave", "swizzle", "l2_promotion", "oob_fill")):
        assert getattr(initializer.attrs, name) == int(original_call.args[3 + 4 * rank + offset])
    (sequence,) = [
        sequence for sequence in collect_nodes(body, ir.SeqStmt) if any(statement.same_as(allocation) for statement in sequence.seq)
    ]
    assert sequence.seq[0].same_as(allocation)
    assert sequence.seq[1].same_as(binding)
    assert sequence.seq[2].value.same_as(initializer)
    launch = sequence.seq[3]
    assert isinstance(launch, ir.RegionStmt) and launch.op.name == "tirx.launch_thread"
    assert launch.args[0].value == "threadIdx.x"
    guard = launch.body.seq[0]
    assert isinstance(guard, ir.If) and guard.condition.op.name == "tl.tl_shuffle_elect"
    (prefetch,) = collect_calls(guard.then_case, op="tl.prefetch_tma_descriptor")
    assert prefetch.args[0].same_as(descriptor)
    assert launch.body.seq[1].value.same_as(descriptor)
    assert tirx.analysis.check_well_formed(func)
    tvm.ir.assert_structural_equal(func, tvm.ir.load_json(tvm.ir.save_json(func)))


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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    mod = cuda_transform.LowerSharedBarrier()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)

    _assert_shared_barrier_init(mod["main"])


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_descriptor_init_after_alloc_global():
    @T.prim_func
    def before():
        T.func_attr({"tirx.is_entry_func": True, "tl.has_tma": T.bool(True)})
        Output_partial = T.alloc_buffer((32,), "float16", scope="global")
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    (original_call,) = collect_calls(mod["main"].body, op="tl.create_tma_descriptor")
    mod = cuda_transform.LowerHopperIntrin()(mod)
    _assert_tma_descriptor_init_after_alloc(mod["main"], original_call)


if __name__ == "__main__":
    tilelang.testing.main()
