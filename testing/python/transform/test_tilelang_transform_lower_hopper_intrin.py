from tilelang import tvm as tvm
import tilelang as tl
from tilelang.cuda import transform as cuda_transform
from tilelang.backend.target import determine_target
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls, collect_extern_calls, collect_nodes
from tvm import tirx

auto_target = tvm.target.Target(determine_target("auto"))


def _assert_shared_barrier_init(func):
    body = func.body
    assert_call_count(body, op="tirx.ptx_init_barrier_thread_count", count=4)
    assert_call_count(body, op="tl.ptx_fence_barrier_init", count=1)
    assert_call_count(body, op="tirx.tvm_storage_sync", count=1)
    assert_call_count(body, op="tl.tl_shuffle_elect", count=1)

    (alloc,) = collect_nodes(body, tirx.AllocBuffer)
    (decl,) = collect_nodes(body, tirx.DeclBuffer)
    assert alloc.buffer.scope() == decl.buffer.scope() == "shared.barrier"
    assert alloc.buffer.dtype == decl.buffer.dtype == "uint64"
    tvm.ir.assert_structural_equal(alloc.buffer.shape, tvm.runtime.convert([tirx.IntImm("int32", 4)]))
    tvm.ir.assert_structural_equal(decl.buffer.shape, alloc.buffer.shape)
    assert decl.buffer.data.same_as(alloc.buffer.data)

    (guard,) = collect_nodes(body, tirx.IfThenElse)
    # Direct siblings establish execution order; traversal order does not.
    (seq,) = [seq for seq in collect_nodes(body, tirx.SeqStmt) if any(stmt.same_as(alloc) for stmt in seq.seq)]
    assert seq.seq[0].same_as(alloc)
    assert seq.seq[1].same_as(decl)
    assert seq.seq[2].same_as(guard)
    expected = tirx.SeqStmt(
        [
            tirx.IfThenElse(
                tirx.Call("bool", tvm.ir.Op.get("tl.tl_shuffle_elect"), [0]),
                tirx.SeqStmt(
                    [
                        tirx.Evaluate(tirx.Call("handle", "tirx.ptx_init_barrier_thread_count", [tirx.BufferLoad(decl.buffer, [i]), 128]))
                        for i in range(4)
                    ]
                ),
                None,
            ),
            tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.ptx_fence_barrier_init"), [])),
            tirx.Evaluate(tirx.Call("handle", "tirx.tvm_storage_sync", [tirx.StringImm("shared")])),
        ]
    )
    tvm.ir.assert_structural_equal(tirx.SeqStmt(seq.seq[2:5]), expected)


def _assert_tma_descriptor_init_after_alloc(func, original_call):
    body = func.body
    assert not tirx.analysis.undefined_vars(body, func.params)
    assert_call_count(body, op="tl.create_tma_descriptor", count=0)
    assert_call_count(body, op="tl.prefetch_tma_descriptor", count=1)
    assert_call_count(body, op="tirx.tvm_call_packed", count=1)
    assert not collect_extern_calls(body, symbol="tl::prefetch_tma_descriptor")

    (alloc,) = collect_nodes(body, tirx.AllocBuffer)
    assert alloc.buffer.scope() == "global" and alloc.buffer.dtype == "float16"
    tvm.ir.assert_structural_equal(alloc.buffer.shape, tvm.runtime.convert([tirx.IntImm("int32", 32)]))
    assert alloc.buffer.data.same_as(original_call.args[2])
    (bind,) = collect_nodes(body, tirx.Bind)
    descriptor = bind.var
    assert descriptor.type_annotation.storage_scope == "grid_constant"
    init_args = tvm.runtime.convert([tirx.StringImm("__tvm_tensormap_create_tiled"), descriptor, *original_call.args])
    (init_call,) = collect_calls(body, op="tirx.tvm_call_packed")
    tvm.ir.assert_structural_equal(init_call.args, init_args)
    assert init_call.args[4].same_as(alloc.buffer.data)
    assert_call_count(body, op="tirx.tvm_stack_alloca", count=1)

    (seq,) = [seq for seq in collect_nodes(body, tirx.SeqStmt) if any(stmt.same_as(alloc) for stmt in seq.seq)]
    assert seq.seq[0].same_as(alloc)
    expected_init = tirx.SeqStmt(
        [
            tirx.Bind(descriptor, tirx.Call("handle", "tirx.tvm_stack_alloca", [tirx.StringImm("tvm_ffi_any"), 16])),
            tirx.Evaluate(tirx.Call("handle", "tirx.tvm_call_packed", init_args)),
        ]
    )
    tvm.ir.assert_structural_equal(tirx.SeqStmt(seq.seq[1:3]), expected_init)
    # The launch follows initialization in the same scope; the prefetch is
    # guarded inside that launch and uses the very same descriptor Var.
    launch = seq.seq[3]
    assert isinstance(launch, tirx.AttrStmt) and launch.attr_key == "thread_extent"
    assert isinstance(launch.node, tirx.IterVar) and launch.node.thread_tag == "threadIdx.x"
    expected_body = tirx.SeqStmt(
        [
            tirx.IfThenElse(
                tirx.Call("bool", tvm.ir.Op.get("tl.tl_shuffle_elect"), [0]),
                tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.prefetch_tma_descriptor"), [descriptor])),
                None,
            ),
            tirx.Evaluate(descriptor),
        ]
    )
    tvm.ir.assert_structural_equal(launch.body, expected_body)
    (metadata_descriptor,) = list(func.attrs["tma_descriptor_args"])
    assert metadata_descriptor.same_as(descriptor)
    tvm.ir.assert_structural_equal(func.attrs["tma_descriptor_args"][descriptor], init_args)


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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    (original_call,) = collect_calls(mod["main"].body, op="tl.create_tma_descriptor")
    mod = cuda_transform.LowerHopperIntrin()(mod)
    _assert_tma_descriptor_init_after_alloc(mod["main"], original_call)


if __name__ == "__main__":
    tilelang.testing.main()
