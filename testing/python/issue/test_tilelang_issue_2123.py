import tilelang
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls, collect_nodes
import tilelang.language as T
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.cuda.pipeline import CUDAPassPipelineBody
from tilelang.transform import LowerAccessPtr


def issue_2123_atomic_load_repro(num_tiles, threads=32):
    @T.prim_func
    def kernel(status: T.Tensor((num_tiles,), T.int32), out: T.Tensor((1,), T.int32)):
        with T.Kernel(num_tiles, threads=threads) as tile:
            look = T.alloc_var(T.int32)
            state = T.alloc_var(T.int32)
            done = T.alloc_var(T.bool)
            tx = T.get_thread_binding()
            if tx == 0:
                look = tile - 1
                done = look < 0
                state = 0
                while not done:
                    state = T.atomic_load(status[look], memory_order="acquire")
                    if state != 0:
                        done = True
                    else:
                        look -= 1
                        done = look < 0
                if tile == num_tiles - 1:
                    out[0] = state

    return kernel


def _assert_access_ptr_lowered(mod):
    assert tirx.analysis.check_well_formed(mod)
    assert_call_count(mod, op="tl.access_ptr", count=0)
    functions = [func for func in mod.functions.values() if collect_calls(func, op="tl.atomic_load_elem_op")]
    (func,) = functions
    (atomic,) = collect_calls(func, op="tl.atomic_load_elem_op")
    pointer, memory_order = atomic.args
    assert atomic.ty == ir.PrimType("int32")
    assert pointer.ty == ir.PtrType("int32", "global")
    assert int(memory_order) == 2  # acquire
    assert isinstance(pointer, ir.Call) and pointer.op.same_as(ir.Op.get("tirx.address_of"))
    (load,) = pointer.args
    assert isinstance(load, ir.TensorLoad)
    status = load.source
    # The original tensor is contiguous int32: status[look[0]] names byte
    # offset 4 * look[0]. The TensorLoad here describes an address, not a load.
    ir.assert_structural_equal(status.ty, tirx.TensorType((4,), "int32", data_alignment=1))
    if not any(status.same_as(param) for param in func.params):
        # The production pipeline outlines the tensor as a pointer parameter.
        (status_binding,) = [binding for binding in collect_nodes(func, ir.Bind) if binding.var.same_as(status)]
        assert status_binding.value.op.same_as(ir.Op.get("tirx.decl_tensor"))
        (status_ptr,) = [param for param in func.params if param.name == "status_ptr"]
        assert status_binding.value.args[0].same_as(status_ptr)
    else:
        assert status.same_as(next(param for param in func.params if param.name == "status"))
    (index,) = load.indices
    assert isinstance(index, ir.TensorLoad)
    assert index.source.name == "look" and str(index.source.ty.scope) == "local.var"
    assert len(index.indices) == 1 and int(index.indices[0]) == 0
    (loop,) = collect_nodes(func, ir.While)
    (loop_atomic,) = collect_calls(loop.body, op="tl.atomic_load_elem_op")
    assert loop_atomic.same_as(atomic)
    (decrement,) = [store for store in collect_nodes(loop.body, ir.TensorStore) if store.dest.same_as(index.source)]
    assert len(decrement.indices) == 1 and int(decrement.indices[0]) == 0
    ir.assert_structural_equal(decrement.value, index - 1)
    # Preserve the returned atomic value; an ordinary status load cannot
    # replace it, and the index must follow the loop-carried look variable.
    (state_store,) = [
        store for store in collect_nodes(loop.body, ir.TensorStore) if collect_calls(store.value, op="tl.atomic_load_elem_op")
    ]
    assert state_store.dest.name == "state"
    status_loads = [node for node in collect_nodes(func, ir.TensorLoad) if node.source.same_as(status)]
    assert len(status_loads) == 1 and status_loads[0].same_as(load)


def test_issue_2123_atomic_load_lower_access_ptr_direct():
    func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
    mod = tvm_tirx.IRModule.from_expr(func)

    lowered = LowerAccessPtr()(mod)

    _assert_access_ptr_lowered(lowered)


@tilelang.testing.requires_cuda
def test_issue_2123_atomic_load_lower_access_ptr_pipeline():
    target = tvm_tirx.target.Target("cuda", host="llvm")
    func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
    mod = tvm_tirx.IRModule.from_expr(func)

    # Access extent/mask remain live through synchronization and fence passes;
    # the native pipeline erases them at its final address/launch boundary.
    lowered = CUDAPassPipelineBody(mod, target)

    _assert_access_ptr_lowered(lowered)


if __name__ == "__main__":
    tilelang.testing.main()
