import tilelang
from tilelang import tvm
from tvm import s_tir, tirx
from tvm.script import tirx as T


def _collect_launch_ir(mod):
    loops = []
    attrs = []
    roots = []

    def visitor(node):
        if isinstance(node, tirx.For):
            loops.append(node)
        elif isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
            attrs.append(node)
        elif isinstance(node, tirx.SBlockRealize) and node.block.name_hint == "tilelang_root":
            roots.append(node)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visitor)
    return loops, attrs, roots


def test_reserve_root_block_preserves_target_neutral_launch_loops():
    @T.prim_func
    def before(A: T.Buffer((8,), "float32"), B: T.Buffer((8,), "float32")):
        for bx in T.thread_binding(2, thread="blockIdx.x"):
            for tx in T.thread_binding(4, thread="threadIdx.x"):
                with T.sblock("B"):
                    v_i = T.axis.spatial(8, bx * 4 + tx)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = A[v_i]

    mod = tvm.IRModule({"main": before})
    mod = s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.ReserveRootBlock()(mod)

    loops, attrs, roots = _collect_launch_ir(mod)
    assert [loop.thread_binding.thread_tag for loop in loops] == ["threadIdx.x", "blockIdx.x"]
    assert not attrs
    assert len(roots) == 1

    cuda_mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    loops, attrs, roots = _collect_launch_ir(cuda_mod)
    assert not loops
    assert {attr.node.thread_tag for attr in attrs} == {"blockIdx.x", "threadIdx.x"}
    assert len(roots) == 1

    cpu_mod = tilelang.transform.MaterializeKernelLaunch(lower_thread_binding=False)(mod)
    loops, attrs, roots = _collect_launch_ir(cpu_mod)
    assert not attrs
    assert {loop.loop_var.name: int(loop.extent) for loop in loops} == {"bx": 2, "tx": 1}
    assert len(roots) == 1
