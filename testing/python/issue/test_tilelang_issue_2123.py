from __future__ import annotations

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize
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


def _has_op_call(func, op_name):
    found = False

    def _visit(node):
        nonlocal found
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == op_name:
            found = True

    tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
    return found


def _assert_access_ptr_lowered(mod):
    assert _has_op_call(mod["main"], "tir.tvm_access_ptr")
    assert not _has_op_call(mod["main"], "tl.access_ptr")


def test_issue_2123_atomic_load_lower_access_ptr_direct():
    func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)

    lowered = LowerAccessPtr()(mod)

    _assert_access_ptr_lowered(lowered)


def test_issue_2123_atomic_load_lower_access_ptr_pipeline():
    target = tvm.target.Target("cuda", host="llvm")
    func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)

    lowered = LowerAndLegalize(mod, target)

    _assert_access_ptr_lowered(lowered)


if __name__ == "__main__":
    tilelang.testing.main()
