# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.engine.phase import LowerAndLegalize
from tvm import tir


sm100_target = tvm.target.Target("cuda -arch=sm_100")
sm90_target = tvm.target.Target("cuda -arch=sm_90a")


def _apply(func, target=sm100_target):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tl.transform.InjectTcgen05Fence()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    return mod


def _check(original, expected, target=sm100_target):
    mod = _apply(original, target)
    expected_mod = tvm.IRModule.from_expr(expected.with_attr("global_symbol", "main"))
    expected_mod = tvm.tir.transform.BindTarget(target)(expected_mod)
    expected_mod = tir.transform.LowerOpaqueBlock()(expected_mod)
    tvm.ir.assert_structural_equal(mod["main"], expected_mod["main"], True)


def _count_calls(stmt, op_name: str):
    count = 0

    def visitor(node):
        nonlocal count
        if isinstance(node, tir.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            count += 1

    tir.stmt_functor.post_order_visit(stmt, visitor)
    return count


def _count_extern_calls_with_prefix(stmt, prefix: str):
    count = 0

    def visitor(node):
        nonlocal count
        if not isinstance(node, tir.Call):
            return
        op = getattr(node, "op", None)
        if getattr(op, "name", None) != "tir.call_extern":
            return
        if not node.args:
            return
        name = node.args[0]
        if isinstance(name, tir.StringImm) and name.value.startswith(prefix):
            count += 1

    tir.stmt_functor.post_order_visit(stmt, visitor)
    return count


def _tcgen05_ld_call(tmem_ref, local_buf):
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.tcgen05_ld"),
        32,
        128,
        False,
        tmem_ref,
        0,
        T.tvm_access_ptr(T.type_annotation(T.float32), local_buf.data, 0, 128, 2),
    )


def test_storage_sync_is_wrapped_with_tcgen05_fences():
    @T.prim_func
    def before():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.tvm_storage_sync("shared")
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))

    @T.prim_func
    def after():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.tcgen05_before_thread_sync()
            T.tvm_storage_sync("shared")
            T.tcgen05_after_thread_sync()
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))

    _check(before, after)


def test_lower_tmem_copy_uses_tcgen05_ld_intrin():
    @T.prim_func
    def func(X: T.Tensor((256, 256), T.float16), Y: T.Tensor((256, 256), T.float16)):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 128), T.float16)
            B_shared = T.alloc_shared((128, 128), T.float16)
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float32)
            T.copy(X[0, 0], A_shared)
            T.copy(X[0, 0], B_shared)
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                C_tmem,
                transpose_B=True,
                mbar=mbar,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, Y[0, 0])

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with sm100_target:
        mod = LowerAndLegalize(mod, sm100_target)
        mod = tl.transform.LowerSharedTmem()(mod)

    body = mod["main"].body
    assert _count_calls(body, "tl.tcgen05_ld") == 1
    assert _count_extern_calls_with_prefix(body, "tl::tcgen05_ld_") == 0


def test_lower_tmem_copy_uses_tcgen05_st_intrin():
    @T.prim_func
    def func(X: T.Tensor((256, 256), T.bfloat16)):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            S_tmem = T.alloc_tmem([128, 128], T.float32)
            mbar = T.alloc_barrier(1)
            S_local = T.alloc_fragment((128, 128), T.float32)
            P_local = T.alloc_fragment((128, 128), T.bfloat16)
            P_tmem = T.alloc_tmem([128, 128], T.bfloat16)
            B2_shared = T.alloc_shared((128, 128), T.bfloat16)
            D_tmem = T.alloc_tmem([128, 128], T.float32)
            mbar2 = T.alloc_barrier(1)
            T.copy(X[0, 0], A_shared)
            T.copy(X[0, 0], B_shared)
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                S_tmem,
                transpose_B=True,
                mbar=mbar,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(S_tmem, S_local)
            T.copy(S_local, P_local)
            T.copy(P_local, P_tmem)
            T.copy(X[0, 0], B2_shared)
            T.tcgen05_gemm(
                P_tmem,
                B2_shared,
                D_tmem,
                transpose_B=True,
                mbar=mbar2,
                clear_accum=True,
            )

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with sm100_target:
        mod = LowerAndLegalize(mod, sm100_target)
        mod = tl.transform.LowerSharedTmem()(mod)

    body = mod["main"].body
    assert _count_calls(body, "tl.tcgen05_st") == 1
    assert _count_extern_calls_with_prefix(body, "tl::tcgen05_st_") == 0


def test_wait_and_arrive_are_rewritten_only_at_tmem_handoffs():
    @T.prim_func
    def before():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))
            T.ptx_arrive_barrier(mbarrier[0])

    @T.prim_func
    def after():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.tcgen05_after_thread_sync()
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))
            T.tcgen05_before_thread_sync()
            T.ptx_arrive_barrier(mbarrier[0])

    _check(before, after)


def test_wait_and_arrive_scan_across_neutral_statements():
    @T.prim_func
    def before():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.call_extern("handle", "generic_op")
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))
            T.call_extern("handle", "generic_op")
            T.ptx_arrive_barrier(mbarrier[0])

    @T.prim_func
    def after():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.tcgen05_after_thread_sync()
            T.call_extern("handle", "generic_op")
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))
            T.call_extern("handle", "generic_op")
            T.tcgen05_before_thread_sync()
            T.ptx_arrive_barrier(mbarrier[0])

    _check(before, after)


def test_sync_boundary_stops_wait_lookahead():
    @T.prim_func
    def func():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.call_extern("handle", "generic_op")
            T.ptx_arrive_barrier(mbarrier[0])
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))

    mod = _apply(func)
    assert _count_calls(mod["main"].body, "tl.tcgen05_after_thread_sync") == 0


def test_existing_manual_fences_are_not_duplicated():
    @T.prim_func
    def func():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.tcgen05_after_thread_sync()
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))
            T.tcgen05_before_thread_sync()
            T.ptx_arrive_barrier(mbarrier[0])

    mod = _apply(func)
    body = mod["main"].body
    assert _count_calls(body, "tl.tcgen05_after_thread_sync") == 1
    assert _count_calls(body, "tl.tcgen05_before_thread_sync") == 1


def test_non_sm100_targets_are_left_untouched():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            C_local = T.decl_buffer((128,), T.float32, scope="local")
            T.tvm_storage_sync("shared")
            T.evaluate(_tcgen05_ld_call(C_tmem[0], C_local))

    mod = _apply(func, sm90_target)
    assert _count_calls(mod["main"].body, "tl.tcgen05_before_thread_sync") == 0
    assert _count_calls(mod["main"].body, "tl.tcgen05_after_thread_sync") == 0


if __name__ == "__main__":
    tl.testing.main()
