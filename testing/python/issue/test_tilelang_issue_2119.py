import tilelang as tl
import tilelang.language as T
import tilelang.testing
import tvm
from tvm.tir import stmt_functor
from tvm.tir.stmt import IfThenElse


def _count_if_then_else(stmt):
    count = 0

    def visit(node):
        nonlocal count
        if isinstance(node, IfThenElse):
            count += 1

    stmt_functor.post_order_visit(stmt, visit)
    return count


def _guarded_global_atomic(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id < num_bins:
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_algebraic(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id + 1 <= num_bins:
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_reassigned(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads * 2,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id < num_bins:
                bin_id = keys[tid + num_threads]
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_reassigned_static(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id < num_bins:
                bin_id = 3
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_reassigned_algebraic(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            shifted = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id > 0 and bin_id < num_bins:
                shifted = bin_id - 1
                T.atomic_add(counts[shifted], 1)

    return main


def _guarded_global_atomic_reassigned_self_dependent(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id < num_bins:
                bin_id = bin_id + 1
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_reassigned_self_dependent_safe(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id < num_bins - 1:
                bin_id = bin_id + 1
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_nested_self_dependent_join(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id > 0 and bin_id < 13:
                if bin_id > 8:
                    bin_id = bin_id + 1
                else:
                    bin_id = bin_id // 2
                bin_id = bin_id + 1
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_nested_self_dependent_outside_guard(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id > 0 and bin_id < 13:
                if bin_id > 8:
                    bin_id = bin_id + 1
                else:
                    bin_id = bin_id // 2
            bin_id = bin_id + 1
            T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_nested_unknown_branch(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads * 2,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id < num_bins:
                if bin_id > 8:
                    bin_id = keys[tid + num_threads]
                else:
                    bin_id = bin_id // 2
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_nested_unsafe_branch(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id < num_bins:
                if bin_id > 8:
                    bin_id = bin_id + 10
                else:
                    bin_id = bin_id // 2
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_loop_self_dependent_no_leak(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            if bin_id >= 0 and bin_id < num_bins - 1:
                for _ in T.serial(1):
                    bin_id = bin_id + 1
                T.atomic_add(counts[bin_id], 1)

    return main


def _guarded_global_atomic_store_bitwise(num_bins: int = 16, num_threads: int = 128):
    @T.prim_func
    def main(
        keys: T.Tensor((num_threads,), dtype=T.int32),
        counts: T.Tensor((num_bins,), dtype=T.int32),
    ):
        with T.Kernel(1, 1, threads=num_threads) as (bx, by):
            bin_id = T.alloc_var(T.int32)
            masked = T.alloc_var(T.int32)
            tid = T.get_thread_binding()

            bin_id = keys[tid]
            masked = T.bitwise_and(bin_id, 15)
            T.atomic_add(counts[masked], 1)

    return main


def _alias_local_var_reassigned():
    keys = tvm.tir.decl_buffer((128,), "int32", name="keys")
    counts = tvm.tir.decl_buffer((16,), "int32", name="counts")
    bin_id = tvm.tir.decl_buffer((1,), "int32", name="bin_id", scope="local.var")
    alias = tvm.tir.decl_buffer((1,), "int32", name="bin_id_alias", data=bin_id.data, scope="local.var")
    zero = tvm.tir.IntImm("int32", 0)
    one = tvm.tir.IntImm("int32", 1)
    load = tvm.tir.BufferLoad(bin_id, [zero])
    condition = tvm.tir.And(load >= zero, load < tvm.tir.IntImm("int32", 16))
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.BufferStore(bin_id, tvm.tir.BufferLoad(keys, [zero]), [zero]),
            tvm.tir.IfThenElse(
                condition,
                tvm.tir.SeqStmt(
                    [
                        tvm.tir.BufferStore(alias, tvm.tir.BufferLoad(keys, [one]), [zero]),
                        tvm.tir.BufferStore(counts, one, [tvm.tir.BufferLoad(bin_id, [zero])]),
                    ]
                ),
                None,
            ),
        ]
    )
    return tvm.tir.PrimFunc(
        [keys.data, counts.data],
        body,
        buffer_map={keys.data: keys, counts.data: counts},
    )


def _alias_local_var_self_dependent_safe():
    keys = tvm.tir.decl_buffer((128,), "int32", name="keys")
    counts = tvm.tir.decl_buffer((16,), "int32", name="counts")
    bin_id = tvm.tir.decl_buffer((1,), "int32", name="bin_id", scope="local.var")
    alias = tvm.tir.decl_buffer((1,), "int32", name="bin_id_alias", data=bin_id.data, scope="local.var")
    zero = tvm.tir.IntImm("int32", 0)
    one = tvm.tir.IntImm("int32", 1)
    load = tvm.tir.BufferLoad(bin_id, [zero])
    condition = tvm.tir.And(load >= zero, load < tvm.tir.IntImm("int32", 15))
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.BufferStore(bin_id, tvm.tir.BufferLoad(keys, [zero]), [zero]),
            tvm.tir.IfThenElse(
                condition,
                tvm.tir.SeqStmt(
                    [
                        tvm.tir.BufferStore(alias, tvm.tir.BufferLoad(bin_id, [zero]) + one, [zero]),
                        tvm.tir.BufferStore(counts, one, [tvm.tir.BufferLoad(bin_id, [zero])]),
                    ]
                ),
                None,
            ),
        ]
    )
    return tvm.tir.PrimFunc(
        [keys.data, counts.data],
        body,
        buffer_map={keys.data: keys, counts.data: counts},
    )


def _alias_local_var_invalidates_tracked_self_dependent():
    keys = tvm.tir.decl_buffer((128,), "int32", name="keys")
    counts = tvm.tir.decl_buffer((16,), "int32", name="counts")
    bin_id = tvm.tir.decl_buffer((1,), "int32", name="bin_id", scope="local.var")
    alias = tvm.tir.decl_buffer((1,), "int32", name="bin_id_alias", data=bin_id.data, scope="local.var")
    zero = tvm.tir.IntImm("int32", 0)
    one = tvm.tir.IntImm("int32", 1)
    load = tvm.tir.BufferLoad(bin_id, [zero])
    condition = tvm.tir.And(load >= zero, load < tvm.tir.IntImm("int32", 15))
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.BufferStore(bin_id, tvm.tir.BufferLoad(keys, [zero]), [zero]),
            tvm.tir.IfThenElse(
                condition,
                tvm.tir.SeqStmt(
                    [
                        tvm.tir.BufferStore(alias, tvm.tir.BufferLoad(bin_id, [zero]) + one, [zero]),
                        tvm.tir.BufferStore(alias, tvm.tir.BufferLoad(keys, [one]), [zero]),
                        tvm.tir.BufferStore(counts, one, [tvm.tir.BufferLoad(bin_id, [zero])]),
                    ]
                ),
                None,
            ),
        ]
    )
    return tvm.tir.PrimFunc(
        [keys.data, counts.data],
        body,
        buffer_map={keys.data: keys, counts.data: counts},
    )


def _legalize(func):
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    return tl.transform.LegalizeSafeMemoryAccess()(mod)


def test_guarded_global_atomic_reuses_active_guard():
    transformed = _legalize(_guarded_global_atomic())
    assert _count_if_then_else(transformed["main"].body) == 1


def test_guarded_global_atomic_uses_active_guard_implication():
    transformed = _legalize(_guarded_global_atomic_algebraic())
    assert _count_if_then_else(transformed["main"].body) == 1


def test_guarded_global_atomic_reassigned_gets_new_guard():
    transformed = _legalize(_guarded_global_atomic_reassigned())
    assert _count_if_then_else(transformed["main"].body) > 1


def test_guarded_global_atomic_reassigned_static_reuses_store_predicate():
    transformed = _legalize(_guarded_global_atomic_reassigned_static())
    assert _count_if_then_else(transformed["main"].body) == 1


def test_guarded_global_atomic_reassigned_algebraic_uses_z3_store_predicate():
    transformed = _legalize(_guarded_global_atomic_reassigned_algebraic())
    assert _count_if_then_else(transformed["main"].body) == 1


def test_guarded_global_atomic_reassigned_self_dependent_gets_new_guard():
    transformed = _legalize(_guarded_global_atomic_reassigned_self_dependent())
    assert _count_if_then_else(transformed["main"].body) > 1


def test_guarded_global_atomic_reassigned_self_dependent_safe_gets_conservative_guard():
    transformed = _legalize(_guarded_global_atomic_reassigned_self_dependent_safe())
    assert _count_if_then_else(transformed["main"].body) > 1


def test_guarded_global_atomic_nested_self_dependent_join_gets_conservative_guard():
    transformed = _legalize(_guarded_global_atomic_nested_self_dependent_join())
    assert _count_if_then_else(transformed["main"].body) > 2


def test_guarded_global_atomic_nested_self_dependent_outside_guard_gets_new_guard():
    transformed = _legalize(_guarded_global_atomic_nested_self_dependent_outside_guard())
    assert _count_if_then_else(transformed["main"].body) > 2


def test_guarded_global_atomic_nested_unknown_branch_gets_new_guard():
    transformed = _legalize(_guarded_global_atomic_nested_unknown_branch())
    assert _count_if_then_else(transformed["main"].body) > 2


def test_guarded_global_atomic_nested_unsafe_branch_gets_new_guard():
    transformed = _legalize(_guarded_global_atomic_nested_unsafe_branch())
    assert _count_if_then_else(transformed["main"].body) > 2


def test_guarded_global_atomic_loop_self_dependent_no_leak_gets_new_guard():
    transformed = _legalize(_guarded_global_atomic_loop_self_dependent_no_leak())
    assert _count_if_then_else(transformed["main"].body) > 1


def test_guarded_global_atomic_store_bitwise_uses_z3_store_predicate():
    transformed = _legalize(_guarded_global_atomic_store_bitwise())
    assert _count_if_then_else(transformed["main"].body) == 0


def test_alias_local_var_reassigned_invalidates_by_storage():
    mod = tvm.IRModule({"main": _alias_local_var_reassigned()})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    assert _count_if_then_else(transformed["main"].body) > 1


def test_alias_local_var_self_dependent_safe_gets_conservative_guard():
    mod = tvm.IRModule({"main": _alias_local_var_self_dependent_safe()})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    assert _count_if_then_else(transformed["main"].body) > 1


def test_alias_local_var_invalidates_tracked_self_dependent_by_storage():
    mod = tvm.IRModule({"main": _alias_local_var_invalidates_tracked_self_dependent()})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    assert _count_if_then_else(transformed["main"].body) > 1


if __name__ == "__main__":
    tilelang.testing.main()
