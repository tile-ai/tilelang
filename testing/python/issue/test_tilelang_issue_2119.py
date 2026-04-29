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


def test_guarded_global_atomic_store_bitwise_uses_z3_store_predicate():
    transformed = _legalize(_guarded_global_atomic_store_bitwise())
    assert _count_if_then_else(transformed["main"].body) == 0


if __name__ == "__main__":
    tilelang.testing.main()
