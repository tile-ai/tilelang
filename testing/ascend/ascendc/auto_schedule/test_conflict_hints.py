"""Conflict hints affect synchronization at the declared scope and region."""

import pytest
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, schedule, nodes


def _insert_sync(program):
    before = schedule(program, through="ResolveCore", auto_schedule=False)
    return transform.InsertSync()(before)


def _event_count(mod, operation, pipe):
    return sum(call.args[0].value == pipe for call in calls(mod, "tl.ascend_" + operation + "_flag"))


def _hint_program(scope, exact, cross, explicit_root):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            producer = T.alloc_shared((128,), "float32")
            consumer = T.alloc_shared((128,), "float32")
            if scope == "root":
                T.assume_conflict(producer, consumer, level=-1 if explicit_root else None, cross=False)
                T.copy(A, producer)
                T.copy(consumer, C)
            else:
                for _i in T.serial(2):
                    T.assume_conflict(producer[: 128 if exact else 64], consumer[: 128 if exact else 64], cross=cross)
                    T.copy(A, producer)
                    T.copy(consumer, C)

    return main


@pytest.mark.parametrize(
    "scope, exact, cross, explicit_root",
    [
        pytest.param("root", True, False, False, id="default-root"),
        pytest.param("root", True, False, True, id="explicit-root"),
        pytest.param("loop", True, False, False, id="same-iteration"),
        pytest.param("loop", True, True, False, id="cross-iteration"),
        pytest.param("loop", False, False, False, id="partial-region-is-not-a-match"),
    ],
)
def test_hint_scope_and_region(scope, exact, cross, explicit_root):
    mod = _insert_sync(_hint_program(scope, exact, cross, explicit_root))
    for op in ("set", "wait"):
        events = [call for call in calls(mod, "tl.ascend_" + op + "_flag") if call.args[0].value == "MTE2_MTE3"]
        expected = (2 if cross else 1) if exact else 0
        assert len(events) == expected


def test_assume_no_conflict_overrides_alias_conservatism():
    @T.prim_func
    def main(
        A: T.Tensor((2, 4, 16), "float32"),
        C: T.Tensor((2, 64), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            alias = T.reshape(ub, (4, 16))
            for i in T.serial(2):
                T.assume_no_conflict(alias, ub)
                T.copy(A[i, :, :], alias)
                T.copy(ub, C[i, :])

    after = _insert_sync(main)

    (loop,) = nodes(after, tirx.For)
    assert _event_count(loop, "set", "MTE2_MTE3") == 0
    assert _event_count(loop, "wait", "MTE2_MTE3") == 0


def test_assume_conflict_uses_common_projection_domain():
    tile = 64

    @T.prim_func
    def main(
        C: T.Tensor((8 * tile,), "float32"),
    ):
        with T.Kernel(1):
            producer = T.alloc_shared((tile,), "float32")
            consumer = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({producer: (2, "counter")})
            for i in T.Pipelined(8, num_stages=2):
                T.assume_conflict(producer, consumer, cross=False)
                if i % 2 == 0:
                    with T.SimtVF(threads=tile):
                        T.fill(producer, 0)
                    with T.SimtVF(threads=tile):
                        T.fill(producer, producer[0])
                if i % 3 == 0:
                    T.copy(consumer, C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(main)

    assert _event_count(after, "set", "V_MTE3")
    assert _event_count(after, "wait", "V_MTE3")


def test_assume_conflict_keeps_same_and_cross_distances_separate():
    tile = 64

    @T.prim_func
    def main(
        A: T.Tensor((2 * tile,), "float32"),
        C: T.Tensor((2 * tile,), "float32"),
        D: T.Tensor((2 * tile,), "float32"),
    ):
        with T.Kernel(1):
            same_producer = T.alloc_shared((tile,), "float32")
            same_consumer = T.alloc_shared((tile,), "float32")
            cross_producer = T.alloc_shared((tile,), "float32")
            cross_consumer = T.alloc_shared((tile,), "float32")
            for i in T.Pipelined(2, num_stages=2):
                T.assume_conflict(same_producer, same_consumer, cross=False)
                T.assume_conflict(cross_producer, cross_consumer, cross=True)
                for _producer in T.serial(1):
                    T.copy(A[i * tile : (i + 1) * tile], same_producer)
                    with T.SimtVF(threads=tile):
                        T.fill(cross_producer, 0)
                for _consumer in T.serial(1):
                    T.copy(same_consumer, C[i * tile : (i + 1) * tile])
                    T.copy(cross_consumer, D[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(main)

    assert _event_count(after, "wait", "MTE2_MTE3") == 1
    assert _event_count(after, "wait", "V_MTE3") == 2


def test_forward_loop_carried_dependency_without_hint():
    tile = 64

    @T.prim_func
    def main(
        A: T.Tensor((2 * tile,), "float32"),
        C: T.Tensor((2 * tile,), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((3 * tile,), "float32")
            for i in T.Pipelined(2, num_stages=2):
                T.copy(A[i * tile : (i + 1) * tile], ub[(i + 1) * tile : (i + 2) * tile])
                T.copy(ub[i * tile : (i + 1) * tile], C[i * tile : (i + 1) * tile])

    after = _insert_sync(main)

    assert _event_count(after, "set", "MTE2_MTE3")
    assert _event_count(after, "wait", "MTE2_MTE3")


if __name__ == "__main__":
    tilelang.testing.main()
