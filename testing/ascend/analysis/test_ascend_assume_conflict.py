from tilelang import tvm
import tilelang.ascend.language as T

from tilelang.engine.lower import lower


def test_assume_conflict_adds_sync_in_single_loop():
    tile = 64

    @T.prim_func
    def main(
        A: T.Tensor((2 * tile,), "float32"),
        C: T.Tensor((2 * tile,), "float32"),
    ):
        with T.Kernel(1):
            producer = T.alloc_shared((tile,), "float32")
            consumer = T.alloc_shared((tile,), "float32")
            for i in T.serial(2):
                T.assume_conflict(producer, consumer, cross=False)
                T.copy(A[i * tile : (i + 1) * tile], producer)
                T.copy(consumer, C[i * tile : (i + 1) * tile])

    source = lower(main, target="ascend").kernel_source

    loop_body = source[source.index("for (") :]
    set_flag = "asc_sync_notify(PIPE_MTE2, PIPE_MTE3,"
    wait_flag = "asc_sync_wait(PIPE_MTE2, PIPE_MTE3,"
    assert loop_body.count(set_flag) == 1
    assert loop_body.count(wait_flag) == 1
    assert loop_body.index("asc_copy_gm2ub_align") < loop_body.index(set_flag) < loop_body.index(wait_flag)
    assert loop_body.index(wait_flag) < loop_body.index("asc_copy_ub2gm_align")


def test_assume_conflict_adds_forward_cross_iteration_raw():
    tile = 64

    @T.prim_func
    def main(
        A: T.Tensor((2 * tile,), "float32"),
        C: T.Tensor((2 * tile,), "float32"),
    ):
        with T.Kernel(1):
            producer = T.alloc_shared((tile,), "float32")
            consumer = T.alloc_shared((tile,), "float32")
            for i in T.serial(2):
                T.assume_conflict(producer, consumer, cross=True)
                T.copy(A[i * tile : (i + 1) * tile], producer)
                T.copy(consumer, C[i * tile : (i + 1) * tile])

    source = lower(main, target="ascend").kernel_source

    loop_body = source[source.index("for (") :]
    set_flag = "asc_sync_notify(PIPE_MTE2, PIPE_MTE3,"
    wait_flag = "asc_sync_wait(PIPE_MTE2, PIPE_MTE3,"
    assert loop_body.count(set_flag) == 1
    wait = loop_body.index(wait_flag)
    consumer = loop_body.index("asc_copy_ub2gm_align")
    producer = loop_body.index("asc_copy_gm2ub_align")
    set_ = loop_body.index(set_flag)
    assert wait < consumer < producer < set_


def test_assume_conflict_matches_exact_region():
    tile = 64

    @T.prim_func
    def main(
        A: T.Tensor((2 * tile,), "float32"),
        C: T.Tensor((2 * tile,), "float32"),
    ):
        with T.Kernel(1):
            producer = T.alloc_shared((2 * tile,), "float32")
            consumer = T.alloc_shared((2 * tile,), "float32")
            for _ in T.serial(2):
                T.assume_conflict(producer[0 : 2 * tile], consumer[0 : 2 * tile], cross=False)
                T.copy(A, producer)
                T.copy(consumer, C)

    source = lower(main, target="ascend").kernel_source

    assert "asc_sync_notify(PIPE_MTE2, PIPE_MTE3," in source
    assert "asc_sync_wait(PIPE_MTE2, PIPE_MTE3," in source


def test_assume_conflict_requires_exact_region_match():
    tile = 64

    @T.prim_func
    def main(
        A: T.Tensor((2 * tile,), "float32"),
        C: T.Tensor((2 * tile,), "float32"),
    ):
        with T.Kernel(1):
            producer = T.alloc_shared((2 * tile,), "float32")
            consumer = T.alloc_shared((2 * tile,), "float32")
            for i in T.serial(2):
                T.assume_conflict(
                    producer[i * tile : (i + 1) * tile],
                    consumer[i * tile : (i + 1) * tile],
                    cross=False,
                )
                T.copy(A, producer)
                T.copy(consumer, C)

    source = lower(main, target="ascend").kernel_source

    assert "asc_sync_notify(PIPE_MTE2, PIPE_MTE3," not in source
    assert "asc_sync_wait(PIPE_MTE2, PIPE_MTE3," not in source


def test_root_assume_conflict_survives_scheduled_tir_round_trip():
    tile = 64

    @T.prim_func
    def main(
        A: T.Tensor((tile,), "float32"),
        C: T.Tensor((tile,), "float32"),
    ):
        with T.Kernel(1):
            producer = T.alloc_shared((tile,), "float32")
            consumer = T.alloc_shared((tile,), "float32")
            T.assume_conflict(producer, consumer, level=-1, cross=False)
            T.copy(A, producer)
            T.copy(consumer, C)

    source = lower(main, target="ascend").kernel_source

    assert "asc_sync_notify(PIPE_MTE2, PIPE_MTE3," in source
    assert "asc_sync_wait(PIPE_MTE2, PIPE_MTE3," in source


def test_default_level_applies_assume_conflict_at_root():
    tile = 64

    @T.prim_func
    def main(
        A: T.Tensor((tile,), "float32"),
        C: T.Tensor((tile,), "float32"),
    ):
        with T.Kernel(1):
            producer = T.alloc_shared((tile,), "float32")
            consumer = T.alloc_shared((tile,), "float32")
            T.assume_conflict(producer, consumer, cross=False)
            T.copy(A, producer)
            T.copy(consumer, C)

    source = lower(main, target="ascend").kernel_source

    assert "asc_sync_notify(PIPE_MTE2, PIPE_MTE3," in source
    assert "asc_sync_wait(PIPE_MTE2, PIPE_MTE3," in source


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

    source = lower(main, target="ascend").kernel_source

    loop_body = source[source.index("for (") :]
    assert "asc_sync_notify(PIPE_MTE2, PIPE_MTE3," not in loop_body
    assert "asc_sync_wait(PIPE_MTE2, PIPE_MTE3," not in loop_body


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
        source = lower(main, target="ascend").kernel_source

    assert "asc_sync_notify(PIPE_V, PIPE_MTE3," in source
    assert "asc_sync_wait(PIPE_V, PIPE_MTE3," in source


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
        source = lower(main, target="ascend").kernel_source

    same_wait = "asc_sync_wait(PIPE_MTE2, PIPE_MTE3,"
    cross_wait = "asc_sync_wait(PIPE_V, PIPE_MTE3,"
    assert source.count(same_wait) == 1
    assert source.count(cross_wait) == 2


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

    source = lower(main, target="ascend").kernel_source

    assert "asc_sync_notify(PIPE_MTE2, PIPE_MTE3," in source
    assert "asc_sync_wait(PIPE_MTE2, PIPE_MTE3," in source
