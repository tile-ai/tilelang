"""Frontend version-count and indexing-mode contracts, before scheduling."""

import pytest
import tilelang.ascend.language as T
import tilelang.testing
from tvm import tirx
from testing.ascend._ir import nodes


def test_buffer_version_annotations_use_storage_identity():
    @T.prim_func
    def main():
        with T.Kernel(1):
            fixed = T.alloc_shared((64,), "float32")
            alias = T.reshape(fixed, (32, 2))
            explicit = T.alloc_shared((64,), "float32")
            inferred = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({fixed: 2, alias: "counter", explicit: (3, "counter"), inferred: "iteration"})

    root = next(block for block in nodes(main, tirx.SBlock) if block.name_hint == "tilelang_root")
    storages = {buffer.name: buffer.data for buffer in root.alloc_buffers}
    versions = root.annotations["tl.buffer_versions_map"]
    modes = root.annotations["tl.buffer_version_mode"]
    assert len(versions) == 2
    assert int(versions[storages["fixed"]]) == 2
    assert int(versions[storages["explicit"]]) == 3
    assert storages["inferred"] not in versions
    assert len(modes) == 3
    assert str(modes[storages["fixed"]]) == "counter"
    assert str(modes[storages["explicit"]]) == "counter"
    assert str(modes[storages["inferred"]]) == "iteration"


@pytest.mark.parametrize(
    "annotation, message",
    [
        pytest.param("lexical", "buffer version mode", id="unknown-mode"),
        pytest.param((2, "counter", "extra"), r"must be \(num_versions, mode\)", id="tuple-arity"),
        pytest.param((2, None), "buffer version mode", id="tuple-mode"),
    ],
)
def test_invalid_buffer_version_annotation(annotation, message):
    with pytest.raises(ValueError, match=message):

        @T.prim_func
        def main():
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "float32")
                T.annotate_buffer_versions({ub: annotation})


@pytest.mark.parametrize(
    "first, second, message",
    [
        pytest.param(2, 3, "same version count", id="count"),
        pytest.param("counter", "iteration", "aliases of one buffer storage", id="mode"),
    ],
)
def test_conflicting_alias_annotations(first, second, message):
    with pytest.raises(ValueError, match=message):

        @T.prim_func
        def main():
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "float32")
                alias = T.reshape(ub, (32, 2))
                T.annotate_buffer_versions({ub: first, alias: second})


@pytest.mark.parametrize("specification", [None, "auto", "iteration", "counter", "manual"])
def test_placement_requires_an_explicit_version_count(specification):
    from tilelang import tvm
    from testing.ascend._ir import schedule_snapshots

    @T.prim_func
    def main(A: T.Tensor((4, 64), "float32"), C: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_addresses({ub: 32})
            if specification == "manual":
                T.annotate_manual_multi_buffer({ub: 1})
            elif specification is not None:
                T.annotate_buffer_versions({ub: specification})
            for i in T.Pipelined(4, num_stages=2):
                T.copy(A[i, :], ub)
                T.copy(ub, C[i, :])

    with pytest.raises(tvm.error.InternalError, match="explicit version count.*annotate_buffer_versions"):
        schedule_snapshots(main, ["tl.AutoSchedule"])


def test_placement_rejects_unapplied_fixed_versions():
    from tilelang import tvm
    from testing.ascend._ir import schedule_snapshots

    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({ub: 2})
            T.annotate_buffer_addresses({ub: 32})
            T.copy(A, ub)
            T.copy(ub, C)

    with pytest.raises(tvm.error.InternalError, match="requested 2 versions.*selected 1"):
        schedule_snapshots(main, ["tl.AutoSchedule"])


if __name__ == "__main__":
    tilelang.testing.main()
