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


if __name__ == "__main__":
    tilelang.testing.main()
