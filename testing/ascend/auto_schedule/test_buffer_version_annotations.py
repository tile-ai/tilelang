"""Frontend version-count and indexing-mode contracts, before scheduling."""

import pytest
import tilelang.ascend.language as T
import tilelang.testing
from tvm_tirx import ir, tirx
from testing.ascend._ir import allocated_buffer, annotations, calls, nodes


def test_buffer_version_annotations_use_storage_identity():
    @T.prim_func
    def main():
        with T.Kernel(1):
            fixed = T.alloc_shared((64,), "float32")
            alias = T.reshape(fixed, (32, 2))
            explicit = T.alloc_shared((64,), "float32")
            inferred = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({fixed: 2, alias: "counter", explicit: (3, "counter"), inferred: "iteration"})

    # Native annotations carry the view TensorVar. The alias definition keeps
    # the shared storage relation explicit instead of exposing Buffer.data.
    module = ir.IRModule({"main": main})
    module = tilelang.transform.LowerTensorSpec()(module)
    assert tirx.analysis.check_well_formed(module["main"])
    storages = {name: allocated_buffer(module, name) for name in ("fixed", "explicit", "inferred")}
    values = {key: value for fields in annotations(module) for key, value in fields.items()}
    (alias,) = [binding for binding in nodes(module, ir.Bind) if binding.var.name == "alias"]
    assert any(call.args[0].same_as(storages["fixed"]) for call in calls(alias.value, "tirx.tensor_data_ptr"))
    versions = values["tl.buffer_versions_map"]
    modes = values["tl.buffer_version_mode"]
    assert len(versions) == 2
    assert int(versions[storages["fixed"]]) == 2
    assert int(versions[storages["explicit"]]) == 3
    assert storages["inferred"] not in versions
    assert len(modes) == 3
    assert str(modes[alias.var]) == "counter"
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
