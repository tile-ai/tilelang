import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.jit.adapter.utils import maybe_desc_name
from tilelang.jit.adapter.wrapper import TLCPUSourceWrapper, TLCUDASourceWrapper


def test_adapter_source_wrapper_requires_lowered_modules():
    """Reject source wrappers that lack lowered host and device modules."""
    with pytest.raises(ValueError, match="missing: device_mod, host_mod"):
        TLCPUSourceWrapper(
            scheduled_ir_module=tvm.IRModule(),
            source="",
            target=tvm.target.Target("c"),
        )


def test_non_descriptor_name_does_not_modify_descriptor_map():
    """Leave existing mappings intact for an ordinary pointer parameter."""
    descriptor_map = {"A_desc": "A"}
    assert not maybe_desc_name("A", ["half_t", "A_desc_1"], 1, descriptor_map)
    assert descriptor_map == {"A_desc": "A"}


@tilelang.testing.requires_cuda
def test_cuda_wrapper_with_descriptor_named_output():
    """Lower TMA source with an output name that resembles a descriptor."""
    @T.prim_func
    def mixed(
        A: T.StridedTensor((32, 128), (160, 1), "float16"),
        A_desc: T.Tensor((32, 128), "float16"),
    ):
        """Copy strided input through shared memory into a descriptor-like name."""
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((32, 128), "float16")
            T.copy(A, shared, prefer_instruction="tma")
            for i, j in T.Parallel(32, 128):
                A_desc[i, j] = shared[i, j]

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        artifact = tilelang.lower(mixed, target=target, enable_host_codegen=False, enable_device_compile=False)
    assert "CUtensorMap" in artifact.kernel_source
    wrapper = TLCUDASourceWrapper(
        scheduled_ir_module=tvm.IRModule({"main": mixed}),
        source=artifact.kernel_source,
        target=artifact.target,
        host_mod=artifact.host_mod,
        device_mod=artifact.device_mod,
    )
    assert "cuTensorMapEncodeTiled" in wrapper.lib_code


if __name__ == "__main__":
    pytest.main([__file__])
