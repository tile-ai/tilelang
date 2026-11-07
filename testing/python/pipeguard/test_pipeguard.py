"""Optional integration smoke tests; lowering only, no GPU launch."""

import pytest

pytest.importorskip("pipeguard.integration")
import tilelang
import tilelang.language as T
from tilelang.contrib.pipeguard import check


def shared_handoff(synchronized):
    @T.prim_func
    def kernel(Out: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((1,), "float32")
            if tx == 0:
                shared[0] = 1.0
            if synchronized:
                T.sync_threads()
            if tx == 64:
                Out[0] = shared[0]

    return kernel


@pytest.mark.parametrize("synchronized,verdict", [(True, "SAFE"), (False, "DATA_HAZARD")])
def test_shared_handoff(synchronized, verdict):
    report = check(
        shared_handoff(synchronized),
        launch_grid=(1,),
        pass_configs={"tl.disable_thread_storage_sync": True},
    )
    assert report["verdict"] == verdict, report
    assert report["scope"]["kind"] == "selected-cta"
    assert report["scope"]["device_code_checked"] is False


def test_compiled_artifact():
    artifact = tilelang.lower(
        shared_handoff(True), target={"kind": "cuda", "arch": "sm_90a"}, enable_host_codegen=False, enable_device_compile=False
    )
    report = check(artifact, launch_grid=(1,))
    assert report["verdict"] == "SAFE", report


def test_cached_kernel_without_ir():
    with pytest.raises(ValueError, match="retained CompiledArtifact"):
        check(object(), launch_grid=(1,))


def test_tma_host_evidence():
    from pipeguard.capture.tilelang_host import host_descriptors

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), "float16"), Out: T.Tensor((1,), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((32, 32), "float16")
            barrier = T.alloc_barrier([128])
            T.tma_copy(A[0:32, 0:32], shared, barrier=barrier)
            T.mbarrier_arrive(barrier)
            T.mbarrier_wait_parity(barrier, 0)
            if T.get_thread_binding() == 0:
                Out[0] = shared[0, 0]

    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    with target:
        artifact = tilelang.lower(kernel, target=target, enable_host_codegen=False, enable_device_compile=False)
    evidence = host_descriptors(artifact.host_mod, {})
    assert evidence, "expected a host-side TMA descriptor"
    report = check(artifact, launch_grid=(1,))
    assert report["verdict"] == "SAFE", report


def test_jit_kernel_method():
    from tilelang.jit.kernel import JITKernel

    artifact = tilelang.lower(
        shared_handoff(True), target={"kind": "cuda", "arch": "sm_90a"}, enable_host_codegen=False, enable_device_compile=False
    )
    kernel = JITKernel.__new__(JITKernel)
    kernel.artifact = artifact
    report = kernel.check_pipeguard(launch_grid=(1,))
    assert report["verdict"] == "SAFE", report
