"""Per-launch names must survive lowering and name the corresponding calls."""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.jit.adapter.nvrtc import is_nvrtc_available


def _two_stage(first_name=None, second_name=None):
    @T.prim_func
    def main(
        A: T.Tensor((256,), "float32"),
        Tmp: T.Tensor((256,), "float32"),
        B: T.Tensor((256,), "float32"),
    ):
        with T.Kernel(2, threads=128, name=first_name) as bx:
            for i in T.Parallel(128):
                Tmp[bx * 128 + i] = A[bx * 128 + i] + 1.0
        with T.Kernel(2, threads=128, name=second_name) as bx:
            for i in T.Parallel(128):
                B[bx * 128 + i] = Tmp[bx * 128 + i] * 2.0

    return main


@pytest.mark.parametrize(
    "requested, expected",
    [
        ((None, None), ("main_kernel", "main_kernel_1")),
        (("add_one", "double_values"), ("add_one", "double_values")),
        (("add_one", None), ("add_one", "main_kernel")),
        ((None, "double_values"), ("main_kernel", "double_values")),
        (("stage", "stage"), ("stage", "stage_1")),
        (("main", "main"), ("main_1", "main_2")),
        (("main_kernel", None), ("main_kernel", "main_kernel_1")),
        ((None, "main_kernel"), ("main_kernel", "main_kernel_1")),
    ],
)
def test_kernel_names_and_host_calls(requested, expected):
    mod = tvm.IRModule.from_expr(_two_stage(*requested))
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target("cuda", "c"))(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.GlobalVar):
            calls.append(node.op)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert [call.name_hint for call in calls] == list(expected)
    assert {var.name_hint for var in mod.functions} == {"main", *expected}
    for call in calls:
        assert mod[call].attrs["target"].kind.name == "cuda"
    # The naming annotation is consumed, not emitted as a device instruction.
    assert "tl.kernel_name" not in mod.script()


@pytest.mark.parametrize("name", ["", "bad-name", "with space", "1stage", "foo.bar", "ns::stage", "阶段", "stage\n"])
def test_invalid_kernel_name(name):
    with pytest.raises(ValueError, match="Kernel name must be a non-empty ASCII identifier"):
        _two_stage(name)


@pytest.mark.parametrize("name", [1, True, ["stage"]])
def test_kernel_name_requires_string(name):
    with pytest.raises(TypeError, match="Kernel name must be a string"):
        _two_stage(name)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "execution_backend",
    [
        "tvm_ffi",
        "cython",
        pytest.param("nvrtc", marks=pytest.mark.skipif(not is_nvrtc_available, reason="cuda-python is required for NVRTC")),
    ],
)
@pytest.mark.parametrize(
    "requested, expected",
    [
        (("add_one", "double_values"), ("add_one", "double_values")),
        (("stage", "stage"), ("stage", "stage_1")),
        (("add_one", None), ("add_one", "main_kernel")),
    ],
)
def test_named_kernels_execute(requested, expected, execution_backend):
    kernel = tilelang.compile(_two_stage(*requested), target="cuda", execution_backend=execution_backend)
    source = kernel.get_kernel_source()
    for name in expected:
        assert re.search(rf"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?{name}\s*\(", source)

    a = torch.randn(256, device="cuda", dtype=torch.float32)
    tmp = torch.empty_like(a)
    b = torch.empty_like(a)
    kernel(a, tmp, b)
    torch.testing.assert_close(tmp, a + 1.0)
    torch.testing.assert_close(b, (a + 1.0) * 2.0)


@tilelang.testing.requires_cuda
def test_eager_named_kernels_with_shared_memory():
    @tilelang.jit(target="cuda", execution_backend="tvm_ffi")
    def pipeline(A):
        A: T.Tensor[[256], T.float32]
        Tmp = T.empty([256], T.float32)
        B = T.empty([256], T.float32)
        with T.Kernel(1, threads=256, name="shared_stage"):
            shared = T.alloc_shared([256], T.float32)
            T.copy(A, shared)
            for i in T.Parallel(256):
                Tmp[i] = shared[i] + 1.0
        with T.Kernel(1, threads=256, name="finish_stage"):
            for i in T.Parallel(256):
                B[i] = Tmp[i] * 2.0
        return Tmp, B

    a = torch.randn(256, device="cuda", dtype=torch.float32)
    tmp, b = pipeline(a)
    torch.testing.assert_close(tmp, a + 1.0)
    torch.testing.assert_close(b, (a + 1.0) * 2.0)
    source = pipeline.get_kernel_source(a)
    assert "shared_stage(" in source
    assert "finish_stage(" in source


@tilelang.testing.requires_cuda
def test_named_cluster_kernel_codegen():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32")):
        with T.ClusterKernel(2, threads=128, cluster_dims=2, name="cluster_stage") as bx:
            for i in T.Parallel(128):
                A[bx * 128 + i] = 1.0

    with tvm.target.Target({"kind": "cuda", "arch": "sm_90"}) as target:
        artifact = tilelang.lower(main, target=target)
    assert {var.name_hint for var in artifact.device_mod.functions} == {"cluster_stage"}
    assert "cluster_stage(" in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
