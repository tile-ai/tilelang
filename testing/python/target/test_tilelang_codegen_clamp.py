import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine import lower


def _clamp_kernel(dtype="float32", threads=32):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), Lo: T.Tensor((128,), dtype), Hi: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(128):
                C[i] = T.clamp(A[i], Lo[i], Hi[i])

    return main


@pytest.mark.parametrize(
    "target_config,build_func",
    [
        pytest.param(
            {"kind": "cuda", "arch": "sm_80"},
            "target.build.tilelang_cuda_without_compile",
            marks=tilelang.testing.requires_cuda.marks(),
            id="cuda",
        ),
        pytest.param(
            {"kind": "hip", "arch": "gfx942"},
            "target.build.tilelang_hip_without_compile",
            marks=tilelang.testing.requires_rocm.marks(),
            id="hip",
        ),
        pytest.param(
            {"kind": "metal"},
            "target.build.tilelang_metal_without_compile",
            marks=tilelang.testing.requires_metal.marks(),
            id="metal",
        ),
        pytest.param(
            {"kind": "cutedsl", "arch": "sm_80"},
            "target.build.tilelang_cutedsl_without_compile",
            marks=tilelang.testing.requires_cuda.marks(),
            id="cutedsl",
        ),
        pytest.param({"kind": "c"}, "target.build.tilelang_c", id="cpu"),
    ],
)
def test_clamp_backend_codegen(target_config, build_func):
    if tvm.ffi.get_global_func(build_func, allow_missing=True) is None:
        pytest.skip(f"{build_func} is not enabled in this build")
    if target_config["kind"] == "cutedsl":
        from tilelang.cuda.target import normalize_cutedsl_target

        target = normalize_cutedsl_target(target_config)
        # Exercise C++ source generation without requiring the CuTeDSL Python
        # runtime. Both CUDA variants consume the lowered device PrimFunc.
        cuda_target = tvm.target.Target({"kind": "cuda", "arch": target_config["arch"]})
        with cuda_target:
            artifact = lower(_clamp_kernel(), target=cuda_target)
        source = tvm.ffi.get_global_func(build_func)(artifact.device_mod, target).inspect_source()
    else:
        target = tvm.target.Target(target_config)
        with target:
            source = lower(_clamp_kernel(), target=target).kernel_source
    if target_config["kind"] in ("cuda", "c"):
        assert "tl::clamp(" in source
        if target_config["kind"] == "cuda":
            assert "*(float4*)" in source
    else:
        assert "!=" in source  # NaN checks must survive simplification.
        assert "tl.clamp" not in source


@pytest.mark.parametrize("kind", ["llvm", "hip", "metal", "webgpu"])
def test_clamp_fallback_registration(kind):
    # LLVM/WebGPU codegens need not be compiled in to exercise their lowering.
    from tvm import tirx

    x = T.Var("x", "float32x4")
    func = tirx.PrimFunc([x], tirx.Evaluate(T.clamp(x, -1.0, 1.0))).with_attr("target", tvm.target.Target(kind))
    mod = tilelang.transform.LowerIntrin()(tvm.IRModule({"main": func}))
    calls = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda node: calls.append(node) if isinstance(node, tirx.Call) else None)
    assert not any(call.op.same_as(tirx.op.Op.get("tl.clamp")) for call in calls)
    assert any(call.op.same_as(tirx.op.Op.get("tirx.isnan")) for call in calls)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_clamp_cpu_templates(dtype):
    import torch

    target = tvm.target.Target("c")
    with target:
        kernel = tilelang.compile(_clamp_kernel(dtype), target=target, target_host="c", execution_backend="cython")
    a = torch.tensor([float("nan"), -2.0, 0.5, 2.0], dtype=getattr(torch, dtype)).repeat(32)
    lo, hi = torch.full_like(a, -1), torch.full_like(a, 1)
    lo[5], hi[6] = float("nan"), float("nan")
    lo[8:12] = 5
    c = torch.empty_like(a)
    kernel(a, lo, hi, c)
    torch.testing.assert_close(c, torch.clamp(a, lo, hi), equal_nan=True, atol=0, rtol=0)


def test_clamp_cpu_single_evaluation():
    import torch

    @T.prim_func
    def main(A: T.Tensor((3,), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            C[0] = T.clamp(
                T.atomic_add(A[0], 0.25, return_prev=True),
                T.atomic_add(A[1], 0.25, return_prev=True),
                T.atomic_add(A[2], 0.25, return_prev=True),
            )

    target = tvm.target.Target("c")
    with target:
        kernel = tilelang.compile(main, target=target, target_host="c", execution_backend="cython")
    a = torch.tensor([0.0, -1.0, 1.0])
    c = torch.empty(1)
    kernel(a, c)
    torch.testing.assert_close(a, torch.tensor([0.25, -0.75, 1.25]))
    torch.testing.assert_close(c, torch.zeros_like(c))
