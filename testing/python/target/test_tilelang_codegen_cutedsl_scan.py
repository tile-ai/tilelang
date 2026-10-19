import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target


def _lower_cutedsl_scan(op_name, threads):
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_80"})
    assert target is not None

    if op_name == "cumsum":

        @T.prim_func
        def prog(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
            with T.Kernel(1, threads=threads):
                A_shared = T.alloc_shared((32,), "float32")
                T.copy(A, A_shared)
                T.cumsum(A_shared, dim=0)
                T.copy(A_shared, B)

    elif op_name == "cummax":

        @T.prim_func
        def prog(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
            with T.Kernel(1, threads=threads):
                A_shared = T.alloc_shared((32,), "float32")
                T.copy(A, A_shared)
                T.cummax(A_shared, dim=0)
                T.copy(A_shared, B)

    else:
        raise ValueError(f"Unsupported scan op: {op_name}")

    with target:
        return lower(prog.with_attr("global_symbol", "main"), target=target)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("threads", [32, 96])
def test_cutedsl_codegen_supports_cumsum(threads):
    artifact = _lower_cutedsl_scan("cumsum", threads)
    assert f"tl.CumSum1D({threads}, False).run" in artifact.kernel_source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("threads", [32, 96])
def test_cutedsl_codegen_supports_cummax(threads):
    artifact = _lower_cutedsl_scan("cummax", threads)
    assert f"tl.CumMax1D({threads}, False).run" in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
