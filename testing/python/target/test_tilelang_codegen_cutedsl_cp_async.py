import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower
from tvm.target import Target


def test_cutedsl_codegen_supports_tl_ptx_cp_async():
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = Target({"kind": "cuda", "arch": "sm_80", "keys": ["cuda", "gpu", "cutedsl"]})

    @T.prim_func
    def prog(A: T.Tensor((16,), "uint8"), B: T.Tensor((16,), "uint8")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((16,), "uint8", scope="shared")
            T.ptx_cp_async(T.access_ptr(A_shared[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
            B[0] = A_shared[0]

    artifact = lower(prog.with_attr("global_symbol", "main"), target=target)
    assert "tl.cp_async_gs(" in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
