import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine import lower


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_86", "sm_100"])
@pytest.mark.parametrize("threads", [32, 64, 128])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_clamp_symmetric_codegen(arch, threads, dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(128):
                C[i] = T.clamp_symmetric(A[i], 1.0)

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        if arch == "sm_80":
            with pytest.raises(Exception, match="requires CUDA SM86 or newer"):
                lower(main, target=target)
        else:
            source = lower(main, target=target).kernel_source
            helper = "tl::clamp_symmetric2(" if dtype != "float32" and threads < 128 else "tl::clamp_symmetric("
            assert helper in source
            assert "tl::clamp(" not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_clamp_symmetric_types(dtype):
    x = T.Var("x", dtype + "x4")
    result = T.clamp_symmetric(x, 1.0)
    assert result.dtype == dtype + "x4"
    assert all(arg.dtype == dtype + "x4" for arg in result.args)
    with pytest.raises(TypeError, match="float32"):
        T.clamp_symmetric(T.Var("double_x", "float64"), 1.0)
    with pytest.raises(TypeError, match="matching operand dtypes"):
        T.clamp_symmetric(x, T.Var("limit", "float64"))
    with pytest.raises(ValueError, match="matching lanes"):
        T.clamp_symmetric(x, T.Var("limit", dtype + "x2"))
