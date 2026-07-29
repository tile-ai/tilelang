import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target


def _lower_cutedsl_partial_reduce():
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
    assert target is not None

    @T.prim_func
    def prog(A: T.Tensor((1, 512), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            x_frag = T.alloc_fragment((1, 512), "float32")
            sum_frag = T.alloc_fragment((1,), "float32")
            T.annotate_layout(
                {
                    x_frag: T.Fragment(x_frag.shape, forward_fn=lambda i, j: (j // 8, j % 8)),
                    sum_frag: T.Fragment(sum_frag.shape, forward_fn=lambda i, rep: (rep, 0), replicate=64),
                }
            )
            for i, j in T.Parallel(1, 512):
                x_frag[i, j] = A[i, j]
            T.reduce_sum(x_frag, sum_frag, dim=1)
            for i in T.Parallel(1):
                B[i] = sum_frag[i]

    with target:
        return lower(prog.with_attr("global_symbol", "main"), target=target)


def test_cutedsl_codegen_partial_reduce_named_barrier():
    """The partial scalar AllReduce uses its exact participant count."""
    artifact = _lower_cutedsl_partial_reduce()
    assert "tl.NamedBarrier(64)" in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
