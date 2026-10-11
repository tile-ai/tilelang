import pytest

import tilelang
import tilelang.testing
from tilelang import language as T


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("target_host", [None, "c"])
def test_issue_1810_l2_persistent_float16_host_stub_no_half(target_host):
    """Regression for issue #1810.

    `annotate_l2_hit_ratio` lowers to CUDA stream access policy window calls.
    The runtime API only needs an opaque base pointer, but older lowering used a
    typed access pointer. For float16 buffers, TVM's C host codegen prints the
    type as `half`, which is not defined by the stable C ABI headers, causing
    host-side compilation failures when exporting the executable.
    """

    @T.prim_func
    def minimal_kernel(A: T.Buffer((1,), "float16")):
        with T.Kernel():
            T.annotate_l2_hit_ratio({A: 0.9})
            T.evaluate(0)

    kernel = tilelang.compile(minimal_kernel, execution_backend="tvm_ffi", target="cuda", target_host=target_host)
    source = kernel.get_host_source()
    suffix = "_packed" if kernel.target_host.kind.name == "c" else ""
    assert f"__tvm_cuda_stream_set_access_policy_window{suffix}" in source
    assert f"__tvm_cuda_stream_reset_access_policy_window{suffix}" in source
    assert "half*" not in source
    assert "((half*)" not in source


if __name__ == "__main__":
    tilelang.testing.main()
