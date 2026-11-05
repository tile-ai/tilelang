"""Invalid CUDA geometry raises before launch and leaves the context usable."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", ["tvm_ffi", "cython", "nvrtc"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_zero_grid_is_rejected_without_launching(backend, dynamic):
    extent = T.dynamic("extent") if dynamic else 4
    grid = extent if dynamic else 0

    @T.prim_func
    def kernel(output: T.Tensor((extent,), "float32")):
        with T.Kernel(grid, threads=1) as block:
            output[block] = T.float32(1)

    # A static zero grid is also a runtime argument error: compilation itself
    # must succeed, as it does for ordinary native launch fields.
    compiled = tilelang.compile(kernel, target="cuda", target_host="c", execution_backend=backend)
    if dynamic:
        positive = torch.zeros(4, device="cuda")
        compiled(positive)
        torch.cuda.synchronize()
        torch.testing.assert_close(positive, torch.ones_like(positive), rtol=0, atol=0)
    output = torch.zeros(0 if dynamic else 4, device="cuda")
    with pytest.raises((ValueError, RuntimeError), match="positive"):
        compiled(output)
    # An illegal memory access swallowed by a generic raises(Exception) must
    # never count as a correct rejection. Synchronization must remain healthy.
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    healthy = torch.arange(8, device="cuda")
    torch.testing.assert_close(healthy + 1, torch.arange(1, 9, device="cuda"), rtol=0, atol=0)
