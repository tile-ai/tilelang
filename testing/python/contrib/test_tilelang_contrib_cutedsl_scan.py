import pytest
import torch

import tilelang.testing

try:
    from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

    check_cutedsl_available()
except (ImportError, AssertionError):
    pytest.skip("CuTeDSL not installed", allow_module_level=True)

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from tilelang.contrib.cutedsl import reduce as cutedsl_reduce


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize(
    "shape,dim,reverse,thread_offset",
    [
        ((65,), 0, False, 0),
        ((65,), 0, True, 32),
        ((65, 5), 0, False, 32),
        ((5, 65), 1, True, 32),
    ],
)
def test_cutedsl_scan_thread_offset(op, shape, dim, reverse, thread_offset):
    threads = 64
    scan_cls = getattr(cutedsl_reduce, f"Cum{op.title()}{len(shape)}D")
    if len(shape) == 1:
        scan = scan_cls(threads, reverse, thread_offset)
    else:
        scan = scan_cls(threads, dim, reverse, thread_offset)

    @cute.kernel
    def kernel(src: cute.Tensor, dst: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        # Match the enclosing warp-specialized region emitted by TileLang.
        if tidx >= thread_offset and tidx < thread_offset + threads:
            if cutlass.const_expr(len(shape) == 1):
                scan.run(src.iterator, dst.iterator, shape[0])
            else:
                scan.run(src.iterator, dst.iterator, shape[0], shape[1])

    @cute.jit
    def launch(src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
        kernel(src, dst).launch(grid=(1, 1, 1), block=(thread_offset + threads, 1, 1), stream=stream)

    x = torch.randint(-2, 3, shape, dtype=torch.float32, device="cuda")
    out = torch.full_like(x, float("nan"))
    launch(from_dlpack(x), from_dlpack(out), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()

    ref_input = x.flip([dim]) if reverse else x
    ref = ref_input.cumsum(dim) if op == "sum" else ref_input.cummax(dim).values
    if reverse:
        ref = ref.flip([dim])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
