import tilelang
import tilelang.language as T
import tilelang.testing


def _get_sig_line(code: str) -> str:
    # Find the kernel signature line in generated CUDA code
    for line in code.splitlines():
        line = line.strip()
        if line.startswith('extern "C" __global__ void'):
            return line
    raise AssertionError("Kernel signature not found in generated code")


@tilelang.testing.requires_cuda
def test_cuda_restrict_default_has_restrict():
    N = 128

    @T.prim_func
    def kernel(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        with T.Kernel(N, threads=32) as pid:
            y[pid] = x[pid] + 1.0

    artifact = tilelang.lower(kernel, target="cuda")
    sig = _get_sig_line(artifact.kernel_source)
    # By default, kNoAlias is set and both pointers are restrict-qualified
    assert "__restrict__" in sig


@tilelang.testing.requires_cuda
def test_cuda_restrict_annotation_removes_restrict():
    N = 128

    @T.prim_func
    def kernel_body_annot(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        # Explicitly mark buffers that may alias as non-restrict
        with T.Kernel(N, threads=32) as pid:
            T.annotate_restrict_buffers(x, y)
            y[pid] = x[pid] + 1.0

    art1 = tilelang.lower(kernel_body_annot, target="cuda")
    sig1 = _get_sig_line(art1.kernel_source)
    # No parameter should be emitted with __restrict__
    assert "__restrict__" not in sig1


if __name__ == "__main__":
    tilelang.testing.main()
