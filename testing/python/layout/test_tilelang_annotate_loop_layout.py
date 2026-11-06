import re

import tilelang
import tilelang.testing
import tilelang.language as T


def _assert_vector_copy(kernel, shape, element_index, iterations, replicated=False):
    import torch

    code = kernel.get_kernel_source()
    loop = re.search(r"for \(int (\w+) = 0; \1 < " + str(iterations) + r"; \+\+\1\)", code)
    assert loop is not None, code
    index = element_index(loop.group(1))
    load = f"*(float4*)(A_ptr + {index})"
    store = f"*(float4*)(B_ptr + {index}) = {load};"
    assert store in code
    if replicated:
        # Only the canonical owner writes. Under this guard, the former
        # threadIdx.x & 63 address is exactly threadIdx.x.
        assert "if ((((int)threadIdx.x) >> 6) == 0)" in code
    a = torch.arange(shape[0] * shape[1], dtype=torch.float32, device="cuda").reshape(shape)
    b = torch.full_like(a, -12345.0)
    kernel(a, b)
    torch.testing.assert_close(b, a, rtol=0, atol=0)


@tilelang.jit
def loop_layout_kernel(A, B, loop_layout):
    M, N = T.const("M, N")
    A: T.Tensor[(M, N), T.float32]
    B: T.Tensor[(M, N), T.float32]

    with T.Kernel(1, threads=128):
        for i, j in T.Parallel(M, N, loop_layout=loop_layout):
            B[i, j] = A[i, j]


@tilelang.testing.requires_cuda
def test_loop_layout_fragment_vec4():
    def loop_layout_fn(i, j):
        elems = i * 32 + j
        forward_thread = (elems // 4) % 128
        forward_local = elems % 4 + (elems // 512) * 4
        return forward_thread, forward_local

    M, N = 128, 32
    loop_layout = T.Fragment((M, N), forward_fn=loop_layout_fn)
    kernel = loop_layout_kernel.compile(M=M, N=N, loop_layout=loop_layout)

    # Expect vectorized copy along innermost dimension (float4).
    _assert_vector_copy(kernel, (M, N), lambda i: f"(({i} * 512) + (((int)threadIdx.x) * 4))", 8)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_lt(9, 0)
def test_loop_layout_identity():
    def loop_layout_fn(i, j):
        forward_thread = i
        forward_local = j
        return forward_thread, forward_local

    M, N = 128, 32
    loop_layout = T.Fragment((M, N), forward_fn=loop_layout_fn)
    kernel = loop_layout_kernel.compile(M=M, N=N, loop_layout=loop_layout)
    _assert_vector_copy(kernel, (M, N), lambda i: f"((((int)threadIdx.x) * 32) + ({i} * 4))", 8)


@tilelang.jit
def copy_with_layout_kernel(A, B, loop_layout):
    M, N = T.const("M, N")
    A: T.Tensor[(M, N), T.float32]
    B: T.Tensor[(M, N), T.float32]

    with T.Kernel(1, threads=128):
        T.copy(A, B, loop_layout=loop_layout)


@tilelang.testing.requires_cuda
def test_copy_loop_layout_annotated_replicate_vec4():
    def loop_layout_fn(i, j, rep):
        elems = i * 32 + j
        fth = (elems // 4) % 64 + rep * 64
        floc = elems % 4 + (elems // (64 * 4)) * 4
        return fth, floc

    M, N = 128, 32
    loop_layout = T.Fragment((M, N), forward_fn=loop_layout_fn, replicate=2)
    kernel = copy_with_layout_kernel.compile(M=M, N=N, loop_layout=loop_layout)

    _assert_vector_copy(kernel, (M, N), lambda i: f"(({i} * 256) + (((int)threadIdx.x) * 4))", 16, replicated=True)


@tilelang.jit
def replicate_loop_layout_kernel(A, B, loop_layout):
    M, N = T.const("M, N")
    A: T.Tensor[(M, N), T.float32]
    B: T.Tensor[(M, N), T.float32]

    with T.Kernel(1, threads=128):
        for i, j in T.Parallel(M, N, loop_layout=loop_layout):
            B[i, j] = A[i, j]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_annotate_replicate_loop_layout_vec4():
    M, N = 128, 32

    def loop_layout_fn(i, j, rep):
        elems = i * 32 + j
        forward_thread = (elems // 4) % 64 + rep * 64
        forward_local = elems % 4 + (elems // (64 * 4)) * 4
        return forward_thread, forward_local

    loop_layout = T.Fragment((M, N), forward_fn=loop_layout_fn, replicate=2)
    kernel = replicate_loop_layout_kernel.compile(M=M, N=N, loop_layout=loop_layout)
    _assert_vector_copy(kernel, (M, N), lambda i: f"(({i} * 256) + (((int)threadIdx.x) * 4))", 16, replicated=True)


if __name__ == "__main__":
    tilelang.testing.main()
