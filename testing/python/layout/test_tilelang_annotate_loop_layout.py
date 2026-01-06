import tilelang
import tilelang.testing
import tilelang.language as T


# TODO(lei): replicate loop layout and more complicated layout cases
@tilelang.lazy_jit
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
    A = T.Tensor((M, N), T.float32)
    B = T.Tensor((M, N), T.float32)
    loop_layout = T.Fragment((M, N), forward_fn=loop_layout_fn)
    kernel = loop_layout_kernel.compile(A=A, B=B, loop_layout=loop_layout)
    code = kernel.get_kernel_source()

    # Expect vectorized copy along innermost dimension (float4)
    assert "*(float4*)(B + ((i * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(A + ((i * 512) + (((int)threadIdx.x) * 4)));" in code


@tilelang.testing.requires_cuda
def test_loop_layout_identity():
    def loop_layout_fn(i, j):
        forward_thread = i
        forward_local = j
        return forward_thread, forward_local

    M, N = 128, 32
    A = T.Tensor((M, N), T.float32)
    B = T.Tensor((M, N), T.float32)
    loop_layout = T.Fragment((M, N), forward_fn=loop_layout_fn)
    kernel = loop_layout_kernel.compile(A=A, B=B, loop_layout=loop_layout)
    code = kernel.get_kernel_source()

    assert "*(float4*)(B + ((((int)threadIdx.x) * 32) + (i * 4))) = *(float4*)(A + ((((int)threadIdx.x) * 32) + (i * 4)));" in code


@tilelang.lazy_jit
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
    A = T.Tensor((M, N), T.float32)
    B = T.Tensor((M, N), T.float32)
    loop_layout = T.Fragment((M, N), forward_fn=loop_layout_fn, replicate=2)
    kernel = copy_with_layout_kernel.compile(A=A, B=B, loop_layout=loop_layout)
    code = kernel.get_kernel_source()

    assert "*(float4*)(B + ((i * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(A + ((i * 512) + (((int)threadIdx.x) * 4)));" in code


@tilelang.lazy_jit
def replicate_loop_layout_kernel(A, B, loop_layout):
    M, N = T.const("M, N")
    A: T.Tensor[(M, N), T.float32]
    B: T.Tensor[(M, N), T.float32]

    with T.Kernel(1, threads=128):
        for i, j in T.Parallel(M, N, loop_layout=loop_layout):
            B[i, j] = A[i, j]


@tilelang.testing.requires_cuda
def test_annotate_replicate_loop_layout_vec4():
    M, N = 128, 32
    A = T.Tensor((M, N), T.float32)
    B = T.Tensor((M, N), T.float32)

    def loop_layout_fn(i, j, rep):
        elems = i * 32 + j
        forward_thread = (elems // 4) % 64 + rep * 64
        forward_local = elems % 4 + (elems // (64 * 4)) * 4
        return forward_thread, forward_local

    loop_layout = T.Fragment((M, N), forward_fn=loop_layout_fn, replicate=2)

    kernel = replicate_loop_layout_kernel.compile(A, B, loop_layout=loop_layout)
    code = kernel.get_kernel_source()

    assert (
        "*(float4*)(B + ((i * 256) + ((((int)threadIdx.x) & 63) * 4))) = *(float4*)(A + ((i * 256) + ((((int)threadIdx.x) & 63) * 4)));"
        in code
    )


if __name__ == "__main__":
    tilelang.testing.main()
