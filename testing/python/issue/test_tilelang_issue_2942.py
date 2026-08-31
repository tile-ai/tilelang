import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


M, N = 32, 64


def build(off, h):
    @tilelang.jit(out_idx=[-1])
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((M, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=128) as _:
                f = T.alloc_fragment((M, N), "float32")
                T.copy(A, f)
                T.fill(f[off : off + h, :], 7.0)
                T.copy(f, C)

        return main

    return prog


@pytest.mark.parametrize("off, h", [(0, 8), (8, 8), (3, 8)])
@tilelang.testing.requires_cuda
def test_fragment_fill_slice_without_layout_anchor_is_rejected(off, h):
    with pytest.raises(ValueError, match="Fragment buffer slicing is not supported"):
        build(off, h)()


@tilelang.testing.requires_cuda
def test_fragment_copy_slice_without_layout_anchor_is_rejected():
    @tilelang.jit(out_idx=[-2, -1])
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((M, N), "float32"),
            C: T.Tensor((M, N), "float32"),
            D: T.Tensor((8, N), "float32"),
        ):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((M, N), "float32")
                s = T.alloc_shared((8, N), "float32")

                T.copy(A, f)
                T.copy(f[3:11, :], s)
                T.copy(s, D)
                T.copy(f, C)

        return main

    with pytest.raises(ValueError, match="Fragment buffer slicing is not supported"):
        prog()


@tilelang.testing.requires_cuda
def test_explicit_parallel_fragment_slice_is_not_a_region_slice():
    @tilelang.jit(out_idx=[-1])
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((8, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((M, N), "float32")
                for i, j in T.Parallel(8, N):
                    f[i + 3, j] = A[i, j]
                T.copy(f, C)

        return main

    prog()


@tilelang.testing.requires_cuda
def test_fragment_slice_with_manual_layout_anchor_is_allowed():
    fragment_layout = T.Fragment(
        (M, N),
        forward_fn=lambda i, j: (i * 4 + j // 16, j % 16),
    )

    @tilelang.jit(out_idx=[-1])
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((M, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((M, N), "float32")
                T.annotate_layout({f: fragment_layout})

                T.copy(A, f)
                T.fill(f[3:11, :], 7.0)
                T.copy(f, C)

        return main

    prog()


@tilelang.testing.requires_cuda
def test_fragment_slice_with_gemm_layout_anchor_is_allowed():
    size = 16

    @tilelang.jit(out_idx=[-1])
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((size, size), "float16"),
            B: T.Tensor((size, size), "float16"),
            C: T.Tensor((8, size), "float32"),
        ):
            with T.Kernel(1, threads=32):
                a_shared = T.alloc_shared((size, size), "float16")
                b_shared = T.alloc_shared((size, size), "float16")
                partial_shared = T.alloc_shared((8, size), "float32")
                f = T.alloc_fragment((size, size), "float32")

                T.copy(A, a_shared)
                T.copy(B, b_shared)
                T.clear(f)
                T.gemm(a_shared, b_shared, f, transpose_B=True)
                T.copy(f[3:11, :], partial_shared)
                T.copy(partial_shared, C)

        return main

    prog()


@tilelang.testing.requires_cuda
def test_parallel_with_complete_inner_serial_access_is_allowed():
    @tilelang.jit(out_idx=[-1])
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((M, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((M, N), "float32")
                for i in T.Parallel(M):
                    for j in T.serial(N):
                        f[i, j] = A[i, j]
                T.copy(f, C)

        return main

    prog()


@tilelang.testing.requires_cuda
def test_parallel_with_complete_outer_serial_access_is_allowed():
    rows, cols = 4, 256

    @tilelang.jit(out_idx=[-1])
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((rows, cols), "float32"),
            C: T.Tensor((cols,), "float32"),
        ):
            with T.Kernel(1, threads=64):
                f = T.alloc_fragment((rows, cols), "float32")
                out = T.alloc_fragment((cols,), "float32")
                T.copy(A, f)
                T.clear(out)
                for i in T.serial(rows):
                    for j in T.Parallel(cols):
                        out[j] += f[i, j]
                T.copy(out, C)

        return main

    prog()


@tilelang.testing.requires_cuda
def test_replicated_parallel_allows_same_value_global_writes():
    replicated_layout = T.Fragment(
        (1,),
        forward_fn=lambda i, rep: (rep, 0),
        replicate=128,
    )

    @tilelang.jit(out_idx=[-1])
    def prog():
        @T.prim_func
        def main(C: T.Tensor((1,), "float32")):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((1,), "float32")
                T.annotate_layout({f: replicated_layout})
                for i in T.Parallel(1):
                    f[i] = T.float32(1)
                    C[i] = T.float32(1)

        return main

    prog()


if __name__ == "__main__":
    tilelang.testing.main()
