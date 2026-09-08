import pytest
import torch

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
def test_fragment_fill_rectangular_slice_uses_valid_fallback_layout(off, h):
    kernel = build(off, h)()
    a = torch.arange(M * N, dtype=torch.float32, device="cuda").reshape(M, N)
    expected = a.clone()
    expected[off : off + h] = 7.0
    c = kernel(a)
    torch.testing.assert_close(c, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_fragment_fill_non_rectangular_slice_is_rejected():
    with pytest.raises(ValueError, match="No valid layout"):
        build(8, 16)()


@tilelang.testing.requires_cuda
def test_fragment_copy_read_slice_uses_valid_fallback_layout():
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

    kernel = prog()
    a = torch.arange(M * N, dtype=torch.float32, device="cuda").reshape(M, N)
    c, d = kernel(a)
    torch.testing.assert_close(c, a, rtol=0, atol=0)
    torch.testing.assert_close(d, a[3:11], rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_fragment_copy_non_rectangular_write_slice_is_rejected():
    @tilelang.jit
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((M, N), "float32"),
            patch: T.Tensor((16, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((M, N), "float32")
                T.copy(A, f)
                T.copy(patch, f[8:24, :])
                T.copy(f, C)

        return main

    with pytest.raises(ValueError, match="No valid layout"):
        prog()


def build_parallel_fragment_slice(off, h):
    @tilelang.jit
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((M, N), "float32"),
            patch: T.Tensor((h, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((M, N), "float32")
                T.copy(A, f)
                for i, j in T.Parallel(h, N):
                    f[i + off, j] = patch[i, j]
                T.copy(f, C)

        return main

    return prog


def build_unanchored_parallel_fragment_slice(off, h):
    @tilelang.jit(out_idx=[-1])
    def prog():
        @T.prim_func
        def main(
            patch: T.Tensor((h, N), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((M, N), "float32")
                for i, j in T.Parallel(h, N):
                    f[i + off, j] = patch[i, j]
                T.copy(f, C)

        return main

    return prog


@pytest.mark.parametrize("off, h", [(3, 8), (16, 8)])
@tilelang.testing.requires_cuda
def test_explicit_parallel_fragment_slice_uses_valid_fallback_layout(off, h):
    kernel = build_parallel_fragment_slice(off, h)()
    a = torch.arange(M * N, dtype=torch.float32, device="cuda").reshape(M, N)
    patch = -torch.arange(1, h * N + 1, dtype=torch.float32, device="cuda").reshape(h, N)
    expected = a.clone()
    expected[off : off + h] = patch
    c = torch.full_like(a, 777777.0)
    kernel(a, patch, c)
    torch.testing.assert_close(c, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_explicit_parallel_fragment_non_rectangular_slice_is_rejected():
    with pytest.raises(ValueError, match="No valid layout"):
        build_parallel_fragment_slice(8, 16)()


@tilelang.testing.requires_cuda
def test_unanchored_parallel_fragment_rectangular_slice_is_allowed():
    off, h = 3, 8
    kernel = build_unanchored_parallel_fragment_slice(off, h)()
    patch = torch.arange(h * N, dtype=torch.float32, device="cuda").reshape(h, N)
    c = kernel(patch)
    torch.testing.assert_close(c[off : off + h], patch, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_unanchored_parallel_fragment_non_rectangular_slice_is_rejected():
    with pytest.raises(ValueError, match="No valid layout"):
        build_unanchored_parallel_fragment_slice(8, 16)()


@tilelang.testing.requires_cuda
def test_explicit_parallel_fragment_collision_is_rejected():
    @tilelang.jit
    def prog():
        @T.prim_func
        def main(
            A: T.Tensor((16, N), "float32"),
            C: T.Tensor((8, N), "float32"),
        ):
            with T.Kernel(1, threads=128):
                f = T.alloc_fragment((8, N), "float32")
                for i, j in T.Parallel(16, N):
                    f[i % 8, j] = A[i, j]
                T.copy(f, C)

        return main

    with pytest.raises(ValueError, match="not one-to-one"):
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
                # Keep the partial slice aligned with the GEMM accumulator's
                # 8-row ownership period. A shifted slice such as 3:11 also
                # exercises cyclic-layout inversion, which is independent of
                # whether GEMM established a strict layout for the fragment.
                T.copy(f[8:16, :], partial_shared)
                T.copy(partial_shared, C)

        return main

    kernel = prog()
    torch.manual_seed(0)
    a = torch.randn((size, size), dtype=torch.float16, device="cuda")
    b = torch.randn((size, size), dtype=torch.float16, device="cuda")
    c = kernel(a, b)
    expected = (a.float() @ b.float().T)[8:16]
    torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)


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
