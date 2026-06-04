import tilelang.language as T
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.layout
import tilelang as tl
import pytest
import torch


def view_test(N, M, dtype, new_dtype=None):
    new_shape = [N // M, M]
    if new_dtype:
        from tvm import DataType

        dtype_src = DataType(dtype)
        dtype_dst = DataType(new_dtype)
        src_bits = dtype_src.bits
        dst_bits = dtype_dst.bits
        scale = src_bits / dst_bits
        new_shape[-1] = int(M * scale)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor(new_shape, new_dtype if new_dtype else dtype),
    ):
        with T.Kernel(1) as _:
            A_viewed = T.view(A, new_shape, dtype=new_dtype)
            T.copy(A_viewed, B)

    return main


def run_view(N, M, dtype, new_dtype=None):
    program = view_test(N, M, dtype, new_dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        if new_dtype:
            torch_dtype = T.dtype(new_dtype).as_torch()
            return A.view(N // M, M).view(dtype=torch_dtype)
        return A.view(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_view():
    # Test view with same dtype
    run_view(1024, 32, T.float32)
    run_view(2048, 64, T.float16)

    # Test view with dtype conversion
    run_view(1024, 32, T.float32, T.float16)
    run_view(2048, 64, T.float16, T.float32)


def view_shape_mismatch_test(N, M, dtype, new_dtype=None):
    new_shape = [N // M, M + 1]
    if new_dtype:
        from tvm import DataType

        dtype_src = DataType(dtype)
        dtype_dst = DataType(new_dtype)
        src_bits = dtype_src.bits
        dst_bits = dtype_dst.bits
        scale = src_bits / dst_bits
        new_shape[-1] = int(M * scale)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor(new_shape, new_dtype if new_dtype else dtype),
    ):
        with T.Kernel(1) as _:
            A_viewed = T.view(A, new_shape, dtype=new_dtype)
            T.copy(A_viewed, B)

    return main


def test_view_shape_mismatch():
    with pytest.raises(AssertionError):
        view_shape_mismatch_test(1024, 32, T.float32)


def test_view_subbyte_dtype_change():
    A = tvm.tirx.decl_buffer((16, 32), "float4_e2m1fn", name="A")
    A_viewed = T.view(A, (16, 16), dtype=T.uint8)
    assert str(A_viewed.dtype) == "uint8"
    assert tuple(int(dim) for dim in A_viewed.shape) == (16, 16)
    assert A_viewed.data.same_as(A.data)


def test_view_accepts_explicit_strides():
    A = tvm.tirx.decl_buffer((4, 8), "float8_e4m3fn", name="A", strides=[16, 1])
    A_viewed = T.view(A, (4, 2), dtype=T.uint32, strides=(8, 1), elem_offset=4)
    assert str(A_viewed.dtype) == "uint32"
    assert tuple(int(dim) for dim in A_viewed.shape) == (4, 2)
    assert tuple(int(stride) for stride in A_viewed.strides) == (8, 1)
    assert int(A_viewed.elem_offset) == 4
    assert A_viewed.data.same_as(A.data)


def _int_tuple(values):
    return tuple(int(value) for value in values)


def test_layout_reshape_preserves_packed_subtype_lane():
    layout = T.Layout((2, 2, 8), lambda a, b, c: [a, b, c])
    u8_view = layout.reshape((2, 2, 16), 16, 8)
    assert _int_tuple(u8_view.get_output_shape()) == (2, 2, 8, 2)
    assert _int_tuple(u8_view.map_forward_index([0, 0, 0])) == (0, 0, 0, 0)
    assert _int_tuple(u8_view.map_forward_index([0, 0, 1])) == (0, 0, 0, 1)
    assert _int_tuple(u8_view.map_forward_index([0, 0, 2])) == (0, 0, 1, 0)

    u16_view = u8_view.reshape((2, 2, 8), 8, 16)
    assert _int_tuple(u16_view.get_output_shape()) == (2, 2, 8)
    assert _int_tuple(u16_view.map_forward_index([0, 0, 1])) == (0, 0, 1)


def test_fragment_reshape_preserves_packed_subtype_lane():
    fragment = T.Fragment(
        (2, 2, 8),
        forward_thread_fn=lambda a, b, c: a * 2 + b,
        forward_index_fn=lambda a, b, c: c,
    )
    u8_view = fragment.reshape((2, 2, 16), 16, 8)
    assert _int_tuple(u8_view.get_output_shape()) == (8, 2)
    assert _int_tuple(u8_view.map_forward_index([0, 0, 0])) == (0, 0)
    assert _int_tuple(u8_view.map_forward_index([0, 0, 1])) == (0, 1)
    assert _int_tuple(u8_view.map_forward_index([0, 0, 2])) == (1, 0)
    assert _int_tuple(u8_view.map_forward_thread([1, 1, 15])) == (3,)

    u16_view = u8_view.reshape((2, 2, 8), 8, 16)
    assert _int_tuple(u16_view.get_output_shape()) == (8,)
    assert _int_tuple(u16_view.map_forward_index([0, 0, 1])) == (1,)
    assert _int_tuple(u16_view.map_forward_thread([1, 1, 7])) == (3,)


def annotated_fragment_layout_on_dtype_changing_view_test():
    @T.prim_func
    def main(B: T.Tensor((4,), T.uint32)):
        with T.Kernel(1, threads=32) as _:
            sf = T.alloc_fragment((16,), T.float8_e4m3fn)
            sf_words = T.view(sf, (4,), dtype=T.uint32)
            T.annotate_layout(
                {
                    sf_words: T.Fragment(
                        (4,),
                        forward_thread_fn=lambda i: i,
                        forward_index_fn=lambda i: 0,
                    )
                }
            )

            for i in T.Parallel(4):
                sf_words[i] = T.Cast(T.uint32, i + 1)
                B[i] = sf_words[i]

    return main


@tilelang.testing.requires_cuda
def test_annotated_fragment_layout_on_dtype_changing_view_compile():
    program = annotated_fragment_layout_on_dtype_changing_view_test()
    kernel = tl.compile(program, out_idx=-1)
    src = kernel.get_kernel_source()
    assert "uint sf_words[1]" in src
    out = kernel()
    torch.testing.assert_close(out.cpu(), torch.tensor([1, 2, 3, 4], dtype=torch.uint32))


def layout_view_ldmatrix_pointer_test():
    @T.prim_func
    def main(
        A: T.Tensor((16, 16), T.float16),
        B: T.Tensor((4,), T.uint32),
    ):
        with T.Kernel(1, threads=32) as _:
            S = T.alloc_shared((16, 16), T.float16)
            V = T.view(S, (16, 16), dtype=T.float16)
            R = T.alloc_local((4,), T.uint32)
            T.annotate_layout_view({V: T.Layout((16, 16), lambda i, j: [j, i])})

            tx = T.get_thread_binding()
            S[tx // 16, tx % 16] = A[tx // 16, tx % 16]
            T.ptx_ldmatrix(
                T.bool(False),
                4,
                T.access_ptr(V[0, 1], "r", extent=16),
                T.access_ptr(R[0], "w", extent=4),
            )
            if tx < 4:
                B[tx] = R[tx]

    return main


def test_layout_view_applies_to_ldmatrix_access_ptr():
    artifact = tl.lower(
        layout_view_ldmatrix_pointer_test(),
        target="cuda",
        enable_device_compile=False,
    )
    source = str(artifact.kernel_source)
    assert "tl::ptx_ldmatrix_x4((&(V[16])), (&(R[0])))" in source


def replicated_fragment_word_view_test():
    @T.prim_func
    def main(B: T.Tensor((32, 4), T.uint32)):
        with T.Kernel(1, threads=32) as _:
            p = T.alloc_fragment((32,), T.float4_e2m1fn)
            p_words = T.view(p, (4,), dtype=T.uint32)
            T.annotate_layout({p_words: tilelang.layout.make_fully_replicated_layout_fragment(p_words, 32)})
            tx = T.get_thread_binding()
            p_words[0] = T.Cast(T.uint32, tx + 1)
            p_words[1] = T.Cast(T.uint32, tx + 2)
            p_words[2] = T.Cast(T.uint32, tx + 3)
            p_words[3] = T.Cast(T.uint32, tx + 4)
            B[tx, 0] = p_words[0]
            B[tx, 1] = p_words[1]
            B[tx, 2] = p_words[2]
            B[tx, 3] = p_words[3]

    return main


def ws_consumer_fragment_word_view_thread_range_test():
    @T.prim_func
    def main(B: T.Tensor((384,), T.uint32)):
        with T.Kernel(1, threads=384) as _:
            tx = T.get_thread_binding()
            role_tx = tx
            scratch = T.alloc_shared((32,), T.uint32)
            with T.ws(0):
                if tx < 32:
                    scratch[tx] = T.Cast(T.uint32, tx)
            with T.ws(1, 2):
                sf = T.alloc_fragment((8,), T.float8_e4m3fn, role_scoped=True)
                sf_words = T.view(sf, (2,), dtype=T.uint32)
                sf_words[0] = T.lds32(scratch[role_tx & 31])
                B[tx] = sf_words[0]

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_replicated_fragment_fp4_word_view_runs():
    program = replicated_fragment_word_view_test()
    kernel = tl.compile(program, out_idx=-1)
    src = kernel.get_kernel_source()
    assert "uint p_words[4]" in src
    out = kernel().cpu().to(torch.int64)
    expected = torch.arange(32, dtype=torch.int64).reshape(32, 1) + torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    torch.testing.assert_close(out, expected)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_ws_consumer_fragment_word_view_uses_consumer_thread_range():
    program = ws_consumer_fragment_word_view_thread_range_test()
    kernel = tl.compile(program, out_idx=-1)
    src = kernel.get_kernel_source()
    assert "load_shared_32" in src
    assert "threadIdx.x) - 256" not in src


def fp4_to_uint8_view_test(rows_per_cta=16, mask_k=256):
    @T.prim_func
    def main(
        A: T.Tensor((rows_per_cta, mask_k), T.bfloat16),
        B: T.Tensor((rows_per_cta, mask_k // 2), T.uint8),
    ):
        with T.Kernel(1, threads=256) as _:
            A_frag = T.alloc_fragment((rows_per_cta, mask_k), T.bfloat16)
            B_shared_fp4 = T.alloc_shared((rows_per_cta, mask_k), T.float4_e2m1fn)
            B_shared_uint8 = T.view(B_shared_fp4, (rows_per_cta, mask_k // 2), dtype=T.uint8)

            T.copy(A, A_frag)
            for i, j in T.Parallel(rows_per_cta, mask_k):
                B_shared_fp4[i, j] = T.cast(A_frag[i, j], T.float4_e2m1fn)
            T.copy(B_shared_uint8, B)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_view_shared_fp4_to_uint8_compile():
    program = fp4_to_uint8_view_test()
    kernel = tl.compile(program, out_idx=-1)
    src = kernel.get_kernel_source()
    assert "fp4_e2" in src

    dummy_input = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    output = kernel(dummy_input)
    assert output.shape == (16, 128)
    assert output.dtype == torch.uint8


def annotated_layout_on_dtype_changing_view_test():
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), T.float16),
        B: T.Tensor((64, 128), T.int8),
    ):
        with T.Kernel(1, threads=128) as _:
            A_stage = T.alloc_shared((2, 64, 64), T.float16, scope="shared.dyn")
            A_i8 = T.view(A_stage, (2, 64, 128), dtype=T.int8)
            T.annotate_layout({A_i8: T.Layout((2, 64, 128), lambda s, i, j: [s, i, j])})

            for i, j in T.Parallel(64, 64):
                A_stage[0, i, j] = A[i, j]

            for i, j in T.Parallel(64, 128):
                B[i, j] = A_i8[0, i, j]

    return main


@tilelang.testing.requires_cuda
def test_annotated_layout_on_dtype_changing_view_compile():
    program = annotated_layout_on_dtype_changing_view_test()
    kernel = tl.compile(program, out_idx=-1)
    assert kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
