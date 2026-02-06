from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T

tilelang.testing.set_random_seed()


def _make_shared_reduce(M, N, dtype, reduce_cb):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)

            T.copy(A, A_shared)
            reduce_cb(T, A_shared, B_shared)
            T.copy(B_shared, B)

    return main


def _run_program(program, ref_program, atol=1e-2, rtol=1e-2):
    jit_kernel = tl.compile(program, out_idx=-1)
    profiler = jit_kernel.get_profiler()
    profiler.assert_allclose(ref_program, atol=atol, rtol=rtol)


def reduce_max_test(M, N, dtype=T.float16):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            T.reduce_max(A_local, B_local, dim=1)
            T.copy(B_local, B)

    return main


def reduce_sum_test(M, N, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            T.reduce_sum(A_local, B_local, dim=1)
            T.copy(B_local, B)

    return main


def reduce_sum_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_sum(src, dst, dim=1))


def reduce_max_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_max(src, dst, dim=1))


def reduce_min_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_min(src, dst, dim=1))


def reduce_abssum_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_abssum(src, dst, dim=1))


def reduce_absmax_ss(M, N, dtype=T.float32):
    return _make_shared_reduce(M, N, dtype, lambda T, src, dst: T.reduce_absmax(src, dst, dim=1))


def run_reduce_sum(M, N, dtype=T.float32, mode="rr"):
    if mode == "rr":
        program = reduce_sum_test(M, N, dtype)
    elif mode == "ss":
        program = reduce_sum_ss(M, N, dtype)
    else:
        raise NotImplementedError("run_reduce_sum only supports rr and ss")
    _run_program(program, lambda A: A.sum(dim=1))


def run_shared_reduce(program_builder, ref_program, M, N, dtype=T.float32):
    program = program_builder(M, N, dtype)
    _run_program(program, ref_program)


def run_reduce_max(M, N, dtype=T.float16):
    program = reduce_max_test(M, N, dtype)
    _run_program(program, lambda A: A.max(dim=1).values, atol=1e-2, rtol=1e-2)


def test_reduce_sum():
    run_reduce_sum(256, 256)
    run_reduce_sum(512, 128)
    run_reduce_sum(128, 512)


def test_reduce_sum_shared():
    run_reduce_sum(64, 64, mode="ss")


def test_reduce_max():
    run_reduce_max(256, 256, T.float16)
    run_reduce_max(512, 128, T.float16)
    run_reduce_max(256, 256, T.float32)


def test_reduce_max_shared():
    run_shared_reduce(reduce_max_ss, lambda A: A.max(dim=1).values, 64, 64, T.float32)


def test_reduce_min_shared():
    run_shared_reduce(reduce_min_ss, lambda A: A.min(dim=1).values, 64, 64, T.float32)


def test_reduce_abssum_shared():
    run_shared_reduce(reduce_abssum_ss, lambda A: A.abs().sum(dim=1), 64, 64, T.float32)


def test_reduce_absmax_shared():
    run_shared_reduce(reduce_absmax_ss, lambda A: A.abs().max(dim=1).values, 64, 64, T.float32)


def reduce_sum_test_clear(M, N, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            T.fill(B_local, 1)
            T.reduce_sum(A_local, B_local, dim=1, clear=False)
            T.copy(B_local, B)

    return main


def run_reduce_sum_clear(M, N, dtype=T.float32):
    program = reduce_sum_test_clear(M, N, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)

    def ref_program(A):
        return A.sum(dim=1) + 1

    import torch

    dummy_A = torch.randn((M, N), dtype=getattr(torch, dtype)).cuda()
    ref_out = ref_program(dummy_A)
    tl_out = jit_kernel(dummy_A)
    torch.testing.assert_close(tl_out, ref_out, atol=1e-2, rtol=1e-2)


def test_reduce_sum_clear():
    run_reduce_sum_clear(256, 256, T.float32)
    run_reduce_sum_clear(512, 128, T.float32)
    run_reduce_sum_clear(128, 512, T.float32)


def reduce_max_test_clear(M, N, dtype=T.float16):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            T.fill(B_local, -T.infinity(dtype))
            T.reduce_max(A_local, B_local, dim=1, clear=False)
            T.copy(B_local, B)

    return main


def run_reduce_max_clear(M, N, dtype=T.float16):
    program = reduce_max_test_clear(M, N, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)

    def ref_program(A):
        return A.max(dim=1).values

    import torch

    dummy_A = torch.randn((M, N), dtype=getattr(torch, dtype)).cuda()
    ref_out = ref_program(dummy_A)
    tl_out = jit_kernel(dummy_A)
    torch.testing.assert_close(tl_out, ref_out, atol=1e-2, rtol=1e-2)


def test_reduce_max_clear():
    run_reduce_max_clear(256, 256, T.float16)


if __name__ == "__main__":
    tilelang.testing.main()
