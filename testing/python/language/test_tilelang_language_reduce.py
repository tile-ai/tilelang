from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T

tilelang.testing.set_random_seed()
tilelang.disable_cache()


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


def reduce_test(M, N, dtype=T.float16, op="sum", threads=32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            if op == "sum":
                T.reduce_sum(A_local, B_local, dim=1)
            elif op == "max":
                T.reduce_max(A_local, B_local, dim=1)
            elif op == "min":
                T.reduce_min(A_local, B_local, dim=1)
            elif op == "abssum":
                T.reduce_abssum(A_local, B_local, dim=1)
            elif op == "absmax":
                T.reduce_absmax(A_local, B_local, dim=1)
            elif op == "bitand":
                T.reduce_bitand(A_local, B_local, dim=1)
            elif op == "bitor":
                T.reduce_bitor(A_local, B_local, dim=1)
            elif op == "bitxor":
                T.reduce_bitxor(A_local, B_local, dim=1)
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


def run_reduce(M, N, dtype=T.float32, op="sum", mode="rr", threads=32):
    if mode == "rr":
        program = reduce_test(M, N, dtype, op, threads)
    elif mode == "ss":
        assert op == "sum", f"shared reduce only supports sum, got {op}"
        program = reduce_sum_ss(M, N, dtype)
    else:
        raise NotImplementedError(f"run_reduce only supports rr and ss, got {mode}")

    import torch

    def ref_fn(A):
        if op == "sum":
            res = A.sum(dim=1)
        elif op == "max":
            res = A.max(dim=1).values
        elif op == "min":
            res = A.min(dim=1).values
        elif op == "abssum":
            res = A.abs().sum(dim=1)
        elif op == "absmax":
            res = A.abs().max(dim=1).values
        if A.dtype in [torch.uint32, torch.int32, torch.int64]:
            return res.to(A.dtype)
        return res

    _run_program(program, ref_fn)


def run_shared_reduce(program_builder, ref_program, M, N, dtype=T.float32):
    program = program_builder(M, N, dtype)
    _run_program(program, ref_program)


def run_reduce_max(M, N, dtype=T.float16):
    program = reduce_test(M, N, dtype, "max")
    _run_program(program, lambda A: A.max(dim=1).values, atol=1e-2, rtol=1e-2)


def test_reduce_sum():
    MN_zip = [(256, 256), (512, 128), (128, 512)]
    for dtype in [T.float32, T.int32, T.int64]:
        for M, N in MN_zip:
            run_reduce(M, N, dtype, "sum")


def test_reduce_other_op():
    MN_zip = [(256, 256), (512, 128)]
    for op in ["max", "min", "abssum", "absmax"]:
        for dtype in [T.float32, T.int32, T.int64]:
            for M, N in MN_zip:
                run_reduce(M, N, dtype, op)


def test_reduce_sum_threads():
    run_reduce(32, 32, T.float32, "sum", mode="rr", threads=16)
    run_reduce(16, 16, T.float32, "sum", mode="rr", threads=8)


def test_reduce_sum_shared():
    run_reduce(64, 64, op="sum", mode="ss")


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


def run_reduce_sum_clear(M, N, dtype=T.float32, tl_func=reduce_sum_test_clear):
    program = tl_func(M, N, dtype)
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


def reduce_sum_test_clear_B_shared(M, N, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)

            T.copy(A, A_local)
            T.fill(B_shared, 1)
            T.reduce_sum(A_local, B_shared, dim=1, clear=False)
            T.copy(B_shared, B)

    return main


def test_reduce_sum_clear_B_shared():
    run_reduce_sum_clear(256, 256, T.float32, reduce_sum_test_clear_B_shared)


def reduce_sum_test_clear_AB_shared(M, N, dtype=T.float32):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)

            T.copy(A, A_shared, disable_tma=True)
            T.fill(B_shared, 1)
            T.reduce_sum(A_shared, B_shared, dim=1, clear=False)
            T.copy(B_shared, B)

    return main


def test_reduce_sum_clear_AB_shared():
    run_reduce_sum_clear(64, 64, T.float32, reduce_sum_test_clear_AB_shared)


if __name__ == "__main__":
    tilelang.testing.main()
