"""Regression test for GitHub issue #2628.

Allreduce with dtypes such as int8, int16, bfloat16 can produce incorrect
results when the warp-synchronous reduction is not properly synchronized.
This test ensures that the warp-synchronous reduction is properly synchronized
for such dtypes, and that the results are correct.
"""

import torch
import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def make_kernel(reduce_threads, dtype, reduce_op, reduce_id):
    @tl.jit(pass_configs={tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
    def kernel(A):
        K = T.const("K")
        A: T.Tensor((K,), dtype)
        C = T.empty((1,), dtype)
        with T.Kernel(1, threads=(reduce_threads,)):
            tk = T.get_thread_binding(0)  # reduce dim = threadIdx.x, single warp
            v = T.alloc_local((1,), dtype)
            v[0] = A[tk]
            Cr = T.alloc_local((1,), dtype)
            with T.attr(T.comm_reducer(reduce_op, [T.cast(reduce_id, dtype)]), "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle")):
                T.evaluate(T.tvm_thread_allreduce(T.uint32(1), v[0], True, Cr[0], tk, dtype="handle"))
            C[0] = Cr[0]
        return C

    return kernel


INT_VALS = [10, 25, 40, 30, 15, 42, 20, 20]  # sum = 202 (-54 for int8)
BF16_VALS = [3.0, 8.0, 5.0, 2.0, 7.0, 6.0, 4.0, 6.0]  # sum = 41 (exact in bf16)


@pytest.mark.parametrize(
    "dt,vals",
    [
        ("float32", INT_VALS),
        ("int32", INT_VALS),
        ("float16", INT_VALS),
        ("bfloat16", BF16_VALS),
        ("int16", INT_VALS),
        ("int8", INT_VALS),
    ],
)
def test_thread_allreduce_syncwarp_sum(dt, vals):
    dtype = getattr(torch, dt)
    A = torch.tensor(vals, device="cuda").to(dtype)
    got = make_kernel(8, dt, lambda x, y: x + y, 0)(A).to(dtype)[0].item()
    ref = A.sum().to(dtype).item()
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "dt,vals",
    [
        ("float32", INT_VALS),
        ("int32", INT_VALS),
        ("float16", INT_VALS),
        ("bfloat16", BF16_VALS),
        ("int16", INT_VALS),
        ("int8", INT_VALS),
    ],
)
def test_thread_allreduce_syncwarp_mul(dt, vals):
    dtype = getattr(torch, dt)
    A = torch.tensor(vals, device="cuda").to(dtype)
    got = make_kernel(8, dt, lambda x, y: x * y, 1)(A).to(dtype)[0].item()
    ref = torch.prod(A).to(dtype).item()
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
