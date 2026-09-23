"""fp8 vector add, swept over the storage dtype and the per-thread element count.

A thread owns ``num_fp8_per_thread`` fp8 elements, so the width the vectorizer
picks ranges from a single fp8 up to a full small vector. Ascend carries small
fp8 vectors as packed integer typedefs with no arithmetic, so every width past
one element goes through the lane-wise scalarization in the Ascend codegen
rather than a native vector add. Regressing that emits a packed *integer* add
and silently produces wrong values, so each width and dtype is pinned here.
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


THREADS = 128
TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]

# (TileLang dtype, torch dtype)
FP8_DTYPES = [
    pytest.param(T.float8_e4m3fn, torch.float8_e4m3fn, id="e4m3"),
    pytest.param(T.float8_e5m2, torch.float8_e5m2, id="e5m2"),
]


def fp8_vecadd_kernel(n: int, dtype, target="ascend"):
    @tilelang.jit(out_idx=[2], target=target)
    def kernel_factory(n_val: int, dtype_val):
        @T.prim_func
        def main(
            A: T.Tensor((n_val,), dtype_val),
            B: T.Tensor((n_val,), dtype_val),
            C: T.Tensor((n_val,), dtype_val),
        ):
            with T.Kernel(1):
                a_ub = T.alloc_shared((n_val,), dtype_val)
                b_ub = T.alloc_shared((n_val,), dtype_val)
                c_ub = T.alloc_shared((n_val,), dtype_val)

                T.copy(A, a_ub)
                T.copy(B, b_ub)
                with T.SimtVF(threads=THREADS):
                    a_local = T.alloc_fragment((n_val,), dtype_val)
                    b_local = T.alloc_fragment((n_val,), dtype_val)
                    c_local = T.alloc_fragment((n_val,), dtype_val)

                    T.copy(a_ub, a_local)
                    T.copy(b_ub, b_local)
                    for i in T.Parallel(n_val):
                        c_local[i] = a_local[i] + b_local[i]
                    T.copy(c_local, c_ub)
                T.copy(c_ub, C)

        return main

    return kernel_factory(n, dtype)


def _raw_prefix(tensor: torch.Tensor, count: int = 16) -> str:
    try:
        return str(tensor.view(torch.uint8)[:count].cpu())
    except Exception:
        return "<raw view unavailable>"


def run_case(num_fp8_per_thread: int, tl_dtype, torch_dtype, target="ascend") -> bool:
    n = THREADS * num_fp8_per_thread

    kernel = fp8_vecadd_kernel(n, tl_dtype, target=target)

    a_f32 = torch.linspace(-128.0, 128.0, n, device="npu", dtype=torch.float32)
    b_f32 = torch.linspace(64.0, -64.0, n, device="npu", dtype=torch.float32)
    a = a_f32.to(torch_dtype)
    b = b_f32.to(torch_dtype)

    c = kernel(a, b)
    torch.npu.synchronize()

    ref = (a.to(torch.float32) + b.to(torch.float32)).to(torch_dtype)
    c_f32 = c.to(torch.float32)
    ref_f32 = ref.to(torch.float32)
    ok = torch.equal(c_f32, ref_f32)

    if not ok:
        print(f"\n=== {torch_dtype}, num_fp8_per_thread={num_fp8_per_thread}, n={n} ===")
        print(kernel.get_kernel_source())
        mismatch = torch.nonzero(c_f32 != ref_f32).flatten()
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        print(f"first mismatch index: {first}")
        print(f"a f32: {a.to(torch.float32)[first : first + 8].cpu()}")
        print(f"b f32: {b.to(torch.float32)[first : first + 8].cpu()}")
        print(f"tilelang f32: {c_f32[first : first + 8].cpu()}")
        print(f"torch f32: {ref_f32[first : first + 8].cpu()}")
        print(f"tilelang raw: {_raw_prefix(c)}")
        print(f"torch raw: {_raw_prefix(ref)}")
    return ok


@pytest.mark.parametrize("tl_dtype, torch_dtype", FP8_DTYPES)
@pytest.mark.parametrize("num_fp8_per_thread", [16, 8, 4, 2, 1])
@pytest.mark.parametrize("target", TARGETS)
def test_fp8_vecadd(num_fp8_per_thread, tl_dtype, torch_dtype, target):
    assert run_case(num_fp8_per_thread, tl_dtype, torch_dtype, target=target), (
        f"{target} {torch_dtype} vector add mismatch at num_fp8_per_thread={num_fp8_per_thread}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
