import torch
import tilelang
import tilelang.testing
import tilelang.language as T

tilelang.disable_cache()

# The three headline values from issue #2483. Each is one bf16 ULP away
# from a bf16-representable neighbour, so RNE and truncate-toward-zero
# produce different results:
#   RNE  :  259 -> 260, 33000 -> 33024, -259 -> -260
#   trunc:  259 -> 258, 33000 -> 32768, -259 -> -258
ISSUE_2483_VALUES = [259, 33000, -259]
ISSUE_2483_BF16_RNE = [260.0, 33024.0, -260.0]


def test_int32_to_bf16_rounds_to_nearest_even():
    """Faithful port of the issue #2483 reproducer."""
    N = len(ISSUE_2483_VALUES)

    @T.prim_func
    def main(
        A: T.Tensor((N,), "int32"),
        O: T.Tensor((N,), "bfloat16"),
        Oh: T.Tensor((N,), "float16"),
    ):
        with T.Kernel(1, threads=N) as _:
            Al = T.alloc_fragment((N,), "int32")
            Ob = T.alloc_fragment((N,), "bfloat16")
            Ol = T.alloc_fragment((N,), "float16")
            T.copy(A, Al)
            for i in T.Parallel(N):
                Ob[i] = T.Cast("bfloat16", Al[i])  # int -> bf16
                Ol[i] = T.Cast("float16", Al[i])  # int -> fp16
            T.copy(Ob, O)
            T.copy(Ol, Oh)

    kernel = tilelang.compile(main)

    a = torch.tensor(ISSUE_2483_VALUES, dtype=torch.int32, device="cuda")
    o_bf16 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
    o_fp16 = torch.zeros(N, dtype=torch.float16, device="cuda")
    kernel(a, o_bf16, o_fp16)
    torch.cuda.synchronize()

    got = o_bf16.float().tolist()

    assert got == ISSUE_2483_BF16_RNE, f"int->bf16 must round-to-nearest-even; got {got}, expected {ISSUE_2483_BF16_RNE}"

    # Cross-check against torch's native int->bf16 (which is RNE).
    ref = a.to(torch.bfloat16)
    tilelang.testing.torch_assert_close(o_bf16, ref, rtol=0, atol=0)

    ref_fp16 = a.to(torch.float16)
    tilelang.testing.torch_assert_close(o_fp16, ref_fp16, rtol=0, atol=0)


def test_int64_to_bf16_rounds_to_nearest_even():
    N = len(ISSUE_2483_VALUES)

    @T.prim_func
    def main(A: T.Tensor((N,), "int64"), O: T.Tensor((N,), "bfloat16")):
        with T.Kernel(1, threads=N) as _:
            Al = T.alloc_fragment((N,), "int64")
            Ob = T.alloc_fragment((N,), "bfloat16")
            T.copy(A, Al)
            for i in T.Parallel(N):
                Ob[i] = T.Cast("bfloat16", Al[i])
            T.copy(Ob, O)

    kernel = tilelang.compile(main, out_idx=[1])

    a = torch.tensor(ISSUE_2483_VALUES, dtype=torch.int64, device="cuda")
    o = kernel(a)
    ref = a.to(torch.bfloat16)
    tilelang.testing.torch_assert_close(o, ref, rtol=0, atol=0)


def test_int64_to_fp16_compiles_and_matches_torch():
    N = len(ISSUE_2483_VALUES)

    @T.prim_func
    def main(A: T.Tensor((N,), "int64"), O: T.Tensor((N,), "float16")):
        with T.Kernel(1, threads=N) as _:
            Al = T.alloc_fragment((N,), "int64")
            Ol = T.alloc_fragment((N,), "float16")
            T.copy(A, Al)
            for i in T.Parallel(N):
                Ol[i] = T.Cast("float16", Al[i])
            T.copy(Ol, O)

    kernel = tilelang.compile(main, out_idx=[1])

    a = torch.tensor(ISSUE_2483_VALUES, dtype=torch.int64, device="cuda")
    o = kernel(a)
    # Match torch's own int64 -> fp16 (RNE via fp32) exactly.
    ref = a.to(torch.float16)
    tilelang.testing.torch_assert_close(o, ref, rtol=0, atol=0)


def test_float32_to_bf16_still_rounds_to_nearest_even():
    """Regression guard: f32 -> bf16 was already correct and must stay so."""
    N = len(ISSUE_2483_VALUES)

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), O: T.Tensor((N,), "bfloat16")):
        with T.Kernel(1, threads=N) as _:
            Al = T.alloc_fragment((N,), "float32")
            Ob = T.alloc_fragment((N,), "bfloat16")
            T.copy(A, Al)
            for i in T.Parallel(N):
                Ob[i] = T.Cast("bfloat16", Al[i])
            T.copy(Ob, O)

    kernel = tilelang.compile(main, out_idx=[1])

    a = torch.tensor([float(v) for v in ISSUE_2483_VALUES], dtype=torch.float32, device="cuda")
    o = kernel(a)
    ref = a.to(torch.bfloat16)
    tilelang.testing.torch_assert_close(o, ref, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
