import torch
import tilelang
import tilelang.language as T


def test_int64_address():
    @tilelang.jit
    def set_cache_kernel(
        S,
        D,
        pos_ty="int64",
        dtype=T.float32,
    ):
        @T.prim_func
        def main(
            pos: T.Tensor(
                [
                    S,
                ],
                pos_ty,
            ),  # type: ignore  `TypeError: Check failed: (a.dtype() == b.dtype()) is false: mismatched types. int64 vs. int32`
            value: T.Tensor([S, D], dtype),  # type: ignore
            cache: T.Tensor([S, D], dtype),  # type: ignore
        ):
            with T.Kernel(S, threads=128) as bx:
                slot = pos[bx]
                for i in T.Parallel(D):
                    cache[slot, i] = value[bx, i]

        return main

    D = 2
    S = 10
    cache = torch.rand((S, D), device="cuda", dtype=torch.float32)
    value = torch.rand((S, D), device="cuda", dtype=torch.float32)
    pos_int64 = torch.arange(S, device="cuda", dtype=torch.int64)
    pos_int32 = torch.arange(S, device="cuda", dtype=torch.int32)
    kernel_int64 = set_cache_kernel(S, D, "int64")
    kernel_int32 = set_cache_kernel(S, D, T.int32)
    kernel_int64(pos_int64, value, cache)
    torch.testing.assert_close(cache, value)
    kernel_int32(pos_int32, value, cache)
    torch.testing.assert_close(cache, value)


if __name__ == "__main__":
    tilelang.testing.main()
