import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.autotuner import AutoTuner


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("cache_inputs", [True, False])
@pytest.mark.parametrize("symbolic", [False, True])
def test_autotune_reuses_compatible_inputs(cache_inputs, symbolic):
    supplied = []

    def supply(params):
        inputs = [
            torch.ones(tuple(256 if isinstance(dim, T.Var) else int(dim) for dim in p.shape), dtype=p.torch_dtype(), device="cuda")
            for p in params
        ]
        supplied.append(inputs)
        return inputs

    def kernel(threads):
        size = T.dynamic("n") if symbolic else 256

        @T.prim_func
        def main(A: T.Tensor((size,), T.float32), B: T.Tensor((size,), T.float32)):
            with T.Kernel(T.ceildiv(size, 256), threads=threads) as bx:
                for i in T.Parallel(256):
                    B[bx * 256 + i] = A[bx * 256 + i] + 1

        return main

    result = (
        AutoTuner(kernel, configs=[{"threads": 64}, {"threads": 128}])
        .set_compile_args(out_idx=[1], target="cuda")
        .set_profile_args(supply_prog=supply, ref_prog=lambda x: x + 1, cache_input_tensors=cache_inputs)
        .run(warmup=1, rep=1)
    )
    # The reference benchmark has its own inputs.
    assert len(supplied) == (2 if cache_inputs else 3)
    x = supplied[0][0]
    torch.testing.assert_close(result.kernel(x), x + 1)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("change", ["rank", "extent", "dtype"])
def test_autotune_refreshes_incompatible_inputs(change):
    supplied = []

    def supply(params):
        inputs = [torch.ones(tuple(p.shape), dtype=p.torch_dtype(), device="cuda") for p in params]
        supplied.append(inputs)
        return inputs

    def kernel(variant):
        size = 128 if change == "extent" and variant else 256
        dtype = T.float16 if change == "dtype" and variant else T.float32
        shape = (size, 1) if change == "rank" and variant else (size,)

        @T.prim_func
        def main(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)):
            with T.Kernel(1, threads=128):
                T.copy(A, B)

        return main

    result = (
        AutoTuner(kernel, configs=[{"variant": 0}, {"variant": 1}])
        .set_compile_args(out_idx=[1], target="cuda")
        .set_profile_args(supply_prog=supply, ref_prog=lambda x: x, cache_input_tensors=True)
        .run(warmup=1, rep=1)
    )
    assert len(supplied) == 3
    param = result.kernel.params[0]
    x = torch.ones(tuple(param.shape), dtype=param.torch_dtype(), device="cuda")
    torch.testing.assert_close(result.kernel(x), x)


if __name__ == "__main__":
    tilelang.testing.main()
