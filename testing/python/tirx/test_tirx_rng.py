import os

import pytest


@pytest.mark.skipif(os.environ.get("TILELANG_LIGHT_IMPORT") == "1", reason="RNG execution requires the full frontend")
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "nvrtc"])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("generator", ["curandStatePhilox4_32_10_t", "curandStateMRG32k3a_t", "curandStateXORWOW_t"])
def test_native_rng_state_advances_and_is_reproducible(execution_backend, distribution, generator):
    import torch
    import tilelang
    import tilelang.language as T

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for RNG validation")

    @T.prim_func
    def generate(output: T.Tensor((16384,), "float32")):
        with T.Kernel(32, threads=128) as block:
            values = T.alloc_fragment((512,), "float32")
            T.rng_init(123, generator=generator)
            for index in T.Parallel(512):
                values[index] = T.rng_rand_float(dist=distribution)
            T.copy(values, output[block * 512 : block * 512 + 512])

    kernel = tilelang.compile(generate, target="cuda", execution_backend=execution_backend)
    output = torch.empty(16384, dtype=torch.float32, device="cuda")
    replay = torch.empty_like(output)
    kernel(output)
    kernel(replay)
    torch.testing.assert_close(output, replay, rtol=0, atol=0)
    assert torch.isfinite(output).all()
    assert output.unique().numel() > output.numel() * 0.95
    if distribution == "uniform":
        assert ((output > 0) & (output <= 1)).all()
        assert abs(output.mean().item() - 0.5) < 0.03
    else:
        assert abs(output.mean().item()) < 0.05
        assert abs(output.var().item() - 1.0) < 0.1
