import warnings
from tilelang.profiler.bench import _bench_with_cupti
import tilelang.testing


@tilelang.testing.requires_cuda
def test_cupti_counts_torch_zeros_kernel_time():
    import torch

    cache = torch.empty(1024 * 1024, dtype=torch.int, device="cuda")

    def fn():
        return torch.zeros(cache.numel(), dtype=cache.dtype, device=cache.device)

    cache.zero_()
    fn()
    torch.cuda.synchronize()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        assert _bench_with_cupti(fn, cache, n_repeat=3) > 0


@tilelang.testing.requires_cuda
def test_cupti_empty_function_excludes_cache_flush_time():
    import torch

    cache = torch.empty(1024 * 1024, dtype=torch.int, device="cuda")

    def fn():
        pass

    cache.zero_()
    torch.cuda.synchronize()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        assert _bench_with_cupti(fn, cache, n_repeat=3) == 0.0


if __name__ == "__main__":
    tilelang.testing.main()
