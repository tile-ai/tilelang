from tilelang.cache.kernel_cache import KernelCache


class TorchKernelCache(KernelCache):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
