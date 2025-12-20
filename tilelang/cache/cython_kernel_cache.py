from tilelang.cache.kernel_cache import KernelCache


class CythonKernelCache(KernelCache):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
