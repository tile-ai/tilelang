from tilelang.cache.kernel_cache import KernelCache


class TorchKernelCache(KernelCache):
    """In-memory-only kernel cache for in-process shader compilation without disk artifacts."""

    _instance = None
    supports_disk_cache = False
