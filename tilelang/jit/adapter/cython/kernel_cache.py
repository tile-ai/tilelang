from tilelang.cache.kernel_cache import KernelCache


class CythonKernelCache(KernelCache):
    @staticmethod
    def _get_base_key():
        # Old launchers merge independent implicit arguments by name.
        return {**KernelCache._get_base_key(), "cython_launch_abi": 1}
