class TileDevice:
    """
    Represents the architecture of a computing device, capturing various hardware specifications.
    """

    def __init__(self) -> None:
        self.reg_cap: int = 0  # Register capacity: The amount of register memory available
        self.smem_cap: int = 0  # Shared memory capacity: The amount of shared memory available
        self.compute_max_core: int = 0  # The maximum number of computing cores
        self.warp_size: int = 0  # The size of a warp, a group of threads that execute instructions in lockstep
        self.sm_partition: int = 0  # The number of streaming multiprocessor partitions
        self.transaction_size: list[int] = [
            0,
            0,
        ]  # The size of memory transactions, typically in bytes
        self.max_smem_usage: int = 0  # The maximum shared memory usage allowed
        self.bandwidth: list[int] = [
            0,
            0,
        ]  # Bandwidth specifications, possibly including peak and sustained rates
        self.platform: str = "unknown"  # The platform or manufacturer of the device
        self.compute_capability: str = "unknown"  # The compute capability, indicating the feature set and performance level
        self.l2_cache_size_bytes: int = 0
        # the number of transaction size in bytes
        self.transaction_size: list[int] = [0, 0]  # in bytes
        # bandwidth in MB/s, will be used for recommend basic tile size
        self.bandwidth: list[int] = [0, 0]

    def get_avaliable_tensorintrin_shapes(self):
        raise NotImplementedError()
