from __future__ import annotations

from dataclasses import dataclass, field
from tvm.target import Target
from .arch_base import TileDevice
from tilelang.utils.target import is_hexagon_target


class HexagonMemoryScope:
    """String constants for Hexagon-specific TVM memory scopes."""

    DDR = "global"  # off-chip DRAM (host-side tensors)
    VTCM = "global.vtcm"  # on-chip Versatile TCM  (~8 MB on v68+)
    LOCAL_VTCM = "local.vtcm"  # VTCM viewed as thread-local scratch
    HMX_ACC = "global.hmx.acc"  # HMX accumulator register file
    L2_CACHE = "global.l2cache"  # L2 cache hint scope (informational)


@dataclass
class HMXTileShape:
    """Describes one legal HMX MMA tile configuration."""

    M: int  # output rows
    N: int  # output cols
    K: int  # reduction dimension
    a_dtype: str  # input A element type  (e.g. "int8", "uint8", "float16")
    b_dtype: str  # input B element type
    c_dtype: str  # accumulator type       (e.g. "int32", "float32")


# HMX shapes available on Hexagon v73+ (from Qualcomm HKL documentation)
HMX_TILE_SHAPES: list[HMXTileShape] = [
    # Int8 × Int8 → Int32
    HMXTileShape(M=8, N=32, K=32, a_dtype="int8", b_dtype="int8", c_dtype="int32"),
    HMXTileShape(M=16, N=32, K=32, a_dtype="int8", b_dtype="int8", c_dtype="int32"),
    HMXTileShape(M=32, N=32, K=32, a_dtype="int8", b_dtype="int8", c_dtype="int32"),
    # UInt8 × Int8 → Int32  (asymmetric quantisation)
    HMXTileShape(M=8, N=32, K=32, a_dtype="uint8", b_dtype="int8", c_dtype="int32"),
    HMXTileShape(M=32, N=32, K=32, a_dtype="uint8", b_dtype="int8", c_dtype="int32"),
    # Float16 × Float16 → Float32  (v75+ HMX-FP)
    HMXTileShape(M=16, N=16, K=16, a_dtype="float16", b_dtype="float16", c_dtype="float32"),
    HMXTileShape(M=32, N=16, K=16, a_dtype="float16", b_dtype="float16", c_dtype="float32"),
]


@dataclass
class HexagonArch(TileDevice):
    """
    Architecture descriptor for Qualcomm Hexagon DSPs with HMX support.

    Attributes
    ----------
    target : tvm.target.Target
        The TVM target object (must satisfy is_hexagon_target()).
    cpu_version : str
        Hexagon CPU version string, e.g. "hexagonv73", "hexagonv75".
    hvx_vector_bytes : int
        HVX vector width in bytes (128 on v66+).
    vtcm_size_bytes : int
        Total VTCM capacity in bytes (default 8 MB for v73).
    hmx_acc_bits : int
        Total HMX accumulator register file size in bits.
    hmx_tile_shapes : list[HMXTileShape]
        Supported HMX MMA tile configurations.
    """

    target: Target
    cpu_version: str = "hexagonv73"
    hvx_vector_bytes: int = 128  # 1024-bit HVX vectors
    vtcm_size_bytes: int = 8 * 1024 * 1024  # 8 MB default
    hmx_acc_bits: int = 32 * 32 * 32  # conservative accumulator file
    hmx_tile_shapes: list[HMXTileShape] = field(default_factory=lambda: HMX_TILE_SHAPES)

    @property
    def arch(self) -> HexagonArch:  # self IS the arch
        return self

    @property
    def device_name(self) -> str:
        return f"hexagon-{self.cpu_version}"

    @property
    def sm_count(self) -> int:
        # Hexagon has no SMs; return 1 so any code that queries this still works.
        return 1

    @property
    def max_threads_per_block(self) -> int:
        # HVX operates as SIMD over 1 HVX thread; expose 1 "thread"
        return 1

    @property
    def max_shared_memory_per_block(self) -> int:
        return self.vtcm_size_bytes

    def get_hmx_tile(
        self,
        M: int,
        N: int,
        K: int,
        a_dtype: str = "int8",
        b_dtype: str = "int8",
        c_dtype: str = "int32",
    ) -> HMXTileShape | None:
        """
        Return the matching HMXTileShape or None if not natively supported.

        Note: shapes like (64, 64, 32) are NOT native HMX tiles. The caller
        is responsible for tiling into (32, 32, 32) blocks before calling hmx.mma().
        Returns None for any M or N > 32 on int8.
        """
        for ts in self.hmx_tile_shapes:
            if ts.M == M and ts.N == N and ts.K == K and ts.a_dtype == a_dtype and ts.b_dtype == b_dtype and ts.c_dtype == c_dtype:
                return ts
        return None

    def supports_hmx_fp(self) -> bool:
        """True when the target includes FP16 HMX (v75+)."""
        return "v75" in self.cpu_version or "v79" in self.cpu_version


def get_hexagon_arch(target: Target) -> HexagonArch:
    """
    Construct a HexagonArch from a TVM Target.

    Raises
    ------
    ValueError
        If *target* is not a Hexagon target.
    """
    if not is_hexagon_target(target):
        raise ValueError(f"Expected a Hexagon target, got: {target}. Use e.g. tvm.target.Target('llvm -mtriple=hexagon -mcpu=hexagonv73')")

    attrs = target.attrs if hasattr(target, "attrs") else {}
    mcpu = str(attrs.get("mcpu", "hexagonv73")).lower()

    # Derive VTCM size heuristic from version
    vtcm = 8 * 1024 * 1024  # 8 MB baseline (v68/v69/v73)
    if "v75" in mcpu or "v79" in mcpu:
        vtcm = 16 * 1024 * 1024  # 16 MB on newer parts

    return HexagonArch(
        target=target,
        cpu_version=mcpu,
        vtcm_size_bytes=vtcm,
    )
