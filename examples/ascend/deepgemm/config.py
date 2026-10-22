"""Host descriptors and kernel configuration for DeepGEMM examples."""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

import tilelang  # noqa: F401  # Register the native library directory.
from tilelang_deepgemm_heuristics import select_gemm_config as _select_gemm_config


class Major(enum.Enum):
    K = 0
    MN = 1


class GemmType(enum.Enum):
    Normal = 0
    MGroupedContiguousWithPsumLayout = 1
    KGroupedContiguousWithPsumLayout = 2
    Batched = 3


def ceil_div(a, b):
    """Ceiling division for integer metadata and device tensors."""
    return (a + b - 1) // b


@dataclass(frozen=True)
class GemmDesc:
    """GEMM metadata for configuration selection; tensor storage stays in the launch arguments.

    Operand strides count packed elements, including packed bytes for FP4.
    """

    m: int
    n: int
    k: int
    a_dtype: str
    b_dtype: str
    cd_dtype: str
    major_a: Major = Major.K
    major_b: Major = Major.K
    acc: bool = False
    gemm_type: GemmType = GemmType.Normal
    num_groups: int = 0
    expected_m: int = 0
    expected_k: int = 0
    num_cores: int = 32
    epilogue_kind: str = "identity"
    outer_stride_a: int = 0  # Zero means tightly packed.
    outer_stride_b: int = 0
    outer_stride_d: int = 0
    output_alignment: int = 0  # Address modulo 32 bytes.
    compiled_dims: str = "nk"
    stride_batch_a: int = 0
    stride_batch_b: int = 0


@dataclass(frozen=True)
class GemmConfig:
    """Resolved tile geometry, pipeline stages, L2 cache hints, and launch size.

    num_blocks specifies the number of launched AI cores.
    """

    block_m: int
    block_n: int
    block_k: int
    mad_m: int
    mad_n: int
    mad_k: int
    num_l1_stages: int
    num_l0_stages: int
    num_epilogue_stages: int
    num_l1_sf_stages: int
    sf_k_blocks: int
    l2_ctrl_a: int
    l2_ctrl_b: int
    l2_ctrl_store_cd: int
    num_blocks: int
    direct_store: bool = False
    dual_aiv_dequant: bool = False
    resident_a: bool = False
    resident_b: bool = False
    resident_sf: bool = False


def select_gemm_config(d: GemmDesc) -> GemmConfig:
    """Return the DeepGEMM-Ascend heuristic result before applying execution limits."""
    return _select_gemm_config(d, GemmConfig)


def execution_config(d: GemmDesc, config: GemmConfig) -> GemmConfig:
    """Adapt the selected configuration to the supported kernel features."""
    if d.a_dtype not in ("bfloat16", "float8_e4m3fn") or d.a_dtype != d.b_dtype:
        # TODO: implement native FP4 and FP8 x FP4 dequant kernels separately.
        raise NotImplementedError("FP4 and FP8 dequant kernels are not implemented")
    num_blocks = config.num_blocks
    if d.gemm_type in (GemmType.Normal, GemmType.Batched):
        # Launch only cores that own an output tile.
        num_batches = d.num_groups if d.gemm_type == GemmType.Batched else 1
        num_tiles = ceil_div(d.m, config.block_m) * ceil_div(d.n, config.block_n) * num_batches
        num_blocks = min(num_blocks, num_tiles)
    if (
        num_blocks == config.num_blocks
        and not (config.resident_a or config.resident_b or config.resident_sf)
        and config.num_epilogue_stages <= 8
    ):
        return config
    num_l1_stages = config.num_l1_stages
    if config.resident_a and d.a_dtype == "bfloat16":
        # Resident-A can enlarge the B ring; streaming A and B must both fit.
        tile_bytes = (config.block_m + config.block_n) * config.block_k * 2
        num_l1_stages = min(num_l1_stages, 512 * 1024 // tile_bytes)
    return replace(
        config,
        num_blocks=num_blocks,
        # TODO: separate retained L1 operand/SF slots from advancing load/event rings.
        resident_a=False,
        resident_b=False,
        resident_sf=False,
        num_l1_stages=num_l1_stages,
        # TODO: map 16 intra-block stages onto 8 reusable PIPE event IDs.
        # Keep the unclamped value in select_gemm_config for parity checks.
        num_epilogue_stages=min(config.num_epilogue_stages, 8),
    )


@dataclass(frozen=True)
class TransformSFConfig:
    """Scale tile geometry, L2 cache hints, and vector-core launch size."""

    src_block_m: int
    src_block_k: int
    l2_ctrl_load: str
    l2_ctrl_store: str
    num_blocks: int


def select_transform_sf_config(is_float, major, src_shape, gemm_type, gran_mn, num_cores, input_stride_bytes) -> TransformSFConfig:
    num_batches, src_m, src_k = src_shape
    pack = 2 if is_float else 1
    if major == Major.K:
        src_block_m, src_block_k = (128 if gran_mn <= 4 else 256) // gran_mn, 128
    elif gran_mn == 1:
        src_block_m, src_block_k = (
            (512, 16 * pack) if gemm_type == GemmType.MGroupedContiguousWithPsumLayout or src_m <= 512 else (8192, pack)
        )
    else:
        src_block_m, src_block_k = min(512, 8192 // gran_mn), 4 * pack
    num_tiles = num_batches * ceil_div(src_m, src_block_m) * ceil_div(src_k, src_block_k)
    return TransformSFConfig(
        src_block_m=src_block_m,
        src_block_k=src_block_k,
        l2_ctrl_load="NORMAL_FV" if input_stride_bytes <= 32 else "NOTALLOC_KEEP",
        l2_ctrl_store="NORMAL_FV",
        num_blocks=min(num_cores, num_tiles),
    )
