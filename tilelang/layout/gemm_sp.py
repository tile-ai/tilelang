"""Wrapping Layouts."""

# pylint: disable=invalid-name, unsupported-binary-operation
from __future__ import annotations
import tvm
import tilelang.language as T
import warnings

from tilelang.contrib import nvcc
from math import prod


def decompose_col_major(index_1d: int, basis: list[int]) -> list[int]:
    res = []
    for x in basis:
        res.append(index_1d % x)
        index_1d //= x
    return res


def make_cutlass_metadata_layout_sm90(buffer: tvm.tir.Buffer, mma_dtype: str, block_k: int):
    """Make a layout of metadata that is compatible with cutlass sm90 compression kernel. Note that layout atom is the same for smem and gmem.

    Args:
        buffer: metadata buffer shape, for sm90 it should be a 8-bit type
        mma_dtype: dtype of mma operand A, different dtypes result in different layout atom
        block_k: tiling size along K dim, different block_ks results in different layout atom.
    """

    if block_k > 128:
        block_k = 128
        # Ref: https://github.com/NVIDIA/cutlass/blob/c2ad7c5b20f131c4ba33601860f1da3f9c9df0f3/include/cutlass/gemm/collective/builders/sm90_sparse_gmma_builder.inl#L145-L146
        warnings.warn(f"block_k {block_k} is too large, set to 128 for {mma_dtype}.", stacklevel=2)
    if mma_dtype not in [
        T.float16,
        T.bfloat16,
        T.float32,
        T.int8,
        T.float8_e4m3,
        T.float8_e4m3fn,
        T.float8_e4m3fnuz,
        T.float8_e5m2,
        T.float8_e5m2fnuz,
    ]:
        raise NotImplementedError(f"Unsupported dtype: {mma_dtype}")

    if buffer.dtype not in [T.uint8, T.int8]:
        raise ValueError(f"metadata should be 8 bit, got {buffer.dtype}")

    bits_map = {
        "float16": 16,
        "bfloat16": 16,
        "float32": 32,
        "int8": 8,
        "float8_e4m3": 8,
        "float8_e4m3fn": 8,
        "float8_e4m3fnuz": 8,
        "float8_e5m2": 8,
        "float8_e5m2fnuz": 8,
    }

    # ref: https://github.com/NVIDIA/cutlass/blob/c2ad7c5b20f131c4ba33601860f1da3f9c9df0f3/include/cutlass/gemm/collective/builders/sm90_sparse_config.inl#L108-L117
    # get atom layout according to mma dtype
    BlockK = 512 // bits_map[mma_dtype]
    if block_k % BlockK != 0:
        raise ValueError(f"Tile K is too small, which should be at least {BlockK} for {mma_dtype}")
    NumK = block_k // BlockK  # block_k is MinTileShapeK

    def gen_stride(shape_ik, order):
        stride_ik = [None for _ in range(len(shape_ik))]
        order = [(i, o) for i, o in enumerate(order)]
        order.sort(key=lambda x: x[1])
        accu_shape = 1
        for i, (o, _) in enumerate(order):
            if i == 0:
                stride_ik[o] = 1
            else:
                stride_ik[o] = accu_shape
            accu_shape *= shape_ik[o]
        return stride_ik

    if bits_map[mma_dtype] == 32:  # x // 8 is to convert bits into uint8
        shape_ik = [8, 2, 4, 8 // 8, 2, NumK]
        stride_ik = gen_stride(shape_ik, [3, 1, 5, 0, 4, 2])
        shape_i, shape_k = shape_ik[:3], shape_ik[3:]
        stride_i, stride_k = stride_ik[:3], stride_ik[3:]
    elif bits_map[mma_dtype] == 16:
        shape_ik = [8, 2, 4, 16 // 8, 2, NumK]
        stride_ik = gen_stride(shape_ik, [3, 1, 5, 0, 4, 2])
        shape_i, shape_k = shape_ik[:3], shape_ik[3:]
        stride_i, stride_k = stride_ik[:3], stride_ik[3:]
    elif bits_map[mma_dtype] == 8:
        shape_i, shape_k = [64], [block_k // 8]
        stride_i, stride_k = [block_k // 8], [1]
    else:
        raise NotImplementedError(f"Unknown mma type {mma_dtype}")

    shape = buffer.shape

    # repeat to buffer size in col major
    rep_i = (shape[0] + 63) // 64
    rep_k = (shape[1] + prod(shape_k) - 1) // prod(shape_k)
    rep_i_stride = prod(shape_i + shape_k)
    shape_i.append(rep_i)
    stride_i.append(rep_i_stride)
    rep_k_stirde = prod(shape_i + shape_k)
    shape_k.append(rep_k)
    stride_k.append(rep_k_stirde)

    def transform(i: int, k: int) -> int:
        nonlocal shape_i, shape_k, stride_i, stride_k
        i_decomposed = decompose_col_major(i, shape_i)
        k_decomposed = decompose_col_major(k, shape_k)
        i_offset = sum(i_decomposed[k] * stride_i[k] for k in range(len(i_decomposed)))
        k_offset = sum(k_decomposed[k] * stride_k[k] for k in range(len(k_decomposed)))
        return i_offset + k_offset

    return T.Layout(shape, transform)


def make_cutlass_metadata_layout_sm8x(buffer: tvm.tir.Buffer, mma_dtype: str):
    """Make a layout of metadata that is compatible with cutlass sm8x compression kernel. Note that layout atom is the same for smem and gmem.
        ref: https://github.com/pytorch/pytorch/blob/d0c24b392cbb7b213d22e42c52c6c2d1ac2da1bd/torch/sparse/_semi_structured_conversions.py#L5
    Args:
        buffer: metadata buffer shape, for sm80 it should be a 16bit type
    """

    if mma_dtype in [T.float16, T.bfloat16] and buffer.dtype not in [T.uint16, T.int16]:
        raise ValueError(f"metadata should be 16 bit, got {buffer.dtype}")

    if mma_dtype in ["float8_e4m3", "float8_e5m2", T.int8, T.uint8] and buffer.dtype not in [T.uint32, T.int32]:
        raise ValueError(f"metadata should be 32 bit, got {buffer.dtype}")

    m, k = buffer.shape
    group = 32 if buffer.dtype.bits == 16 else 16
    interweave = 4 if buffer.dtype.bits == 16 else 2

    def ColumnMajorInterleaved(i: int, j: int) -> int:
        i = i // group * group + (i % 8) * interweave + (i % group) // 8
        topright = (1 - (i % 2)) & (j % 2)
        bottomleft = (i % 2) & (1 - (j % 2))
        i += topright - bottomleft
        j -= topright - bottomleft
        offset = (j // 2) * m * 2 + i * 2 + (j % 2)
        return offset // k, offset % k

    return T.Layout(buffer.shape, ColumnMajorInterleaved)


def make_cutlass_metadata_layout(buffer: tvm.tir.Buffer, mma_dtype: str = T.float16, arch: str | None = None, **extra_args):
    if arch is None:
        arch = nvcc.get_target_compute_version()

    compute_version = nvcc.parse_compute_version(arch)

    if compute_version >= (9, 0):
        return make_cutlass_metadata_layout_sm90(buffer=buffer, mma_dtype=mma_dtype, **extra_args)
    elif compute_version >= (8, 0):
        return make_cutlass_metadata_layout_sm8x(buffer=buffer, mma_dtype=mma_dtype)
    else:
        raise NotImplementedError(f"Unsupported architecture: {arch}")
