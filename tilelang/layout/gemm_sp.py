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


def _make_metadata_layout_sm90_cutlass(buffer: tvm.tir.Buffer, mma_dtype: str, block_k: int):
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
    if mma_dtype not in ["float16", "bfloat16", "float32", "int8", "float8"]:
        raise NotImplementedError(f"Unsupported dtype: {mma_dtype}")

    if buffer.dtype not in ["uint8", "int8"]:
        raise ValueError(f"metadata should be 8 bit, got {buffer.dtype}")

    bits_map = {
        "float16": 16,
        "bfloat16": 16,
        "float32": 32,
        "int8": 8,
        "float8": 8,
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
        shape_i, shape_k = [64], [BlockK]
        stride_i, stride_k = [BlockK], [1]
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


def _make_metadata_layout_sm8x_cutlass(buffer: tvm.tir.Buffer, mma_dtype: str):
    """Make a layout of metadata that is compatible with cutlass sm8x compression kernel. Note that layout atom is the same for smem and gmem.

    Args:
        buffer: metadata buffer shape, for sm80 it should be a 16bit type
    """

    # ref: https://github.com/nvidia/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/gemm/threadblock/default_mma_core_sparse_sm80.h#L651
    #      https://github.com/nvidia/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/layout/matrix.h#L405
    #      https://github.com/nvidia/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/gemm/warp/mma_sparse_tensor_op.h#L172

    if mma_dtype in ["float16", "bfloat16"] and buffer.dtype not in ["uint16", "int16"]:
        raise ValueError(f"metadata should be 16 bit, got {buffer.dtype}")

    if mma_dtype in ["float8", "int8", "uint8"] and buffer.dtype not in ["uint32", "int32"]:
        raise ValueError(f"metadata should be 32 bit, got {buffer.dtype}")

    kInterleaved = 2
    stride = buffer.shape[0] * kInterleaved

    def ColumnMajorInterleaved(i: int, j: int) -> int:
        column_major = j // kInterleaved
        column_minor = j % kInterleaved
        return column_major * stride + i * kInterleaved + column_minor

    return T.Layout(buffer.shape, ColumnMajorInterleaved)


def make_metadata_layout(buffer: tvm.tir.Buffer,
                         mma_dtype: str = "float16",
                         backend: str = 'cutlass',
                         arch: str | None = None,
                         **extra_args):
    if arch is None:
        arch = nvcc.get_target_compute_version()

    compute_version = nvcc.parse_compute_version(arch)

    if compute_version >= (9, 0):
        if backend == 'cutlass':
            return _make_metadata_layout_sm90_cutlass(
                buffer=buffer, mma_dtype=mma_dtype, **extra_args)
        else:
            raise NotImplementedError(f"Arch {arch}, Unsupported backend: {backend}")
    elif compute_version >= (8, 0):
        if backend == 'cutlass':
            return _make_metadata_layout_sm8x_cutlass(buffer=buffer, mma_dtype=mma_dtype)
        else:
            raise NotImplementedError(f"Arch {arch}, Unsupported backend: {backend}")
    else:
        raise NotImplementedError(f"Unsupported architecture: {arch}")
