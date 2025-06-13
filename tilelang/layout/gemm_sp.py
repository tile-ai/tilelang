# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

import tvm
import tilelang.language as T
import warnings

from typing import List
from math import prod


def decompose_col_major(index_1d: int, basis: List[int]) -> List[int]:
    res = []
    for x in basis:
        res.append(index_1d % x)
        index_1d //= x
    return res


def __make_metadata_layout_sm90_cutlass(buffer: tvm.tir.Buffer, mma_dtype: str, block_k: int):
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


def make_metadata_layout(buffer: tvm.tir.Buffer,
                         mma_dtype: str = "float16",
                         arch: str = "sm90",
                         backend: str = 'cutlass',
                         **extra_args):
    if arch == "sm90":
        if backend == 'cutlass':
            return __make_metadata_layout_sm90_cutlass(buffer, mma_dtype, **extra_args)
        else:
            raise NotImplementedError(f"Arch {arch}, Unsupported backend: {backend}")
    else:
        raise NotImplementedError(f"Unsupported architecture: {arch}")
