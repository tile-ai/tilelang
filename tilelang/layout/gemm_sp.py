# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

import tvm
import tilelang.language as T

from typing import List
from math import prod
from tilelang import _ffi_api


def decompose_col_major(index_1d: int, basis: List[int]) -> List[int]:
    res = []
    for x in basis:
        res.append(index_1d % x)
        index_1d //= x
    return res


def __make_metadata_layout_sm90_cutlass_16bit(buffer: tvm.tir.Buffer):
    assert buffer.dtype in ["uint8", "int8"], f"metadata should be 8 bit, got {buffer.dtype}"

    shape = buffer.shape
    if shape[0] < 64 or shape[1] < 16:
        raise ValueError(f"Buffer shape {shape} is too small for sm90 cutlass 16-bit layout")

    # atom layout
    i_basis = [8, 2, 4]
    j_basis = [2, 2, 4]
    stride_i = [16, 2, 256]
    stride_j = [1, 128, 4]

    # repeat to buffer size
    rep_i = (shape[0] + 63) // 64
    rep_j = (shape[1] + 15) // 16
    rep_i_stride = prod(i_basis + j_basis)
    i_basis.append(rep_i)
    stride_i.append(rep_i_stride)
    rep_j_stirde = prod(i_basis + j_basis)
    j_basis.append(rep_j)
    stride_j.append(rep_j_stirde)

    def transform_sm90_cutlass_16bit(i: int, j: int) -> int:
        nonlocal i_basis, j_basis, stride_i, stride_j
        # (128, 16) in int8
        # E shared layout:
        # (((_8,_2,_4), rep_i),((_2,_2,_4), rep_j)):(((_16,_2,_256), rep_i_stride),((_1,_128,_4), rep_j_stride))
        # rep_i, rep_j are repeated in col-major order
        i_decomposed = decompose_col_major(i, i_basis)
        j_decomposed = decompose_col_major(j, j_basis)
        i_offset = sum(i_decomposed[k] * stride_i[k] for k in range(len(i_decomposed)))
        j_offset = sum(j_decomposed[k] * stride_j[k] for k in range(len(j_decomposed)))
        return i_offset + j_offset

    return T.Layout(
        shape,
        transform_sm90_cutlass_16bit
    )



def __make_metadata_layout_sm90_cutlass(buffer: tvm.tir.Buffer, mma_dtype: str = "float16"):
    if mma_dtype in ["float16", "bfloat16"]:
        return __make_metadata_layout_sm90_cutlass_16bit(buffer)
    else:
        raise NotImplementedError(f"Unsupported dtype: {mma_dtype}")


def make_metadata_layout(buffer: tvm.tir.Buffer, mma_dtype: str = "float16", arch: str = "sm90", backend: str = 'cutlass'):
    if arch == "sm90":
        if backend == 'cutlass':
            return __make_metadata_layout_sm90_cutlass(buffer, mma_dtype)
        else:
            raise NotImplementedError(f"Arch {arch}, Unsupported backend: {backend}")
    else:
        raise NotImplementedError(f"Unsupported architecture: {arch}")
