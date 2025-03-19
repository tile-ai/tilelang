# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm.script import tir as T


def alloc_shared(shape, dtype, scope="shared.dyn"):
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_local(shape, dtype, scope="local"):
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_fragment(shape, dtype, scope="local.fragment"):
    return T.alloc_buffer(shape, dtype, scope=scope)
