# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm.script import tir as T


def atomic_add(dst, value):
    return T.call_extern("handle", "AtomicAdd", T.address_of(dst), value)


def atomic_addx2(dst, value):
    return T.call_extern("handle", "AtomicAddx2", T.address_of(dst), T.address_of(value))


def dp4a(A, B, C):
    return T.call_extern("handle", "DP4A", T.address_of(A), T.address_of(B), T.address_of(C))

def clamp(dst, min=None, max=None):
    if min is None and max is None:
        raise ValueError("min and max can't both be None.")
    elif min is not None and max is None:
        return T.max(dst, min)
    elif min is None and max is not None:
        return T.min(dst, max)
    else:
        if min > max:
            raise ValueError("min must be less than or equal to max.")
        return T.min(T.max(dst, min), max)
