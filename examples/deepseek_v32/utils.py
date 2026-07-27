# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import contextlib
import functools
import logging
import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from packaging import version


def _is_equal(a, b):
    if isinstance(a, torch.Tensor):
        return a is b
    # Whitelist of types that are safe to compare by value for caching.
    if isinstance(a, (int, float, str, bool, type(None))) and isinstance(b, (int, float, str, bool, type(None))):
        return a == b
    # For other types, we cannot guarantee a cheap and safe comparison, so we fail the cache check.
    return False


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                # For Tensors, check for object identity. For other types, check for equality.
                # Python caches small integers, so `is` works for them but not for large integers like 4096.
                if (
                    all(_is_equal(a, b) for a, b in zip(args, last_args))
                    and set(kwargs.keys()) == set(last_kwargs.keys())
                    and all(_is_equal(v, last_kwargs[k]) for k, v in kwargs.items())
                ):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def cal_seq_idx_from_cu_seqlens(cu_seqlens: torch.LongTensor, seq_len: int):
    seq_idx = cu_seqlens.new_zeros(seq_len + 1)
    seq_idx.scatter_add_(0, cu_seqlens[1:].long(), torch.ones_like(seq_idx))
    seq_idx.cumsum_(0)
    return seq_idx[:-1]


@tensor_cache
def cal_seq_idx_for_q(cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor, seq_len: int) -> torch.IntTensor:
    seq_idx_for_q = torch.full((seq_len,), len(cu_seqlens_qs), dtype=torch.int32, device=cu_seqlens_qs.device)
    for i in range(len(cu_seqlens_qs)):
        seq_idx_for_q[cu_seqlens_qs[i] : cu_seqlens_qe[i]] = i
    return seq_idx_for_q


@tensor_cache
def cal_cu_seqlen_ks_for_q(
    cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor, cu_seqlens_ks: torch.LongTensor, seq_len: int
) -> torch.IntTensor:
    cu_seqlen_ks_for_each_q = torch.gather(
        input=torch.cat([cu_seqlens_ks, torch.full((1,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=cu_seqlens_qs.device)]),
        dim=0,
        index=cal_seq_idx_for_q(cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
    )
    return cu_seqlen_ks_for_each_q.int()


@tensor_cache
def cal_cu_seqlen_ke_for_q(
    cu_seqlens_qs: torch.LongTensor,
    cu_seqlens_qe: torch.LongTensor,
    cu_seqlens_ks: torch.LongTensor,
    cu_seqlens_ke: torch.LongTensor,
    q_start_idxs: torch.LongTensor,
    seq_len: int,
    kv_stride: int,
) -> torch.IntTensor:
    cu_seqlen_ke_for_each_q = torch.gather(
        input=torch.cat([cu_seqlens_ke, torch.zeros(1, dtype=torch.int32, device=cu_seqlens_qs.device)]),
        dim=0,
        index=cal_seq_idx_for_q(cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
    )
    casual_cu_seqlen_ke_for_each_q = torch.zeros((seq_len,), dtype=torch.int32, device=cu_seqlens_qs.device)
    for i in range(len(cu_seqlens_qs)):
        casual_cu_seqlen_ke_for_each_q[cu_seqlens_qs[i] : cu_seqlens_qe[i]] = (
            torch.arange(
                q_start_idxs[i], q_start_idxs[i] + cu_seqlens_qe[i] - cu_seqlens_qs[i], dtype=torch.int32, device=cu_seqlens_qs.device
            )
            + 1
        ) // kv_stride + cu_seqlens_ks[i]
    cu_seqlen_ke_for_each_q = torch.minimum(casual_cu_seqlen_ke_for_each_q, cu_seqlen_ke_for_each_q)
    return cu_seqlen_ke_for_each_q.int()


@tensor_cache
def cal_ks_ke_from_cu_seqlen_qk(
    cu_seqlens_q: torch.LongTensor,
    cu_seqlens_k: torch.LongTensor = None,
    offs_q: torch.LongTensor = None,
    *,
    seq_len: int,
    kv_stride: int = 1,
    cp_rank: int = 0,
    cp_size: int = 1,
    balanced_cp=False,
):
    """
    seq_len: seq len per cp rank
    balanced cp slice assignment: 0 1 2 3 3 2 1 0
    """
    n_seq = len(cu_seqlens_q) - 1
    assert n_seq > 0
    assert cu_seqlens_q.shape == (n_seq + 1,)
    seq_idx = cal_seq_idx_from_cu_seqlens(cu_seqlens_q.long(), seq_len * cp_size)
    qs = cu_seqlens_q.gather(0, seq_idx)
    pos = torch.arange(len(qs), dtype=qs.dtype, device=qs.device) - qs
    if offs_q is not None:
        assert offs_q.shape == (n_seq,), offs_q.shape
        qoff = offs_q.gather(0, seq_idx)
        pos += qoff
    if cu_seqlens_k is None or cu_seqlens_k is cu_seqlens_q:
        ks = qs
    else:
        assert cu_seqlens_k.shape == (n_seq + 1,)
        ks = cu_seqlens_k.gather(0, seq_idx)
    ke = ks + (pos + 1) // kv_stride

    if cp_size == 1:
        pass
    elif balanced_cp:
        assert cp_size % 2 == 0, cp_size

        def f(x: torch.Tensor):
            chunks = x.chunk(cp_size * 2)
            return torch.cat(
                [
                    chunks[cp_rank],
                    chunks[cp_size - cp_rank - 1],
                ]
            )

        ks = f(ks)
        ke = f(ke)
    else:
        ks = ks.chunk(cp_size)[cp_rank]
        ke = ke.chunk(cp_size)[cp_rank]

    return ks, ke


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple[int], use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def generate_random_cu_seqlens(per_cp_seqlen, cp_size=4, cp_rank=3, kv_stride=1, average_q_len=512):
    total_seqlen = per_cp_seqlen * cp_size

    cu_seqlens = torch.randint(0, average_q_len * 2, (total_seqlen // average_q_len * 2,)).cuda()
    last_seq_id = torch.where(cu_seqlens.cumsum(0) >= total_seqlen)[0][0]
    cu_seqlens = cu_seqlens[:last_seq_id]

    if cu_seqlens.sum() < total_seqlen:
        cu_seqlens = torch.cat([cu_seqlens, torch.tensor([total_seqlen - cu_seqlens.sum()]).cuda()])

    cu_seqlens_cumsum = torch.cumsum(cu_seqlens, dim=0)
    cu_seqlens_k_cumsum = torch.cumsum(cu_seqlens // kv_stride, dim=0)
    cu_seqlens_qs = torch.cat([torch.tensor([0]).cuda(), cu_seqlens_cumsum[:-1]])
    cu_seqlens_ks = torch.cat([torch.tensor([0]).cuda(), cu_seqlens_k_cumsum[:-1]])
    cu_seqlens_qe = cu_seqlens_cumsum.clone()
    cu_seqlens_ke = cu_seqlens_k_cumsum.clone()

    cu_seqlens_ks_for_each_q = cal_cu_seqlen_ks_for_q(
        cu_seqlens_qs=cu_seqlens_qs,
        cu_seqlens_qe=cu_seqlens_qe,
        cu_seqlens_ks=cu_seqlens_ks,
        seq_len=total_seqlen,
    )
    cu_seqlens_ke_for_each_q = cal_cu_seqlen_ke_for_q(
        cu_seqlens_qs=cu_seqlens_qs,
        cu_seqlens_qe=cu_seqlens_qe,
        cu_seqlens_ks=cu_seqlens_ks,
        cu_seqlens_ke=cu_seqlens_ke,
        q_start_idxs=torch.zeros_like(cu_seqlens_qs),
        seq_len=total_seqlen,
        kv_stride=kv_stride,
    )

    assert per_cp_seqlen % 2 == 0
    per_chunk_seqlen = per_cp_seqlen // 2
    slice_short = slice(cp_rank * per_chunk_seqlen, (cp_rank + 1) * per_chunk_seqlen)
    slice_long = slice(
        total_seqlen - (cp_rank + 1) * per_chunk_seqlen,
        total_seqlen - cp_rank * per_chunk_seqlen,
    )
    ks = torch.cat(
        [
            cu_seqlens_ks_for_each_q[slice_short],
            cu_seqlens_ks_for_each_q[slice_long],
        ]
    )
    ke = torch.cat(
        [
            cu_seqlens_ke_for_each_q[slice_short],
            cu_seqlens_ke_for_each_q[slice_long],
        ]
    )
    assert len(ks) == len(ke) == per_cp_seqlen
    return ks, ke


def calculate_tensor_similarity(x, y, name="tensor"):
    """
    Calculate similarity between two tensors using a normalized dot product metric.

    Unlike torch.testing.assert_close which uses absolute/relative tolerance based on
    element-wise differences, this function computes a global similarity score:
        sim = 2 * <x, y> / (||x||^2 + ||y||^2)

    This metric is scale-invariant and measures the cosine-like similarity normalized
    by the magnitude of both tensors. It returns 1 for identical tensors and values
    closer to 0 for dissimilar ones. This is particularly useful for comparing tensors
    with varying magnitudes where relative errors matter more than absolute differences.

    Args:
        x: First tensor to compare
        y: Second tensor to compare
        name: Name of the tensor for logging purposes

    Returns:
        Similarity score in range [0, 1] where 1 means identical
    """
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print(f"\033[33mWARNING: {name} all zero\033[0m")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_tensors_similar(x, y, eps=1e-8, name="tensor", raise_assert=True):
    """
    Assert that two tensors are similar using a global similarity metric.

    Key differences from torch.testing.assert_close:
    - torch.testing.assert_close: Uses element-wise comparison with rtol/atol, checking
      that |x - y| <= atol + rtol * |y| for each element. It's sensitive to outliers
      and requires all elements to satisfy the tolerance.
    - assert_tensors_similar: Uses a single global similarity score (1 - sim) where sim is the
      normalized dot product. It's more robust to outliers and focuses on overall
      tensor similarity rather than element-wise precision. This is better suited for
      comparing large tensors where a few outlier elements shouldn't fail the test.

    Args:
        x: First tensor to compare
        y: Second tensor to compare
        eps: Maximum allowed difference (1 - similarity), default 1e-8
        name: Name of the tensor for error messages
        raise_assert: Whether to raise assertion error on failure
    """
    sim = calculate_tensor_similarity(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print(f"\033[31mERROR: {name} similarity check failed, diff={diff:.2e} (threshold={eps:.2e})\033[0m")
        if raise_assert:
            assert False  # noqa: B011


if __name__ == "__main__":
    seq_len = 32768
    cu_seqlens = torch.randint(128, 4096, (1000,), dtype=torch.int32, device="cuda")
    last_idx = torch.where(cu_seqlens.cumsum(dim=0) >= seq_len)[0][0]
    cu_seqlens_cumsum = cu_seqlens[:last_idx].cumsum(dim=0)
    cu_seqlens_qs = torch.cat([torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device), cu_seqlens_cumsum])
    cu_seqlens_qe = torch.cat([cu_seqlens_cumsum, torch.ones(1, dtype=torch.int32, device=cu_seqlens.device) * seq_len])

    from tilelang.profiler import do_bench

    fn = lambda: cal_seq_idx_for_q(cu_seqlens_qs, cu_seqlens_qe, seq_len)  # noqa: E731
    ms = do_bench(fn, warmup=25, rep=100)
