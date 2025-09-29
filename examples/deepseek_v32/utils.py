# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
# -*- coding: utf-8 -*-

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

def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
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
                if all(_is_equal(a, b) for a, b in zip(args, last_args)) and \
                   set(kwargs.keys()) == set(last_kwargs.keys()) and \
                   all(_is_equal(v, last_kwargs[k]) for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper

@tensor_cache
def cal_seq_idx_from_cu_seqlens(cu_seqlens: torch.LongTensor, seq_len: int):
    seq_idx = cu_seqlens.new_zeros(seq_len+1)
    seq_idx.scatter_add_(0, cu_seqlens[1:].long(), torch.ones_like(seq_idx))
    seq_idx.cumsum_(0)
    return seq_idx[:-1]

@tensor_cache
def cal_seq_idx_for_q(cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor, seq_len: int) -> torch.IntTensor:
    seq_idx_for_q = torch.full((seq_len,), len(cu_seqlens_qs), dtype=torch.int32, device=cu_seqlens_qs.device)
    for i in range(len(cu_seqlens_qs)):
        seq_idx_for_q[cu_seqlens_qs[i]:cu_seqlens_qe[i]] = i
    return seq_idx_for_q

@tensor_cache
def cal_cu_seqlen_ks_for_q(cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor, cu_seqlens_ks: torch.LongTensor, seq_len: int) -> torch.IntTensor:
    cu_seqlen_ks_for_each_q = torch.gather(input=torch.cat([cu_seqlens_ks, torch.full((1,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=cu_seqlens_qs.device)]), dim=0, index=cal_seq_idx_for_q(cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long())
    return cu_seqlen_ks_for_each_q.int()

@tensor_cache
def cal_cu_seqlen_ke_for_q(cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor, cu_seqlens_ks: torch.LongTensor, cu_seqlens_ke: torch.LongTensor, q_start_idxs: torch.LongTensor, seq_len: int, kv_stride: int) -> torch.IntTensor:
    cu_seqlen_ke_for_each_q = torch.gather(input=torch.cat([cu_seqlens_ke, torch.zeros(1, dtype=torch.int32, device=cu_seqlens_qs.device)]), dim=0, index=cal_seq_idx_for_q(cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long())
    casual_cu_seqlen_ke_for_each_q = torch.zeros((seq_len,), dtype=torch.int32, device=cu_seqlens_qs.device)
    for i in range(len(cu_seqlens_qs)):
        casual_cu_seqlen_ke_for_each_q[cu_seqlens_qs[i]:cu_seqlens_qe[i]] = (torch.arange(q_start_idxs[i], q_start_idxs[i] + cu_seqlens_qe[i] - cu_seqlens_qs[i], dtype=torch.int32, device=cu_seqlens_qs.device) + 1) // kv_stride + cu_seqlens_ks[i]
    cu_seqlen_ke_for_each_q = torch.minimum(casual_cu_seqlen_ke_for_each_q, cu_seqlen_ke_for_each_q)
    return cu_seqlen_ke_for_each_q.int()

@tensor_cache
def cal_ks_ke_from_cu_seqlen_qk(cu_seqlens_q: torch.LongTensor, cu_seqlens_k: torch.LongTensor = None, offs_q: torch.LongTensor = None, *, seq_len: int, kv_stride:int=1, cp_rank:int=0, cp_size:int=1, balanced_cp=False):
    '''
    seq_len: seq len per cp rank
    balanced cp slice assignment: 0 1 2 3 3 2 1 0
    '''
    n_seq = len(cu_seqlens_q) - 1
    assert n_seq > 0
    assert cu_seqlens_q.shape == (n_seq + 1,)
    seq_idx = cal_seq_idx_from_cu_seqlens(cu_seqlens_q.long(), seq_len*cp_size)
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
            chunks = x.chunk(cp_size*2)
            return torch.cat([
                chunks[cp_rank],
                chunks[cp_size-cp_rank-1],
            ])
        ks = f(ks)
        ke = f(ke)
    else:
        ks = ks.chunk(cp_size)[cp_rank]
        ke = ke.chunk(cp_size)[cp_rank]

    return ks, ke


def print_red_warning(message):
    print(f"\033[31mWARNING: {message}\033[0m")


def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f'{name} all zero')
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_similar(x, y, eps=1e-8, name="tensor"):
    sim = calc_sim(x, y, name)
    diff = 1. - sim
    if not (0 <= diff <= eps):
        print_red_warning(f'{name} Error: {diff}')
        assert False


if __name__ == "__main__":
    seq_len = 32768
    cu_seqlens = torch.randint(128, 4096, (1000,), dtype=torch.int32, device="cuda")
    last_idx = torch.where(cu_seqlens.cumsum(dim=0) >= seq_len)[0][0]
    cu_seqlens_cumsum = cu_seqlens[:last_idx].cumsum(dim=0)
    cu_seqlens_qs = torch.cat([torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device), cu_seqlens_cumsum])
    cu_seqlens_qe = torch.cat([cu_seqlens_cumsum, torch.ones(1, dtype=torch.int32, device=cu_seqlens.device) * seq_len])

    from tilelang.profiler import do_bench

    fn = lambda: cal_seq_idx_for_q(cu_seqlens_qs, cu_seqlens_qe, seq_len)
    ms = do_bench(fn, warmup=25, rep=100)
