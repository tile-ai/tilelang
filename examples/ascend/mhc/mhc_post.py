# SPDX-License-Identifier: MIT
# Copyright (c) 2026 DeepSeek.
# Copyright (c) 2026 Q4Kernels contributors.
"""mHC Post forward, adapted from TileKernels' Ascend implementation."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

__all__ = ["mhc_post"]

_VEC = 64


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def _mhc_post_kernel(comb_res_mix, residual, post_layer_mix, x, out, *, hidden: int, h_blk: int, num_vec_cores: int):
    num_tokens = T.dynamic("num_tokens")
    mhc = 4
    comb_res_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    post_layer_mix: T.Tensor[(num_tokens, mhc), T.float32]
    x: T.Tensor[(num_tokens, hidden), T.bfloat16]
    out: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC
    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        residual_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        comb_res_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        # DMA transfers require a whole 32-byte block in UB.
        post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)
        out_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        T.annotate_buffer_versions(
            {
                residual_ub: num_stages,
                x_ub: num_stages,
                comb_res_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                out_ub: num_stages,
            }
        )

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(comb_res_mix[pid_n, 0, 0], comb_res_mix_ub)
            T.copy(post_layer_mix[pid_n, 0], post_layer_mix_ub[0:mhc])
            T.copy(residual[pid_n, 0, pid_h * h_blk], residual_ub)
            T.copy(x[pid_n, pid_h * h_blk], x_ub)

            with T.SimdVF():
                comb_mix = S.alloc_local((mhc * mhc,), T.float32)
                post_mix = S.alloc_local((mhc,), T.float32)
                for mi in T.unroll(mhc, explicit=True):
                    for mo in T.unroll(mhc, explicit=True):
                        comb_mix[mi * mhc + mo] = S.vld(comb_res_mix_ub[mi, mo], dist="BRC_B32")
                for mo in T.unroll(mhc, explicit=True):
                    post_mix[mo] = S.vld(post_layer_mix_ub[mo], dist="BRC_B32")
                out_reg = S.alloc_local((mhc,), T.float32)
                res = S.alloc_local((mhc,), T.float32)
                for c in T.serial(num_chunks):
                    col = c * _VEC
                    x_f32 = S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32, part=0)
                    for mi in T.unroll(mhc, explicit=True):
                        res[mi] = S.vcvt(S.vld(residual_ub[mi, col], dist="UNPK_B16"), T.float32, part=0)
                    for mo in T.unroll(mhc, explicit=True):
                        out_reg[mo] = S.vmul(x_f32, post_mix[mo])
                    for mi in T.unroll(mhc, explicit=True):
                        for mo in T.unroll(mhc, explicit=True):
                            S.vmula(out_reg[mo], res[mi], comb_mix[mi * mhc + mo])
                    for mo in T.unroll(mhc, explicit=True):
                        S.vsts(out_ub[mo, col], S.vcvt(out_reg[mo], T.bfloat16), dist="PK_B32")

            T.copy(out_ub, out[pid_n, 0, pid_h * h_blk])


@functools.lru_cache(maxsize=32)
def _compiled_kernel(hidden, num_vec_cores):
    h_blk = hidden if hidden <= 4096 else min((hidden + 1) // 2, 4096)
    h_blk = (h_blk + _VEC - 1) // _VEC * _VEC
    return _mhc_post_kernel.compile(hidden=hidden, h_blk=h_blk, num_vec_cores=num_vec_cores)


def mhc_post(x, residual, post_layer_mix, comb_res_mix, *, num_vec_cores=56):
    """Combine the layer output and residual streams, returning fresh BF16 storage.

    ``x[..., H]`` and ``residual[..., 4, H]`` are BF16. FP32 mixes have shapes
    ``[..., 4, 1]`` (or ``[..., 4]``) and ``[..., 4, 4]``. The leading batch
    dimensions must match; H is 4096, 5120 or 7168. Inputs are contiguous on
    one NPU. The combination matrix is indexed as [input stream, output stream].
    """
    import torch

    if residual.ndim < 3 or residual.shape[-2] != 4:
        raise ValueError("Expected residual[...,4,H] with at least one batch dimension")
    batch, hidden = tuple(residual.shape[:-2]), residual.shape[-1]
    n = math.prod(batch)
    if not 0 < n <= 2**31 - 4096 or hidden not in (4096, 5120, 7168):
        raise ValueError("Expected positive token count and H in (4096, 5120, 7168)")
    if type(num_vec_cores) is not int or num_vec_cores <= 0:
        raise ValueError("num_vec_cores must be a positive integer")
    if tuple(post_layer_mix.shape) not in (batch + (4,), batch + (4, 1)):
        raise ValueError("Expected post_layer_mix[...,4] or [...,4,1] matching residual batch dimensions")
    for name, tensor, shape, dtype in (
        ("x", x, batch + (hidden,), torch.bfloat16),
        ("residual", residual, batch + (4, hidden), torch.bfloat16),
        ("post_layer_mix", post_layer_mix, tuple(post_layer_mix.shape), torch.float32),
        ("comb_res_mix", comb_res_mix, batch + (4, 4), torch.float32),
    ):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{name}: expected shape={shape}, dtype={dtype}")
        if tensor.device.type != "npu" or tensor.device != residual.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the same NPU")

    kernel = _compiled_kernel(hidden, num_vec_cores)
    out = torch.empty_like(residual)
    kernel(
        comb_res_mix.view(n, 4, 4),
        residual.view(n, 4, hidden),
        post_layer_mix.view(n, 4),
        x.view(n, hidden),
        out.view(n, 4, hidden),
    )
    return out
