"""Output operators matching DeepGEMM's public epilogue API."""

from dataclasses import dataclass

import torch


class Epilogue:
    kind = "identity"


@dataclass(frozen=True)
class Identity(Epilogue):
    pass


@dataclass(frozen=True)
class Alpha(Epilogue):
    kind = "alpha"
    alpha: float


@dataclass(frozen=True)
class _Bias(Epilogue):
    kind = "bias"
    tensor: torch.Tensor


@dataclass(frozen=True)
class FP8Quantization(Epilogue):
    kind = "fp8"
    sfd: torch.Tensor


@dataclass(frozen=True)
class BF16StochasticRounding(Epilogue):
    """Deterministic quartet-hash rounding of the FP32 accumulator to BF16."""

    kind = "stochastic"


def resolve(d, c=None, alpha=None, epilogue=None, *, allow_fp8=False, dense=False):
    sfd = None
    if isinstance(d, (tuple, list)):
        if not allow_fp8 or len(d) != 2:
            raise TypeError("A (D, SFD) output pair is supported only by fp8_bmm/fp8_einsum")
        d, sfd = d
    if sum(x is not None for x in (epilogue, alpha, sfd)) > 1:
        raise ValueError("epilogue, alpha and the FP8 output pair are mutually exclusive")
    resolved = (
        epilogue
        if epilogue is not None
        else Alpha(float(alpha))
        if alpha is not None
        else FP8Quantization(sfd)
        if sfd is not None
        else Identity()
    )
    if not isinstance(resolved, (Identity, Alpha, FP8Quantization, BF16StochasticRounding)):
        raise TypeError(f"Unsupported epilogue: {type(resolved).__name__}")
    if isinstance(resolved, BF16StochasticRounding):
        if c is not None or d.dtype != torch.bfloat16:
            raise ValueError("Stochastic rounding requires a direct BF16 output without accumulation")
    elif isinstance(resolved, FP8Quantization):
        if not allow_fp8:
            raise NotImplementedError("FP8 quantization is supported only by fp8_bmm/fp8_einsum")
        if c is not None or d.dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 quantization requires an E4M3 output without accumulation")
        if d.ndim not in (2, 3) or d.shape[-1] % 128:
            raise ValueError("FP8 output requires rank 2 or 3 and N divisible by 128")
        m, n = d.shape[-2:]
        batches = d.shape[0] if d.ndim == 3 else 1
        if d.stride(-1) != 1 or (d.ndim == 3 and d.stride(0) != n):
            raise ValueError("FP8 batched output must have the interleaved [M, batch, N] storage layout")
        sf = resolved.sfd
        if (
            sf.dtype != torch.int16
            or sf.device != d.device
            or sf.ndim != 2
            or tuple(sf.shape) != (m, batches * n // 64)
            or (sf.stride(0) != 1 and m != 1)
            or sf.stride(1) != m
        ):
            raise ValueError("SFD must be MN-major int16 with shape [M, batch * N / 64] on the output device")
    elif d.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Identity/Alpha epilogues require a BF16 or FP32 output")
    if dense and c is not None and c.data_ptr() != d.data_ptr():
        if isinstance(resolved, Alpha):
            raise ValueError("alpha and bias cannot be used together")
        if c.ndim != 2 or tuple(c.shape) != (1, d.shape[-1]) or not c.is_contiguous():
            raise ValueError("Dense C must alias D for accumulation, or be a separate contiguous [1, N] bias")
        resolved = _Bias(c)
    return d, resolved


def arguments(epilogue, *, dense=False):
    # Dense entry points always carry alpha; other GEMMs only need it for Alpha.
    args = (float(epilogue.alpha),) if isinstance(epilogue, Alpha) else (1.0,) if dense else ()
    if isinstance(epilogue, _Bias):
        return (*args, epilogue.tensor)
    if isinstance(epilogue, FP8Quantization):
        return (*args, epilogue.sfd.T)
    return args
