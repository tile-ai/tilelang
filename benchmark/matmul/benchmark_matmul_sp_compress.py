import argparse
from typing import Optional

import torch
from tilelang.profiler import do_bench
from tilelang.utils.sparse import compress, randn_semi_sparse, randint_semi_sparse, torch_compress

SUPPORTED_DTYPE_NAMES = ["float16", "bfloat16", "float32", "int8"]
SUPPORTED_META_DTYPE_NAMES = ["int8", "int16", "int32"]


def _resolve_torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


def _generate_semi_sparse(m: int, k: int, dtype: torch.dtype, device: str = "cuda") -> torch.Tensor:
    if dtype in (torch.int8, torch.uint8):
        return randint_semi_sparse(m, k, low=-64, high=64, dtype=dtype, device=device)
    return randn_semi_sparse(m, k, dtype=dtype, device=device)


def _compress_bytes(input_tensor: torch.Tensor, sparse_tensor: torch.Tensor, meta_tensor: torch.Tensor) -> int:
    return (
        input_tensor.numel() * input_tensor.element_size()
        + sparse_tensor.numel() * sparse_tensor.element_size()
        + meta_tensor.numel() * meta_tensor.element_size()
    )


def benchmark_compress(
    m: int,
    k: int,
    dtype: torch.dtype,
    meta_dtype: Optional[torch.dtype] = None,  # noqa: FA100
):
    a0 = _generate_semi_sparse(m, k, dtype)

    sparse0, meta0 = compress(a0, meta_dtype=meta_dtype)
    ref_sparse0, ref_meta0 = torch_compress(a0, meta_dtype=meta_dtype)

    bytes_per_compress = _compress_bytes(a0, sparse0, meta0)
    bytes_per_torch = _compress_bytes(a0, ref_sparse0, ref_meta0)

    tl_latency_ms = do_bench(lambda: compress(a0, meta_dtype=meta_dtype))
    torch_latency_ms = do_bench(lambda: torch_compress(a0, meta_dtype=meta_dtype))

    tl_latency_s = tl_latency_ms * 1e-3
    torch_latency_s = torch_latency_ms * 1e-3
    tl_throughput_gbps = bytes_per_compress / tl_latency_s / 1e9
    torch_throughput_gbps = bytes_per_torch / torch_latency_s / 1e9

    return bytes_per_compress, bytes_per_torch, tl_latency_ms, torch_latency_ms, tl_throughput_gbps, torch_throughput_gbps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark two TileLang compress operators by memory throughput")
    parser.add_argument("--m", type=int, default=16384, help="Matrix rows")
    parser.add_argument("--k", type=int, default=16384, help="Matrix columns")
    parser.add_argument("--dtype", type=str, default="float16", choices=SUPPORTED_DTYPE_NAMES, help="Input dtype")
    parser.add_argument(
        "--meta_dtype",
        type=str,
        default=None,
        choices=SUPPORTED_META_DTYPE_NAMES,
        help="Metadata dtype (defaults to the library choice for the input dtype)",
    )
    args = parser.parse_args()

    dtype = _resolve_torch_dtype(args.dtype)
    meta_dtype = _resolve_torch_dtype(args.meta_dtype) if args.meta_dtype is not None else None

    bytes_per_compress, bytes_per_torch, tl_latency_ms, torch_latency_ms, tl_gbps, torch_gbps = benchmark_compress(
        args.m,
        args.k,
        dtype=dtype,
        meta_dtype=meta_dtype,
    )

    print(f"M={args.m} K={args.k} dtype={args.dtype} meta={args.meta_dtype or 'default'}")
    print(f"tilelang: {tl_latency_ms:.4f} ms, {tl_gbps:.3f} GB/s (bytes={bytes_per_compress})")
    print(f"torch:    {torch_latency_ms:.4f} ms, {torch_gbps:.3f} GB/s (bytes={bytes_per_torch})")
