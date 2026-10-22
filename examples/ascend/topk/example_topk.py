"""DeepSelect-compatible dynamic BF16 TopK on Huawei Ascend."""

from functools import lru_cache
import struct
import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.algorithm import make_topk_macro


@lru_cache(maxsize=12)
def build_topk(capacity, sorted_index=False, abort_when_nan_found=True):
    rows, n, k = T.dynamic("rows"), T.dynamic("n"), T.dynamic("k")
    xs, istride, vs = T.dynamic("xs", "int64"), T.dynamic("istride", "int64"), T.dynamic("vs", "int64")
    select = make_topk_macro(capacity, sorted_index=sorted_index, abort_when_nan_found=abort_when_nan_found)

    @T.prim_func
    def main(
        X: T.StridedTensor((rows, n), (xs, 1), "bfloat16"),
        I: T.StridedTensor((rows, k), (istride, 1), "int32"),
        V: T.StridedTensor((rows, k), (vs, 1), "bfloat16"),
        E: T.Tensor((rows,), "int32"),
        O: T.Tensor((rows,), "int32"),
        return_value: T.int32,
        has_end: T.int32,
        has_offset: T.int32,
        index_fill: T.int32,
        value_fill: T.int32,
    ):
        with T.Kernel(rows) as row:
            select(X, I, V, E, O, row, n, k, return_value, has_end, has_offset, index_fill, value_fill)

    return tilelang.compile(
        main,
        target="ascend",
        execution_backend="tvm_ffi",
        pass_configs={"tl.disable_thread_storage_sync": True, "tl.enable_auto_schedule": False},
    )


def topk(
    input,
    topk,
    sorted=False,
    begin=None,
    end=None,
    indices_type=torch.int64,
    sorted_index=False,
    hint=None,
    output_idx=None,
    output_idx_offset=None,
    idx_oob_fill_value=2147483647,
    value_oob_fill_value=float("-inf"),
    return_value=True,
    abort_when_nan_found=True,
):
    import torch

    if begin is not None or hint is not None or sorted:
        raise ValueError("Huawei Ascend TopK requires begin=None, hint=None and sorted=False")
    if input.device.type != "npu" or input.dtype != torch.bfloat16 or input.ndim != 2:
        raise ValueError("TopK input must be a two-dimensional NPU BF16 tensor")
    rows, n = input.shape
    if rows < 1 or not 0 <= n < 2**21 or type(topk) is not int or not 1 <= topk <= 4096:
        raise ValueError("TopK requires B>=1, 0<=N<2**21 and 1<=K<=4096")
    if input.stride(1) != 1 or input.stride(0) % 16 or input.stride(0) < n:
        raise ValueError("TopK input requires unit inner stride and a non-overlapping 32B-aligned row stride")
    if indices_type != torch.int32:
        raise ValueError("Huawei Ascend TopK supports only INT32 indices")
    if not -(2**31) <= idx_oob_fill_value < 2**31:
        raise ValueError("TopK index fill must fit INT32")
    for name, tensor in (("end", end), ("output_idx_offset", output_idx_offset)):
        if tensor is not None and (
            tensor.device != input.device or tensor.dtype != torch.int32 or tensor.shape != (rows,) or tensor.stride(0) != 1
        ):
            raise ValueError(f"{name} must be contiguous INT32[B] on the input device")
    if output_idx is not None and (
        output_idx.device != input.device
        or output_idx.dtype != torch.int32
        or output_idx.shape != (rows, topk)
        or output_idx.stride(1) != 1
        or output_idx.stride(0) % 8
        or output_idx.stride(0) < topk
    ):
        raise ValueError("output_idx must be INT32[B,K] with unit inner stride and a non-overlapping 32B-aligned row stride")

    def allocate(dtype, alignment):
        return torch.empty((rows, (topk + alignment - 1) // alignment * alignment), dtype=dtype, device=input.device)[:, :topk]

    indices = output_idx if output_idx is not None else allocate(torch.int32, 8)
    values = allocate(torch.bfloat16, 16) if return_value else None
    # Disabled outputs/inputs use metadata-only aliases; the kernel never accesses them.
    value_arg = values if return_value else torch.as_strided(indices.view(torch.bfloat16), (rows, topk), (indices.stride(0) * 2, 1))
    dummy = torch.as_strided(indices, (rows,), (1,))
    raw = struct.unpack("<I", struct.pack("<f", value_oob_fill_value))[0]
    fill = 0x7FC0 if (raw & 0x7FFFFFFF) > 0x7F800000 else ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16) & 0xFFFF
    capacity = 512 if topk <= 512 else (1024 if topk <= 1024 else 4096)
    kernel = build_topk(capacity, bool(sorted_index), bool(abort_when_nan_found))
    kernel(
        input,
        indices,
        value_arg,
        end if end is not None else dummy,
        output_idx_offset if output_idx_offset is not None else dummy,
        int(return_value),
        int(end is not None),
        int(output_idx_offset is not None),
        idx_oob_fill_value,
        fill,
    )
    return values, indices


def main():
    import argparse
    import torch
    import torch_npu  # noqa: F401
    from tilelang.profiler import do_bench

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--n", type=int, default=131072)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--sorted-index", action="store_true")
    args = parser.parse_args()
    stride = (args.n + 15) // 16 * 16
    x = torch.randn((args.rows, stride), device="npu", dtype=torch.bfloat16)[:, : args.n]
    values, indices = topk(x, args.k, indices_type=torch.int32, sorted_index=args.sorted_index)
    torch.testing.assert_close(x.cpu().gather(1, indices.cpu().long()), values.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(values.cpu().sort().values, x.cpu().topk(args.k).values.sort().values, rtol=0, atol=0)
    latency = do_bench(lambda: topk(x, args.k, indices_type=torch.int32, sorted_index=args.sorted_index), backend="event", device=x.device)
    print(f"TileLang TopK: {latency * 1000:.3f} us")


if __name__ == "__main__":
    main()
