"""Real-device coverage for the parameterized DeepSelect macro."""

import pytest
import torch
from example_topk import build_topk as compile_topk, topk
import struct


def build_topk(
    rows,
    n,
    k,
    *,
    input_stride=None,
    return_value=True,
    sorted_index=False,
    has_end=False,
    has_offset=False,
    abort_when_nan_found=False,
    idx_oob_fill_value=2147483647,
    value_oob_fill_bits=None,
    value_oob_fill_value=float("-inf"),
):
    if value_oob_fill_bits is not None:
        value_oob_fill_value = struct.unpack("<f", struct.pack("<I", value_oob_fill_bits << 16))[0]

    def run(x, end, offset):
        values, indices = topk(
            x[:, :n],
            k,
            indices_type=torch.int32,
            sorted_index=sorted_index,
            end=end if has_end else None,
            output_idx_offset=offset if has_offset else None,
            return_value=return_value,
            abort_when_nan_found=abort_when_nan_found,
            idx_oob_fill_value=idx_oob_fill_value,
            value_oob_fill_value=value_oob_fill_value,
        )
        return indices, values

    return run


def make_input(rows, columns, distribution):
    generator = torch.Generator().manual_seed(0)
    if distribution == "ties":
        x = torch.randint(-4, 5, (rows, columns), generator=generator).float()
    else:
        x = torch.randn((rows, columns), generator=generator)
        if distribution == "special":
            x[:, ::7] = float("-inf")
            x[:, ::11] = float("inf")
            x[:, ::13] = -0.0
            x[:, ::17] = 0.0
    return x.to(torch.bfloat16).contiguous()


def check(x, k, ends, offsets, actual, values, sorted_index=False, fill=-37, value_fill=-2.0):
    ai = actual[0].cpu()[:, :k]
    av = actual[1].cpu()[:, :k] if values else None
    x = x.cpu().float()
    for r, end in enumerate(ends.tolist()):
        count = min(end, k)
        if x[r, :end].isnan().any():
            assert (ai[r] == 0x3F3F3F3F).all()
            continue
        if sorted_index:
            assert (ai[r, 1:count] >= ai[r, : max(0, count - 1)]).all()
        ids = ai[r, :count].long() - int(offsets[r])
        assert ((ids >= 0) & (ids < end)).all()
        assert ids.unique().numel() == count
        torch.testing.assert_close(x[r, ids].sort().values, x[r, :end].topk(count).values.sort().values, rtol=0, atol=0)
        assert (ai[r, count:] == fill).all()
        if values:
            torch.testing.assert_close(av[r, :count].float(), x[r, ids], rtol=0, atol=0)
            assert (av[r, count:] == value_fill).all()


@pytest.mark.parametrize("k", [1, 7, 17, 511, 512, 513, 1000, 1024, 1025, 4095, 4096])
@pytest.mark.parametrize("sorted_index", [False, True])
@pytest.mark.parametrize("values", [False, True])
def test_general(k, sorted_index, values):
    n = 131073
    pivot = 32832 if k <= 512 else (36800 if k <= 1024 else 10432)
    ends = torch.tensor([0, max(0, k - 1), k, k + 1, pivot, pivot + 1, 65537, n], dtype=torch.int32)
    offsets = torch.tensor([100, -100, 13, 0, -7, 1024, -2048, 4096], dtype=torch.int32)
    stride = (n + 15) // 16 * 16 + 16
    e, o = ends.npu(), offsets.npu()
    kernel = build_topk(
        8,
        n,
        k,
        input_stride=stride,
        return_value=values,
        sorted_index=sorted_index,
        has_end=True,
        has_offset=True,
        idx_oob_fill_value=-37,
        value_oob_fill_bits=0xC000,
    )
    for distribution in ["normal", "ties", "special"]:
        x = make_input(8, stride, distribution).npu()
        actual = kernel(x, e, o)
        check(x, k, ends, offsets, actual, values, sorted_index)


@pytest.mark.parametrize("n,k", [(0, 1), (1, 17), (2, 1), (3, 1), (3, 2), (17, 7), (511, 512), (4097, 4096), (8193, 1025)])
def test_static_lengths(n, k):
    stride = max(16, (n + 15) // 16 * 16)
    if 1 < n < 4 and k < n:
        storage = make_input(1, stride + n, "normal").npu()
        x = torch.as_strided(storage, (2, n), (stride, 1))
        assert x.untyped_storage().nbytes() == (stride + n) * 2
    else:
        x = make_input(2, stride, "normal").npu()
    e = torch.full((2,), n, dtype=torch.int32, device="npu")
    o = torch.zeros_like(e)
    kernel = build_topk(
        2, n, k, input_stride=stride, return_value=True, sorted_index=True, idx_oob_fill_value=-37, value_oob_fill_bits=0xC000
    )
    actual = kernel(x, e, o)
    check(x, k, e.cpu(), o.cpu(), actual, True, sorted_index=True)


@pytest.mark.parametrize("k", [7, 513, 4096])
def test_nan_sentinel(k):
    n = 65537
    stride = (n + 15) // 16 * 16
    x = make_input(3, stride, "normal")
    x[0, 0] = float("nan")
    x[1, n - 1] = float("nan")
    x = x.npu()
    e = torch.tensor([min(k, n), n, n], device="npu", dtype=torch.int32)
    o = torch.zeros_like(e)
    kernel = build_topk(3, n, k, input_stride=stride, return_value=False, has_end=True)
    actual = kernel(x, e, o)
    check(x, k, e.cpu(), o.cpu(), actual, False)


@pytest.mark.parametrize("k", [1024, 4096])
def test_maximum_length(k):
    n = 2**21 - 1
    stride = 2**21
    x = make_input(1, stride, "normal").npu()
    e = torch.tensor([n], dtype=torch.int32, device="npu")
    o = torch.zeros_like(e)
    actual = build_topk(1, n, k, input_stride=stride)(x, e, o)
    ids, values = actual[0][0, :k].cpu().long(), actual[1][0, :k].cpu()
    source = x.cpu()[0, :n]
    assert ((ids >= 0) & (ids < n)).all()
    assert ids.unique().numel() == k
    torch.testing.assert_close(source[ids], values, rtol=0, atol=0)
    torch.testing.assert_close(values.sort().values, source.topk(k).values.sort().values, rtol=0, atol=0)


def test_abort_enabled_without_nan():
    x = make_input(2, 8208, "normal").npu()
    e = torch.tensor([8193, 8193], dtype=torch.int32, device="npu")
    o = torch.zeros_like(e)
    actual = build_topk(2, 8193, 513, input_stride=8208, abort_when_nan_found=True)(x, e, o)
    check(x, 513, e.cpu(), o.cpu(), actual, True)


def test_float_fill_rounding():
    x = torch.zeros((1, 16), dtype=torch.bfloat16, device="npu")
    e = torch.zeros((1,), dtype=torch.int32, device="npu")
    i, v = build_topk(1, 0, 7, input_stride=16, value_oob_fill_value=1.00390625)(x, e, e)
    assert (i[:, :7] == 2147483647).all()
    assert (v[:, :7] == 1.0).all()


@pytest.mark.parametrize("capacity,ks", [(512, [1, 17, 512]), (1024, [513, 700, 1024]), (4096, [1025, 2049, 4096])])
@pytest.mark.parametrize("sorted_index", [False, True])
def test_reuse_compiled_kernel(capacity, ks, sorted_index):
    kernel = compile_topk(capacity, sorted_index, False)
    misses = compile_topk.cache_info().misses
    for variant, k in enumerate(ks):
        rows, n = variant + 1, [8193, 40001, 131073][variant]
        stride = (n + 15) // 16 * 16 + variant * 16
        x = make_input(rows, stride, "ties").npu()[:, :n]
        ends = torch.full((rows,), n, dtype=torch.int32)
        ends[0] = k - 1
        offsets = torch.arange(rows, dtype=torch.int32) - 10
        output = torch.empty((rows, (k + 7) // 8 * 8 + 8 * variant), device="npu", dtype=torch.int32)[:, :k]
        for enabled in [False, True]:
            values, indices = topk(
                x,
                k,
                indices_type=torch.int32,
                end=ends.npu() if enabled else None,
                output_idx_offset=offsets.npu() if enabled else None,
                return_value=enabled,
                sorted_index=sorted_index,
                output_idx=output,
                idx_oob_fill_value=-37 - variant,
                value_oob_fill_value=-2.0 - variant,
                abort_when_nan_found=False,
            )
            assert indices is output
            assert (values is None) == (not enabled)
            assert indices.shape == (rows, k)
            check(
                x,
                k,
                ends if enabled else torch.full_like(ends, n),
                offsets if enabled else torch.zeros_like(offsets),
                (indices, values),
                enabled,
                sorted_index,
                fill=-37 - variant,
                value_fill=-2.0 - variant,
            )
            assert compile_topk(capacity, sorted_index, False) is kernel
            assert compile_topk.cache_info().misses == misses
