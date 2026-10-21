"""Numerical checks for DeepGEMM schedulers and scale conversion."""

import os

import pytest
import torch
import tilelang.testing

from . import api


def _input(shape, seed):
    # Small integers keep the CPU reference exact in BF16 and FP8.
    return torch.randint(-2, 3, shape, generator=torch.Generator().manual_seed(seed)).float()


@tilelang.testing.requires_ascend
def test_bf16_dense_alpha_accumulation():
    a, b = _input((17, 64), 0), _input((32, 64), 1)
    c = _input((17, 32), 2)
    d = c.npu()
    api.bf16_gemm_nt(a.to(device="npu", dtype=torch.bfloat16), b.to(device="npu", dtype=torch.bfloat16), d, c=d, alpha=0.5)
    torch.testing.assert_close(d.cpu(), 0.5 * (a @ b.T) + c, rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_fp8_batched_scales():
    a, b = _input((2, 17, 64), 0), _input((2, 32, 64), 1)
    sfa = torch.full((2, 17, 2), 0.5, device="npu")
    sfb = torch.full((2, 32, 2), 0.25, device="npu")
    d = torch.empty((2, 17, 32), dtype=torch.float32, device="npu")
    api.fp8_bmm(
        (a.to(torch.float8_e4m3fn).npu(), sfa),
        (b.to(torch.float8_e4m3fn).npu(), sfb),
        d,
        epilogue=api.epilogue.Alpha(0.5),
    )
    torch.testing.assert_close(d.cpu(), (a @ b.transpose(-2, -1)) * 0.0625, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.skipif(
    os.environ.get("TILELANG_DEFAULT_TARGET") == "pto",
    reason="PTO L0C->UB MTE does not support quant_pre != 0 yet",
)
def test_bf16_m_grouped_padding():
    a, b = _input((512, 64), 0), _input((2, 32, 64), 1)
    a[17:256] = 0
    a[289:] = 0
    layout = torch.tensor([17, 289], dtype=torch.int32, device="npu")
    d = torch.empty((512, 32), dtype=torch.bfloat16, device="npu")
    api.m_grouped_bf16_gemm_nt_contiguous(
        a.to(device="npu", dtype=torch.bfloat16),
        b.to(device="npu", dtype=torch.bfloat16),
        d,
        layout,
        use_psum_layout=True,
    )
    expected = torch.cat((a[:256] @ b[0].T, a[256:] @ b[1].T))
    torch.testing.assert_close(d.cpu(), expected.bfloat16(), rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("accumulate", [False, True])
def test_fp8_k_grouped_scale_tail(accumulate):
    # The final group ends before its 256-aligned boundary; scales only cover K.
    a, b = _input((385, 16), 0), _input((385, 32), 1)
    a[65:256] = 0
    b[65:256] = 0
    sfa, sfb = torch.ones((13, 16)), torch.ones((13, 32))
    # Padding scales must be cleared before the power-of-two validation.
    sfa[3:8] = -3
    sfb[3:8] = -3
    c = _input((4, 16, 32), 2)
    d = torch.full(c.shape, float("nan"), dtype=torch.float32, device="npu")
    layout = torch.tensor([0, 65, 256, 385], dtype=torch.int32, device="npu")
    api.k_grouped_fp8_gemm_tn_contiguous(
        (a.to(torch.float8_e4m3fn).npu(), sfa.npu()),
        (b.to(torch.float8_e4m3fn).npu(), sfb.npu()),
        d,
        ks_cpu=[0, 65, 0, 129],
        grouped_layout=layout,
        c=c.npu() if accumulate else None,
    )
    expected = torch.zeros_like(c)
    expected[1], expected[3] = a[:65].T @ b[:65], a[256:].T @ b[256:]
    if accumulate:
        expected += c
    torch.testing.assert_close(d.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_bf16_dense_bias():
    a, b = _input((17, 64), 0), _input((70, 64), 1)
    bias = _input((1, 70), 2) * 0.25
    d = torch.empty((17, 70), dtype=torch.bfloat16, device="npu")
    api.bf16_gemm_nt(a.bfloat16().npu(), b.bfloat16().npu(), d, c=bias.bfloat16().npu())
    torch.testing.assert_close(d.cpu(), (a @ b.T + bias).bfloat16(), rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.skipif(
    os.environ.get("TILELANG_DEFAULT_TARGET") == "pto",
    reason="PTO L0C->UB MTE does not support FP32-to-BF16 conversion yet",
)
def test_fp8_quantization_epilogue():
    batches, m, n, k = 2, 17, 256, 64
    a, b = _input((batches, m, k), 0), _input((batches, n, k), 1)
    sfa = torch.ones((batches, m, k // 32), device="npu")
    sfb = torch.ones((batches, n, k // 32), device="npu")
    d = torch.empty((m, batches, n), dtype=torch.float8_e4m3fn, device="npu").permute(1, 0, 2)
    sfd = torch.empty((batches * n // 64, m), dtype=torch.int16, device="npu").T
    api.fp8_bmm(
        (a.to(torch.float8_e4m3fn).npu(), sfa),
        (b.to(torch.float8_e4m3fn).npu(), sfb),
        d,
        epilogue=api.epilogue.FP8Quantization(sfd),
    )
    values = (a @ b.transpose(-2, -1)).bfloat16().float().reshape(batches, m, n // 32, 32)
    exponent = torch.ceil(torch.log2(values.abs().amax(-1) / 448)).clamp_min(-22)
    expected = (values / torch.pow(2.0, exponent)[..., None]).reshape(batches, m, n).to(torch.float8_e4m3fn)
    expected_sf = (exponent + 127).to(torch.uint8).view(torch.int16).transpose(0, 1).reshape(m, batches * n // 64)
    torch.testing.assert_close(d.float().cpu(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(sfd.cpu(), expected_sf, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.skipif(
    os.environ.get("TILELANG_DEFAULT_TARGET") == "pto",
    reason="PTO codegen does not support tl.simd.vst2 yet",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_rounding(dtype):
    a = (_input((17, 128), 0) / 7).to(dtype)
    b = (_input((70, 128), 1) / 7).to(dtype)
    d = torch.empty((17, 70), dtype=torch.bfloat16, device="npu")
    epilogue = api.epilogue.BF16StochasticRounding()
    if dtype == torch.bfloat16:
        api.bf16_gemm_nt(a.npu(), b.npu(), d, epilogue=epilogue)
    else:
        api.fp8_gemm_nt(
            (a.npu(), torch.ones((17, 4), device="npu")),
            (b.npu(), torch.ones((70, 4), device="npu")),
            d,
            epilogue=epilogue,
        )
    # Stochastic rounding may choose either adjacent BF16 value (one ULP).
    torch.testing.assert_close(d.float().cpu(), a.float() @ b.float().T, rtol=8e-3, atol=1e-5)


@pytest.mark.parametrize(
    "dtype,major,gran",
    [
        (torch.float32, "k", 1),
        (torch.float32, "mn", 1),
        (torch.float32, "k", 8),
        (torch.float32, "mn", 128),
        (torch.int16, "k", 1),
        (torch.int16, "mn", 8),
    ],
)
@tilelang.testing.requires_ascend
def test_transform_sf_strided(dtype, major, gran):
    mn, k = 257, 1025
    cols = (k + (31 if dtype == torch.float32 else 63)) // (32 if dtype == torch.float32 else 64)
    shape = (2, (mn + gran - 1) // gran, cols)
    generator = torch.Generator().manual_seed(17)
    if dtype == torch.float32:
        exponent = torch.randint(0, 256, shape, dtype=torch.int32, generator=generator)
        values = (exponent << 23).view(torch.float32)
        padded = torch.nn.functional.pad(exponent, (0, cols % 2))
        expected = (padded[..., 0::2] | (padded[..., 1::2] << 8)).to(torch.int16)
    else:
        values = torch.randint(-32768, 32768, shape, dtype=dtype, generator=generator)
        expected = values
    expected = expected.repeat_interleave(gran, dim=-2)[..., :mn, :]
    physical = values if major == "k" else values.transpose(-2, -1)
    storage = torch.full((2, physical.shape[-2] + 1, physical.shape[-1] + 7), -3, dtype=dtype, device="npu")
    view = storage[:, 1:, 3 : 3 + physical.shape[-1]]
    view.copy_(physical)
    scale = view if major == "k" else view.transpose(-2, -1)
    output = api._transform_sf_into_required_layout(scale, mn, k, (gran, gran, 32), is_sfa=True, num_groups=2)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert output.stride() == (mn * ((k + 63) // 64), 1, mn)


@pytest.mark.parametrize("k_grouped,major", [(False, "k"), (False, "mn"), (True, "k"), (True, "mn")])
@tilelang.testing.requires_ascend
def test_transform_sf_group_padding(k_grouped, major):
    mn, k = (513, 8193) if k_grouped else (20001, 97)
    ends = [65, 4097, 8065] if k_grouped else [17, 8449, 16897]
    values = torch.full((mn, (k + 31) // 32), 0.5)
    valid = torch.zeros(values.shape, dtype=torch.bool)
    begin = 0
    for end in ends:
        if k_grouped:
            valid[:, begin // 32 : (end + 31) // 32] = True
        else:
            valid[begin:end, :] = True
        begin = (end + 255) // 256 * 256
    values[~valid] = -3  # Padding must be cleared before validating FP32 bits.
    exponent = torch.where(valid, 126, 0)
    exponent = torch.nn.functional.pad(exponent, (0, exponent.shape[-1] % 2))
    expected = (exponent[:, 0::2] | (exponent[:, 1::2] << 8)).to(torch.int16)
    scale = values.npu()
    if major == "mn":
        scale = scale.T.contiguous().T
    layout = torch.tensor(ends, dtype=torch.int32, device="npu")
    output = api._transform_sf_into_required_layout(scale, mn, k, (1, 1, 32), is_sfa=True, grouped_layout=layout, k_grouped=k_grouped)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_transform_sf_passthrough_and_empty():
    packed = torch.full((2, 17), -1, dtype=torch.int16, device="npu").T
    assert api._transform_sf_into_required_layout(packed, 17, 65, (1, 1, 32), is_sfa=True) is packed
    for mn, k in [(0, 65), (17, 0)]:
        scale = torch.empty((mn, (k + 31) // 32), device="npu")
        output = api._transform_sf_into_required_layout(scale, mn, k, (1, 1, 32), is_sfa=True)
        assert output.shape == (mn, (k + 63) // 64)
        assert output.dtype == torch.int16
