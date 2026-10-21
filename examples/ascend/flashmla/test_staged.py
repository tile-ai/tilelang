"""Device regression for automatic versions across rescale, jobs, and launches."""

import math

import pytest
import torch

from examples.ascend.flashmla.staged import compile_prefill


@pytest.mark.parametrize("pipeline_stages", [True, False], ids=["pipeline", "flat"])
def test_last_tile_rescale_across_jobs_and_launches(pipeline_stages):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    nq, nk, topk = 33, 128, 128
    q = torch.zeros((nq, 64, 512), dtype=torch.bfloat16, device="npu")
    q[:, 32:, :] = 1
    kv = torch.zeros((nk, 512), dtype=torch.bfloat16, device="npu")
    indices = torch.arange(topk, dtype=torch.int32, device="npu").expand(nq, topk).contiguous()
    sink = torch.zeros(64, dtype=torch.float32, device="npu")
    lengths = torch.full((nq,), topk, dtype=torch.int32, device="npu")
    kernel = compile_prefill(nq, nk, topk, sink=False, scale=0.5, pipeline_stages=pipeline_stages)

    # Only AIV1 requests rescaling, on the last KV tile. The second query job
    # reuses all storage; changing KV between launches detects stale state.
    for value in (1, 2, 1, 2, 1):
        kv[64:] = value
        out, maximum, lse = kernel(q, kv, indices, sink, lengths)
        torch.npu.synchronize()
        expected = torch.full((nq, 64, 512), value / 2, dtype=torch.bfloat16)
        expected[:, 32:] = value
        expected_max = torch.zeros((nq, 64), dtype=torch.float32)
        expected_max[:, 32:] = 256 * value
        expected_lse = torch.full((nq, 64), math.log(128), dtype=torch.float32)
        expected_lse[:, 32:] = 256 * value + math.log(64)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(maximum.cpu(), expected_max, rtol=0, atol=0)
        torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=2e-5)


def test_index_group_tail_and_query_state():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    # Eleven KV tiles cross the 640-index group boundary and rotate versions
    # by an odd count. Block 0 processes three distinct queries.
    nq, nk, topk = 65, 704, 704
    q = torch.zeros((nq, 64, 512), dtype=torch.bfloat16, device="npu")
    row_values = (torch.arange(nk, device="npu") % 7).to(torch.bfloat16)
    kv = row_values[:, None].expand(nk, 512).contiguous()
    indices = (torch.arange(topk)[None, :] + torch.arange(nq)[:, None] * 11) % nk
    indices[:, ::17] = -1
    indices = indices.to(device="npu", dtype=torch.int32)
    lengths = torch.tensor([704, 640, 64, 0], dtype=torch.int32).repeat(17)[:nq].to("npu")
    sink = torch.zeros(64, dtype=torch.float32, device="npu")
    kernel = compile_prefill(nq, nk, topk, sink=False, variable_lengths=True)
    out, maximum, lse = kernel(q, kv, indices, sink, lengths)
    torch.npu.synchronize()

    indices_cpu, lengths_cpu = indices.cpu(), lengths.cpu()
    valid = (indices_cpu >= 0) & (torch.arange(topk)[None, :] < lengths_cpu[:, None])
    count = valid.sum(dim=1)
    values = row_values.cpu()[indices_cpu.clamp(min=0)].float()
    average = (values * valid).sum(dim=1) / count.clamp(min=1)
    expected = average.to(torch.bfloat16)[:, None, None].expand(nq, 64, 512)
    expected_max = torch.where(count > 0, 0.0, -torch.inf)[:, None].expand(nq, 64)
    expected_lse = torch.where(count > 0, count.float().log(), torch.inf)[:, None].expand(nq, 64)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0.015625)
    torch.testing.assert_close(maximum.cpu(), expected_max, rtol=0, atol=0)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=2e-5)


def test_alternating_rescale_and_accumulate():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    nq, nk = 65, 448
    q = torch.zeros((nq, 64, 512), dtype=torch.bfloat16, device="npu")
    q[:, 32:, :] = 1
    q[1::2, 32:, :] = -1
    tile_values = torch.tensor([0, 1, 1, 2, 2, 3, 3], dtype=torch.bfloat16)
    values = tile_values.repeat_interleave(64)
    kv = values[:, None].expand(nk, 512).contiguous().to("npu")
    indices = torch.arange(nk, dtype=torch.int32).expand(nq, nk).clone()
    indices[2::4] = indices[2::4].flip(1)
    indices[3::4] = indices[3::4].flip(1)
    indices = indices.to("npu")
    sink = torch.zeros(64, dtype=torch.float32, device="npu")
    lengths = torch.full((nq,), nk, dtype=torch.int32, device="npu")
    kernel = compile_prefill(nq, nk, nk, sink=False, scale=0.5)

    # Alternating large score jumps exercise conditional handoffs; reversed
    # queries swap which AIV1 rows rescale. Odd tile count rotates ring slots.
    expected = torch.full((nq, 64, 512), 12 / 7, dtype=torch.bfloat16)
    expected[::2, 32:, :] = 3
    expected[1::2, 32:, :] = 0
    expected_max = torch.zeros((nq, 64), dtype=torch.float32)
    expected_max[::2, 32:] = 768
    expected_lse = torch.full((nq, 64), math.log(nk), dtype=torch.float32)
    expected_lse[::2, 32:] = 768 + math.log(128)
    expected_lse[1::2, 32:] = math.log(64)
    for _ in range(3):
        out, maximum, lse = kernel(q, kv, indices, sink, lengths)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0.0078125)
        torch.testing.assert_close(maximum.cpu(), expected_max, rtol=0, atol=0)
        torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=8e-5)


@pytest.mark.parametrize("nb", [5, 15, 16], ids=["metadata-lead-start", "q2-limit", "single-q-fallback"])
def test_query_prefetch_boundaries(nb):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    # Three query jobs force the two-Q path to reuse its first physical slot.
    nq, nk, topk = 65, nb * 64, nb * 64
    q = torch.zeros((nq, 64, 512), dtype=torch.bfloat16, device="npu")
    row_values = (torch.arange(nk, device="npu") % 7).to(torch.bfloat16)
    kv = row_values[:, None].expand(nk, 512).contiguous()
    indices = (torch.arange(topk)[None, :] + torch.arange(nq)[:, None] * 11) % nk
    indices[:, ::17] = -1
    indices = indices.to(device="npu", dtype=torch.int32)
    lengths = torch.tensor([0, 1, 63, 64, 65, topk - 1, topk], dtype=torch.int32).repeat(10)[:nq].to("npu")
    sink = torch.zeros(64, dtype=torch.float32, device="npu")
    kernel = compile_prefill(nq, nk, topk, sink=False, variable_lengths=True)
    out, maximum, lse = kernel(q, kv, indices, sink, lengths)
    torch.npu.synchronize()

    indices_cpu, lengths_cpu = indices.cpu(), lengths.cpu()
    valid = (indices_cpu >= 0) & (torch.arange(topk)[None, :] < lengths_cpu[:, None])
    count = valid.sum(dim=1)
    values = row_values.cpu()[indices_cpu.clamp(min=0)].float()
    average = (values * valid).sum(dim=1) / count.clamp(min=1)
    expected = average.to(torch.bfloat16)[:, None, None].expand(nq, 64, 512)
    expected_max = torch.where(count > 0, 0.0, -torch.inf)[:, None].expand(nq, 64)
    expected_lse = torch.where(count > 0, count.float().log(), torch.inf)[:, None].expand(nq, 64)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0.015625)
    torch.testing.assert_close(maximum.cpu(), expected_max, rtol=0, atol=0)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=2e-5)


@pytest.fixture
def flashmla_reference(monkeypatch):
    """The same separately built reference used by run_decode.py."""
    import importlib
    import os
    from pathlib import Path

    location = os.environ.get("FLASHMLA_REFERENCE")
    if not location:
        pytest.skip("Set FLASHMLA_REFERENCE to the pinned FlashMLA reference checkout")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    reference = Path(location).resolve()
    monkeypatch.syspath_prepend(str(reference / "tests"))
    monkeypatch.syspath_prepend(str(reference))
    modules = tuple(importlib.import_module(name) for name in ("lib", "ref", "quant", "kernelkit"))
    for module in modules:
        assert Path(module.__file__).resolve().is_relative_to(reference), f"Unexpected reference module: {module.__file__}"
    return modules


@pytest.mark.parametrize("extra_format", ["fp8", "fp4"])
@pytest.mark.parametrize("main_lengths", [True, False], ids=["main-lengths", "extra-lengths"])
def test_decode_independent_lengths_and_group_tails(flashmla_reference, extra_format, main_lengths):
    from dataclasses import replace

    from examples.ascend.flashmla.decode import prepare_decode

    lib, ref, quant, kk = flashmla_reference
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    fp8 = quant.KVCacheLayout.V41_FP8Sparse
    fmt = quant.KVCacheLayout.V41_FP4Sparse if extra_format == "fp4" else fp8
    rp = lib.RawTestParamForDecode(
        35,
        64,
        3,
        1,
        1153,
        False,
        704,
        extra_s_k=1537,
        extra_topk=768,
        block_size=53,
        extra_block_size=61,
        have_topk_length=main_lengths,
        have_extra_topk_length=not main_lengths,
        have_zero_seqlen_k=False,
        enable_attn_sink=True,
        kvcache_layout=fp8,
        extra_kvcache_layout=fmt,
        seed=42,
    )
    p = rp.to_test_param()
    # Generate every selected KV entry before imposing boundary lengths.
    # Otherwise the reference fixture poisons entries beyond random lengths.
    fixture_param = replace(p, have_topk_length=False, decode=replace(p.decode, have_extra_topk_length=False))
    old_dtype = torch.get_default_dtype()
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_default_dtype(torch.bfloat16)
        torch.set_float32_matmul_precision("high")
        with torch.device("npu:0"):
            inputs = lib.generate_testcase_for_decode(fixture_param)
            inputs.p = p
            inputs.kv_scope.t = p
            inputs.extra_kv_scope.t = p
            max_length = rp.topk if rp.have_topk_length else rp.extra_topk
            boundary_lengths = [0, 1, 63, 64, 65, 639, 640, 641, max_length]
            pattern = torch.tensor(boundary_lengths, dtype=torch.int32, device=inputs.q.device)
            active_lengths = pattern.repeat((rp.b + len(boundary_lengths) - 1) // len(boundary_lengths))[: rp.b].contiguous()
            if rp.have_topk_length:
                inputs.kv_scope.topk_length = active_lengths
            else:
                inputs.extra_kv_scope.topk_length = active_lengths
            assert (inputs.kv_scope.topk_length is not None) != (inputs.extra_kv_scope.topk_length is not None)
            assert not inputs.q.is_contiguous()
            assert not inputs.kv_scope.indices_in_kvcache.is_contiguous()
            assert not inputs.extra_kv_scope.indices_in_kvcache.is_contiguous()
            torch.npu.synchronize()
            _, run = prepare_decode(p, inputs, implementation="staged")
            out, lse = run()
            torch.npu.synchronize()
            expected = ref.ref_sparse_attn_decode(p, inputs)
            assert kk.check_is_allclose("staged out", out, expected[0], abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
            assert kk.check_is_allclose("staged lse", lse, expected[1], abs_tol=1e-6, rel_tol=8.01 / 65536)
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_float32_matmul_precision(old_precision)
