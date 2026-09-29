"""DeepGEMM GEMM and scale-conversion benchmarks for Ascend."""

import torch

from tilelang.profiler import do_bench
from examples.ascend.deepgemm.config import GemmDesc, Major, select_gemm_config
from examples.ascend.deepgemm.kernels import load_kernels


def run_regression_perf(m, n, k, dtype="bfloat16", loop_order="mnk"):
    """Check one shape, then return GEMM-only latency in ms with L2 flushed."""
    if dtype not in ("bfloat16", "float8_e4m3fn"):
        raise ValueError(f"unsupported benchmark dtype: {dtype}")
    builders = load_kernels(loop_order)
    device = torch.device("npu", torch.npu.current_device())
    desc = GemmDesc(
        m,
        n,
        k,
        dtype,
        dtype,
        "float32",
        expected_m=m,
        expected_k=k,
        num_cores=int(torch.npu.get_device_properties(device.index).cube_core_num),
        outer_stride_a=k,
        outer_stride_b=k,
    )
    config = select_gemm_config(desc)
    fp8 = dtype == "float8_e4m3fn"
    build = builders.build_fp8_gemm if fp8 else builders.build_bf16_gemm
    # Match the dense API's default: static N/K, runtime M and outer strides.
    kernel = build(None, n, k, Major.K, Major.K, "float32", False, config)

    # Small integers are exactly representable in both input formats and FP32 sums.
    a = torch.randint(-2, 3, (m, k), device=device, dtype=torch.float32).to(getattr(torch, dtype))
    b = torch.randint(-2, 3, (n, k), device=device, dtype=torch.float32).to(getattr(torch, dtype))
    d = torch.empty((m, n), device=device, dtype=torch.float32)
    if fp8:
        # Two UE8M0 exponents of 127 encode unit scales, in physical [K/64, MN].
        sfa = torch.full(((k + 63) // 64, m), 0x7F7F, dtype=torch.int16, device=device)
        sfb = torch.full(((k + 63) // 64, n), 0x7F7F, dtype=torch.int16, device=device)
        args = (a, b, sfa, sfb, d, 1.0)
    else:
        args = (a, b, d, 1.0)
    kernel(*args)
    torch.testing.assert_close(d, a.float() @ b.float().T, rtol=0, atol=0)
    torch.npu.synchronize()

    prof = do_bench(lambda: kernel(*args), backend="msprof_detail", _n_warmup=30, _n_repeat=50)
    print(f"    [{dtype}, {loop_order}, {m}x{n}x{k}] {prof.dur_us:.2f} us/iter | {prof.tflops(2.0 * m * n * k):.1f} TFLOPS")
    return prof.dur_ns / 1e6


def run_transform_sf_perf(mn=32768, k=7168, dtype="float32", major="k", gran_mn=1):
    """Check scale packing, then report cold-L2 device latency in milliseconds."""
    from examples.ascend.deepgemm.api import _transform_sf_into_required_layout

    device = torch.device("npu", torch.npu.current_device())
    cols = (k + (31 if dtype == "float32" else 63)) // (32 if dtype == "float32" else 64)
    shape = ((mn + gran_mn - 1) // gran_mn, cols)
    scales = torch.full(shape, 1.0 if dtype == "float32" else 0x7F7F, dtype=getattr(torch, dtype), device=device)
    if major == "mn":
        scales = scales.T.contiguous().T

    def run():
        return _transform_sf_into_required_layout(scales, mn, k, (gran_mn, gran_mn, 32), is_sfa=True)

    expected = torch.full((mn, (k + 63) // 64), 0x7F7F, dtype=torch.int16)
    if dtype == "float32" and cols % 2:
        expected[:, -1] = 0x007F
    output = run()
    assert output.data_ptr() != scales.data_ptr(), "compact int16 MN-major input bypasses transform_sf"
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    prof = do_bench(run, backend="msprof_detail", cache_size=8192, _n_warmup=5, _n_repeat=12, early_stop_baseline=None)
    assert prof.dur_ns > 0 and prof.aiv_total_cycles > 0
    num_bytes = scales.numel() * scales.element_size() + output.numel() * output.element_size()
    print(f"    [transform_sf, {dtype}, {major}, gran={gran_mn}] {prof.dur_us:.2f} us/iter | {prof.gbps(num_bytes):.2f} GB/s")
    return prof.dur_ns / 1e6
