"""Test TL_ENABLE_FAST_MATH control for Ascend fp32 division."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench
from tilelang.transform import PassConfigKey


TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


def make_div_kernel(N, enable_fast_math=False, target=None):
    """Create a kernel that divides two vectors element-wise."""
    n_cores = 64
    num_stages = 2

    jit_kwargs = dict(
        out_idx=[2],
        pass_configs={PassConfigKey.TL_ENABLE_FAST_MATH: enable_fast_math},
    )
    if target is not None:
        jit_kwargs["target"] = target

    @tilelang.jit(**jit_kwargs)
    def div_kernel_factory(N_val):
        num_tokens = T.dynamic("num_tokens")

        @T.prim_func
        def kernel(
            a: T.Tensor((num_tokens, N_val), T.float32),
            b: T.Tensor((num_tokens, N_val), T.float32),
            out: T.Tensor((num_tokens, N_val), T.float32),
        ):
            with T.Kernel(n_cores) as core_id:
                for row in T.Persistent(
                    [num_tokens],
                    n_cores,
                    core_id,
                    group_size=1,
                    num_stages=num_stages,
                ):
                    a_ub = T.alloc_shared((1, N_val), T.float32)
                    b_ub = T.alloc_shared((1, N_val), T.float32)
                    out_ub = T.alloc_shared((1, N_val), T.float32)
                    T.annotate_buffer_versions({a_ub: num_stages, b_ub: num_stages, out_ub: num_stages})
                    T.copy(a[row, 0], a_ub)
                    T.copy(b[row, 0], b_ub)
                    with T.SimdVF():
                        mask_f32 = T.simd.pset(32)
                        for j in range(N_val // 64):
                            col = j * 64
                            va = T.simd.vld(a_ub[0, col])
                            vb = T.simd.vld(b_ub[0, col])
                            vc = T.simd.vdiv(va, vb, mask_f32)
                            T.simd.vsts(out_ub[0, col], vc, mask_f32)
                    T.copy(out_ub, out[row, 0])

        return kernel

    return div_kernel_factory(N)


def compare_bits(out, ref, label, desc_list=None, max_print=16):
    """Compare two fp32 tensors bit-for-bit."""
    out_cpu = out.flatten().cpu().contiguous()
    ref_cpu = ref.flatten().cpu().contiguous()
    out_bits = out_cpu.view(torch.int32)
    ref_bits = ref_cpu.view(torch.int32)
    n = out_cpu.numel()

    mismatches = []
    for i in range(n):
        got_bits = out_bits[i].item() & 0xFFFFFFFF
        ref_bits_i = ref_bits[i].item() & 0xFFFFFFFF
        if got_bits != ref_bits_i:
            desc = desc_list[i] if desc_list is not None and i < len(desc_list) else str(i)
            mismatches.append((i, desc, out_cpu[i].item(), ref_cpu[i].item(), got_bits, ref_bits_i))

    if mismatches:
        print(f"\n  {label}: {len(mismatches)} mismatches out of {n}")
        for idx, desc, got, exp, got_bits, ref_bits_i in mismatches[:max_print]:
            print(f"    [{idx:4d}] {desc:24s} got={got:15.8e} ref={exp:15.8e} got_bits=0x{got_bits:08x} ref_bits=0x{ref_bits_i:08x}")
        if len(mismatches) > max_print:
            print(f"    ... and {len(mismatches) - max_print} more")
    else:
        print(f"  {label}: ALL {n} cases match exactly")

    return len(mismatches)


def check_kernel_sources(kernel_precise, kernel_fast):
    source_precise = kernel_precise.get_kernel_source()
    source_fast = kernel_fast.get_kernel_source()
    # ASC emits AscendC (simd_inst::vdiv); PTO emits pto.vdiv.
    precise_marker = "vdiv_precise" in source_precise
    fast_marker = ("simd_inst::vdiv(" in source_fast) or ("pto.vdiv(" in source_fast)
    print(f"\n  Precise kernel uses vdiv_precise: {precise_marker}")
    print(f"  Fast kernel uses plain vdiv:      {fast_marker}")


@pytest.mark.parametrize("target", TARGETS)
def test_random_cases(target):
    """Random finite inputs. Precise mode should match torch.npu."""
    print("=" * 70)
    print("  Random Finite Division Test")
    print("=" * 70)

    M, N = 64, 512
    torch.manual_seed(42)
    a_npu = torch.randn(M, N, device="npu", dtype=torch.float32) * 128.0
    b_npu = torch.randn(M, N, device="npu", dtype=torch.float32)
    b_npu = torch.where(
        b_npu.abs() < 0.01,
        torch.where(b_npu < 0, -0.01, 0.01),
        b_npu,
    )

    ref_npu = a_npu / b_npu

    kernel_precise = make_div_kernel(N, enable_fast_math=False, target=target)
    kernel_fast = make_div_kernel(N, enable_fast_math=True, target=target)
    out_precise = kernel_precise(a_npu, b_npu)
    out_fast = kernel_fast(a_npu, b_npu)

    check_kernel_sources(kernel_precise, kernel_fast)

    n_precise_npu = compare_bits(out_precise, ref_npu, "PRECISE random vs torch.npu")
    n_fast_npu = compare_bits(out_fast, ref_npu, "FAST random vs torch.npu", max_print=8)

    assert n_precise_npu == 0, f"Precise random must match torch.npu, got {n_precise_npu}"
    if n_fast_npu:
        print("  [INFO] Fast random differs from torch.npu for raw vdiv")


def build_special_vectors():
    cases = [
        (1.0, 0.0, "1 / 0"),
        (-1.0, 0.0, "-1 / 0"),
        (1.0, -0.0, "1 / -0"),
        (0.0, 0.0, "0 / 0"),
        (-0.0, 0.0, "-0 / 0"),
        (float("inf"), 1.0, "inf / 1"),
        (float("-inf"), 2.0, "-inf / 2"),
        (1.0, float("inf"), "1 / inf"),
        (1.0, float("-inf"), "1 / -inf"),
        (float("inf"), float("inf"), "inf / inf"),
        (float("inf"), float("-inf"), "inf / -inf"),
        (float("nan"), 1.0, "nan / 1"),
        (1.0, float("nan"), "1 / nan"),
        (float("nan"), float("nan"), "nan / nan"),
        (float("nan"), 0.0, "nan / 0"),
        (0.0, float("nan"), "0 / nan"),
    ]
    return [x[0] for x in cases], [x[1] for x in cases], [x[2] for x in cases]


@pytest.mark.parametrize("target", TARGETS)
def test_special_cases(target):
    """Constructed divide-by-zero, inf, and nan inputs."""
    print("\n" + "=" * 70)
    print("  Special Division Test")
    print("=" * 70)

    a_list, b_list, desc_list = build_special_vectors()
    N = ((len(a_list) + 63) // 64) * 64
    while len(a_list) < N:
        a_list.append(1.0)
        b_list.append(1.0)
        desc_list.append("padding")

    a_npu = torch.tensor(a_list, dtype=torch.float32, device="npu").unsqueeze(0)
    b_npu = torch.tensor(b_list, dtype=torch.float32, device="npu").unsqueeze(0)

    ref_npu = a_npu / b_npu

    kernel_precise = make_div_kernel(N, enable_fast_math=False, target=target)
    kernel_fast = make_div_kernel(N, enable_fast_math=True, target=target)
    out_precise = kernel_precise(a_npu, b_npu)
    out_fast = kernel_fast(a_npu, b_npu)

    n_precise_npu = compare_bits(out_precise, ref_npu, "PRECISE special vs torch.npu", desc_list)
    n_fast_npu = compare_bits(out_fast, ref_npu, "FAST special vs torch.npu", desc_list)

    assert n_precise_npu == 0, f"Precise special must match torch.npu, got {n_precise_npu}"
    if n_fast_npu:
        print("  [INFO] Fast special differs from torch.npu for raw vdiv")


def bench_performance(target):
    """Benchmark precise vs fast division."""
    print("\n" + "=" * 70)
    print("  Performance Benchmark")
    print("=" * 70)

    M, N = 8192, 8192
    a = torch.randn(M, N, device="npu", dtype=torch.float32)
    b = torch.randn(M, N, device="npu", dtype=torch.float32).clamp(min=0.01)

    kernel_precise = make_div_kernel(N, enable_fast_math=False, target=target)
    kernel_fast = make_div_kernel(N, enable_fast_math=True, target=target)

    kernel_precise(a, b)
    kernel_fast(a, b)

    total_bytes = M * N * 4 * 3
    io_gb = total_bytes / 1e9

    t_precise = do_bench(lambda: kernel_precise(a, b), backend="msprof", _n_warmup=10, _n_repeat=50)
    t_fast = do_bench(lambda: kernel_fast(a, b), backend="msprof", _n_warmup=10, _n_repeat=50)

    bw_precise = io_gb / (t_precise / 1e3)
    bw_fast = io_gb / (t_fast / 1e3)

    print(f"\n  Shape: ({M}, {N}) float32")
    print(f"  Precise: {t_precise * 1000:.1f} us, {bw_precise:.1f} GB/s")
    print(f"  Fast:    {t_fast * 1000:.1f} us, {bw_fast:.1f} GB/s")
    print(f"  Speedup (fast/precise): {t_precise / t_fast:.2f}x")


if __name__ == "__main__":
    for target in TARGETS:
        test_random_cases(target)
        test_special_cases(target)
        bench_performance(target)
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED")
    print("=" * 70)
