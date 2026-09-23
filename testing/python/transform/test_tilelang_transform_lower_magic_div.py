"""Tests for tl.LowerMagicDiv (tl.enable_magic_div).

Covers: magic params + mul-shift expansion, divisor dedup, d == 1 select,
per-site fallback, safe index widening, and control/data dependence safety.
"""

import ctypes

import pytest
import tilelang
from tilelang import libinfo
import tilelang.language as T
import tilelang.testing
import torch
from tilelang.transform import PassConfigKey
from tilelang.tools.compile_only import cuda_codegen_available
from tvm.target import Target

MAGIC_CONFIG = {
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    PassConfigKey.TL_ENABLE_MAGIC_DIV: True,
}

requires_cuda_codegen = pytest.mark.skipif(not cuda_codegen_available(), reason="CUDA codegen is not built")
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA or ROCm GPU")


def _scales_kernel():
    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel(in_scales, out_scales, block: int = 256):
        num_experts = T.dynamic("num_experts")
        size_n = T.dynamic("size_n")
        num_groups = T.dynamic("num_groups")
        in_scales: T.Tensor[(num_experts, size_n, num_groups), T.uint8]
        out_scales: T.Tensor[(num_experts, num_groups, size_n), T.uint8]
        total = num_experts * num_groups * size_n
        with T.Kernel(T.ceildiv(total, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < total:
                    e = idx // (num_groups * size_n)
                    f = idx - e * num_groups * size_n
                    g = f // size_n
                    n = f - g * size_n
                    out_scales[e, g, n] = in_scales[e, n, g]

    return kernel


def _lower(func, target_str):
    from tilelang.tools.compile_only import resolve_target

    target = Target(resolve_target(target_str))
    with tilelang.transform.PassContext(opt_level=3, config=MAGIC_CONFIG), target:
        artifact = tilelang.lower(func, target=target, enable_device_compile=False)
    return artifact.kernel_source


@requires_cuda_codegen
def test_magic_div_cuda_codegen():
    src = _lower(_scales_kernel().get_tir(), "cuda -arch=sm_90")
    assert "__umulhi" in src
    # two unique divisors (num_groups*size_n and size_n) -> two m/s param pairs
    assert "tl_magic_m_0" in src and "tl_magic_s_0" in src
    assert "tl_magic_m_1" in src and "tl_magic_s_1" in src
    # d == 1 select is emitted in the expansion
    assert "== 1) ?" in src or "== 1 ?" in src
    # Each div/mod pair shares one quotient instead of issuing two mul-highs.
    assert src.count("__umulhi") == 2
    assert "tl_magic_r_0" in src and "tl_magic_r_1" in src
    remainder_bind = next(line for line in src.splitlines() if "tl_magic_r_0" in line)
    assert "tl_magic_floormod_i32" in remainder_bind
    # Keep the complete output unflatten chain transparent to FlattenBuffer;
    # opaque magic values here prevent NVCC from recovering the linear idx.
    output_store = next(line for line in src.splitlines() if "out_scales[" in line)
    output_index = output_store.split("out_scales[", 1)[1].split("]", 1)[0]
    assert "tl_magic" not in output_index
    assert "/" not in output_index and "%" not in output_index


def test_magic_div_cpu_codegen():
    src = _lower(_scales_kernel().get_tir(), "c")
    assert "tl_magic_m_0" in src
    assert ">> 32 >>" in src


def _fallback_kernel():
    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel(A, B, block: int = 128):
        n = T.dynamic("n")
        A: T.Tensor[(n,), T.int32]
        B: T.Tensor[(n,), T.int32]
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    # constant divisor: left to the backend
                    a = A[idx] // 4
                    # thread-dependent divisor: not launch-invariant
                    b = A[idx] // (i + 1)
                    B[idx] = a + b

    return kernel


def _mod_only_kernel():
    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel(A, B, block: int = 128):
        n = T.dynamic("n")
        divisor = T.dynamic("divisor")
        A: T.Tensor[(n,), T.int32]
        B: T.Tensor[(divisor,), T.int32]
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    A[idx] = idx % divisor

    return kernel


def _condition_reuse_kernel():
    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel(A, divisor_buffer, limit_buffer, block: int = 128):
        n = T.dynamic("n")
        divisor = T.dynamic("divisor")
        limit = T.dynamic("limit")
        A: T.Tensor[(n,), T.int32]
        divisor_buffer: T.Tensor[(divisor,), T.uint8]
        limit_buffer: T.Tensor[(limit,), T.uint8]
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    q = idx // divisor
                    r = idx % divisor
                    if q < limit and r != 3:
                        A[idx] = q + r

    return kernel


def _trunc_condition_no_reuse_kernel():
    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel(A, divisor_buffer, limit_buffer, block: int = 128):
        n = T.dynamic("n")
        divisor = T.dynamic("divisor")
        limit = T.dynamic("limit")
        A: T.Tensor[(n,), T.int32]
        divisor_buffer: T.Tensor[(divisor,), T.uint8]
        limit_buffer: T.Tensor[(limit,), T.uint8]
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    q = T.truncdiv(idx, divisor)
                    if T.truncdiv(idx, divisor) < limit:
                        A[idx] = q

    return kernel


def _condition_runtime_fallback_kernel():
    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel(A, divisor_buffer, limit_buffer, block: int = 128):
        n = T.dynamic("n")
        divisor = T.dynamic("divisor")
        limit = T.dynamic("limit")
        A: T.Tensor[(n,), T.int32]
        divisor_buffer: T.Tensor[(divisor,), T.uint8]
        limit_buffer: T.Tensor[(limit,), T.uint8]
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    # x is negative for idx == 0, so the runtime validity
                    # predicate must select the exact floor-div fallback.
                    x = idx - 1
                    q = x // divisor
                    r = x % divisor
                    if q < limit:
                        A[idx] = q + r

    return kernel


def _write_then_read_kernel():
    capacity = 256
    block = 128
    n = T.dynamic("n")
    divisor = T.dynamic("divisor")

    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor((capacity,), T.int32),
            B: T.Tensor((capacity,), T.int32),
            n_buffer: T.Tensor((n,), T.uint8),
            divisor_buffer: T.Tensor((divisor,), T.uint8),
        ):
            with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
                for tx in T.Parallel(block):
                    idx = bx * block + tx
                    if idx < n:
                        A[idx] = 7 * divisor
                        B[idx] = T.max(A[idx], 0) // divisor

        return main

    return kernel()


def _widened_dividend_kernel():
    lanes = 32
    n = T.dynamic("n")
    size = T.dynamic("size")

    @tilelang.jit(pass_configs={**MAGIC_CONFIG, PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor((size,), T.uint8),
            divisor_buffer: T.Tensor((n,), T.uint8),
            B: T.Tensor((lanes,), T.uint8),
            base: T.int32,
        ):
            with T.Kernel(1, threads=lanes):
                tx = T.get_thread_binding()
                T.assume(base >= 0)
                B[tx] = A[((tx + base) * 4) // n]

        return main

    return kernel()


def _unsafe_host_divisor_kernel():
    block = 32

    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((block,), T.int32), p: T.int32):
            with T.Kernel(1, threads=block):
                tx = T.get_thread_binding()
                if p > 0:
                    A[tx] = tx // (7 // p + 1)

        return main

    return kernel()


def _loop_dependent_dividend_kernel():
    n = T.dynamic("n")
    divisor = T.dynamic("divisor")
    block = 32

    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel():
        @T.prim_func
        def main(
            B: T.Tensor((n, block), T.int32),
            divisor_buffer: T.Tensor((divisor,), T.uint8),
        ):
            with T.Kernel(1, threads=block):
                tx = T.get_thread_binding()
                for k in T.serial(n):
                    B[k, tx] = k // divisor

        return main

    return kernel()


@requires_cuda_codegen
def test_magic_div_fallback_sites_untouched():
    src = _lower(_fallback_kernel().get_tir(), "cuda -arch=sm_90")
    assert "tl_magic_m_0" not in src
    assert "__umulhi" not in src


@requires_cuda_codegen
def test_magic_mod_without_div_keeps_direct_mod_path():
    src = _lower(_mod_only_kernel().get_tir(), "cuda -arch=sm_90")
    assert "__umulhi" in src
    assert "tl_magic_floormod_i32" in src
    assert "int tl_magic_q_" not in src
    remainder_bind = next(line for line in src.splitlines() if "int tl_magic_r_" in line)
    assert "__umulhi" in remainder_bind


@requires_cuda_codegen
def test_magic_condition_reuses_hoisted_divmod_and_validity():
    src = _lower(_condition_reuse_kernel().get_tir(), "cuda -arch=sm_90")
    lines = src.splitlines()
    divisor_validity_lines = [line for line in lines if "bool tl_magic_divisor_valid_" in line]
    group_validity_lines = [line for line in lines if "bool tl_magic_valid_" in line and "divisor_valid" not in line.split("=", 1)[0]]
    assert len(divisor_validity_lines) == 1
    assert len(group_validity_lines) == 1
    assert ">= 0" in group_validity_lines[0]
    assert "tl_magic_divisor_valid_0" in group_validity_lines[0]

    quotient_line = next(line for line in lines if "int tl_magic_q_" in line)
    remainder_line = next(line for line in lines if "int tl_magic_r_" in line)
    assert "tl_magic_valid_0" in quotient_line
    assert "tl_magic_valid_0" in remainder_line
    assert ">= 0" not in quotient_line and ">= 0" not in remainder_line
    assert "2147483646" not in quotient_line and "2147483646" not in remainder_line

    reused_condition = next(line for line in lines if line.lstrip().startswith("if (") and "tl_magic_q_0" in line)
    assert " / divisor" not in reused_condition
    assert " % divisor" in reused_condition


@requires_cuda_codegen
def test_magic_condition_does_not_reuse_truncating_division():
    src = _lower(_trunc_condition_no_reuse_kernel().get_tir(), "cuda -arch=sm_90")
    condition = next(line for line in src.splitlines() if line.lstrip().startswith("if (") and "limit" in line)
    assert "tl_magic_q_" not in condition
    assert " / divisor" in condition


@requires_cuda_codegen
def test_magic_condition_uses_runtime_fallback_when_nonnegative_is_unproven():
    src = _lower(_condition_runtime_fallback_kernel().get_tir(), "cuda -arch=sm_90")
    validity = next(line for line in src.splitlines() if "bool tl_magic_valid_" in line)
    quotient = next(line for line in src.splitlines() if "int tl_magic_q_" in line)
    remainder = next(line for line in src.splitlines() if "int tl_magic_r_" in line)
    condition = next(line for line in src.splitlines() if line.lstrip().startswith("if (") and "limit" in line)
    assert ">= 0" in validity
    assert "__umulhi" in quotient and "tl_magic_floordiv_i32" in quotient
    assert "tl_magic_floormod_i32" in remainder
    assert "tl_magic_q_" in condition and " / divisor" not in condition


@requires_gpu
def test_magic_hoist_preserves_write_then_read_and_partial_block_guard():
    kernel = _write_then_read_kernel()
    n = 129
    divisor = 128
    A = torch.zeros(256, dtype=torch.int32, device="cuda")
    B = torch.full((256,), -1, dtype=torch.int32, device="cuda")
    n_buffer = torch.empty(n, dtype=torch.uint8, device="cuda")
    divisor_buffer = torch.empty(divisor, dtype=torch.uint8, device="cuda")

    kernel(A, B, n_buffer, divisor_buffer)
    torch.cuda.synchronize()

    torch.testing.assert_close(B[:n], torch.full_like(B[:n], 7), rtol=0, atol=0)
    torch.testing.assert_close(B[n:], torch.full_like(B[n:], -1), rtol=0, atol=0)
    source = kernel.get_kernel_source()
    assert "int tl_magic_q_" not in source
    assert "__umulhi" not in source


@requires_gpu
def test_magic_dividend_widens_before_overflow():
    kernel = _widened_dividend_kernel()
    base = 2**30
    divisor = 2**20
    expected_indices = ((torch.arange(32, dtype=torch.int64) + base) * 4) // divisor
    size = int(expected_indices.max()) + 1
    A = torch.zeros(size, dtype=torch.uint8, device="cuda")
    A[:32] = 11
    A[expected_indices.cuda()] = 29
    divisor_buffer = torch.empty(divisor, dtype=torch.uint8, device="cuda")
    B = torch.empty(32, dtype=torch.uint8, device="cuda")

    kernel(A, divisor_buffer, B, base)
    torch.cuda.synchronize()

    torch.testing.assert_close(B, torch.full_like(B, 29), rtol=0, atol=0)
    source = kernel.get_kernel_source()
    quotient = next(line for line in source.splitlines() if "int64_t tl_magic_q_" in line)
    assert "tl_magic_floordiv_i64" in quotient
    assert "__umulhi" in quotient


@requires_gpu
def test_magic_preserves_inactive_unsafe_host_divisor_expression():
    kernel = _unsafe_host_divisor_kernel()
    A = torch.full((32,), 123, dtype=torch.int32, device="cuda")
    for p in (0, -1):
        kernel(A, p)
        torch.cuda.synchronize()
        torch.testing.assert_close(A, torch.full_like(A, 123), rtol=0, atol=0)

    source = kernel.get_kernel_source()
    assert "if (0 < p)" in source


def test_magic_host_helper_rejects_out_of_range_divisors():
    runtime = ctypes.CDLL(libinfo.find_lib_path("tvm_runtime"))
    get_mul = runtime.TileLangHostFastDivmodU32Mul
    get_shift = runtime.TileLangHostFastDivmodU32Shift
    get_mul.argtypes = [ctypes.c_uint32]
    get_mul.restype = ctypes.c_uint32
    get_shift.argtypes = [ctypes.c_uint32]
    get_shift.restype = ctypes.c_int32

    assert get_mul(0) == 0
    assert get_shift(0) == 0
    assert get_mul(1) == 0
    assert get_shift(1) == 0
    assert get_mul(0x7FFFFFFF) == 0x80000002
    assert get_shift(0x7FFFFFFF) == 30
    for divisor in (0x80000000, 0xFFFFFFFF):
        assert get_mul(divisor) == 0
        assert get_shift(divisor) == 0


@requires_gpu
def test_magic_loop_dependent_dividend_stays_inside_loop():
    source = _loop_dependent_dividend_kernel().get_kernel_source()
    loop_pos = source.index("for (int k")
    magic_pos = source.index("__umulhi", loop_pos)
    assert loop_pos < magic_pos
    assert "int tl_magic_q_" not in source


if __name__ == "__main__":
    tilelang.testing.main()
