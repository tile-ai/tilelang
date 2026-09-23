"""Tests for tl.LowerMagicDiv (tl.enable_magic_div).

Covers: magic params + mul-shift expansion, divisor dedup, d == 1 select,
per-site fallback, safe index widening, and control/data dependence safety.
"""

import ctypes

import pytest
import tilelang
import tvm
from tilelang import libinfo
import tilelang.language as T
import tilelang.testing
import torch
from tilelang.transform import PassConfigKey
from tilelang.tools.compile_only import cuda_codegen_available
from tvm import te, tirx
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
    assert "if (b == 0) return 0;" in src


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


def _shape_product_widened_dividend_kernel():
    lanes = 32
    n = T.dynamic("n")
    m = T.dynamic("m")

    @tilelang.jit(pass_configs={**MAGIC_CONFIG, PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor((8,), T.int32),
            n_buffer: T.Tensor((n,), T.uint8),
            m_buffer: T.Tensor((m,), T.uint8),
            B: T.Tensor((lanes,), T.int32),
            divisor: T.int32,
        ):
            with T.Kernel(1, threads=lanes):
                tx = T.get_thread_binding()
                T.assume(divisor > 0)
                B[tx] = A[(n * m + tx) // divisor]

        return main

    return kernel()


def _shape_constant_widened_dividend_kernel():
    lanes = 32
    n = T.dynamic("n")

    @tilelang.jit(pass_configs={**MAGIC_CONFIG, PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor((8,), T.int32),
            n_buffer: T.Tensor((n,), T.uint8),
            B: T.Tensor((lanes,), T.int32),
            divisor: T.int32,
        ):
            with T.Kernel(1, threads=lanes):
                tx = T.get_thread_binding()
                T.assume(divisor > 0)
                B[tx] = A[(n * 50_000 + tx) // divisor]

        return main

    return kernel()


def _shape_sum_widened_dividend_kernel():
    lanes = 32
    n = T.dynamic("n")
    m = T.dynamic("m")

    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor((8,), T.int32),
            n_buffer: T.Tensor((n, 0), T.uint8),
            m_buffer: T.Tensor((m, 0), T.uint8),
            B: T.Tensor((lanes,), T.int32),
            divisor: T.int32,
        ):
            with T.Kernel(1, threads=lanes):
                tx = T.get_thread_binding()
                T.assume(divisor > 0)
                B[tx] = A[(n + m + tx) // divisor]

        return main

    return kernel()


def _trunc_overflow_dividend_kernel():
    lanes = 32
    n = T.dynamic("n")

    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel():
        @T.prim_func
        def main(
            n_buffer: T.Tensor((n,), T.uint8),
            quotient: T.Tensor((lanes,), T.int32),
            remainder: T.Tensor((lanes,), T.int32),
            base: T.int32,
        ):
            with T.Kernel(1, threads=lanes):
                tx = T.get_thread_binding()
                T.assume(base >= 0)
                quotient[tx] = T.truncdiv(tx + base, n)
                remainder[tx] = T.truncmod(tx + base, n)

        return main

    return kernel()


def _trunc_overflow_divisor_kernel():
    lanes = 32

    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel():
        @T.prim_func
        def main(
            quotient: T.Tensor((lanes,), T.int32),
            remainder: T.Tensor((lanes,), T.int32),
            base: T.int32,
        ):
            with T.Kernel(1, threads=lanes):
                tx = T.get_thread_binding()
                T.assume(base >= 0)
                quotient[tx] = T.truncdiv(tx, base + 1)
                remainder[tx] = T.truncmod(tx, base + 1)

        return main

    return kernel()


def _guarded_zero_divisor_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def main(
        n_buffer: T.Tensor((n,), T.uint8),
        output: T.Tensor((32,), T.int32),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx < n:
                output[tx] = tx // n

    return main


def _combined_guard_zero_divisor_kernel():
    n = T.dynamic("n")
    gate = T.dynamic("gate")

    @T.prim_func
    def main(
        n_buffer: T.Tensor((n,), T.uint8),
        gate_buffer: T.Tensor((gate,), T.uint8),
        output: T.Tensor((32,), T.int32),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx < n and gate > 0:
                output[tx] = tx // n

    return main


def _unrelated_guard_zero_divisor_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def main(
        n_buffer: T.Tensor((n,), T.uint8),
        output: T.Tensor((32,), T.int32),
        gate: T.int32,
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if gate > 0:
                output[tx] = tx // n

    return main


def _select_guard_zero_divisor_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def main(
        n_buffer: T.Tensor((n,), T.uint8),
        output: T.Tensor((32,), T.int32),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            output[tx] = T.if_then_else(tx < n, tx // n, 0)

    return main


def _nested_block_binding_func():
    n = tirx.Var("n", "int32")
    data = tirx.decl_buffer((n,), "uint8", name="D")
    output = tirx.decl_buffer((64,), "int32", name="O")
    bx = te.thread_axis("blockIdx.x")
    by = te.thread_axis("blockIdx.y")
    inner = tirx.AttrStmt(
        by,
        "thread_extent",
        2,
        tirx.BufferStore(
            output,
            tirx.floordiv(by.var, n + 1),
            [bx.var * 2 + by.var],
        ),
    )
    body = tirx.AttrStmt(
        bx,
        "thread_extent",
        2,
        tirx.SeqStmt([tirx.Evaluate(tirx.floordiv(bx.var, n + 1)), inner]),
    )
    return tirx.PrimFunc(
        [data.data, output.data, n],
        body,
        buffer_map={data.data: data, output.data: output},
    ).with_attr("global_symbol", "main")


def _magic_named_local_binding_func():
    n = tirx.Var("n", "int32")
    output = tirx.decl_buffer((32,), "int32", name="O")
    tx = te.thread_axis("threadIdx.x")
    user_var = tirx.Var("tl_magic_user", "int32")
    body = tirx.AttrStmt(
        tx,
        "thread_extent",
        32,
        tirx.SeqStmt(
            [
                tirx.Bind(user_var, tx.var),
                tirx.BufferStore(output, tirx.floordiv(user_var, n + 1), [tx.var]),
            ]
        ),
    )
    return tirx.PrimFunc([output.data, n], body, buffer_map={output.data: output}).with_attr("global_symbol", "main")


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
def test_magic_shape_product_widens_before_index_overflow():
    kernel = _shape_product_widened_dividend_kernel()
    n = 50_000
    m = 50_000
    divisor = 1_000_000_000
    n_buffer = torch.empty(n, dtype=torch.uint8, device="cuda")
    m_buffer = torch.empty(m, dtype=torch.uint8, device="cuda")
    backing = torch.arange(32, dtype=torch.int32, device="cuda") * 100 + 5
    A = backing[8:16]
    B = torch.empty(32, dtype=torch.int32, device="cuda")

    kernel(A, n_buffer, m_buffer, B, divisor)
    torch.cuda.synchronize()

    expected = backing[10]
    torch.testing.assert_close(B, expected.expand_as(B), rtol=0, atol=0)
    source = kernel.get_kernel_source()
    store = next(line for line in source.splitlines() if " = A[" in line)
    assert "int64_t" in store
    assert "tl_magic_floordiv_i64" in store
    assert "__umulhi" in store


@requires_gpu
def test_magic_shape_times_constant_widens_before_index_overflow():
    kernel = _shape_constant_widened_dividend_kernel()
    n = 50_000
    divisor = 1_000_000_000
    n_buffer = torch.empty(n, dtype=torch.uint8, device="cuda")
    backing = torch.arange(32, dtype=torch.int32, device="cuda") * 100 + 5
    A = backing[8:16]
    B = torch.empty(32, dtype=torch.int32, device="cuda")

    kernel(A, n_buffer, B, divisor)
    torch.cuda.synchronize()

    expected = backing[10]
    torch.testing.assert_close(B, expected.expand_as(B), rtol=0, atol=0)
    source = kernel.get_kernel_source()
    store = next(line for line in source.splitlines() if " = A[" in line)
    assert "int64_t" in store
    assert "tl_magic_floordiv_i64" in store


@requires_gpu
def test_magic_shape_times_constant_is_widened_by_default_safe_access():
    n = T.dynamic("n")

    @tilelang.jit(pass_configs=MAGIC_CONFIG)
    def kernel(A, n_buffer, B, divisor: int):
        A: T.Tensor[(8,), T.int32]
        n_buffer: T.Tensor[(n,), T.uint8]
        B: T.Tensor[(32,), T.int32]
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.assume(divisor > 0)
            B[tx] = A[(n * 50_000 + tx) // divisor]

    n_value = 50_000
    divisor = 1_000_000_000
    n_buffer = torch.empty(n_value, dtype=torch.uint8, device="cuda")
    backing = torch.arange(32, dtype=torch.int32, device="cuda") * 100 + 5
    A = backing[8:16]
    B = torch.empty(32, dtype=torch.int32, device="cuda")

    kernel(A, n_buffer, B, divisor)
    torch.cuda.synchronize()
    torch.testing.assert_close(B, backing[10].expand_as(B), rtol=0, atol=0)


@requires_gpu
def test_magic_shape_sum_widens_before_index_overflow():
    kernel = _shape_sum_widened_dividend_kernel()
    n = 1_200_000_000
    m = 1_200_000_000
    divisor = 1_000_000_000
    n_buffer = torch.empty((n, 0), dtype=torch.uint8, device="cuda")
    m_buffer = torch.empty((m, 0), dtype=torch.uint8, device="cuda")
    backing = torch.arange(32, dtype=torch.int32, device="cuda") * 100 + 5
    A = backing[8:16]
    B = torch.empty(32, dtype=torch.int32, device="cuda")

    kernel(A, n_buffer, m_buffer, B, divisor)
    torch.cuda.synchronize()

    torch.testing.assert_close(B, backing[10].expand_as(B), rtol=0, atol=0)
    source = kernel.get_kernel_source()
    store = next(line for line in source.splitlines() if " = A[" in line)
    assert "int64_t" in store
    assert "tl_magic_floordiv_i64" in store


@requires_gpu
def test_magic_preserves_trunc_divmod_overflow_semantics():
    quotient = torch.empty(32, dtype=torch.int32, device="cuda")
    remainder = torch.empty_like(quotient)

    dividend_kernel = _trunc_overflow_dividend_kernel()
    n_buffer = torch.empty(7, dtype=torch.uint8, device="cuda")
    dividend_kernel(n_buffer, quotient, remainder, 2**31 - 16)
    torch.cuda.synchronize()
    assert (quotient[16].item(), remainder[16].item()) == (-306_783_378, -2)
    assert "__umulhi" not in dividend_kernel.get_kernel_source()

    divisor_kernel = _trunc_overflow_divisor_kernel()
    divisor_kernel(quotient, remainder, 2**31 - 1)
    torch.cuda.synchronize()
    assert (quotient[1].item(), remainder[1].item()) == (0, 1)
    assert "__umulhi" not in divisor_kernel.get_kernel_source()


def test_magic_hoist_does_not_escape_nested_thread_binding():
    n = tirx.Var("n", "int32")
    data = tirx.decl_buffer((n,), "uint8", name="D")
    output = tirx.decl_buffer((64,), "int32", name="O")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    inner = tirx.AttrStmt(
        ty,
        "thread_extent",
        32,
        tirx.BufferStore(
            output,
            tirx.floordiv(ty.var, n + 1),
            [tx.var * 32 + ty.var],
        ),
    )
    body = tirx.AttrStmt(
        tx,
        "thread_extent",
        2,
        tirx.SeqStmt([tirx.Evaluate(tirx.floordiv(tx.var, n + 1)), inner]),
    )
    func = tirx.PrimFunc(
        [data.data, output.data, n],
        body,
        buffer_map={data.data: data, output.data: output},
    ).with_attr("global_symbol", "main")

    with tilelang.transform.PassContext(config=MAGIC_CONFIG):
        lowered = tilelang.transform.LowerMagicDiv()(tvm.IRModule.from_expr(func))
        hoisted = tilelang.transform.MagicCallHoist()(lowered)

    undefined = tirx.analysis.undefined_vars(hoisted["main"].body, hoisted["main"].params)
    assert [var.name for var in undefined] == []
    assert "tl_magic_q_" not in hoisted["main"].script()


def test_magic_hoist_does_not_escape_nested_block_binding():
    func = _nested_block_binding_func()
    with tilelang.transform.PassContext(config=MAGIC_CONFIG):
        lowered = tilelang.transform.LowerMagicDiv()(tvm.IRModule.from_expr(func))
        hoisted = tilelang.transform.MagicCallHoist()(lowered)

    undefined = tirx.analysis.undefined_vars(hoisted["main"].body, hoisted["main"].params)
    assert [var.name for var in undefined] == []
    assert "tl_magic_q_" not in hoisted["main"].script()


def test_magic_hoist_respects_user_bind_with_internal_prefix():
    func = _magic_named_local_binding_func()
    with tilelang.transform.PassContext(config=MAGIC_CONFIG):
        lowered = tilelang.transform.LowerMagicDiv()(tvm.IRModule.from_expr(func))
        hoisted = tilelang.transform.MagicCallHoist()(lowered)

    undefined = tirx.analysis.undefined_vars(hoisted["main"].body, hoisted["main"].params)
    assert [var.name for var in undefined] == []
    assert "tl_magic_q_" not in hoisted["main"].script()


@requires_cuda_codegen
def test_magic_hoist_stays_inside_zero_divisor_guard():
    source = _lower(_guarded_zero_divisor_kernel(), "cuda -arch=sm_90")
    lines = source.splitlines()
    guard_index = next(i for i, line in enumerate(lines) if line.lstrip().startswith("if (") and "threadIdx.x" in line)
    store_index = next(i for i, line in enumerate(lines) if "output[" in line)

    assert "int tl_magic_q_" not in source
    assert guard_index < store_index
    assert "__umulhi" in lines[store_index]
    assert "tl_magic_floordiv_i32" in lines[store_index]


@requires_cuda_codegen
@pytest.mark.parametrize(
    "kernel_factory",
    [_combined_guard_zero_divisor_kernel, _select_guard_zero_divisor_kernel],
)
def test_magic_hoist_stays_inside_compound_zero_divisor_guard(kernel_factory):
    source = _lower(kernel_factory(), "cuda -arch=sm_90")
    assert "int tl_magic_q_" not in source
    assert "tl_magic_floordiv_i32" in source


@requires_cuda_codegen
def test_magic_zero_divisor_fallback_is_safe_under_unrelated_guard():
    source = _lower(_unrelated_guard_zero_divisor_kernel(), "cuda -arch=sm_90")
    helper = source.index("static __device__ __noinline__ int tl_magic_floordiv_i32")
    helper_end = source.index("static __device__ __noinline__ int tl_magic_floormod_i32")
    guard = next(i for i, line in enumerate(source.splitlines()) if line.lstrip().startswith("if (") and "gate" in line)
    quotient = next(i for i, line in enumerate(source.splitlines()) if "int tl_magic_q_" in line)
    assert "if (b == 0) return 0;" in source[helper:helper_end]
    assert quotient < guard


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
