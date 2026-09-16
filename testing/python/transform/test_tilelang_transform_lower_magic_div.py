"""Compile-only tests for tl.LowerMagicDiv (tl.enable_magic_div).

Covers: magic params + mul-shift expansion, divisor dedup, d == 1 select,
and per-site fallback (negative / thread-dependent / constant divisors are
not rewritten). No GPU required.
"""

import tilelang
import tilelang.language as T
from tilelang.transform import PassConfigKey
from tvm.target import Target

MAGIC_CONFIG = {
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    PassConfigKey.TL_ENABLE_MAGIC_DIV: True,
}


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


def test_magic_div_fallback_sites_untouched():
    src = _lower(_fallback_kernel().get_tir(), "cuda -arch=sm_90")
    assert "tl_magic_m_0" not in src
    assert "__umulhi" not in src


def test_magic_mod_without_div_keeps_direct_mod_path():
    src = _lower(_mod_only_kernel().get_tir(), "cuda -arch=sm_90")
    assert "__umulhi" in src
    assert "tl_magic_floormod_i32" in src
    assert "int tl_magic_q_" not in src
    remainder_bind = next(line for line in src.splitlines() if "int tl_magic_r_" in line)
    assert "__umulhi" in remainder_bind


if __name__ == "__main__":
    test_magic_div_cuda_codegen()
    test_magic_div_cpu_codegen()
    test_magic_div_fallback_sites_untouched()
    test_magic_mod_without_div_keeps_direct_mod_path()
    print("ALL PASS")
