import re
import random

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def divmod_kernel(size=4096):
    @T.prim_func
    def main(A: T.Tensor((size,), "int32"), Q: T.Tensor((size,), "int32"), R: T.Tensor((size,), "int32"), d: T.int32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            for i in T.Parallel(128):
                x = A[bx * 128 + i]
                Q[bx * 128 + i] = x // d
                R[bx * 128 + i] = x % d

    return main


@tilelang.testing.requires_cuda
def test_invariant_div():
    kernel = tilelang.compile(
        divmod_kernel(), target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    source = kernel.get_kernel_source()
    signature = re.search(r"void main_kernel\((.*?)\)", source).group(1)
    assert signature.count("fastdiv_multiplier") == 1
    assert signature.count("fastdiv_shift") == 1
    assert "fastdiv_k" not in source and "fastdiv_d" not in source
    a = torch.randint(-(2**31) + 1, 2**31, (4096,), dtype=torch.int32, device="cuda")
    a[:7] = torch.tensor([0, 1, -1, 2**31 - 1, -(2**31), 7, -7], device="cuda", dtype=torch.int32)
    q, r = torch.empty_like(a), torch.empty_like(a)
    for d in (1, 2, 3, 7, 31, 65536, 65537, 2**30 + 1, 2**31 - 1, -3, -(2**31), 7):
        kernel(a, q, r, d)
        expected_q = torch.div(a.to(torch.int64), d, rounding_mode="floor").to(torch.int32)
        expected_r = (a.to(torch.int64) % d).to(torch.int32)
        torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(r, expected_r)


@tilelang.testing.requires_cuda
def test_invariant_div_skips_device_divisor():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = i // A[i]

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "fastdiv_multiplier" not in kernel.get_kernel_source()
    a = torch.arange(1, 129, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    kernel(a, b)
    torch.testing.assert_close(b, torch.zeros_like(b))


@tilelang.testing.requires_cuda
def test_invariant_div_guarded_launch():
    @T.prim_func
    def main(B: T.Tensor((128,), "int32"), d: T.int32):
        if d > 0:
            with T.Kernel(1, threads=128):
                for i in T.Parallel(128):
                    B[i] = i // d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "fastdiv_multiplier" in kernel.get_kernel_source()
    b = torch.full((128,), -999, dtype=torch.int32, device="cuda")
    kernel(b, 0)
    torch.testing.assert_close(b, torch.full_like(b, -999))
    kernel(b, 7)
    torch.testing.assert_close(b, torch.arange(128, dtype=torch.int32, device="cuda") // 7)


@tilelang.testing.requires_cuda
def test_invariant_div_dynamic_shape():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), B: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = i // n + A[0]

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "fastdiv_multiplier" in kernel.get_kernel_source()
    b = torch.empty(128, dtype=torch.int32, device="cuda")
    for n in (1, 7, 31):
        a = torch.ones(n, dtype=torch.int32, device="cuda")
        kernel(a, b)
        torch.testing.assert_close(b, torch.arange(128, dtype=torch.int32, device="cuda") // n + 1)


@tilelang.testing.requires_cuda
def test_invariant_div_multiple_divisors():
    @T.prim_func
    def main(B: T.Tensor((128,), "int32"), d: T.int32, e: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = i // d + i % d + i // e + i % e

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signature = re.search(r"void main_kernel\((.*?)\)", kernel.get_kernel_source()).group(1)
    assert signature.count("fastdiv_multiplier") == 2
    assert signature.count("fastdiv_shift") == 2
    b = torch.empty(128, dtype=torch.int32, device="cuda")
    x = torch.arange(128, dtype=torch.int32, device="cuda")
    for d, e in ((3, 7), (7, 3), (1, 31), (31, 31)):
        kernel(b, d, e)
        torch.testing.assert_close(b, x // d + x % d + x // e + x % e)


@tilelang.testing.requires_cuda
def test_invariant_div_rejects_unsupported_adapter():
    with pytest.raises(ValueError, match="tl.enable_invariant_arithmetic requires CUDA"):
        tilelang.compile(
            divmod_kernel(),
            target="cuda",
            target_host="c",
            execution_backend="cython",
            pass_configs={"tl.enable_invariant_arithmetic": True},
        )


@tilelang.testing.requires_cuda
def test_invariant_div_int64():
    @T.prim_func
    def main(A: T.Tensor((128,), "int64"), B: T.Tensor((128,), "int64"), d: T.int64):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = A[i] // d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "barrett_reciprocal" in kernel.get_kernel_source()
    a = torch.arange(128, dtype=torch.int64, device="cuda") + 2**40
    b = torch.empty_like(a)
    kernel(a, b, 7)
    torch.testing.assert_close(b, a // 7)


@tilelang.testing.requires_cuda
def test_invariant_div_leaves_constant_divisor_unchanged():
    @T.prim_func
    def main(B: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = i // 7

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "fastdiv_multiplier" not in kernel.get_kernel_source()
    b = torch.empty(128, dtype=torch.int32, device="cuda")
    kernel(b)
    torch.testing.assert_close(b, torch.arange(128, dtype=torch.int32, device="cuda") // 7)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("unsigned", [False, True])
def test_barrett_remainder(unsigned):
    dtype = "uint32" if unsigned else "int32"

    @T.prim_func
    def main(A: T.Tensor((1024,), dtype), B: T.Tensor((1024,), dtype), d: T.dtype(dtype)):
        with T.Kernel(8, threads=128) as bx:
            for i in T.Parallel(128):
                B[bx * 128 + i] = A[bx * 128 + i] % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "barrett_reciprocal" in kernel.get_kernel_source()
    values = torch.randint(0 if unsigned else -(2**31), 2**32 if unsigned else 2**31, (1024,), dtype=torch.int64, device="cuda")
    values[:5] = torch.tensor([0, 1, 2**31 - 1, 2**32 - 1 if unsigned else -1, 2**31 if unsigned else -(2**31)], device="cuda")
    a = values.to(torch.uint32 if unsigned else torch.int32)
    b = torch.empty_like(a)
    divisors = [1, 2, 3, 7, 65535, 65536, 65537, 2**31 - 1]
    divisors += [2**31, 2**32 - 1] if unsigned else [-3, -(2**31)]
    for d in divisors:
        kernel(a, b, d)
        torch.testing.assert_close(b.to(torch.int64), values % d)


@tilelang.testing.requires_cuda
def test_proven_exact_division():
    @T.prim_func
    def main(A: T.Tensor((1024,), "int32"), B: T.Tensor((1024,), "int32"), d: T.int32):
        with T.Kernel(8, threads=128) as bx:
            for i in T.Parallel(128):
                x = T.bind(A[bx * 128 + i])
                if x % d == 0:
                    B[bx * 128 + i] = x // d
                else:
                    B[bx * 128 + i] = -777

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    source = kernel.get_kernel_source()
    assert "exact_inverse" in source and "exact_shift" in source
    a = torch.arange(-512, 512, dtype=torch.int32, device="cuda")
    a[:2] = torch.tensor([-(2**31), 2**31 - 1], dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    for d in (1, 2, 3, 6, 7, 12, 128, 65536, 2**30, 2**31 - 1, -3, -(2**31)):
        kernel(a, b, d)
        values = a.to(torch.int64)
        expected = torch.where(values % d == 0, values // d, -777).to(torch.int32)
        torch.testing.assert_close(b, expected)


@tilelang.testing.requires_cuda
def test_composite_layout_divisors():
    @T.prim_func
    def main(B: T.Tensor((1024,), "int32"), height: T.int32, width: T.int32):
        with T.Kernel(8, threads=128) as bx:
            for i in T.Parallel(128):
                linear = bx * 128 + i
                B[linear] = linear // (height * width) + linear % (height * width)

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signature = re.search(r"void main_kernel\((.*?)\)", kernel.get_kernel_source()).group(1)
    assert signature.count("fastdiv_multiplier") == 1
    b = torch.empty(1024, dtype=torch.int32, device="cuda")
    x = torch.arange(1024, dtype=torch.int32, device="cuda")
    for h, w in ((3, 7), (1, 31), (7, 3)):
        kernel(b, h, w)
        torch.testing.assert_close(b, x // (h * w) + x % (h * w))


@tilelang.testing.requires_cuda
def test_unsigned_fast_div_and_rem():
    @T.prim_func
    def main(A: T.Tensor((1024,), "uint32"), Q: T.Tensor((1024,), "uint32"), R: T.Tensor((1024,), "uint32"), d: T.uint32):
        with T.Kernel(8, threads=128) as bx:
            for i in T.Parallel(128):
                Q[bx * 128 + i] = A[bx * 128 + i] // d
                R[bx * 128 + i] = A[bx * 128 + i] % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signature = re.search(r"void main_kernel\((.*?)\)", kernel.get_kernel_source()).group(1)
    assert signature.count("barrett_reciprocal") == 1
    values = torch.randint(0, 2**32, (1024,), dtype=torch.int64, device="cuda")
    values[:4] = torch.tensor([0, 1, 2**31, 2**32 - 1], device="cuda")
    a = values.to(torch.uint32)
    q, r = torch.empty_like(a), torch.empty_like(a)
    for d in (1, 2, 3, 7, 65537, 2**31 - 1, 2**31, 2**32 - 1):
        kernel(a, q, r, d)
        torch.testing.assert_close(q.to(torch.int64), values // d)
        torch.testing.assert_close(r.to(torch.int64), values % d)


@tilelang.testing.requires_cuda
def test_truncating_division_and_remainder():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), Q: T.Tensor((128,), "int32"), R: T.Tensor((128,), "int32"), d: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                Q[i] = T.truncdiv(A[i], d)
                R[i] = T.truncmod(A[i], d)

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "fastdiv_multiplier" in kernel.get_kernel_source()
    a = torch.arange(-64, 64, dtype=torch.int32, device="cuda")
    q, r = torch.empty_like(a), torch.empty_like(a)
    for d in (1, 3, 7, -3, -7):
        kernel(a, q, r, d)
        expected_q = torch.div(a, d, rounding_mode="trunc")
        torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(r, a - expected_q * d)


@tilelang.testing.requires_cuda
def test_exact_proof_does_not_escape_branch():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32"), d: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                x = T.bind(A[i])
                if x % d == 0:
                    B[i] = x // d + 10
                else:
                    B[i] = x // d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    source = kernel.get_kernel_source()
    assert "exact_inverse" in source and "fastdiv_multiplier" in source
    a = torch.arange(-64, 64, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    for d in (3, 6, 7):
        kernel(a, b, d)
        torch.testing.assert_close(b, a // d + (a % d == 0).to(torch.int32) * 10)


@tilelang.testing.requires_cuda
def test_unsigned_exact_division():
    @T.prim_func
    def main(A: T.Tensor((128,), "uint32"), B: T.Tensor((128,), "uint32"), d: T.uint32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                x = T.bind(A[i])
                if x % d == 0:
                    B[i] = x // d
                else:
                    B[i] = T.uint32(0xFFFFFFFF)

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "exact_inverse" in kernel.get_kernel_source()
    b = torch.empty(128, dtype=torch.uint32, device="cuda")
    for d in (1, 2, 3, 6, 7, 65536, 2**31, 2**32 - 1):
        values = torch.arange(128, dtype=torch.int64, device="cuda") * d
        values &= 0xFFFFFFFF
        values[-1] = 2**32 - 1
        a = values.to(torch.uint32)
        kernel(a, b, d)
        expected = torch.where(values % d == 0, values // d, 0xFFFFFFFF)
        torch.testing.assert_close(b.to(torch.int64), expected)


@tilelang.testing.requires_cuda
def test_exact_proof_does_not_apply_to_mutated_buffer():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32"), d: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if A[i] % d == 0:
                    A[i] = A[i] + 1
                    B[i] = A[i] // d
                else:
                    B[i] = -777

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "exact_inverse" not in kernel.get_kernel_source()
    a = torch.arange(128, dtype=torch.int32, device="cuda") * 3
    expected = (a + 1) // 3
    b = torch.empty_like(a)
    kernel(a, b, 3)
    torch.testing.assert_close(b, expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("remainder_only", [False, True])
@pytest.mark.parametrize("truncating", [False, True])
def test_widened_divisor_preserves_int64_arithmetic(remainder_only, truncating):
    @T.prim_func
    def main(A: T.Tensor((128,), "int64"), Q: T.Tensor((128,), "int64"), R: T.Tensor((128,), "int64"), d: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if not remainder_only:
                    Q[i] = T.truncdiv(A[i], T.int64(d)) if truncating else A[i] // T.int64(d)
                R[i] = T.truncmod(A[i], T.int64(d)) if truncating else A[i] % T.int64(d)

    kernel = tilelang.compile(
        main,
        target="cuda",
        target_host="c",
        execution_backend="tvm_ffi",
        pass_configs={"tl.enable_invariant_arithmetic": True},
    )
    source = kernel.get_kernel_source()
    assert "barrett_reciprocal" in source
    assert "fastdiv_multiplier" not in source
    values = [-(2**63) + 1, -(2**40), -(2**31), -1, 0, 1, 2**31 - 1, 2**31, 2**32, 2**40, 2**63 - 1]
    a = torch.tensor((values * 12)[:128], device="cuda", dtype=torch.int64)
    q, r = torch.empty_like(a), torch.empty_like(a)
    for d in (1, 2, 3, 7, 65537, 2**31 - 1, -1, -3, -(2**31)):
        kernel(a, q, r, d)
        expected_q = torch.div(a, d, rounding_mode="trunc" if truncating else "floor")
        if not remainder_only:
            torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(r, a - expected_q * d)


@tilelang.testing.requires_cuda
def test_widened_layout_gather():
    @T.prim_func
    def main(
        A: T.Tensor((8192,), "float32"),
        Offsets: T.Tensor((128,), "int32"),
        B: T.Tensor((128,), "float32"),
        d: T.int32,
        width: T.int32,
        pitch: T.int32,
    ):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                offset = T.bind(Offsets[i])
                if offset % d == 0:
                    element = offset // d
                    B[i] = A[element // width * pitch + element % width]
                else:
                    B[i] = -1.0

    kernel = tilelang.compile(
        main,
        target="cuda",
        target_host="c",
        execution_backend="tvm_ffi",
        pass_configs={"tl.enable_invariant_arithmetic": True},
    )
    source = kernel.get_kernel_source()
    assert "tl::fast_div(((int64_t)" in source
    a = torch.arange(8192, device="cuda", dtype=torch.float32)
    b = torch.empty(128, device="cuda", dtype=torch.float32)
    for d, width, pitch in ((12, 7, 11), (3, 13, 17), (1, 5, 9)):
        offsets = torch.arange(128, device="cuda", dtype=torch.int32) * d
        offsets[::3] += 1
        kernel(a, offsets, b, d, width, pitch)
        element = offsets.long() // d
        expected = torch.where(offsets % d == 0, a[element // width * pitch + element % width], -1.0)
        torch.testing.assert_close(b, expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize("truncating", [False, True])
@pytest.mark.parametrize("remainder_only", [False, True])
def test_native_wide_divmod(dtype, truncating, remainder_only):
    # TVM-FFI's Python scalar ABI is signed int64. Exercise the full unsigned
    # domain through an explicit bit-preserving input cast, not a smaller range.
    signed_abi = dtype == "uint64"
    parameter_dtype = "int64" if signed_abi else dtype

    @T.prim_func
    def main(A: T.Tensor((128,), dtype), Q: T.Tensor((128,), dtype), R: T.Tensor((128,), dtype), d: T.dtype(parameter_dtype)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                divisor = T.cast(d, dtype)
                if not remainder_only:
                    Q[i] = T.truncdiv(A[i], divisor) if truncating else A[i] // divisor
                R[i] = T.truncmod(A[i], divisor) if truncating else A[i] % divisor

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signature = re.search(r"void main_kernel\((.*?)\)", kernel.get_kernel_source()).group(1)
    assert signature.count("barrett_reciprocal") == 1
    unsigned = dtype == "uint64"
    limit = 2**64 - 1 if unsigned else 2**63 - 1
    values = [0, 1, 2, 3, 2**31 - 1, 2**32, 2**40, limit - 1, limit]
    if not unsigned:
        values += [-1, -7, -(2**40), -(2**63) + 1]
    rng = random.Random(0)
    values += [rng.randrange(0 if unsigned else -(2**63) + 1, limit + 1) for _ in range(128 - len(values))]
    a = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
    q, r = torch.empty_like(a), torch.empty_like(a)
    divisors = [1, 2, 3, 7, 65537, 2**32 - 1, 2**32, 2**32 + 1, 2**62, 2**63 - 1]
    if unsigned:
        divisors += [2**63, 2**64 - 1]
    else:
        divisors += [-1, -3, -(2**63)]
    divisors += [rng.randrange(1, limit + 1) for _ in range(12)]
    for d in divisors:
        kernel(a, q, r, d - 2**64 if signed_abi and d >= 2**63 else d)
        quotients = [(abs(x) // abs(d)) * (-1 if (x < 0) != (d < 0) else 1) if truncating else x // d for x in values]
        if not remainder_only:
            assert q.cpu().tolist() == quotients
        assert r.cpu().tolist() == [x - quotient * d for x, quotient in zip(values, quotients)]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16"])
def test_narrow_divmod(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), Q: T.Tensor((128,), dtype), R: T.Tensor((128,), dtype), d: T.dtype(dtype)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                x = T.bind(A[i])
                Q[i] = x // d
                R[i] = x % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signed = dtype.startswith("int")
    bits = 8 if "8" in dtype else 16
    high = 2 ** (bits - int(signed)) - 1
    low = -(2 ** (bits - 1)) if signed else 0
    values = ([low, low + 1, 0, 1, 7, high - 1, high] * 19)[:128]
    a = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
    q, r = torch.empty_like(a), torch.empty_like(a)
    for d in [1, 2, 3, 7, high, -3, low] if signed else [1, 2, 3, 7, high]:
        kernel(a, q, r, d)
        assert q.cpu().tolist() == [x // d for x in values]
        assert r.cpu().tolist() == [x % d for x in values]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype,wide", [("int32", "int64"), ("uint32", "uint64"), ("uint32", "int64"), ("int32", "uint64")])
def test_divmod_preparation_shared_across_widths(dtype, wide):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((4, 128), wide), d: T.dtype(dtype)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                x = T.bind(A[i])
                B[0, i] = T.cast(x // d, wide)
                B[1, i] = T.cast(x % d, wide)
                wide_x = T.bind(T.cast(x, wide))
                B[2, i] = wide_x // T.cast(d, wide)
                B[3, i] = wide_x % T.cast(d, wide)

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    source = kernel.get_kernel_source()
    signature = re.search(r"void main_kernel\((.*?)\)", source).group(1)
    if dtype == "uint32":
        assert signature.count("barrett_reciprocal") == 1
        # Zero extension proves the whole dividend lies in the uint32 range.
        assert "4294967295" not in source
    else:
        assert signature.count("fastdiv_multiplier") == 1
        assert signature.count("fastdiv_shift") == 1
        # Signed-to-unsigned conversion is not a value-preserving extension.
        assert signature.count("barrett_reciprocal") == int(wide == "uint64")
    values = [0, 1, 7, 2**31 - 1]
    values += [-(2**31), -7, -1] if dtype == "int32" else [2**31, 2**32 - 1]
    values = (values * 26)[:128]
    a = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
    b = torch.empty((4, 128), dtype=getattr(torch, wide), device="cuda")
    divisors = [1, 3, 7, -3] if dtype == "int32" else [1, 3, 7, 2**31 + 1, 2**32 - 1]

    def convert(x):
        return x & (2**64 - 1) if wide == "uint64" else x

    for d in divisors:
        kernel(a, b, d)
        expected = [
            [convert(x // d) for x in values],
            [convert(x % d) for x in values],
            [convert(x) // convert(d) for x in values],
            [convert(x) % convert(d) for x in values],
        ]
        assert b.cpu().tolist() == expected


@tilelang.testing.requires_cuda
def test_wide_range_proof_does_not_survive_buffer_mutation():
    @T.prim_func
    def main(A: T.Tensor((128,), "uint64"), B: T.Tensor((128,), "uint64"), d: T.uint32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if A[i] < T.uint64(2**32):
                    A[i] = A[i] + T.uint64(2**40)
                    B[i] = A[i] // T.uint64(d)

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    a = torch.arange(128, device="cuda", dtype=torch.int64).to(torch.uint64)
    b = torch.empty_like(a)
    kernel(a, b, 7)
    assert b.cpu().tolist() == [(x + 2**40) // 7 for x in range(128)]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("wide", ["int64", "uint64"])
def test_narrow_dividend_wide_divisor(wide):
    @T.prim_func
    def main(A: T.Tensor((128,), "uint32"), Q: T.Tensor((128,), wide), R: T.Tensor((128,), wide), bits: T.int64):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                d = T.cast(bits, wide)
                Q[i] = T.cast(A[i], wide) // d
                R[i] = T.cast(A[i], wide) % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signature = re.search(r"void main_kernel\((.*?)\)", kernel.get_kernel_source()).group(1)
    assert signature.count("barrett_reciprocal") == 1
    assert "uint barrett_reciprocal" in signature
    values = ([0, 1, 7, 2**31, 2**32 - 2, 2**32 - 1] * 22)[:128]
    a = torch.tensor(values, dtype=torch.uint32, device="cuda")
    q = torch.empty(128, dtype=getattr(torch, wide), device="cuda")
    r = torch.empty_like(q)
    divisors = [1, 2, 3, 7, 2**32 - 1, 2**32, 2**32 + 1, 2**63 - 1]
    divisors += [2**63, 2**64 - 1] if wide == "uint64" else [-3, -(2**63)]
    for d in divisors:
        kernel(a, q, r, d - 2**64 if d >= 2**63 else d)
        assert q.cpu().tolist() == [x // d for x in values]
        assert r.cpu().tolist() == [x % d for x in values]


@tilelang.testing.requires_cuda
def test_dedup_keeps_widened_product_distinct():
    @T.prim_func
    def main(A: T.Tensor((128,), "uint32"), B: T.Tensor((2, 128), "uint64"), h: T.uint32, w: T.uint32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[0, i] = T.uint64(A[i] // (h * w))
                B[1, i] = T.uint64(A[i]) // (T.uint64(h) * T.uint64(w))

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signature = re.search(r"void main_kernel\((.*?)\)", kernel.get_kernel_source()).group(1)
    assert signature.count("barrett_reciprocal") == 2
    values = [0, 1, 2**31, 2**32 - 1] * 32
    a = torch.tensor(values, dtype=torch.uint32, device="cuda")
    b = torch.empty((2, 128), dtype=torch.uint64, device="cuda")
    for h, w in ((7, 3), (65537, 65537), (2**31, 3), (2**32 - 1, 2**32 - 1)):
        kernel(a, b, h, w)
        assert b.cpu().tolist() == [[x // ((h * w) & 0xFFFFFFFF) for x in values], [x // (h * w) for x in values]]


@tilelang.testing.requires_cuda
def test_invariant_arithmetic_materialization_preserves_lazy_guards():
    @T.prim_func
    def main(B: T.Tensor((128,), "int32"), d: T.int32, e: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if d != 0 and e != 0:
                    q = i // d
                    r = q % e
                    B[i] = T.if_then_else(q + r >= 0 and q + r < 128, q + r, -1)
                else:
                    B[i] = -9

    kernel = tilelang.compile(
        main,
        target="cuda",
        target_host="c",
        execution_backend="tvm_ffi",
        pass_configs={"tl.enable_invariant_arithmetic": True},
    )
    source = kernel.get_kernel_source()
    assert "invariant_value" in source
    assert "tl::fast_div" in source  # Materialization must not expand fallback math.
    out = torch.empty(128, dtype=torch.int32, device="cuda")
    x = torch.arange(128, dtype=torch.int32, device="cuda")
    for d, e in ((0, 0), (0, 7), (7, 0), (1, 1), (7, 3), (-3, 7), (7, -3)):
        kernel(out, d, e)
        if d and e:
            value = x // d + (x // d) % e
            expected = torch.where((value >= 0) & (value < 128), value, -1)
        else:
            expected = torch.full_like(x, -9)
        torch.testing.assert_close(out, expected)


@tilelang.testing.requires_cuda
def test_invariant_arithmetic_materialization_does_not_cache_loads():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32"), d: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = A[i] // d
                A[i] = -A[i]
                B[i] = B[i] + A[i] % d

    kernel = tilelang.compile(
        main,
        target="cuda",
        target_host="c",
        execution_backend="tvm_ffi",
        pass_configs={"tl.enable_invariant_arithmetic": True},
    )
    values = torch.arange(-64, 64, dtype=torch.int32, device="cuda")
    out = torch.empty_like(values)
    for d in (1, 7, -3):
        data = values.clone()
        kernel(data, out, d)
        torch.testing.assert_close(data, -values)
        torch.testing.assert_close(out, values // d + (-values) % d)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("divisor_dtype", ["int32", "uint32"])
@pytest.mark.parametrize(
    "base,dtype",
    [
        (2**31, "int64"),
        (2**48, "int64"),
        (2**64 - 128, "uint64"),
        (-(2**31) - 128, "int64"),
        (-(2**32) - 128, "int64"),
        (-(2**48), "int64"),
    ],
)
def test_wide_index_narrow_divisor_uses_barrett(divisor_dtype, base, dtype):
    @T.prim_func
    def main(Q: T.Tensor((128,), dtype), R: T.Tensor((128,), dtype), d: T.dtype(divisor_dtype)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                x = (T.uint64(base) if dtype == "uint64" else T.int64(base)) + T.cast(i, dtype)
                Q[i] = x // T.cast(d, dtype)
                R[i] = x % T.cast(d, dtype)

    kernel = tilelang.compile(
        main,
        target="cuda",
        target_host="c",
        execution_backend="tvm_ffi",
        pass_configs={"tl.enable_invariant_arithmetic": True},
    )
    source = kernel.get_kernel_source()
    signature = re.search(r"void main_kernel\((.*?)\)", source).group(1)
    assert signature.count("barrett_reciprocal") == 1
    assert "fastdiv_multiplier" not in source
    assert "fastdiv_shift" not in source
    q = torch.empty(128, dtype=getattr(torch, dtype), device="cuda")
    r = torch.empty_like(q)
    divisors = [1, 2, 37, 2**31 - 1]
    divisors += [-3, -(2**31)] if divisor_dtype == "int32" else [2**32 - 1]
    for d in divisors:
        kernel(q, r, d)
        effective_d = d & (2**64 - 1) if dtype == "uint64" else d
        assert q.cpu().tolist() == [(base + i) // effective_d for i in range(128)]
        assert r.cpu().tolist() == [(base + i) % effective_d for i in range(128)]


@tilelang.testing.requires_cuda
def test_wide_remainder_swizzle_keeps_narrow_arithmetic():
    @T.prim_func
    def main(A: T.Tensor((128,), "int64"), Q: T.Tensor((128,), "int64"), R: T.Tensor((128,), "int64"), d: T.int32, e: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                swizzle = (A[i] % T.int64(d)) ^ T.int64((i & 7) * 4)
                Q[i] = swizzle // T.int64(e)
                R[i] = swizzle % T.int64(e)

    kernel = tilelang.compile(
        main,
        target="cuda",
        target_host="c",
        execution_backend="tvm_ffi",
        pass_configs={"tl.enable_invariant_arithmetic": True},
    )
    source = kernel.get_kernel_source()
    signature = re.search(r"void main_kernel\((.*?)\)", source).group(1)
    assert signature.count("barrett_reciprocal") == 1
    assert signature.count("fastdiv_multiplier") == 1
    assert "2147483647" not in source  # No redundant runtime upper-bound guard.
    values = [-(2**63), -(2**40), -1, 0, 1, 2**31, 2**48, 2**63 - 1] * 16
    a = torch.tensor(values, dtype=torch.int64, device="cuda")
    q, r = torch.empty_like(a), torch.empty_like(a)
    for d in [1, 37, 2**31 - 1, -3, -(2**31)]:
        for e in [1, 7, -7]:
            kernel(a, q, r, d, e)
            expected = [(x % d) ^ ((i & 7) * 4) for i, x in enumerate(values)]
            assert q.cpu().tolist() == [x // e for x in expected]
            assert r.cpu().tolist() == [x % e for x in expected]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("truncating", [False, True])
def test_invariant_divmod_exhaustive_byte_domain(dtype, truncating):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), Q: T.Tensor((256,), dtype), R: T.Tensor((256,), dtype), d: T.dtype(dtype)):
        with T.Kernel(2, threads=128) as bx:
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                Q[i] = T.truncdiv(A[i], d) if truncating else A[i] // d
                R[i] = T.truncmod(A[i], d) if truncating else A[i] % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    low, high = (-128, 127) if dtype == "int8" else (0, 255)
    q = torch.empty(256, dtype=getattr(torch, dtype), device="cuda")
    r = torch.empty_like(q)
    for d in range(low, high + 1):
        if d == 0:
            continue
        # Exclude the sole unrepresentable signed quotient, MIN / -1.
        values = [0 if x == low and d == -1 else x for x in range(low, high + 1)]
        a = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
        kernel(a, q, r, d)
        expected_q = [((abs(x) // abs(d)) * (-1 if (x < 0) != (d < 0) else 1)) if truncating else x // d for x in values]
        assert q.cpu().tolist() == expected_q
        assert r.cpu().tolist() == [x - y * d for x, y in zip(values, expected_q)]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("truncating", [False, True])
def test_invariant_signed_domain_boundaries(dtype, truncating):
    @T.prim_func
    def main(A: T.Tensor((512,), dtype), Q: T.Tensor((512,), dtype), R: T.Tensor((512,), dtype), d: T.dtype(dtype)):
        with T.Kernel(4, threads=128) as bx:
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                Q[i] = T.truncdiv(A[i], d) if truncating else A[i] // d
                R[i] = T.truncmod(A[i], d) if truncating else A[i] % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    bits = 32 if dtype == "int32" else 64
    low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    boundaries = {low, low + 1, -1, 0, 1, high - 1, high}
    for bit in range(bits - 1):
        for sign in [-1, 1]:
            for delta in [-1, 0, 1]:
                x = sign * (2**bit) + delta
                if low <= x <= high:
                    boundaries.add(x)
    rng = random.Random(3261)
    values = sorted(boundaries)
    values += [rng.randint(low, high) for _ in range(512 - len(values))]
    divisors = sorted(boundaries - {0}) + [rng.randint(1, high) for _ in range(16)]
    q = torch.empty(512, dtype=getattr(torch, dtype), device="cuda")
    r = torch.empty_like(q)
    for d in divisors:
        inputs = [0 if x == low and d == -1 else x for x in values]
        a = torch.tensor(inputs, dtype=getattr(torch, dtype), device="cuda")
        kernel(a, q, r, d)
        expected_q = [((abs(x) // abs(d)) * (-1 if (x < 0) != (d < 0) else 1)) if truncating else x // d for x in inputs]
        assert q.cpu().tolist() == expected_q
        assert r.cpu().tolist() == [x - y * d for x, y in zip(inputs, expected_q)]


@tilelang.testing.requires_cuda
def test_exact_division_negative_divisor():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32"), d: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                x = T.bind(A[i])
                if d != 0:
                    if x % d == 0:
                        B[i] = x // d
                    else:
                        B[i] = 123
                else:
                    B[i] = 123

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "exact_inverse" in kernel.get_kernel_source()
    values = [-(2**31), -(2**30), -65536, -42, -1, 0, 42, 2**30] * 16
    a = torch.tensor(values, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    for d in [0, 1, 2, 7, -(2**31), -(2**30), -65536, -7, -3, -2]:
        kernel(a, b, d)
        assert b.cpu().tolist() == [x // d if d and x % d == 0 else 123 for x in values]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("constraint", ["none", "nonzero", "positive", "branch"])
def test_invariant_condition_propagation(constraint):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32"), d: T.int32):
        with T.Kernel(1, threads=128):
            if constraint == "positive":
                T.assume(d > 0)
            if constraint == "nonzero":
                T.assume(d != 0)
            for i in T.Parallel(128):
                x = T.bind(A[i])
                if constraint == "branch":
                    if d > 0:
                        B[i] = ((x % d) ^ ((x % 8) * 4)) // d
                    else:
                        B[i] = ((x % d) ^ ((x % 8) * 4)) // d + 1
                else:
                    B[i] = ((x % d) ^ ((x % 8) * 4)) // d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    source = kernel.get_kernel_source()
    calls = [line for line in source.splitlines() if "tl::fast_div(" in line or "tl::fast_rem(" in line]
    assert calls
    if constraint == "positive":
        assert all(line.rstrip().endswith("(bool)1);") for line in calls)
    elif constraint in ("none", "nonzero"):
        assert all(line.rstrip().endswith("(bool)0);") for line in calls)
    a = torch.arange(-64, 64, dtype=torch.int32, device="cuda") * 1000003
    b = torch.empty_like(a)
    divisors = [1, 7, 37, 2**31 - 1]
    if constraint != "positive":
        divisors += [-1, -7, -37, -(2**31)]
    for d in divisors:
        kernel(a, b, d)
        expected = ((a.to(torch.int64) % d) ^ ((a.to(torch.int64) % 8) * 4)) // d
        if constraint == "branch" and d < 0:
            expected += 1
        torch.testing.assert_close(b, expected.to(torch.int32), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_invariant_positive_factors_do_not_prove_positive_product():
    @T.prim_func
    def main(Q: T.Tensor((128,), "int32"), R: T.Tensor((128,), "int32"), a: T.int32, b: T.int32):
        with T.Kernel(1, threads=128):
            T.assume(a > 0)
            T.assume(b > 0)
            for i in T.Parallel(128):
                Q[i] = i // (a * b)
                R[i] = i % (a * b)

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    calls = [line for line in kernel.get_kernel_source().splitlines() if "tl::fast_div(" in line or "tl::fast_rem(" in line]
    assert calls and all(line.rstrip().endswith("(bool)0);") for line in calls)
    q = torch.empty(128, dtype=torch.int32, device="cuda")
    r = torch.empty_like(q)
    x = torch.arange(128, dtype=torch.int64, device="cuda")
    for a, b in [(7, 13), (50000, 50000), (65537, 65537)]:
        d = (a * b + 2**31) % 2**32 - 2**31
        kernel(q, r, a, b)
        torch.testing.assert_close(q, (x // d).to(torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(r, (x % d).to(torch.int32), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("constraint", ["none", "assume", "clamped"])
def test_invariant_swizzle_remainder(dtype, constraint):
    @T.prim_func
    def main(B: T.Tensor((256,), dtype), d: T.dtype(dtype)):
        with T.Kernel(2, threads=128) as bx:
            if constraint == "assume":
                T.assume(d >= 32)
            divisor = T.min(T.max(d, 32), 1024) if constraint == "clamped" else d
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                channel = T.bind((T.cast(i, dtype) * 1000003) % divisor)
                mask = T.cast((i % 8) * 4, dtype)
                B[i] = (channel ^ mask) % divisor

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    source = kernel.get_kernel_source()
    assert ("tl::bounded_rem(" in source) == (constraint != "none")
    b = torch.empty(256, dtype=getattr(torch, dtype), device="cuda")
    i = torch.arange(256, dtype=torch.int64, device="cuda")
    divisors = [32, 33, 37, 63, 64, 65, 1024, 2**31 - 1]
    if constraint != "assume":
        divisors += [1, 2, 7, 28, 31, -1, -7, -37, -(2**31)]
    for d in divisors:
        kernel(b, d)
        divisor = min(max(d, 32), 1024) if constraint == "clamped" else d
        expected = (((i * 1000003) % divisor) ^ ((i % 8) * 4)) % divisor
        torch.testing.assert_close(b, expected.to(b.dtype), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("truncating", [False, True])
def test_invariant_bounded_remainder_limits(dtype, truncating):
    bits = 32 if dtype == "int32" else 64
    lower = 2 ** (bits - 2)
    upper = 2 ** (bits - 1) - 2

    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype), d: T.dtype(dtype)):
        with T.Kernel(1, threads=128):
            T.assume(d >= T.cast(lower, dtype))
            for i in T.Parallel(128):
                x = T.bind(A[i])
                T.assume(x >= 0)
                T.assume(x <= T.cast(upper, dtype))
                B[i] = T.truncmod(x, d) if truncating else x % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    source = kernel.get_kernel_source()
    assert "tl::bounded_rem(" in source
    assert "barrett_reciprocal" not in source and "fastdiv_multiplier" not in source
    a = torch.tensor([0, 1, lower - 1, lower, lower + 1, upper - 1, upper, 7] * 16, dtype=getattr(torch, dtype), device="cuda")
    b = torch.empty_like(a)
    for d in [lower, lower + 1, upper, upper + 1]:
        kernel(a, b, d)
        torch.testing.assert_close(b, a % d, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("bounded", [False, True])
def test_invariant_layout_decode_recomposition(dtype, bounded):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype), a: T.dtype(dtype), b: T.dtype(dtype)):
        with T.Kernel(1, threads=128):
            T.assume(a > 0)
            T.assume(b > 0)
            if bounded:
                T.assume(a <= 1024)
                T.assume(b <= 1024)
            for i in T.Parallel(128):
                x = T.bind(A[i])
                B[i] = (x // (a * b)) * b + (x // a) % b

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signature = re.search(r"void main_kernel\((.*?)\)", kernel.get_kernel_source()).group(1)
    assert bool(re.search(r"\bb\b", signature)) != bounded
    bits = 32 if dtype == "int32" else 64
    low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    values = [low, low + 1, -1000003, -1, 0, 1, high - 1, high] * 16
    x = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
    out = torch.empty_like(x)
    pairs = [(1, 1), (7, 13), (1024, 1024)]
    if not bounded:
        pairs += [(50000, 50000), (65537, 65537)] if bits == 32 else [(2**32, 2**31 + 1)]
    for a, b in pairs:
        kernel(x, out, a, b)
        product = (a * b - low) % (2**bits) + low
        expected = [((v // product) * b + (v // a) % b - low) % (2**bits) + low for v in values]
        assert out.cpu().tolist() == expected


@tilelang.testing.requires_cuda
def test_invariant_layout_shared_quotient():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), Q: T.Tensor((128,), "int32"), R: T.Tensor((128,), "int32"), a: T.int32, b: T.int32):
        with T.Kernel(1, threads=128):
            T.assume(a >= 1)
            T.assume(a <= 1024)
            T.assume(b >= 1)
            T.assume(b <= 1024)
            for i in T.Parallel(128):
                x = T.bind(A[i])
                Q[i] = x // (a * b)
                R[i] = x // a

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    source = kernel.get_kernel_source()
    assert "(a * b)" not in source and "(b * a)" not in source
    x = torch.arange(-64, 64, dtype=torch.int32, device="cuda") * 1000003
    q, r = torch.empty_like(x), torch.empty_like(x)
    for a, b in [(1, 1), (7, 13), (1024, 1024)]:
        kernel(x, q, r, a, b)
        torch.testing.assert_close(q, x // (a * b), rtol=0, atol=0)
        torch.testing.assert_close(r, x // a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_invariant_layout_decode_widened_wrapped_product():
    @T.prim_func
    def main(A: T.Tensor((128,), "int64"), B: T.Tensor((128,), "int64"), a: T.int32, b: T.int32):
        with T.Kernel(1, threads=128):
            T.assume(a > 0)
            T.assume(b > 0)
            for i in T.Parallel(128):
                x = T.bind(A[i])
                B[i] = (x // T.int64(a * b)) * T.int64(b) + (x // T.int64(a)) % T.int64(b)

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    signature = re.search(r"void main_kernel\((.*?)\)", kernel.get_kernel_source()).group(1)
    assert re.search(r"\bb\b", signature)
    values = [-(2**63), -(2**63) + 1, -1000003, -1, 0, 1, 2**63 - 2, 2**63 - 1] * 16
    x = torch.tensor(values, dtype=torch.int64, device="cuda")
    out = torch.empty_like(x)
    for a, b in [(7, 13), (50000, 50000), (65537, 65537)]:
        kernel(x, out, a, b)
        product = (a * b + 2**31) % 2**32 - 2**31
        expected = [((v // product) * b + (v // a) % b + 2**63) % 2**64 - 2**63 for v in values]
        assert out.cpu().tolist() == expected


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_invariant_reassociated_quotient_remainder_identity(dtype):
    @T.prim_func
    def main(B: T.Tensor((128,), dtype), base: T.dtype(dtype), a: T.dtype(dtype), b: T.dtype(dtype)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                x = T.cast(i, dtype) + base
                B[i] = (x // (a * b)) * b * a + x % (a * b)

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert "tl::fast_div" not in kernel.get_kernel_source()
    out = torch.empty(128, dtype=getattr(torch, dtype), device="cuda")
    bits = 32 if dtype == "int32" else 64
    for base in [-(2 ** (bits - 1)), -64, 2 ** (bits - 1) - 128]:
        for a, b in [(7, 13), (-7, 13), (65537, 65537)]:
            kernel(out, base, a, b)
            expected = torch.arange(128, dtype=out.dtype, device="cuda") + base
            torch.testing.assert_close(out, expected)


@tilelang.testing.requires_cuda
def test_invariant_inherits_dynamic_shape_positivity():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                if i < n:
                    B[i] = A[i] // n

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    division = next(line for line in kernel.get_kernel_source().splitlines() if "tl::fast_div(" in line)
    assert ", (bool)1)" in division  # Positive divisor from shape metadata, not T.assume.
    for size in [1, 7, 129]:
        a = torch.arange(size, dtype=torch.int32, device="cuda") * 37 - 1000
        b = torch.empty_like(a)
        kernel(a, b)
        torch.testing.assert_close(b, a // size)


@tilelang.testing.requires_cuda
def test_invariant_guarded_host_expression():
    @T.prim_func
    def main(B: T.Tensor((128,), "int32"), p: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if p > 0:
                    B[i] = i // (7 // p + 1) + i // p

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    out = torch.full((128,), -9, dtype=torch.int32, device="cuda")
    for p in [0, -1, -(2**31)]:
        kernel(out, p)
        torch.testing.assert_close(out, torch.full_like(out, -9))
    for p in [1, 7, 13]:
        kernel(out, p)
        i = torch.arange(128, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(out, i // (7 // p + 1) + i // p)


@tilelang.testing.requires_cuda
def test_invariant_partial_block_write_then_read():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                if i < n:
                    A[i] = 7 * n
                    B[i] = T.max(A[i], 0) // n

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    for size in [1, 127, 128, 129]:
        a = torch.zeros(size, dtype=torch.int32, device="cuda")
        b = torch.empty_like(a)
        kernel(a, b)
        torch.testing.assert_close(b, torch.full_like(b, 7))
    source = kernel.get_kernel_source()
    # The load remains after its store, inside the tail guard.
    body = source[source.index("__launch_bounds__") :]
    assert body.index("if (") < body.index("A[") < body.index("tl::fast_div")


@tilelang.testing.requires_cuda
def test_invariant_serial_loop_scope():
    n = T.dynamic("n")

    @T.prim_func
    def main(B: T.Tensor((n, 128), "int32"), d: T.int32):
        with T.Kernel(1, threads=128):
            T.assume(d > 0)
            for k in T.serial(n):
                for i in T.Parallel(128):
                    B[k, i] = k // d + k % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    out = torch.empty((19, 128), dtype=torch.int32, device="cuda")
    for d in [1, 7, 23]:
        kernel(out, d)
        k = torch.arange(19, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(out, (k // d + k % d)[:, None].expand_as(out))


@tilelang.testing.requires_cuda
def test_invariant_preserves_index_intermediate_widening():
    @T.prim_func
    def main(A: T.Tensor((614000000,), "uint8"), B: T.Tensor((128,), "uint8"), base: T.int32, d: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if d > 0 and base >= 0:
                    B[i] = A[((i + base) * 4) // d]

    a = torch.full((614000000,), 29, dtype=torch.uint8, device="cuda")
    a[:128] = 11
    b = torch.empty(128, dtype=torch.uint8, device="cuda")
    for enabled in [False, True]:
        kernel = tilelang.compile(
            main,
            target="cuda",
            target_host="c",
            execution_backend="tvm_ffi",
            pass_configs={"tl.enable_invariant_arithmetic": enabled, "tl.disable_safe_memory_legalize": True},
        )
        for base, d in [(2**30, 7), (0, 7)]:
            kernel(a, b, base, d)
            indices = (torch.arange(128, dtype=torch.int64, device="cuda") + base) * 4 // d
            torch.testing.assert_close(b, a[indices])


@tilelang.testing.requires_cuda
def test_invariant_launch_extent_bounds_do_not_narrow_wide_indices():
    n = T.dynamic("n")

    @T.prim_func
    def main(Q: T.Tensor((n,), "int64"), R: T.Tensor((n,), "int64"), base: T.int64, d: T.int64):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                i = bx * 128 + lane
                x = T.int64(bx) * 128 + T.int64(lane) + base
                if i < n:
                    Q[i] = x // d
                    R[i] = x % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    q = torch.empty(129, dtype=torch.int64, device="cuda")
    r = torch.empty_like(q)
    for base in [0, 2**31, 2**48, -(2**48)]:
        for d in [7, -7, 2**40, -(2**40)]:
            kernel(q, r, base, d)
            x = torch.arange(129, dtype=torch.int64, device="cuda") + base
            torch.testing.assert_close(q, x // d)
            torch.testing.assert_close(r, x % d)


@tilelang.testing.requires_cuda
def test_invariant_positive_addition_can_wrap():
    @T.prim_func
    def main(Q: T.Tensor((128,), "int64"), R: T.Tensor((128,), "int64"), base: T.int32, d: T.int32):
        with T.Kernel(1, threads=128):
            T.assume(base > 0)
            for i in T.Parallel(128):
                x = base + i
                Q[i] = x // d
                R[i] = x % d

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    q = torch.empty(128, dtype=torch.int64, device="cuda")
    r = torch.empty_like(q)
    base = 2**31 - 64
    x = (torch.arange(128, dtype=torch.int64, device="cuda") + base).to(torch.int32).to(torch.int64)
    for d in [7, -7, 2**31 - 1]:
        kernel(q, r, base, d)
        torch.testing.assert_close(q, x // d)
        torch.testing.assert_close(r, x % d)


@tilelang.testing.requires_cuda
def test_invariant_materialization_reuses_dominating_condition():
    @T.prim_func
    def main(B: T.Tensor((128,), "int32"), d: T.int32):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if i // d > 5:
                    B[i] = i // d + 1
                else:
                    B[i] = i // d - 1

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    assert kernel.get_kernel_source().count("tl::fast_div(") == 1
    out = torch.empty(128, dtype=torch.int32, device="cuda")
    x = torch.arange(128, dtype=torch.int32, device="cuda")
    for d in [1, 7, -7, 2**31 - 1]:
        kernel(out, d)
        q = x // d
        torch.testing.assert_close(out, torch.where(q > 5, q + 1, q - 1))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("mask", [False, True])
def test_invariant_recovers_nonnegative_range_after_wrap(mask):
    @T.prim_func
    def main(B: T.Tensor((128,), "int32"), base: T.int32, d: T.int32, e: T.int32):
        with T.Kernel(1, threads=128):
            T.assume(base > 0)
            T.assume(d > 0)
            T.assume(e > 0)
            for i in T.Parallel(128):
                x = ((base + i) & 255) if mask else (base + i) % d
                B[i] = (x ^ (i % 8)) // e

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    calls = [line for line in kernel.get_kernel_source().splitlines() if "tl::fast_div(" in line]
    assert len(calls) == 1
    assert "(bool)1, (bool)0, (bool)1, (bool)1)" in calls[0]
    out = torch.empty(128, dtype=torch.int32, device="cuda")
    lanes = torch.arange(128, dtype=torch.int64, device="cuda")
    for base in [1, 2**31 - 64]:
        wrapped = (lanes + base).to(torch.int32).to(torch.int64)
        for d, e in [(7, 3), (2**31 - 1, 7), (1, 1)]:
            kernel(out, base, d, e)
            x = (wrapped & 255) if mask else wrapped % d
            torch.testing.assert_close(out, ((x ^ (lanes % 8)) // e).to(torch.int32))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("radix", [4, 32, 64, 31])
def test_invariant_recovers_reassociated_layout_remainder(dtype, radix):
    @T.prim_func
    def main(B: T.Tensor((384,), dtype), base: T.int64, d: T.int32):
        with T.Kernel(3, threads=128) as bx:
            for lane in T.Parallel(128):
                x = T.cast(bx * 128 + lane, dtype) + T.cast(base, dtype)
                # The omitted block offset is divisible by each power-of-two
                # radix, but not by 31. Keep that negative control.
                B[bx * 128 + lane] = (T.cast(lane, dtype) + T.cast(base, dtype) - (x // T.cast(d, dtype)) * T.cast(d, dtype)) % radix

    kernel = tilelang.compile(
        main, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
    )
    bits = 32 if dtype == "int32" else 64
    modulus = 1 << bits

    def wrap(value):
        return (value + modulus // 2) % modulus - modulus // 2

    out = torch.empty(384, dtype=getattr(torch, dtype), device="cuda")
    for base in [0, -257, 2**31 - 64, -(2**31), 2**48 - 31]:
        for d in [1, 3, -3, 7, -7, 2**31 - 1, -(2**31)]:
            expected = torch.tensor(
                [wrap(i % 128 + wrap(base) - (wrap(i + wrap(base)) // d) * d) % radix for i in range(384)],
                dtype=out.dtype,
                device="cuda",
            )
            kernel(out, base, d)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
