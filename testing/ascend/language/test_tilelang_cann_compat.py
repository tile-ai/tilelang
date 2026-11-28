"""CANN compatibility tests for Ascend code generation and device operations.

Each numerical check includes destination guards. The loops reuse one compiled
kernel across empty, sparse and full predicates; they do not relax tolerances.
Integer division excludes divide-by-zero and signed MIN/-1. BF16 min's NaN
policy is not specified here; NaN is covered by the separate BF16 ne test.
vld2 off is an unsigned HardwareLoop ELEMENT STRIDE, not a signed immediate
address displacement. Three real iterations distinguish it from plain loading.
"""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.engine.lower import lower


MODES = ("MODE_ZEROING", "MODE_MERGING")
INTEGER_DTYPES = ("int16", "uint16", "int32", "uint32")


def _masks(lanes):
    lane = torch.arange(lanes)
    return (
        ("empty", lane < 0),
        ("one", lane == 0),
        ("tail", lane < lanes // 2 + 1),
        ("sparse", lane % 3 == 1),
        ("full", lane < lanes),
    )


def _assert_bits(actual, expected, context):
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)), context


def arithmetic_kernel(op_name, dtype, mode):
    bits = tvm.DataType(dtype).bits
    lanes = 2048 // bits
    pred_dtype = f"uint{bits}"

    @T.prim_func
    def kernel(
        A: T.Tensor((lanes,), dtype),
        B: T.Tensor((lanes,), dtype),
        Initial: T.Tensor((3 * lanes,), dtype),
        Predicate: T.Tensor((lanes,), pred_dtype),
        Out: T.Tensor((3 * lanes,), dtype),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((lanes,), dtype)
            b_ub = T.alloc_shared((lanes,), dtype)
            pred_ub = T.alloc_shared((lanes,), pred_dtype)
            dst_ub = T.alloc_shared((3 * lanes,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(Predicate, pred_ub)
            T.copy(Initial, dst_ub)
            with T.SimdVF():
                full = T.simd.pset(bits)
                mask = T.simd.vcmps(T.simd.vld(pred_ub[0]), T.cast(0, pred_dtype), op="gt")
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                dst = T.simd.alloc_local((1,), dtype)
                dst[0] = T.simd.vld(dst_ub[lanes])
                if op_name == "div":
                    dst[0] = T.simd.vdiv(a, b, mask, mode=mode)
                elif op_name == "highest":
                    dst[0] = T.simd.vdupv(a, mask, pos="POS_HIGHEST", mode=mode)
                else:
                    dst[0] = T.simd.vmins(a, T.cast(0, dtype), mask, mode=mode)
                T.simd.vsts(dst_ub[lanes], dst[0], full, dist=f"NORM_B{bits}")
                T.simd.mem_bar("VST_VLD")
            T.copy(dst_ub, Out)

    return kernel


def _exercise_arithmetic(kernel, a, b, active, mode, context):
    lanes = a.numel()
    pred_dtype = getattr(torch, f"uint{a.element_size() * 8}")
    compiled = tilelang.compile(kernel, target="ascend")
    device_a, device_b = a.to("npu"), b.to("npu")
    for sentinel in (83, 117):
        initial = torch.full((3 * lanes,), sentinel, dtype=a.dtype)
        for name, mask in _masks(lanes):
            expected = initial.clone()
            # ZEROING zeros inactive vector lanes, including BF16 +0 bits.
            middle = torch.zeros_like(a) if mode == "MODE_ZEROING" else initial[lanes : 2 * lanes].clone()
            # PyTorch does not implement boolean index_put for uint16/uint32;
            # copy active storage bytes so the host oracle covers those dtypes.
            middle.view(torch.uint8).reshape(lanes, -1)[mask] = active.view(torch.uint8).reshape(lanes, -1)[mask]
            expected[lanes : 2 * lanes] = middle
            out = torch.empty_like(initial, device="npu")
            compiled(device_a, device_b, initial.to("npu"), mask.to(pred_dtype).to("npu"), out)
            _assert_bits(out.cpu(), expected, (context, mode, name, sentinel))
    _assert_bits(device_a.cpu(), a, (context, "read-only A"))
    _assert_bits(device_b.cpu(), b, (context, "read-only B"))


def _division_inputs(dtype):
    bits = tvm.DataType(dtype).bits
    signed = dtype.startswith("int")
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    pairs = [(0, 3), (1, 2), (7, 3), (hi, 1), (hi, 3), (hi - 1, 7), (hi, hi)]
    if signed:
        pairs += [(-7, 3), (7, -3), (-7, -3), (lo, 2), (lo, 3), (lo + 1, -1), (lo, lo)]
    lanes = 2048 // bits
    pairs = [pairs[i % len(pairs)] for i in range(lanes)]
    # Python integer arithmetic avoids FP rounding and Python // negative-floor semantics.
    quotient = [(abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1) for x, y in pairs]
    td = getattr(torch, dtype)
    return (torch.tensor([x for x, _ in pairs], dtype=td), torch.tensor([y for _, y in pairs], dtype=td), torch.tensor(quotient, dtype=td))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_integer_division_truncates_toward_zero_and_preserves_guards(dtype, mode):
    a, b, active = _division_inputs(dtype)
    _exercise_arithmetic(arithmetic_kernel("div", dtype, mode), a, b, active, mode, ("integer div", dtype))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "mode,dtype",
    [
        ("MODE_ZEROING", "uint8"),
        ("MODE_ZEROING", "int16"),
        ("MODE_ZEROING", "bfloat16"),
        ("MODE_ZEROING", "float32"),
        ("MODE_MERGING", "uint8"),
        ("MODE_MERGING", "int16"),
    ],
)
def test_highest_broadcast_uses_fixed_source_lane(dtype, mode):
    td = getattr(torch, dtype)
    lanes = 2048 // tvm.DataType(dtype).bits
    # Source lane zero and the final lane differ; a one-lane destination mask
    # must still broadcast the highest PHYSICAL source lane.
    a = ((torch.arange(lanes) * 7 + 3) % 61).to(td)
    a[0], a[-1] = 11, 53
    active = torch.full_like(a, 53)
    _exercise_arithmetic(arithmetic_kernel("highest", dtype, mode), a, torch.zeros_like(a), active, mode, ("highest broadcast", dtype))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("mode", MODES)
def test_bf16_min_scalar_finite_infinity_and_signed_zero(mode):
    values = [-float("inf"), -11.5, -1.0, -0.0, 0.0, 1.0, 11.5, float("inf")]
    a = torch.tensor(values * 16, dtype=torch.bfloat16)
    active = a.clone()
    active[a > 0] = 0
    # In particular min(-0,+0) keeps -0. NaN min is intentionally not assigned
    # an undocumented expected payload or propagation policy by this test.
    _exercise_arithmetic(arithmetic_kernel("min_scalar", "bfloat16", mode), a, torch.zeros_like(a), active, mode, "BF16 scalar min")


def bf16_ne_kernel():
    lanes = 128

    @T.prim_func
    def kernel(
        A: T.Tensor((lanes,), "bfloat16"),
        B: T.Tensor((lanes,), "bfloat16"),
        Predicate: T.Tensor((lanes,), "uint16"),
        Initial: T.Tensor((3 * lanes,), "uint16"),
        Out: T.Tensor((3 * lanes,), "uint16"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((lanes,), "bfloat16")
            b = T.alloc_shared((lanes,), "bfloat16")
            p = T.alloc_shared((lanes,), "uint16")
            dst = T.alloc_shared((3 * lanes,), "uint16")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(Predicate, p)
            T.copy(Initial, dst)
            with T.SimdVF():
                full = T.simd.pset(16)
                mask = T.simd.vcmps(T.simd.vld(p[0]), T.uint16(0), op="gt")
                unequal = T.simd.vcmp(T.simd.vld(a[0]), T.simd.vld(b[0]), mask, op="ne")
                ones = T.simd.vdup(T.uint16(1), "uint16", full)
                zeros = T.simd.vdup(T.uint16(0), "uint16", full)
                selected = T.simd.vsel(ones, zeros, unequal)
                T.simd.vsts(dst[lanes], selected, full, dist="NORM_B16")
                T.simd.mem_bar("VST_VLD")
            T.copy(dst, Out)

    return kernel


@tilelang.testing.requires_ascend
def test_bf16_ne_nan_zero_and_predicate_lane_layout():
    # Equality treats +/-0 as equal and every NaN comparison as unequal.
    # Public asc_ne constraints explicitly specify these two cases and mask=0:
    # https://asc.gitcode.com/api/SIMD-API/c_api/reg_compute/reg_compare/asc_ne.html
    # This defines our oracle; it is not evidence of any SDK's device behavior.
    av = [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"), 2.0]
    bv = [-0.0, 0.0, 2.0, -1.0, float("inf"), float("inf"), float("nan"), float("nan")]
    a, b = [torch.tensor(v * 16, dtype=torch.bfloat16) for v in (av, bv)]
    compiled = tilelang.compile(bf16_ne_kernel(), target="ascend")
    for sentinel in (83, 117):
        initial = torch.full((384,), sentinel, dtype=torch.uint16)
        for name, mask in _masks(128):
            expected = initial.clone()
            expected[128:256] = (torch.ne(a.float(), b.float()) & mask).to(torch.uint16)
            out = torch.empty_like(initial, device="npu")
            compiled(a.to("npu"), b.to("npu"), mask.to(torch.uint16).to("npu"), initial.to("npu"), out)
            _assert_bits(out.cpu(), expected, ("BF16 ne", name, sentinel))


def e8m0_broadcast_kernel(mode, pos="POS_LOWEST"):
    @T.prim_func
    def kernel(
        A: T.Tensor((256,), "uint8"),
        Predicate: T.Tensor((256,), "uint8"),
        Initial: T.Tensor((768,), "uint8"),
        Out: T.Tensor((768,), "uint8"),
    ):
        with T.Kernel(1):
            src = T.alloc_shared((256,), "uint8")
            pred = T.alloc_shared((256,), "uint8")
            dst = T.alloc_shared((768,), "uint8")
            T.copy(A, src)
            T.copy(Predicate, pred)
            T.copy(Initial, dst)
            with T.SimdVF():
                full = T.simd.pset(8)
                mask = T.simd.vcmps(T.simd.vld(pred[0]), T.uint8(0), op="gt")
                scale = T.reinterpret(T.simd.vld(src[0]), "float8_e8m0fnux256")
                result = T.simd.alloc_local((1,), "float8_e8m0fnu")
                result[0] = T.reinterpret(T.simd.vld(dst[256]), "float8_e8m0fnux256")
                result[0] = T.simd.vdupv(scale, mask, pos=pos, mode=mode)
                T.simd.vsts(dst[256], T.reinterpret(result[0], "uint8x256"), full, dist="NORM_B8")
                T.simd.mem_bar("VST_VLD")
            T.copy(dst, Out)

    return kernel


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pos", ["POS_LOWEST", "POS_HIGHEST"])
def test_e8m0_broadcast_bits_including_zero_and_nan_encoding(mode, pos):
    compiled = tilelang.compile(e8m0_broadcast_kernel(mode, pos), target="ascend")
    # E8M0 is an encoded scale, not a normal floating scalar. Includes its NaN
    # encoding as a bit-copy test; inactive ZEROING lanes must contain raw 0.
    for code in (0, 1, 127, 254, 255):
        a = torch.arange(256).to(torch.uint8)
        a[0 if pos == "POS_LOWEST" else -1] = code
        initial = torch.full((768,), 83 if code % 2 else 117, dtype=torch.uint8)
        for name, mask in _masks(256):
            expected = initial.clone()
            middle = torch.zeros(256, dtype=torch.uint8) if mode == "MODE_ZEROING" else initial[256:512].clone()
            middle[mask] = code
            expected[256:512] = middle
            out = torch.empty_like(initial, device="npu")
            compiled(a.to("npu"), mask.to(torch.uint8).to("npu"), initial.to("npu"), out)
            _assert_bits(out.cpu(), expected, ("E8M0 vector broadcast", pos, code, mode, name))


def iter_load_kernel(dtype, stride):
    bits = tvm.DataType(dtype).bits
    lanes = 2048 // bits

    @T.prim_func
    def kernel(
        A: T.Tensor((8 * lanes,), dtype), Initial: T.Tensor((8 * lanes,), dtype), Out: T.Tensor((8 * lanes,), dtype), repeats: T.int32
    ):
        with T.Kernel(1):
            src = T.alloc_shared((8 * lanes,), dtype)
            dst = T.alloc_shared((8 * lanes,), dtype)
            T.copy(A, src)
            T.copy(Initial, dst)
            with T.SimdVF():
                full = T.simd.pset(bits)
                # VAG needs the canonical i16 HardwareLoop induction variable.
                # Keep a runtime bound (our calls use 1 or 3), not an unrolled
                # compile-time load. No nested branches in the HardwareLoop.
                for j in T.serial(T.uint16(0), T.uint16(repeats)):
                    even, odd = T.simd.vld2(src[lanes], dist=f"DINTLV_B{bits}", off=T.uint32(stride))
                    T.simd.vsts(dst[(1 + 2 * j) * lanes], even, full, dist=f"NORM_B{bits}")
                    T.simd.vsts(dst[(2 + 2 * j) * lanes], odd, full, dist=f"NORM_B{bits}")
                T.simd.mem_bar("VST_VLD")
            T.copy(dst, Out)

    return kernel


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "stride_kind,dtype",
    [("zero", "uint8"), ("contiguous", "uint8"), ("gapped", "uint8"), ("gapped", "int16"), ("gapped", "int32")],
)
def test_vld2_address_register_stride_and_both_outputs(dtype, stride_kind):
    td = getattr(torch, dtype)
    bits = tvm.DataType(dtype).bits
    lanes = 2048 // bits
    stride = {"zero": 0, "contiguous": 2 * lanes, "gapped": 2 * lanes + 256 // bits}[stride_kind]
    a = ((torch.arange(8 * lanes) * 73 + torch.arange(8 * lanes) // lanes * 19) % 251).to(td)
    compiled = tilelang.compile(iter_load_kernel(dtype, stride), target="ascend")
    for repeats in (1, 3):
        initial = torch.full((8 * lanes,), 117 if repeats == 1 else 83, dtype=td)
        expected = initial.clone()
        for j in range(repeats):
            src = a[lanes + j * stride : lanes + j * stride + 2 * lanes]
            expected[(1 + 2 * j) * lanes : (2 + 2 * j) * lanes] = src[0::2]
            expected[(2 + 2 * j) * lanes : (3 + 2 * j) * lanes] = src[1::2]
        out = torch.empty_like(initial, device="npu")
        compiled(a.to("npu"), initial.to("npu"), out, repeats)
        _assert_bits(out.cpu(), expected, ("vld2 HWLoop", dtype, stride_kind, repeats))


def atomic_kernel(op):
    blocks, lanes = 4, 64

    @T.prim_func
    def kernel(
        A: T.Tensor((blocks, lanes), "int32"), Reduced: T.Tensor((3 * lanes,), "int32"), Plain: T.Tensor((blocks, 3 * lanes), "int32")
    ):
        with T.Kernel(blocks) as bx:
            values = T.alloc_shared((lanes,), "int32")
            T.copy(A[bx, 0:lanes], values)
            T.ascend_pipe_barrier("PIPE_MTE3")
            T.set_atomic(op, "int32")
            # All four blocks target the same GM region. Host initialization
            # happens before launch; there is no racing in-kernel zero fill.
            T.copy(values, Reduced[lanes : 2 * lanes])
            T.ascend_pipe_barrier("PIPE_MTE3")
            T.set_atomic_none()
            # A separate per-block write proves that atomic mode was cleared.
            T.copy(values, Plain[bx, lanes : 2 * lanes])
            T.ascend_pipe_barrier("PIPE_MTE3")

    return kernel


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("op", ["add", "max", "min"])
def test_int32_store_atomic_multicore_and_none_restores_plain_store(op):
    compiled = tilelang.compile(atomic_kernel(op), target="ascend")
    a = ((torch.arange(4 * 64).reshape(4, 64) * 31) % 1009 - 504).to(torch.int32)
    # Positive and negative sentinels expose leaked MAX and MIN modes, while
    # ADD must include the initialized destination exactly once.
    for sentinel in (-12345, 12345):
        initial = torch.full((192,), sentinel, dtype=torch.int32)
        seed = (torch.arange(64) % 9 - 4).to(torch.int32)
        initial[64:128] = seed
        expected = initial.clone()
        if op == "add":
            expected[64:128] = (seed.to(torch.int64) + a.to(torch.int64).sum(0)).to(torch.int32)
        elif op == "max":
            expected[64:128] = torch.maximum(seed, a.max(0).values)
        else:
            expected[64:128] = torch.minimum(seed, a.min(0).values)
        plain = torch.full((4, 192), sentinel, dtype=torch.int32)
        plain_expected = plain.clone()
        plain_expected[:, 64:128] = a
        reduced_device, plain_device = initial.to("npu"), plain.to("npu")
        compiled(a.to("npu"), reduced_device, plain_device)
        _assert_bits(reduced_device.cpu(), expected, ("multicore INT32 atomic", op, sentinel))
        _assert_bits(plain_device.cpu(), plain_expected, ("atomic none", op, sentinel))


def test_global_hf32_extern_preserves_imported_overload():
    helper = "__aicore__ inline int32_t asc_disable_hf32(int32_t x) { return x + 1; }"

    @T.prim_func
    def kernel(out: T.Tensor((1,), "int32"), value: T.int32):
        with T.Kernel(1):
            T.import_source(helper)
            T.evaluate(T.call_extern("void", "::asc_disable_hf32"))
            out[0] = T.call_extern("int32", "::asc_disable_hf32", value)

    body = lower(kernel, target="ascend").kernel_source.split("TL_CANN_USE_CORE_NAMES", 1)[1]
    assert "  asc_disable_hf32();" in body
    assert "::asc_disable_hf32(value)" in body


def test_native_extern_routes_follow_execution_domain():
    @T.prim_func
    def kernel(out: T.Tensor((3,), "float32"), powf: T.float32):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.evaluate(T.call_extern("void", "::asc_disable_hf32"))
            T.evaluate(T.call_extern("void", "asc_div", ub.data, ub.data, ub.data, 64))
            T.evaluate(T.call_extern("void", "vendor::asc_disable_hf32"))
            with T.SimdVF():
                v = T.simd.vld(ub[0])
                mask = T.simd.pset(32)
                q = T.call_extern("float32x64", "::asc_div", v, v, mask)
                T.simd.vsts(ub[0], q, mask)
            with T.SimtVF(threads=1):
                out[0] = T.call_extern("float32", "::powf", powf, powf)
                out[1] = T.call_extern("float32", "vendor::powf", powf, powf)
            out[2] = T.call_extern("float32", "::sqrtf", powf)

    source = lower(kernel, target="ascend").kernel_source
    simd = source.split("TL_CANN_USE_SIMD_NAMES", 1)[1].split("\n}", 1)[0]
    kernel_body = source.split('extern "C" __global__', 1)[1]
    assert "TL_CANN_USE_CORE_NAMES" in kernel_body
    assert "TL_CANN_USE_SIMD_NAMES" not in kernel_body
    assert "asc_disable_hf32();" in kernel_body
    assert "vendor::asc_disable_hf32(" in source
    assert simd.count("asc_div(") == 1
    assert kernel_body.count("asc_div(") == 1
    simt = source.split("TL_CANN_USE_SIMT_MATH_NAMES", 1)[1].split("\n}", 1)[0]
    assert " = powf(" in simt and "vendor::powf(" in simt
    assert "AscendC::Std::sqrt(" in source


def native_div_kernel():
    @T.prim_func
    def kernel(a: T.Tensor((64,), "uint32"), b: T.Tensor((64,), "uint32"), out: T.Tensor((64,), "uint32")):
        with T.Kernel(1):
            lhs = T.alloc_shared((64,), "uint32")
            rhs = T.alloc_shared((64,), "uint32")
            T.copy(a, lhs)
            T.copy(b, rhs)
            with T.SimdVF():
                x = T.simd.vld(lhs[0])
                y = T.simd.vld(rhs[0])
                one = T.simd.pset(32, "PAT_VL1")
                full = T.simd.pset(32)
                q = T.call_extern("uint32x64", "asc_div", x, y, one)
                T.simd.vsts(lhs[0], q, full)
            T.copy(lhs, out)

    return kernel


@tilelang.testing.requires_ascend
def test_native_div_extern_returns_one_masked_register():
    kernel = tilelang.compile(native_div_kernel(), target="ascend")
    lhs = torch.arange(65, 129).to(torch.uint32)
    rhs = torch.full((64,), 3, dtype=torch.uint32)
    out = torch.full_like(lhs, 99).to("npu")
    kernel(lhs.to("npu"), rhs.to("npu"), out)
    expected = torch.zeros_like(lhs)
    expected[0] = 65 // 3
    assert torch.equal(out.cpu(), expected)
