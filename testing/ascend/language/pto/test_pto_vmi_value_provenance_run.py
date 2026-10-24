"""Device acceptance: PTO VMI value provenance and scalar bridge boundaries.

Each case compiles through the real TileLang -> PTODSL -> PTOAS VPTO ->
Bisheng pipeline and executes on the NPU.  References are computed on the CPU
from Python integers/floats, independently of the generated kernel.

Coverage:
* int32/uint32 ``vcmax``/``vcmin`` -> ``vbrc(group=1)``: signed negatives,
  exact integer boundaries, unsigned high-bit values (> ``INT32_MAX``), and
  both reduction directions.  20 independent trials per integer case.
* float32 control: the same reduce/broadcast route for a non-integer VMI
  source.
* true scalar integer bridges: compile-time literal and runtime scalar
  sources through ``vbrc`` + ``vci``.
* ordinary scalar memory read after a VMI store (storage contract, not the
  last ``BufferStore`` RHS).
"""

import os
import re
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import torch_npu  # noqa: E402,F401  (import before tilelang/ptodsl)
import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402


def _require_npu_runtime():
    # These device tests require an available NPU runtime and a successful
    # host-to-device copy probe; a missing runtime or probe mismatch fails.
    assert hasattr(torch, "npu"), "torch.npu is unavailable in this environment"
    assert torch.npu.is_available(), "torch.npu is unavailable in the current environment"
    source = torch.arange(8, dtype=torch.float32)
    copied = source.to(torch.device("npu"))
    torch.npu.synchronize()
    assert torch.equal(copied.cpu(), source), "NPU host->device copy probe mismatch"


@pytest.fixture(scope="module")
def npu():
    _require_npu_runtime()
    return torch.device("npu")


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _compile(func):
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    return kernel, kernel.get_kernel_source()


def _run_trials(kernel, make_case, device, trials):
    outputs = []
    for trial in range(trials):
        inputs, expected = make_case(trial)
        device_inputs = [_to_device(value, device) for value in inputs]
        output = kernel(*device_inputs)
        torch.npu.synchronize()
        got = output.cpu()
        assert got.dtype == expected.dtype, f"trial {trial}: {got.dtype} != {expected.dtype}"
        assert torch.equal(got, expected), f"trial {trial}: numeric mismatch\n got={got.tolist()[:16]}\n exp={expected.tolist()[:16]}"
        outputs.append(got)
    return outputs


# ---------------------------------------------------------------------------
# Reduce -> vbrc(group=1): lane-1 VMI data must stay VMI data
# ---------------------------------------------------------------------------
def _reduce_broadcast(dtype, reduce_name):
    reduce = getattr(T.vmi, reduce_name)

    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, work[0:128])
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                r = reduce(x, mask)
                y = T.vmi.vbrc(r, size=128, group=1)
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, C)

    return main


def _int32_input(trial):
    generator = torch.Generator().manual_seed(20261017 + trial)
    values = torch.randint(-(2**31), 2**31 - 1, (128,), generator=generator).to(torch.int32)
    values[0] = -(2**31)
    values[1] = 2**31 - 1
    values[2] = -1
    values[3] = 0
    values[4] = 1
    values[5] = -(2**31) + 1 if trial % 2 else 2**31 - 2
    return values


def _uint32_input(trial):
    generator = torch.Generator().manual_seed(20261017 + trial)
    values = torch.randint(0, 2**31 - 1, (128,), generator=generator).to(torch.uint32)
    values[0] = 2**32 - 1
    values[1] = 2**31  # INT32_MAX + 1, high bit set
    values[2] = 2**31 - 1
    values[3] = 0
    values[4] = 1
    values[5] = 2**32 - 2 if trial % 2 else 2**31 + 1
    return values


def _uint32_mutable_input(trial):
    """Vary the unsigned extremes so stale slot contents cannot pass.

    The shared helper keeps UINT32_MAX in every input, which would make every
    mutable reduction produce the same value.  Keep one boundary trial and
    move both the high-bit maximum and the minimum for the rest.
    """
    generator = torch.Generator().manual_seed(20261017 + trial)
    values = torch.randint(1000, 2**31 - 1, (128,), generator=generator).to(torch.uint32)
    values[0] = 2**32 - 1 if trial == 0 else 2**31 + 1 + trial * 7919
    values[1] = 2**31  # INT32_MAX + 1, high bit set
    values[2] = 2**31 - 1
    values[3] = 0 if trial == 0 else trial * 11 + 2
    values[4] = 1 if trial == 0 else trial * 13 + 5
    values[5] = 2**31 + 7 + trial * 31
    return values


def _float32_input(trial):
    generator = torch.Generator().manual_seed(20261017 + trial)
    values = torch.randn(128, generator=generator) * 1000.0
    values[0] = 3.0e38
    values[1] = -3.0e38
    values[2] = 0.0
    values[3] = -0.0
    return values


INPUTS = {"int32": _int32_input, "uint32": _uint32_input}
REDUCE_OPS = {"vcmax": max, "vcmin": min}
INTEGER_REDUCE_CASES = [
    ("int32", "vcmax"),
    ("int32", "vcmin"),
    ("uint32", "vcmax"),
    ("uint32", "vcmin"),
]


@pytest.mark.parametrize(("dtype", "reduce_name"), INTEGER_REDUCE_CASES)
def test_integer_reduce_broadcast_is_bit_exact(npu, dtype, reduce_name):
    kernel, source = _compile(_reduce_broadcast(dtype, reduce_name))
    vbrc_line = next(line.strip() for line in source.splitlines() if "pto.vmi.vbrc(" in line)
    assert "pto.cast(" not in vbrc_line, vbrc_line
    assert "group=1" in vbrc_line, vbrc_line
    reduce = REDUCE_OPS[reduce_name]

    def make_case(trial):
        a = INPUTS[dtype](trial)
        expected_value = reduce(int(v) for v in a.tolist())
        expected = torch.full((128,), expected_value, dtype=a.dtype)
        return [a], expected

    _run_trials(kernel, make_case, npu, trials=20)


def test_float32_reduce_broadcast_control(npu):
    kernel, source = _compile(_reduce_broadcast("float32", "vcmax"))
    vbrc_line = next(line.strip() for line in source.splitlines() if "pto.vmi.vbrc(" in line)
    assert "pto.cast(" not in vbrc_line, vbrc_line

    def make_case(trial):
        a = _float32_input(trial)
        expected = torch.full((128,), float(a.max().item()), dtype=torch.float32)
        return [a], expected

    _run_trials(kernel, make_case, npu, trials=5)


# ---------------------------------------------------------------------------
# True scalar integer bridges (literal + runtime scalar)
# ---------------------------------------------------------------------------
def _scalar_bridge(literal):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), scale: T.int32, C: T.Tensor((128,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((128,), "int32")
            out = T.alloc_shared((128,), "int32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                base = T.int32(7) if literal else scale
                y = T.vmi.vbrc(base, size=128)
                z = T.vmi.vci(base, size=128)
                T.vmi.vstore(T.vmi.vadd(y, z, mask), out[0], mask)
            T.copy(out, C)

    return main


@pytest.mark.parametrize("literal", [True, False], ids=["literal", "runtime"])
def test_scalar_integer_bridge_device(npu, literal):
    kernel, source = _compile(_scalar_bridge(literal))
    if literal:
        assert "pto.vmi.vbrc(pto.si32(7), size=128)" in source, source
    else:
        assert "pto.cast(base, pto.i32)" in source, source

    def make_case(trial):
        a = torch.arange(128, dtype=torch.int32) + trial
        scale = 7 if literal else 5 + trial * 3
        expected = (2 * scale + torch.arange(128, dtype=torch.int32)).to(torch.int32)
        return [a, scale], expected

    _run_trials(kernel, make_case, npu, trials=5)


# ---------------------------------------------------------------------------
# Ordinary scalar memory read after a VMI store
# ---------------------------------------------------------------------------
def _scalar_load_after_vmi_store():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), C: T.Tensor((128,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((128,), "int32")
            out = T.alloc_shared((128,), "int32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                T.vmi.vstore(x, work[0], mask)
                scalar = work[0]
                y = T.vmi.vbrc(scalar, size=128)
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, C)

    return main


def test_scalar_buffer_load_after_vmi_store_device(npu):
    kernel, source = _compile(_scalar_load_after_vmi_store())
    vbrc_line = next(line.strip() for line in source.splitlines() if "pto.vmi.vbrc(" in line)
    assert re.search(r"pto\.cast\(scalar_\d+, pto\.i32\)", vbrc_line), vbrc_line

    def make_case(trial):
        a = torch.arange(128, dtype=torch.int32) * (trial + 1) - 10
        expected = torch.full((128,), int(a[0]), dtype=torch.int32)
        return [a], expected

    _run_trials(kernel, make_case, npu, trials=3)


# ---------------------------------------------------------------------------
# Explicit mutable VMI data slots (allocation marker)
# ---------------------------------------------------------------------------
def _mutable_slot(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, work[0:128])
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                slot = T.vmi.alloc_var(dtype, size=1)
                slot = T.vmi.vcmax(x, mask)
                y = T.vmi.vbrc(slot, size=128, group=1)
                T.vmi.vstore(y, out[0], mask)
            T.copy(out, C)

    return main


def _mutable_slot_branch(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), scale: T.int32, C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((128,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                x = T.vmi.vload(work[0], size=128)
                slot = T.vmi.alloc_var(dtype, size=1)
                if scale > 0:
                    slot = T.vmi.vcmax(x, mask)
                else:
                    slot = T.vmi.vcmin(x, mask)
                T.vmi.vstore(T.vmi.vbrc(slot, size=128, group=1), out[0], mask)
            T.copy(out, C)

    return main


def _mutable_register(dtype, lanes, group):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((128,), dtype)
            dst = T.alloc_shared((128,), dtype)
            T.copy(A, src)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                slot = T.vmi.alloc_var(dtype, size=lanes)
                slot = T.vmi.vload(src[0], size=lanes)
                value = T.vmi.vbrc(slot, size=128, group=group)
                T.vmi.vstore(value, dst[0], mask)
            T.copy(dst, C)

    return main


@pytest.mark.parametrize("dtype", ["int32", "uint32", "float32"])
def test_single_lane_mutable_slot_device(npu, dtype):
    kernel, source = _compile(_mutable_slot(dtype))
    assert "slot = pto.vmi.vreg(1, " in source
    assert "slot = pto.vmi.vcmax(x, mask)" in source

    def make_case(trial):
        if dtype == "uint32":
            a = _uint32_mutable_input(trial)
            expected = torch.full((128,), max(int(value) for value in a.tolist()), dtype=a.dtype)
            return [a], expected
        if dtype == "int32":
            a = _int32_input(trial)
        else:
            a = _float32_input(trial)
        expected = torch.full((128,), a.max().item(), dtype=a.dtype)
        return [a], expected

    _run_trials(kernel, make_case, npu, trials=5 if dtype == "float32" else 10)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_mutable_slot_branch_device(npu, dtype):
    kernel, source = _compile(_mutable_slot_branch(dtype))
    assert "if tl.as_logical_bool(0 < scale):" in source

    def make_case(trial):
        if dtype == "int32":
            a = _int32_input(trial)
        else:
            a = _float32_input(trial)
        positive = trial % 2 == 0
        value = a.max().item() if positive else a.min().item()
        expected = torch.full((128,), value, dtype=a.dtype)
        return [a, 1 if positive else -1], expected

    _run_trials(kernel, make_case, npu, trials=4)


def test_multi_lane_mutable_slot_device(npu):
    kernel, source = _compile(_mutable_register("float32", 4, 4))
    assert "slot = pto.vmi.vreg(4, pto.f32)" in source

    def make_case(trial):
        a = _float32_input(trial)
        # The VMI group-broadcast contract expands each group slot back to its
        # logical group; size=128 and group=4 therefore give four 32-lane
        # groups, and source lane k fills destination lanes [32k, 32k+32).
        expected = a[:4].repeat_interleave(32)
        return [a], expected

    _run_trials(kernel, make_case, npu, trials=4)


# ---------------------------------------------------------------------------
# Register arrays: pure SIMD reinterpret and VMI multidim updates
# ---------------------------------------------------------------------------
def _simd_array_reinterpret():
    @T.prim_func
    def main(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "int32")):
        with T.Kernel(1):
            src = T.alloc_shared((64,), "float32")
            dst = T.alloc_shared((64,), "int32")
            T.copy(X, src)
            with T.SimdVF():
                regs = T.simd.alloc_local((2,), "float32")
                regs[0] = T.simd.vld(src[0])
                y = T.reinterpret("int32x64", regs[0])
                T.simd.vsts(dst[0], y)
            T.copy(dst, Y)

    return main


def test_pure_simd_register_array_reinterpret_device(npu):
    kernel, source = _compile(_simd_array_reinterpret())
    assert "pto.vbitcast(" in source, source
    assert "pto.vmi." not in source, source

    def make_case(trial):
        generator = torch.Generator().manual_seed(20261021 + trial)
        values = torch.randn((64,), generator=generator).to(torch.float32)
        values[0] = -0.0 if trial % 2 else float("inf")
        values[1] = 1.0
        return [values], values.view(torch.int32)

    _run_trials(kernel, make_case, npu, trials=4)


def _uint32_values(values):
    signed = [value - 2**32 if value >= 2**31 else value for value in values]
    return torch.tensor(signed, dtype=torch.int32).view(torch.uint32)


def _uint32_array_input(trial):
    generator = torch.Generator().manual_seed(20261021 + trial)
    values = [int(value) & 0xFFFFFFFF for value in torch.randint(-(2**31), 2**31 - 1, (256,), generator=generator).tolist()]
    values[0] = 0xFFFFFFFF
    values[64] = 0x80000000
    values[128] = 0x7FFFFFFF
    values[192] = 0xFFFFFFFE - (trial % 3)
    values[193] = 0x80000001
    values[194] = 0
    return _uint32_values(values)


def _vmi_array_multidim_update():
    @T.prim_func
    def main(A: T.Tensor((256,), "uint32"), C: T.Tensor((320,), "uint32")):
        with T.Kernel(1):
            work = T.alloc_shared((256,), "uint32")
            out = T.alloc_shared((320,), "uint32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                regs = T.vmi.alloc_local((2, 2), T.vmi.vreg(64, "uint32"))
                for i in T.unroll(2, explicit=True):
                    for j in T.unroll(2, explicit=True):
                        regs[i, j] = T.vmi.vload(work[64 * (2 * i + j)], size=64)
                slot = T.vmi.alloc_var("uint32", size=1)
                slot = T.vmi.vcmax(regs[1, 0], mask)
                for _ in T.serial(3):
                    regs[1, 1] = T.vmi.vadds(regs[1, 1], 1, mask)
                T.vmi.vstore(regs[0, 0], out[0], mask)
                T.vmi.vstore(regs[0, 1], out[64], mask)
                T.vmi.vstore(regs[1, 0], out[128], mask)
                T.vmi.vstore(regs[1, 1], out[192], mask)
                T.vmi.vstore(T.vmi.vbrc(slot, size=64, group=1), out[256], mask)
            T.copy(out, C)

    return main


def test_vmi_register_array_multidim_update_device(npu):
    kernel, source = _compile(_vmi_array_multidim_update())
    assert "pto.vmi.vreg(4" not in source, source
    assert "slot = pto.vmi.vreg(1, pto.ui32)" in source, source
    assert re.search(r"regs_tl_slot_3 = pto\.vmi\.vadds\(", source), source

    def make_case(trial):
        values = _uint32_array_input(trial)
        lanes = values.tolist()
        expected = list(lanes)
        for index in range(192, 256):
            expected[index] = (lanes[index] + 3) % 2**32
        expected.extend([max(lanes[128:192])] * 64)
        return [values], _uint32_values(expected)

    _run_trials(kernel, make_case, npu, trials=4)
