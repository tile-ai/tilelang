"""Device acceptance: SIMT local integer signless storage and adapters.

Each case compiles through the real PTO pipeline (PTODSL -> PTOAS VPTO ->
Bisheng) and runs on the NPU.  The reference is computed independently on
the CPU with explicit target-width wrapping, so a byte-copy-only implementation
cannot pass: the kernel performs a signed/unsigned comparison and arithmetic
after reloading the value from the local buffer, plus exact boundary values.

Vector access offsets are 32-byte aligned (b8: 32 lanes, b32: 8 lanes), the
hardware-legal SIMT vector shape.  All source lanes read by the kernel are
initialized by the GM->UB DMA and the destination is fully initialized by a
second DMA before the stores.
"""

import os
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

CASES = {
    # dtype: (torch dtype, vector lanes, boundary values, signed?, high bit)
    "int8": (torch.int8, 32, (127, -1, -128, 0), True, 0),
    "uint8": (torch.uint8, 32, (128, 255, 0, 1), False, 128),
    "int32": (torch.int32, 8, (2147483647, -1, -2147483648, 0), True, 0),
    "uint32": (torch.uint32, 8, (2147483648, 4294967295, 0, 1), False, 2147483648),
}


def _require_npu_runtime():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("torch.npu is unavailable in the current environment")
    try:
        source = torch.arange(8, dtype=torch.float32)
        copied = source.to(torch.device("npu"))
        torch.npu.synchronize()
        assert torch.equal(copied.cpu(), source)
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


@pytest.fixture(scope="module")
def npu():
    _require_npu_runtime()
    return torch.device("npu")


def _wrap(value, bits, signed):
    value &= (1 << bits) - 1
    if signed and value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _condition(x, dtype, signed, highbit):
    """Signedness-sensitive comparison evaluated while tracing the kernel."""
    if signed:
        return x < T.cast(0, dtype)
    return x >= T.cast(highbit, dtype)


def _local_integer_kernel(dtype, lanes, bounds, signed, highbit):
    b0, b1, b2, b3 = bounds
    scalar_index = lanes + 4

    @T.prim_func
    def main(A: T.Tensor((64,), dtype), C: T.Tensor((64,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((64,), dtype)
            dst = T.alloc_shared((64,), dtype)
            T.copy(A, src)
            T.copy(A, dst)  # every destination lane is initialized
            with T.SimtVF(threads=1):
                local = T.alloc_local((64,), dtype)
                # Contiguous builtin-vector store/load landing.
                local[T.Ramp(0, 1, lanes)] = src[T.Ramp(0, 1, lanes)]
                dst[T.Ramp(0, 1, lanes)] = local[T.Ramp(0, 1, lanes)]
                # Reload one vector, load a second, and compute v + w.
                local[T.Ramp(lanes, 1, lanes)] = src[T.Ramp(lanes, 1, lanes)]
                left = local[T.Ramp(0, 1, lanes)]
                right = local[T.Ramp(lanes, 1, lanes)]
                dst[T.Ramp(0, 1, lanes)] = left + right
                # Scalar store/load landing.  The index is outside the
                # vector destination range so the vector result stays
                # observable.
                local[scalar_index] = src[scalar_index]
                dst[scalar_index] = local[scalar_index]
                # Signed/unsigned comparison and arithmetic after reload.
                value = local[scalar_index]
                dst[scalar_index + 1] = T.if_then_else(
                    _condition(value, dtype, signed, highbit),
                    value + T.cast(1, dtype),
                    value + T.cast(2, dtype),
                )
                # Exact boundary values through scalar landings.
                local[40] = T.cast(b0, dtype)
                dst[40] = local[40]
                local[41] = T.cast(b1, dtype)
                dst[41] = local[41]
                local[42] = T.cast(b2, dtype)
                dst[42] = local[42]
                local[43] = T.cast(b3, dtype)
                dst[43] = local[43]
            T.copy(dst, C)

    return main


def _expected(a, lanes, bounds, signed):
    out = a.clone()
    bits = a.dtype.itemsize * 8
    for lane in range(lanes):
        out[lane] = _wrap(int(a[lane]) + int(a[lanes + lane]), bits, signed)
    scalar_index = lanes + 4
    out[scalar_index] = a[scalar_index]
    x = int(a[scalar_index])
    takes_true = x < 0 if signed else x >= (1 << (bits - 1))
    x = x + 1 if takes_true else x + 2
    out[scalar_index + 1] = _wrap(x, bits, signed)
    for index, boundary in enumerate(bounds):
        out[40 + index] = _wrap(boundary, bits, signed)
    return out


def _trial_input(torch_dtype, signed, trial, lanes):
    """Fresh input per trial.

    Vector lanes at [0,4) and [lanes,lanes+4) use boundary/high-bit patterns
    whose wrapped sums are computed independently by ``_expected``.  The scalar
    comparison value alternates between the true and false arms across trials.
    """
    bits = torch_dtype.itemsize * 8
    generator = torch.Generator().manual_seed(7000 + trial)
    base = torch.randint(0, 50, (64,), generator=generator, dtype=torch.int64) + trial * 3
    values = [int(value) for value in base]
    maxv = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    minv = -(1 << (bits - 1)) if signed else 0
    high = 1 << (bits - 1)
    if signed:
        first_pattern = [maxv, minv, -1, 0]
        second_pattern = [1, -1, 1, minv]
    else:
        first_pattern = [maxv, 0, high, 1]
        second_pattern = [1, high, 1, maxv]
    variants = [
        (first_pattern, second_pattern),
        (second_pattern, first_pattern),
        (list(reversed(first_pattern)), list(reversed(second_pattern))),
    ]
    first, second = variants[trial % len(variants)]
    values[0:4] = first
    values[lanes : lanes + 4] = second
    scalar_index = lanes + 4
    if trial % 2 == 0:
        values[scalar_index] = -5 if signed else high
    else:
        values[scalar_index] = 0
    return torch.tensor(values, dtype=torch_dtype)


@pytest.mark.parametrize("dtype", list(CASES), ids=list(CASES))
def test_simtvf_local_integer_roundtrip_arithmetic_and_boundaries(npu, dtype):
    torch_dtype, lanes, bounds, signed, highbit = CASES[dtype]
    kernel = tilelang.compile(
        _local_integer_kernel(dtype, lanes, bounds, signed, highbit),
        target="pto",
        out_idx=-1,
    )
    source = kernel.get_kernel_source()

    backing = "pto.i8" if dtype in ("int8", "uint8") else "pto.i32"
    logical = {"int8": "pto.si8", "uint8": "pto.ui8", "int32": "pto.si32", "uint32": "pto.ui32"}[dtype]
    # Live allocation and all four adapter landings are present in the final
    # generated source (they survived lowering/optimization).
    assert f"pto.alloc_buffer((64,), {backing})" in source, source
    scalar_index = lanes + 4
    assert f"pto.cast(pto.load(local, {scalar_index}), {logical})" in source, source
    assert f"pto.store(pto.cast(pto.load(local, 0, contiguous={lanes}), {logical})" in source, source
    assert f", {backing}), local, 0, contiguous={lanes})" in source, source
    assert f", {backing}), local, {scalar_index})" in source, source

    arms = set()
    for trial in range(3):
        a = _trial_input(torch_dtype, signed, trial, lanes)
        x = int(a[scalar_index])
        arms.add(x < 0 if signed else x >= highbit)
        out = kernel(a.to(npu))
        torch.npu.synchronize()
        got = out.cpu()
        expected = _expected(a, lanes, bounds, signed)
        assert torch.equal(got, expected), f"{dtype} trial {trial} mismatch\n got={got.tolist()}\n exp={expected.tolist()}"
        # Guarded output: every lane outside the intended writes still holds
        # the independently initialized input value.
        touched = set(range(lanes)) | {scalar_index, scalar_index + 1, 40, 41, 42, 43}
        for index in range(64):
            if index not in touched:
                assert got[index].item() == a[index].item(), (dtype, trial, index)
    assert arms == {True, False}, f"{dtype} comparison arms exercised: {arms}"


def _shared_control_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((8,), dtype), C: T.Tensor((8,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((8,), dtype)
            dst = T.alloc_shared((8,), dtype)
            T.copy(A, src)
            T.copy(A, dst)
            with T.SimtVF(threads=1):
                dst[0] = src[0]
            T.copy(dst, C)

    return main


@pytest.mark.parametrize("dtype,torch_dtype", [("uint8", torch.uint8), ("int32", torch.int32)])
def test_simtvf_shared_integer_access_control(npu, dtype, torch_dtype):
    kernel = tilelang.compile(_shared_control_kernel(dtype), target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    line = next(line for line in source.splitlines() if "pto.store(pto.load(" in line)
    assert "pto.cast(" not in line, line

    a = torch.arange(-4, 4, dtype=torch.int32).to(torch_dtype)
    out = kernel(a.to(npu))
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), a)
