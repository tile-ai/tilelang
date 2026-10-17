"""Explicit-mask PTO VMI store alignment verification.

The read-only ``VerifyPTOCodegenLegality`` PrimFuncPass runs at the end of
the PTO pipeline, after ``LowerDeviceKernelLaunch``, the buffer-version
rewrites and UB merging.  It consumes the final device TIR, resolves the real
destination byte address and proves the alignment required by PTO store
operations:

* an ordinary default/continuous explicit-mask ``tl.vmi.vstore`` requires a
  provably 32-byte-aligned destination;
* the implicit per-point writeback (``1PT_B8/B16/B32``) selected when the
  value has 1/2/4/8 logical lanes and the store mask is a recognized
  full-active seed with the same logical lane count requires only the stored
  element's natural alignment (1/2/4 bytes).

Covered here:

* matcher boundaries: default and continuous full-mask stores pass; stores
  with no mask, a ``dintlv``/``group`` distribution annotation or a scatter
  op stay untouched; the diagnostics distinguish a provably misaligned
  destination from one whose alignment cannot be proven;
* the instruction offset's actual resulting-pointer element unit: inherited
  offsets stay in bytes and are counted exactly once across typed alias
  layers and merged UB arena descriptors;
* mask-seed precision: a recognized full-active seed with a matching lane
  count lets a 1/2/4/8-lane store use point writeback with only natural
  element alignment; partial, dynamic and arbitrary all-true arithmetic
  seeds stay ordinary;
* scoped Analyzer facts: loop ranges and non-zero loop mins;
* the final-pipeline integration: the pass sees the final merged UB arena,
  descriptor and version-slot addresses; compact ``(3,131)`` FP16/FP32 and
  flat/column-offset stores fail before codegen; padded, row-0-only, tail
  and dynamic-offset controls pass; the point-writeback reductions, the
  buffer-version layout and a non-32B-aligned FP16/FP32 load with a
  physically padded source feeding an aligned store compile and run on the
  device with numeric checks; dynamic-mask and single-active stores are
  rejected.

Every accepted direct-pass case asserts the returned IR is structurally
identical to its input.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch
import torch_npu  # noqa: F401,E402  (must import before tilelang/ptodsl)

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower

tirx = tvm.tirx
_PASS = tvm.ffi.get_global_func("tl.transform.VerifyPTOCodegenLegality")()
_PTO_TARGET = normalize_pto_target("pto")
_MARKER = "[VerifyPTOCodegenLegality]"
_INTERNAL_ERROR = tvm.error.InternalError
_OP_VSTORE = tvm.ir.Op.get("tl.vmi.vstore")
_OP_VSCATTER = tvm.ir.Op.get("tl.vmi.vscatter")
_OP_CREATE_MASK = tvm.ir.Op.get("tl.vmi.create_mask")


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


# ---------------------------------------------------------------------------
# Raw TIR builders (direct read-only pass tests)
# ---------------------------------------------------------------------------
def _imm(dtype, value):
    return tirx.IntImm(dtype, value)


def _float_value(dtype, lanes):
    base = tirx.FloatImm(dtype, 1.0)
    return base if lanes == 1 else tirx.Broadcast(base, lanes)


def _int_value(dtype, lanes):
    base = tirx.IntImm(dtype, 1)
    return base if lanes == 1 else tirx.Broadcast(base, lanes)


def _value(dtype, lanes):
    if str(dtype).startswith(("float", "bfloat")):
        return _float_value(dtype, lanes)
    return _int_value(dtype, lanes)


def _seed(lanes, active=None):
    active = lanes if active is None else active
    return tirx.Call(
        f"boolx{lanes}",
        _OP_CREATE_MASK,
        [active if not isinstance(active, int) else _imm("int32", active)],
        annotations={"size": _imm("int32", lanes)},
    )


def _store(value, ptr, offset, mask, annotations=None):
    return tirx.Evaluate(tirx.Call(value.dtype, _OP_VSTORE, [value, ptr, offset, mask], annotations=annotations))


def _root(name="root", dtype="uint8", size=8192, scope="shared.dyn"):
    var = tirx.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))
    buffer = tirx.decl_buffer((size,), dtype, data=var, scope=scope)
    return var, tirx.AllocBuffer(buffer)


def _typed_int_base(name, dtype, base, scope="shared.dyn"):
    var = tirx.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))
    bind = tirx.Bind(var, tirx.reinterpret("handle", tirx.IntImm("int64", base)))
    return var, bind


def _access(dtype, data, offset, extent=64):
    return tirx.tvm_access_ptr(
        tirx.type_annotation(dtype),
        data,
        offset if not isinstance(offset, int) else _imm("int32", offset),
        _imm("int32", extent),
        _imm("int32", 2),
    )


def _func(body, params=(), target=None):
    return tirx.PrimFunc(params, body).with_attr("target", target if target is not None else _PTO_TARGET)


def _run(func):
    return _PASS(tvm.IRModule({"main_kernel": func}))


def _accept(func):
    out = _run(func)
    assert tvm.ir.structural_equal(out["main_kernel"], func), "the alignment pass must return the input IR unchanged"
    return out


def _reject(func):
    with pytest.raises(_INTERNAL_ERROR) as excinfo:
        _run(func)
    text = str(excinfo.value)
    assert _MARKER in text, text
    return text


def _compile_ptoas_artifacts(source):
    """Compile PTODSL to PTO, VPTO and LLVM artifacts."""
    import importlib.util
    import re
    import subprocess
    import tempfile
    import uuid

    decorator = re.search(r'@pto\.jit\(name="([^"]+)"', source)
    name = decorator.group(1) if decorator is not None else "main"
    work = tempfile.mkdtemp(prefix="masked-store-alignment-", dir=os.environ.get("TMPDIR") or None)
    source_path = os.path.join(work, "kernel.ptodsl.py")
    pto_path = os.path.join(work, "kernel.pto")
    vpto_path = os.path.join(work, "kernel.vpto.mlir")
    llvm_path = os.path.join(work, "kernel.ll")
    with open(source_path, "w", encoding="utf-8") as file:
        file.write(source)
    spec = importlib.util.spec_from_file_location("masked_store_alignment_" + uuid.uuid4().hex, source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pto_text = getattr(module, name).compile().mlir_text()
    with open(pto_path, "w", encoding="utf-8") as file:
        file.write(pto_text)
    vpto_result = subprocess.run(
        [
            "ptoas",
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--emit-vpto",
            pto_path,
            "-o",
            vpto_path,
        ],
        text=True,
        capture_output=True,
    )
    llvm_result = subprocess.run(
        [
            "ptoas",
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--emit-vpto-llvm-ir",
            pto_path,
            "-o",
            llvm_path,
        ],
        text=True,
        capture_output=True,
    )

    def read(path):
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8") as file:
            return file.read()

    stderr = "\n".join(error for error in (vpto_result.stderr, llvm_result.stderr) if error)
    return (
        vpto_result.returncode or llvm_result.returncode,
        stderr,
        pto_text,
        read(vpto_path),
        read(llvm_path),
    )


def _ordinary_case(dtype="float32", lanes=64, offset=0):
    """Ordinary masked store through a UB root; returns the PrimFunc."""
    root, alloc = _root(dtype="uint8")
    value = _value(dtype, lanes)
    mask = _seed(lanes)
    ptr = _access(dtype, root, offset)
    return _func(tirx.SeqStmt([alloc, _store(value, ptr, 0, mask)]))


# ---------------------------------------------------------------------------
# Matcher boundaries
# ---------------------------------------------------------------------------
class TestMatcher:
    def test_default_and_continuous_full_mask_pass(self):
        _accept(_ordinary_case(offset=8))
        root, alloc = _root()
        value = _value("float32", 64)
        func = _func(
            tirx.SeqStmt(
                [
                    alloc,
                    _store(
                        value,
                        _access("float32", root, 8),
                        _imm("int32", 0),
                        _seed(64),
                        annotations={"dist_mode": tirx.StringImm("continuous")},
                    ),
                ]
            )
        )
        _accept(func)

    def test_ordinary_store_misaligned_and_unknown_reasons_differ(self):
        misaligned = _reject(_ordinary_case(offset=7))
        assert "destination is misaligned" in misaligned
        assert "byte remainder" in misaligned
        assert "required alignment=32" in misaligned
        root, alloc = _root()
        unknown_offset = tirx.Var("dynamic_offset", "int32")
        func = _func(
            tirx.SeqStmt(
                [
                    alloc,
                    _store(
                        _value("float32", 64),
                        _access("float32", root, unknown_offset),
                        _imm("int32", 0),
                        _seed(64),
                    ),
                ]
            ),
            params=[unknown_offset],
        )
        unknown = _reject(func)
        assert "cannot prove destination alignment" in unknown
        assert "destination is misaligned" not in unknown
        assert "cannot prove alignment for every execution" in unknown

    @pytest.mark.parametrize("case", ["no_mask", "dintlv", "group", "vscatter"])
    def test_unmatched_forms_are_untouched(self, case):
        root, alloc = _root()
        ptr = _access("float32", root, 1)
        value = _value("float32", 64)
        mask = _seed(64)
        statements = [alloc]
        if case == "no_mask":
            statements.append(tirx.Evaluate(tirx.Call(value.dtype, _OP_VSTORE, [value, ptr, _imm("int32", 1)])))
        elif case == "dintlv":
            statements.append(
                _store(
                    value,
                    ptr,
                    _imm("int32", 1),
                    mask,
                    annotations={"dist_mode": tirx.StringImm("dintlv")},
                )
            )
        elif case == "group":
            statements.append(
                _store(
                    value,
                    ptr,
                    _imm("int32", 1),
                    mask,
                    annotations={"group": _imm("int32", 2)},
                )
            )
        else:  # vscatter
            statements.append(
                tirx.Evaluate(
                    tirx.Call(
                        value.dtype,
                        _OP_VSCATTER,
                        [value, ptr, _imm("int32", 1), _imm("int32", 0), mask],
                    )
                )
            )
        _accept(_func(tirx.SeqStmt(statements)))


# ---------------------------------------------------------------------------
# Instruction-offset element unit and address resolution
# ---------------------------------------------------------------------------
class TestInstructionUnit:
    def test_cross_type_inherited_offset_counted_once(self):
        root, alloc = _root()
        base = tirx.Var("base", "handle")
        alias = tirx.Var("alias", "handle")
        prefix = [alloc, tirx.Bind(base, _access("uint8", root, 32)), tirx.Bind(alias, base)]
        value = _value("float32", 64)
        # 32 + 6*4 + 2*4 = 64 bytes: aligned only when every layer is counted
        # exactly once with its own element unit.
        aligned = _func(
            tirx.SeqStmt(
                prefix
                + [
                    _store(
                        value,
                        _access("float32", alias, 6),
                        _imm("int32", 2),
                        _seed(64),
                    )
                ]
            )
        )
        _accept(aligned)
        misaligned = _func(
            tirx.SeqStmt(
                prefix
                + [
                    _store(
                        value,
                        _access("float32", alias, 6),
                        _imm("int32", 0),
                        _seed(64),
                    )
                ]
            )
        )
        text = _reject(misaligned)
        assert "byte remainder=T.int64(24)" in text

    def test_merged_arena_offset_counted_once(self):
        # A descriptor offset of 400 f32 elements = 1600 bytes is 32B aligned;
        # 401 elements = 1604 bytes leaves a remainder of 4.
        root, alloc = _root()
        _accept(
            _func(
                tirx.SeqStmt(
                    [
                        alloc,
                        _store(
                            _value("float32", 64),
                            _access("float32", root, 400),
                            _imm("int32", 0),
                            _seed(64),
                        ),
                    ]
                )
            )
        )
        text = _reject(
            _func(
                tirx.SeqStmt(
                    [
                        alloc,
                        _store(
                            _value("float32", 64),
                            _access("float32", root, 401),
                            _imm("int32", 0),
                            _seed(64),
                        ),
                    ]
                )
            )
        )
        assert "byte remainder=T.int64(4)" in text


# ---------------------------------------------------------------------------
# Mask seeds and 1PT classification
# ---------------------------------------------------------------------------
class TestMaskSeed:
    @pytest.mark.parametrize(
        ("dtype", "natural"),
        [("uint8", 1), ("float16", 2), ("float32", 4)],
        ids=["b8", "b16", "b32"],
    )
    def test_point_writeback_needs_only_natural_alignment(self, dtype, natural):
        ptr, bind = _typed_int_base("mt", dtype, natural)
        _accept(_func(tirx.SeqStmt([bind, _store(_value(dtype, 1), ptr, _imm("int32", 0), _seed(1))])))
        if natural > 1:
            ptr, bind = _typed_int_base("mt", dtype, natural - 1)
            _reject(_func(tirx.SeqStmt([bind, _store(_value(dtype, 1), ptr, _imm("int32", 0), _seed(1))])))

    @pytest.mark.parametrize(
        "mask_factory",
        [
            lambda: _seed(4, active=2),
            lambda: _seed(1, active=tirx.Var("active", "int32")),
            lambda: tirx.Or(_seed(1), _seed(1)),
        ],
        ids=["partial", "dynamic_active", "all_true_arith"],
    )
    def test_non_full_seeds_stay_ordinary(self, mask_factory):
        root, alloc = _root()
        mask = mask_factory()
        lanes = mask.dtype.lanes
        value = _value("float32", lanes)
        text = _reject(
            _func(
                tirx.SeqStmt(
                    [
                        alloc,
                        _store(value, _access("float32", root, 1), _imm("int32", 0), mask),
                    ]
                )
            )
        )
        assert "ordinary masked vstore" in text
        _accept(
            _func(
                tirx.SeqStmt(
                    [
                        alloc,
                        _store(value, _access("float32", root, 8), _imm("int32", 0), mask),
                    ]
                )
            )
        )


# ---------------------------------------------------------------------------
# Scoped analysis facts
# ---------------------------------------------------------------------------
def _loop_with_offset(offset_builder):
    root, alloc = _root()
    loop_var = tirx.Var("i", "int32")
    body = _store(
        _value("float32", 64),
        _access("float32", root, offset_builder(loop_var)),
        _imm("int32", 0),
        _seed(64),
    )
    loop = tirx.For(
        loop_var,
        _imm("int32", 0),
        _imm("int32", 4),
        tirx.ForKind.SERIAL,
        body,
    )
    return _func(tirx.SeqStmt([alloc, loop]))


class TestAnalysisScope:
    def test_loop_range_facts(self):
        _accept(_loop_with_offset(lambda i: i * _imm("int32", 8)))
        _reject(_loop_with_offset(lambda i: i))

    def test_nonzero_loop_min_with_step(self):
        i = tirx.Var("i", "int32")
        offset = (i - _imm("int32", 4)) * _imm("int32", 2)
        root, alloc = _root()
        body = _store(
            _value("float32", 64),
            _access("float32", root, offset),
            _imm("int32", 0),
            _seed(64),
        )
        loop = tirx.For(
            i,
            _imm("int32", 4),
            _imm("int32", 8),
            tirx.ForKind.SERIAL,
            body,
            step=_imm("int32", 4),
        )
        _accept(_func(tirx.SeqStmt([alloc, loop])))


# ---------------------------------------------------------------------------
# Final pipeline integration
# ---------------------------------------------------------------------------
@contextmanager
def _pass_boundary_capture():
    captured = {}
    original = ascend_transform.VerifyPTOCodegenLegality

    def factory():
        real = original()

        def run(mod):
            captured["module"] = mod
            return real(mod)

        return run

    ascend_transform.VerifyPTOCodegenLegality = factory
    try:
        yield captured
    finally:
        ascend_transform.VerifyPTOCodegenLegality = original


def _lower_capture(func):
    target = normalize_pto_target("pto")
    with _pass_boundary_capture() as captured, target:
        _lower(func, target=target)
    return captured["module"]


def _lower_expect_error(func):
    with pytest.raises(Exception) as excinfo:  # noqa: B017 - kernel errors vary
        tilelang.lower(func, target="pto")
    text = str(excinfo.value)
    assert _MARKER in text, text
    assert "ptoas" not in text.lower()
    return text


def _parallel_kernel(dtype, m, n, lanes=256, row=None):
    if row is None:

        @T.prim_func
        def main(A: T.Tensor((m, n), dtype), C: T.Tensor((m, n), dtype)):
            with T.Kernel(1):
                a = T.alloc_shared((m, n), dtype)
                c = T.alloc_shared((m, n), dtype)
                T.copy(A, a)
                with T.SimdVF(lanes=lanes):
                    for i, j in T.Parallel(m, n):
                        c[i, j] = a[i, j] + T.cast(3, dtype)
                T.copy(c, C)

        return main

    @T.prim_func
    def main(A: T.Tensor((m, n), dtype), C: T.Tensor((m, n), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((m, n), dtype)
            c = T.alloc_shared((m, n), dtype)
            T.copy(A[row, :], a[row, :])
            with T.SimdVF(lanes=lanes):
                for j in T.Parallel(n):
                    c[row, j] = a[row, j] + T.cast(3, dtype)
            T.copy(c[row, :], C[0, :])

    return main


def _tail_kernel(dtype="float32", n=131, lanes=64):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((n,), dtype)
            c = T.alloc_shared((n,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for j in T.Parallel(n):
                    c[j] = a[j] + T.cast(3, dtype)
            T.copy(c, C)

    return main


def _flat_kernel(m=3, n=131, dtype="float32", lanes=256):
    @T.prim_func
    def main(A: T.Tensor((m * n,), dtype), C: T.Tensor((m * n,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((m * n,), dtype)
            c = T.alloc_shared((m * n,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(m, n):
                    c[i * n + j] = a[i * n + j] + T.cast(3, dtype)
            T.copy(c, C)

    return main


def _column_offset_kernel(n=256, dtype="float32", lanes=256):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), C: T.Tensor((n + 1,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((n,), dtype)
            c = T.alloc_shared((n + 1,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for j in T.Parallel(n):
                    c[j + 1] = a[j] + T.cast(3, dtype)
            T.copy(c[1 : n + 1], C[1 : n + 1])

    return main


_REDUCE_OFFSET = 4100
_REDUCE_WINDOW = (_REDUCE_OFFSET // 64) * 64
_REDUCE_INDEX = _REDUCE_OFFSET - _REDUCE_WINDOW


_SENTINEL_LANES = 128


def _manual_reduce_kernel(group=None, size=64, dynamic=False):
    out_lanes = 1 if group is None else group
    window_start = _REDUCE_WINDOW
    window_end = window_start + _SENTINEL_LANES

    if dynamic:

        @T.prim_func
        def main(
            A: T.Tensor((size,), "float32"),
            S: T.Tensor((_SENTINEL_LANES,), "float32"),
            active: T.int32,
            C: T.Tensor((_SENTINEL_LANES,), "float32"),
        ):
            with T.Kernel(1):
                work = T.alloc_shared((size,), "float32")
                out = T.alloc_shared((8192,), "float32")
                T.copy(A, work)
                T.copy(S, out[window_start:window_end])
                with T.SimdVF():
                    mask = T.vmi.create_mask(size, size=size)
                    x = T.vmi.vload(work[0], size=size)
                    r = T.vmi.vcadd(x, mask, reassoc=True)
                    slot = T.alloc_var("int32")
                    slot = active
                    store_mask = T.vmi.create_mask(slot, size=out_lanes)
                    T.vmi.vstore(r, out[_REDUCE_OFFSET], store_mask)
                T.copy(out[window_start:window_end], C)

        return main

    @T.prim_func
    def main(
        A: T.Tensor((size,), "float32"),
        S: T.Tensor((_SENTINEL_LANES,), "float32"),
        C: T.Tensor((_SENTINEL_LANES,), "float32"),
    ):
        with T.Kernel(1):
            work = T.alloc_shared((size,), "float32")
            out = T.alloc_shared((8192,), "float32")
            T.copy(A, work)
            # Independently initialized surrounding sentinels: a swapped,
            # duplicated or out-of-footprint write inside this window changes
            # a sentinel slot and fails the comparison.
            T.copy(S, out[window_start:window_end])
            with T.SimdVF():
                mask = T.vmi.create_mask(size, size=size)
                x = T.vmi.vload(work[0], size=size)
                if group is None:
                    r = T.vmi.vcadd(x, mask, reassoc=True)
                else:
                    r = T.vmi.vcadd(x, mask, group=group, reassoc=True)
                store_mask = T.vmi.create_mask(out_lanes, size=out_lanes)
                T.vmi.vstore(r, out[_REDUCE_OFFSET], store_mask)
            T.copy(out[window_start:window_end], C)

    return main


def _manual_single_active_ordinary(lanes=64):
    @T.prim_func
    def main(A: T.Tensor((lanes,), "float32"), C: T.Tensor((lanes,), "float32")):
        with T.Kernel(1):
            work = T.alloc_shared((lanes,), "float32")
            out = T.alloc_shared((8192,), "float32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(1, size=lanes)
                x = T.vmi.vload(work[0], size=lanes)
                T.vmi.vstore(x, out[_REDUCE_OFFSET], mask)
            T.copy(
                out[_REDUCE_WINDOW : _REDUCE_WINDOW + lanes],
                C,
            )

    return main


def _serial_version_kernel():
    @T.prim_func
    def main(A: T.Tensor((512,), "float32"), C: T.Tensor((512,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            c = T.alloc_shared((128,), "float32")
            T.annotate_buffer_versions({a: 2, c: 2})
            for k in T.serial(4):
                T.copy(A[k * 128 : (k + 1) * 128], a)
                with T.SimdVF(lanes=128):
                    for j in T.Parallel(128):
                        c[j] = a[j] + T.cast(3, "float32")
                T.copy(c, C[k * 128 : (k + 1) * 128])

    return main


def _non32b_load_kernel(dtype):
    source_len = 131 + 256

    @T.prim_func
    def main(S: T.Tensor((source_len,), dtype), C: T.Tensor((256,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((source_len + 128,), dtype)
            out = T.alloc_shared((256,), dtype)
            T.copy(S, work[0:source_len])
            with T.SimdVF(lanes=256):
                mask = T.vmi.create_mask(256, size=256)
                x = T.vmi.vload(work[131], size=256)
                T.vmi.vstore(x, out[0], mask)
            T.copy(out[0:256], C)

    return main


def _dynamic_offset_kernel(mult):
    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        k: T.int32,
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            work = T.alloc_shared((1024,), "float32")
            T.copy(A, work[0:64])
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                x = T.vmi.vload(work[0], size=64)
                T.vmi.vstore(x, work[k * mult], mask)
            T.copy(work[0:64], C)

    return main


class TestFinalPipeline:
    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_compact_three_by_131_fails_before_codegen(self, dtype):
        text = _lower_expect_error(_parallel_kernel(dtype, 3, 131))
        assert "ordinary masked vstore" in text
        assert "required alignment=32" in text
        assert "root=" in text
        assert "normalized byte offset=" in text

    def test_flat_same_address_fails(self):
        text = _lower_expect_error(_flat_kernel())
        assert "cannot prove destination alignment" in text

    def test_nonzero_column_offset_fails(self):
        text = _lower_expect_error(_column_offset_kernel())
        assert "destination is misaligned" in text

    @pytest.mark.parametrize(
        ("name", "kernel"),
        [
            ("padded", _parallel_kernel("float32", 3, 136)),
            ("row0_only", _parallel_kernel("float32", 3, 131, row=0)),
            ("single_row_tail_vl64", _tail_kernel()),
        ],
    )
    def test_aligned_controls_full_compile(self, name, kernel):
        compiled = tilelang.compile(kernel, target="pto", out_idx=-1)
        source = compiled.get_kernel_source()
        assert "vmi.vstore" in source, name

    def test_dynamic_offset_controls(self):
        aligned = _dynamic_offset_kernel(8)
        module = _lower_capture(aligned)
        assert module is not None
        tilelang.compile(aligned, target="pto", out_idx=-1)
        text = _lower_expect_error(_dynamic_offset_kernel(1))
        assert "cannot prove destination alignment" in text

    @pytest.mark.parametrize(
        ("group", "size", "lanes"),
        [(None, 64, 1), (2, 128, 2), (4, 256, 4), (8, 256, 8)],
    )
    def test_point_writeback_compiles_and_runs(self, group, size, lanes):
        _require_npu_runtime()
        kernel = tilelang.compile(
            _manual_reduce_kernel(group=group, size=size),
            target="pto",
            out_idx=-1,
        )
        source = kernel.get_kernel_source()
        assert "vmi.vstore" in source
        values = torch.arange(size, dtype=torch.float32)
        sentinels = torch.arange(_SENTINEL_LANES, dtype=torch.float32) * -3.0 - 500.0
        output = kernel(values.to("npu"), sentinels.to("npu"))
        torch.npu.synchronize()
        got = output.cpu()
        # The group reduction produces one result per consecutive
        # size/lanes-element chunk. Every other slot in the initialized
        # 128-element window must retain its sentinel. Swapped, duplicated or
        # out-of-footprint writes must fail even when the total sum matches.
        chunk = size // lanes
        expected_slots = [float(values[index * chunk : (index + 1) * chunk].sum()) for index in range(lanes)]
        expected = sentinels.clone()
        expected[_REDUCE_INDEX : _REDUCE_INDEX + lanes] = torch.tensor(expected_slots, dtype=torch.float32)
        torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-3)

    def test_actual_version_layout_address_is_checked_and_runs(self):
        _require_npu_runtime()
        module = _lower_capture(_serial_version_kernel())
        stores = []

        def visit(node):
            if isinstance(node, tirx.Call) and hasattr(node.op, "name") and node.op.name == "tl.vmi.vstore":
                stores.append(node)

        tirx.stmt_functor.post_order_visit(module["main_kernel"].body, visit)
        assert stores, "expected a versioned VMI store"
        for store in stores:
            descriptor_offset = store.args[1].args[2]
            version_terms = []

            def offset_visit(node, terms=version_terms):
                if isinstance(node, tirx.FloorMod):
                    terms.append(node)

            tirx.stmt_functor.post_order_visit(descriptor_offset, offset_visit)
            assert version_terms, f"expected the final descriptor to contain the version-slot index, got {descriptor_offset}"
        kernel = tilelang.compile(_serial_version_kernel(), target="pto", out_idx=-1)
        values = torch.arange(512, dtype=torch.float32)
        output = kernel(values.to("npu"))
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), values + 3.0, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_non32b_aligned_load_with_padded_source_runs(self, dtype):
        _require_npu_runtime()
        kernel = tilelang.compile(_non32b_load_kernel(dtype), target="pto", out_idx=-1)
        source = kernel.get_kernel_source()
        load_line = next(line for line in source.splitlines() if "vmi.vload" in line)
        assert "131" in load_line and "addptr" in load_line, load_line
        returncode, stderr, pto_text, vpto_text, llvm_text = _compile_ptoas_artifacts(source)
        assert returncode == 0, stderr
        assert "pto.vldas" in vpto_text, vpto_text
        assert "pto.vldus" in vpto_text, vpto_text
        assert "131" in pto_text, pto_text
        assert "llvm.hivm.vldas" in llvm_text, llvm_text
        assert "llvm.hivm.vldus" in llvm_text, llvm_text
        values = torch.arange(387, dtype=torch.float32).to(getattr(torch, dtype))
        output = kernel(values.to("npu"))
        torch.npu.synchronize()
        expected = values[131:387].float().cpu()
        torch.testing.assert_close(output.cpu().float(), expected, rtol=0.0, atol=0.0)

    def test_dynamic_and_single_active_stores_reject(self):
        text = _lower_expect_error(_manual_reduce_kernel(dynamic=True))
        assert "ordinary masked vstore" in text
        assert "required alignment=32" in text
        text = _lower_expect_error(_manual_single_active_ordinary())
        assert "ordinary masked vstore" in text
        assert "required alignment=32" in text
