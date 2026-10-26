"""VMI memory-access descriptors and interface adaptation (structural).

Hardware independent.  Each case lowers the real TileLang PTO pipeline (no
device execution) and inspects the produced TIR/PTODSL.  Cases that the
backend can express also call ``tilelang.compile``, which runs PTODSL -> PTOAS
(VPTO) -> Bisheng; that is compile evidence, not numerical evidence (device
results live in ``test_pto_vmi_memory_access_run.py``).

The frontend keeps an ordinary ``T.vmi.vload``/``T.vmi.vstore`` address as the
raw ``BufferLoad``; ``LegalizePTOVMIMemoryAccess`` builds the descriptor in TIR
after ``InjectAssumes``, so the proof sees branch facts and the result is
visible to the scheduler.  This file verifies:

* Descriptor construction: the true start address and the per-access-mode
  effective extent.  The VMI instruction offset is zero unless a relocation
  candidate moves the source position into it; descriptor base + instruction
  offset always equals the original access address after the normal
  negative-index legalization.
* Physical envelopes: ordinary loads round up to full physical vectors; group
  envelopes cover all groups (a slot counts one element per group, a static
  negative stride relocates the lowest address and compensates with the
  instruction offset, a padded row-major view declares its logical prefix from
  the buffer origin, and a whole tight buffer is used only when its envelope
  and every axis start are proven compatible with the buffer).
  A dynamic group or block stride is rejected instead of falling back to a
  fabricated range; a static negative stride is rejected only when no complete
  descriptor preserves the address; ``block_stride`` extents are measured in
  32B blocks.  Genuine FP4 descriptors stay logical and are packed exactly
  once; packed (PK) store extents stay logical target elements; gather/scatter
  preserve non-zero bases.
* Coverage diagnostics from the descriptor pass: a local descriptor must prove
  the row-major carry from its own start (there is no whole-buffer fallback);
  dynamic indices may use active loop domains and stable integer lets, but a
  mutable read or an unknown scalar is never bound as an equality; dynamic
  inner dimensions and symbolic multi-dimensional extents require an in-row
  proof.  An explicit ``access_ptr`` keeps its caller-declared descriptor only
  when its extent contains the envelope and its declared region still covers
  the access.  API, UB-scope, mask, direction and element-unit checks stay in
  the frontend.
* Masks never shrink the may-write extent, a logical ``size=80`` load declares
  both physical reads, and the descriptor pass runs exactly once on the PTO
  path in the documented position (and never for plain AscendC).
* Code generation rejects an unlowered outermost raw address at the correct
  VMI operand; a nested scalar read inside a formal descriptor stays a real
  read.
* Removed interface parameters (``repeat_stride``, vload ``unpack`` /
  ``to_dtype``, vstore ``pmode``) fail in a legal scope.

``size=80`` VMI vectors are not constructible by the PTODSL VMI API
(lane counts are 1/2/4/8/64/128/256).  Their real lowering is verified with a
direct PTOAS compile of the PTO form the descriptor describes; the TileLang
diagnostics/descriptors are verified structurally.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import tempfile
from contextlib import contextmanager

import pytest
import torch  # noqa: F401  (torch/torch_npu must import before tilelang/ptodsl)
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_ascend_target, normalize_pto_target
from tilelang.engine.lower import lower as _lower
from tilelang.instrumentation import (
    PassEventObserver,
    PassInstrumentationTool,
    StackedPassInstrument,
    compile_pass_instrumentation,
    create_pass_instruments,
)

FP4 = "float4_e2m1fn"
F16 = "float16"
F32 = "float32"
F8_E4M3 = "float8_e4m3fn"


# ---------------------------------------------------------------------------
# Pipeline/diagnostic helpers
# ---------------------------------------------------------------------------
def _lower_pto(func):
    """Lower *func* for the PTO target and return the compiled artifact."""
    target = normalize_pto_target("pto")
    with target:
        return _lower(func, target=target)


def _compile_pto(func):
    """Compile through the PTODSL -> PTOAS (VPTO) -> Bisheng pipeline."""
    return tilelang.compile(func, target="pto", out_idx=-1)


class _TraceObserver(PassEventObserver):
    """Record visited pass names and optionally capture a module boundary."""

    def __init__(self, capture_start=None, capture_finish=None):
        self.names = []
        self.capture_start = capture_start
        self.capture_finish = capture_finish
        self.started_module = None
        self.finished_module = None

    def pass_started(self, mod, event):
        self.names.append(event.name)
        if event.name == self.capture_start and self.started_module is None:
            self.started_module = mod

    def pass_finished(self, mod, event, state):
        if event.name == self.capture_finish and self.finished_module is None:
            self.finished_module = mod


class _TraceTool(PassInstrumentationTool):
    def __init__(self, observer):
        self.observer = observer

    def create_pass_instrument(self):
        return StackedPassInstrument(self.observer)


_TRACE_IDS = itertools.count()


def _lower_pto_traced(
    func,
    *,
    target_name="pto",
    auto_schedule=True,
    capture_start=None,
    capture_finish=None,
):
    """Lower *func* with the real product pipeline and record its pass trace."""
    observer = _TraceObserver(capture_start=capture_start, capture_finish=capture_finish)
    name = f"vmi-descriptor-trace-{next(_TRACE_IDS)}"
    target = normalize_pto_target("pto") if target_name == "pto" else normalize_ascend_target("ascend")
    with (
        compile_pass_instrumentation(name=name, tools=[_TraceTool(observer)]),
        tvm.transform.PassContext(
            config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: auto_schedule},
            instruments=create_pass_instruments(context=name),
        ),
        target,
    ):
        artifact = _lower(func, target=target)
    return artifact, observer


def _pto_source(func):
    """Generate PTODSL for *func* without compiling, through the real backend."""
    target = normalize_pto_target("pto")
    build = tvm.ffi.get_global_func("target.build.tilelang_pto_without_compile")
    with target:
        return build(tvm.IRModule({"main": func}), target).inspect_source()


def _iter_calls(artifact, op_name):
    for func in artifact.device_mod.functions.values():
        found = []

        def visit(node, found=found):
            if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == op_name:
                found.append(node)

        tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
        for node in found:
            yield func, node


def _resolve_binding(expr, func):
    """Resolve a local handle binding (``p = access_ptr(...)``) to its value."""
    if not isinstance(expr, tvm.tirx.Var):
        return expr
    found = {}

    def visit(node):
        if isinstance(node, (tvm.tirx.Let, tvm.tirx.Bind)) and node.var is not None and node.var.same_as(expr):
            found["value"] = node.value

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return found.get("value", expr)


def _const(expr):
    from tvm import arith

    simplified = arith.Analyzer().simplify(expr)
    assert isinstance(simplified, tvm.tirx.IntImm), f"expected a constant, got {simplified!r}"
    return int(simplified.value)


def _single(artifact, op_name):
    pairs = list(_iter_calls(artifact, op_name))
    assert len(pairs) == 1, f"expected exactly one {op_name}, got {len(pairs)}"
    return pairs[0]


def _assert_descriptor_error(build, op, reason, *, explicit=False, descriptor=True):
    """Assert the descriptor pass rejects an unprovable footprint.

    ``descriptor=False`` covers diagnostics emitted directly by the pass before
    it selects a plan (dynamic/negative group and block strides); those do not
    carry the descriptor-context sentence or the buffer context.
    """
    with pytest.raises(tvm.error.TVMError) as excinfo:
        _lower_pto(build())
    message = str(excinfo.value)
    assert "LegalizePTOVMIMemoryAccess" in message, message
    if descriptor:
        prefix = (
            "cannot verify the explicit access_ptr descriptor"
            if explicit
            else "cannot construct a descriptor covering the complete physical access"
        )
        assert f"{op} {prefix}" in message, message
        assert "buffer '" in message and ": shape=[" in message, message
    assert f"{op} " in message, message
    assert reason in message, message
    return message


def _ptr_position(op_name):
    return 1 if op_name == "tl.vmi.vstore" else 0


def _descriptor(artifact, op_name):
    """Return (descriptor_offset, extent, rw_mask, instruction_offset)."""
    func, call = _single(artifact, op_name)
    ptr_pos = _ptr_position(op_name)
    ptr = _resolve_binding(call.args[ptr_pos], func)
    assert isinstance(ptr, tvm.tirx.Call) and ptr.op.name.endswith("tvm_access_ptr"), (
        f"{op_name} address is not an access descriptor: {ptr!r}"
    )
    return (
        _const(ptr.args[2]),
        _const(ptr.args[3]),
        _const(ptr.args[4]),
        _const(call.args[ptr_pos + 1]),
    )


def _vload_descriptor(func):
    return _descriptor(_lower_pto(func), "tl.vmi.vload")


def _vstore_descriptor(func):
    return _descriptor(_lower_pto(func), "tl.vmi.vstore")


@contextmanager
def _capture_rewrite_input():
    """Capture the module handed to RewriteFp4ToFp4x2, then delegate."""
    captured = {}
    original = ascend_transform.RewriteFp4ToFp4x2

    def factory():
        rewrite = original()

        def run(mod):
            captured["module"] = mod
            return rewrite(mod)

        return run

    ascend_transform.RewriteFp4ToFp4x2 = factory
    try:
        yield captured
    finally:
        ascend_transform.RewriteFp4ToFp4x2 = original


def _descriptor_in_module(mod, op_name):
    class _Artifact:
        device_mod = mod

    return _descriptor(_Artifact(), op_name)


def _ptoas_vpto(pto_text: str) -> str:
    """Compile direct PTO IR through PTOAS and return the VPTO text."""
    work = tempfile.mkdtemp(prefix="vmi-vpto-", dir=os.environ.get("TMPDIR") or None)
    pto_path = os.path.join(work, "case.pto")
    vpto_path = os.path.join(work, "case.vpto.mlir")
    with open(pto_path, "w", encoding="utf-8") as file:
        file.write(pto_text)
    result = subprocess.run(
        ["ptoas", "--pto-arch=a5", "--pto-backend=vpto", "--emit-vpto", pto_path, "-o", vpto_path],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, f"PTOAS rejected the probe:\n{result.stderr}"
    with open(vpto_path, encoding="utf-8") as file:
        return file.read()


# ---------------------------------------------------------------------------
# Ordinary non-zero BufferLoad start
# ---------------------------------------------------------------------------
def _load_store_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((320,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work[0:256])
            T.fill(work[256:320], T.float32(-5.0))
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                v = T.vmi.vload(work[64], size=64)
                T.vmi.vstore(v, work[192], mask)
            T.copy(work[0:320], C)

    return main


def _func_calls(func, op_name):
    """Calls of *op_name* in an unlowered frontend PrimFunc."""
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == op_name:
            found.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


def test_frontend_keeps_ordinary_raw_address():
    """The frontend keeps automatic addresses raw until the descriptor pass.

    ``vload``/``vstore`` address operands stay ``BufferLoad`` with their
    original scalar starts; an explicit binding keeps the original ``p`` Var.
    No automatic ``tl.access_ptr`` is constructed before lowering.
    """
    ordinary = _load_store_kernel()
    (load,) = _func_calls(ordinary, "tl.vmi.vload")
    assert isinstance(load.args[0], tvm.tirx.BufferLoad), load.args[0]
    assert int(load.args[0].indices[0]) == 64
    assert _const(load.args[1]) == 0
    (store,) = _func_calls(ordinary, "tl.vmi.vstore")
    assert isinstance(store.args[1], tvm.tirx.BufferLoad), store.args[1]
    assert int(store.args[1].indices[0]) == 192
    assert not _func_calls(ordinary, "tl.access_ptr")

    explicit = _explicit_pointer_kernel()
    (bound_load,) = _func_calls(explicit, "tl.vmi.vload")
    assert isinstance(bound_load.args[0], tvm.tirx.Var), bound_load.args[0]
    assert _const(bound_load.args[1]) == 16
    (pointer,) = _func_calls(explicit, "tl.access_ptr")
    assert isinstance(pointer.args[0], tvm.tirx.BufferLoad)
    assert _const(pointer.args[1]) == 128
    assert _const(pointer.args[2]) == 1


def _runtime_branch_kernel():
    """Tight (2,128) view whose row-0 coverage follows from a real branch."""

    @T.prim_func
    def main(A: T.Tensor((256,), F32), k: T.int32, C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            view = tvm.tirx.decl_buffer((2, 128), F32, data=work.data, scope="shared.dyn")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                if k >= 0 and k <= 1:
                    value = T.vmi.vload(view[0, k * 64], size=64)
                    T.vmi.vstore(value, view[1, 0], mask)
            T.copy(work, C)

    return main


def test_runtime_branch_compiles_through_pipeline():
    """The descriptor pass proves coverage from the enclosing branch facts."""
    artifact = _lower_pto(_runtime_branch_kernel())
    func, call = _single(artifact, "tl.vmi.vload")
    ptr = _resolve_binding(call.args[0], func)
    assert isinstance(ptr, tvm.tirx.Call) and ptr.op.name.endswith("tvm_access_ptr"), ptr
    assert _const(ptr.args[3]) == 64
    assert _const(ptr.args[4]) == 1
    assert _const(call.args[1]) == 0


def test_load_descriptor_originates_at_actual_load():
    """vload(work[64], size=64) declares [64,128) and instruction offset 0."""
    artifact = _lower_pto(_load_store_kernel())
    assert _descriptor(artifact, "tl.vmi.vload") == (64, 64, 1, 0)

    source = artifact.kernel_source
    vload_line = next(ln.strip() for ln in source.splitlines() if "vmi.vload" in ln)
    assert "addptr" in vload_line and ", 64)" in vload_line, vload_line
    assert ", 0, size=64)" in vload_line, vload_line


def test_store_descriptor_originates_at_actual_destination():
    """vstore(..., work[192]) declares may-write [192,256) with offset 0."""
    artifact = _lower_pto(_load_store_kernel())
    assert _descriptor(artifact, "tl.vmi.vstore") == (192, 64, 2, 0)

    source = artifact.kernel_source
    vstore_line = next(ln.strip() for ln in source.splitlines() if "vmi.vstore" in ln)
    assert "addptr" in vstore_line and ", 192)" in vstore_line, vstore_line
    assert "), 0, mask)" in vstore_line, vstore_line


def test_nonzero_load_and_store_compile_through_backend():
    kernel = _compile_pto(_load_store_kernel())
    source = kernel.get_kernel_source()
    assert "pto.vmi.vload(" in source and "pto.vmi.vstore(" in source, source


# ---------------------------------------------------------------------------
# Buffer / BufferRegion / elem_offset
# ---------------------------------------------------------------------------
def _buffer_region_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                v = T.vmi.vload(work[0:64], size=64)
                T.vmi.vstore(v, work[128], mask)
            T.copy(work, C)

    return main


def _elem_offset_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            with T.SimdVF():
                view = tvm.tirx.decl_buffer((128,), F32, data=work.data, elem_offset=16, scope="shared.dyn")
                mask = T.vmi.create_mask(64, size=64)
                v = T.vmi.vload(view[32], size=64)
                T.vmi.vstore(v, view[32], mask)
            T.copy(work, C)

    return main


def test_buffer_region_source_declares_from_region_minimum():
    assert _vload_descriptor(_buffer_region_kernel()) == (0, 64, 1, 0)


def test_ramp_source_uses_the_public_buffer_region_base():
    """An automatic BufferLoad with a Ramp index consumes ``Ramp.base``.

    This is the public ``get_buffer_region_from_load``/``T.access_ptr``
    convention.  The Ramp lanes and stride describe the vector value, not the
    VMI physical envelope, so they never participate in the descriptor.
    """

    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            with T.SimdVF():
                v = T.vmi.vload(work[tvm.tirx.Ramp(32, 1, 64)], size=64)
                T.vmi.vstore(v, work[128])
            T.copy(work, C)

    artifact = _lower_pto(main)
    assert _descriptor(artifact, "tl.vmi.vload") == (32, 64, 1, 0)
    vload_line = next(line.strip() for line in artifact.kernel_source.splitlines() if "vmi.vload" in line)
    assert "addptr" in vload_line and ", 32)" in vload_line, vload_line


def test_elem_offset_counted_exactly_once():
    """A view with elem_offset=16 loaded at index 32 starts at element 48."""
    artifact = _lower_pto(_elem_offset_kernel())
    assert _descriptor(artifact, "tl.vmi.vload") == (48, 64, 1, 0)
    assert _descriptor(artifact, "tl.vmi.vstore") == (48, 64, 2, 0)


# ---------------------------------------------------------------------------
# Physical-vector rounding of ordinary loads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dtype", "size", "expected"),
    [
        ("int8", 300, 512),  # b8 VL = 256
        (F16, 200, 256),  # b16 VL = 128
        (F32, 80, 128),  # b32 VL = 64 (structural: PTODSL cannot build 80)
    ],
    ids=["b8", "b16", "b32-size80"],
)
def test_ordinary_load_covers_full_physical_vectors(dtype, size, expected):
    @T.prim_func
    def main(A: T.Tensor((1024,), dtype), C: T.Tensor((1024,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((1024,), dtype)
            T.copy(A, work)
            with T.SimdVF():
                v = T.vmi.vload(work[64], size=size)
                T.vmi.vstore(v, work[512])
            T.copy(work, C)

    assert _vload_descriptor(main) == (64, expected, 1, 0)
    assert _vstore_descriptor(main) == (512, size, 2, 0)


# ---------------------------------------------------------------------------
# Logical size 80 / masks never shrink the footprint
# ---------------------------------------------------------------------------
def test_size80_load_declares_two_full_physical_reads():
    """b32 size 80 loads two 64-lane physical reads, so the extent is 128."""

    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            with T.SimdVF():
                v = T.vmi.vload(work[64], size=80)
                T.vmi.vstore(v, work[128])
            T.copy(work, C)

    assert _vload_descriptor(main) == (64, 128, 1, 0)
    assert _vstore_descriptor(main) == (128, 80, 2, 0)


def test_size80_backend_lowering_covers_second_physical_chunk():
    """Direct PTOAS compile of the size-80 PTO form the descriptor describes.

    The PTODSL VMI API cannot construct an 80-lane VMI vreg
    (``_check_vmi_lane_count``), so this verifies the backend lowering itself:
    two full vlds at offsets 64 and 128.
    """
    pto_text = """module {
  func.func @group1_n80_s80() attributes {pto.kernel} {
    %start = arith.constant 64 : index
    %stride = arith.constant 80 : index
    %base = arith.constant 0 : i64
    %dest = arith.constant 4096 : i64
    %src = pto.castptr %base : i64 -> !pto.ptr<f32, ub>
    %dst = pto.castptr %dest : i64 -> !pto.ptr<f32, ub>
    pto.vecscope {
      %v = pto.vmi.vload %src[%start], %stride {group = 1 : i64} : !pto.ptr<f32, ub> -> !pto.vmi.vreg<80xf32>
      pto.vmi.vstore %v, %dst[%start], %stride {group = 1 : i64} : !pto.vmi.vreg<80xf32>, !pto.ptr<f32, ub>
    }
    return
  }
}
"""
    vpto = _ptoas_vpto(pto_text)
    assert "pto.vlds %0[%c64]" in vpto, vpto
    assert "pto.vlds %0[%c128]" in vpto, vpto


def test_store_mask_activity_does_not_shrink_may_write():
    """A partial or all-false mask keeps the full 64-lane may-write span."""

    def kernel(mask_active):
        @T.prim_func
        def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    mask = T.vmi.create_mask(mask_active, size=64)
                    v = T.vmi.vload(work[0], size=64)
                    T.vmi.vstore(v, work[128], mask)
                T.copy(work, C)

        return main

    for active in (0, 10, 64):
        assert _vstore_descriptor(kernel(active)) == (128, 64, 2, 0), active


# ---------------------------------------------------------------------------
# Group envelopes, negative and dynamic stride
# ---------------------------------------------------------------------------
def test_group_ordinary_envelope_covers_all_groups():
    """A=64, N=128, G=2, S=128 => [64,256)."""

    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                v = T.vmi.vload(work[64], size=128, group=2, stride=128)
                T.vmi.vstore(v, work[256], mask)
            T.copy(work, C)

    assert _vload_descriptor(main) == (64, 192, 1, 0)


def test_group_slot_uses_one_element_per_group():
    @T.prim_func
    def slot(A: T.Tensor((768,), F32), C: T.Tensor((768,), F32)):
        with T.Kernel(1):
            # G=8/S=64: source touches index 512; the destination starting at
            # 256 touches 256 + 7*64 = 704.  Both need legal, initialized
            # storage, so allocate/initialize 768 elements.
            work = T.alloc_shared((768,), F32)
            T.copy(A, work)
            with T.SimdVF():
                v = T.vmi.vload(work[64], size=8, group=8, stride=64)
                T.vmi.vstore(v, work[256], group=8, stride=64)
            T.copy(work, C)

    assert _vload_descriptor(slot) == (64, 449, 1, 0)
    assert _vstore_descriptor(slot) == (256, 449, 2, 0)


def test_group_static_negative_stride_relocates_lowest_address():
    """A=192, N=128, G=2, S=-128 => envelope [64,256) and instruction offset 128."""

    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            with T.SimdVF():
                v = T.vmi.vload(work[192], size=128, group=2, stride=-128)
                T.vmi.vstore(v, work[192], group=2, stride=-128)
            T.copy(work, C)

    artifact = _lower_pto(main)
    assert _descriptor(artifact, "tl.vmi.vload") == (64, 192, 1, 128)
    assert _descriptor(artifact, "tl.vmi.vstore") == (64, 192, 2, 128)


@pytest.mark.parametrize("is_load", [True, False], ids=["load", "store"])
def test_group_dynamic_stride_has_no_proven_bounds(is_load):
    """A runtime group stride has no finite envelope, even on a tight buffer."""

    def build():
        @T.prim_func
        def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32), stride: T.int32):
            with T.Kernel(1):
                work = T.alloc_shared((512,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    if is_load:
                        mask = T.vmi.create_mask(128, size=128)
                        v = T.vmi.vload(work[192], size=128, group=2, stride=stride)
                        T.vmi.vstore(v, work[0], mask)
                    else:
                        v = T.vmi.vload(work[0], size=128)
                        T.vmi.vstore(v, work[192], group=2, stride=stride)
                T.copy(work, C)

        return main

    op = "T.vmi.vload" if is_load else "T.vmi.vstore"
    _assert_descriptor_error(build, op, "group stride has no proven bounds", descriptor=False)


def test_group_negative_stride_below_view_origin_is_rejected():
    """The envelope is finite, but no view descriptor expresses its low start."""

    def build():
        @T.prim_func
        def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((512,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    view = tvm.tirx.decl_buffer((2, 64), F32, data=work.data, strides=(128, 1), elem_offset=128, scope="shared.dyn")
                    v = T.vmi.vload(view[0, 0], size=128, group=2, stride=-128)
                    T.vmi.vstore(v, work[0])
                T.copy(work, C)

        return main

    _assert_descriptor_error(build, "T.vmi.vload", "finite access envelope starts below the buffer view origin")


def test_group_negative_stride_shifted_index_below_zero_is_rejected():
    """Relocating a 1-D descriptor must not produce a negative buffer index."""

    def build():
        @T.prim_func
        def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((512,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    v = T.vmi.vload(work[64], size=128, group=2, stride=-128)
                    T.vmi.vstore(v, work[0])
                T.copy(work, C)

        return main

    _assert_descriptor_error(build, "T.vmi.vload", "finite access envelope starts below the buffer view origin")


def test_group_negative_stride_dense_root_form_keeps_finite_descriptor():
    """The equivalent root[128] form stays supported with address-preserving offset."""

    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            with T.SimdVF():
                v = T.vmi.vload(work[128], size=128, group=2, stride=-128)
                T.vmi.vstore(v, work[128], group=2, stride=-128)
            T.copy(work, C)

    artifact = _lower_pto(main)
    assert _descriptor(artifact, "tl.vmi.vload") == (0, 192, 1, 128)
    assert _descriptor(artifact, "tl.vmi.vstore") == (0, 192, 2, 128)


def test_group_negative_stride_nd_uses_padded_origin_plan():
    """A dense 2-D buffer meets the padded-origin layout condition."""

    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((4, 64), F32)
            T.copy(A, work)
            with T.SimdVF():
                v = T.vmi.vload(work[3, 0], size=128, group=2, stride=-64)
                T.vmi.vstore(v, work[3, 0], group=2, stride=-64)
            T.copy(work, C)

    artifact = _lower_pto(main)
    assert _descriptor(artifact, "tl.vmi.vload") == (0, 256, 1, 192)


@pytest.mark.parametrize("is_load", [True, False], ids=["load", "store"])
def test_group_negative_stride_negative_coordinate_uses_legalized_address(is_load):
    """work[2,-32] is normalized to work[2,32] (linear 160) before the pass.

    The descriptor relocates to the tight buffer origin with instruction offset
    160, so the group-1 address (96) still matches the normalized coordinate.
    """

    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((4, 64), F32)
            T.copy(A, work)
            with T.SimdVF():
                if is_load:
                    mask = T.vmi.create_mask(128, size=128)
                    v = T.vmi.vload(work[2, -32], size=128, group=2, stride=-64)
                    T.vmi.vstore(v, work[0, 0], mask)
                else:
                    v = T.vmi.vload(work[0, 0], size=128)
                    T.vmi.vstore(v, work[2, -32], group=2, stride=-64)
            T.copy(work, C)

    artifact = _lower_pto(main)
    if is_load:
        assert _descriptor(artifact, "tl.vmi.vload") == (0, 256, 1, 160)
    else:
        assert _descriptor(artifact, "tl.vmi.vstore") == (0, 256, 2, 160)


# ---------------------------------------------------------------------------
# Padded static negative group stride
# ---------------------------------------------------------------------------
def _padded_view_kernel(row=1, col=0):
    view_data = {}

    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            view = T.StridedTensor((2, 128), (256, 1), F32, data=work.data, scope="shared.dyn")
            view_data["var"] = view.data
            with T.SimdVF():
                v = T.vmi.vload(view[row, col], size=128, group=2, stride=-256)
                T.vmi.vstore(v, view[row, col], group=2, stride=-256)
            T.copy(work, C)

    return main, view_data["var"]


def test_padded_group_negative_stride_uses_buffer_origin():
    """view[1,0] with group stride -256 reads [0,320): the buffer origin covers it."""
    main, view_data = _padded_view_kernel(1, 0)
    artifact = _lower_pto(main)
    load_func, load_call = _single(artifact, "tl.vmi.vload")
    store_func, store_call = _single(artifact, "tl.vmi.vstore")
    load_ptr = _resolve_binding(load_call.args[0], load_func)
    store_ptr = _resolve_binding(store_call.args[1], store_func)

    assert _descriptor(artifact, "tl.vmi.vload") == (0, 320, 1, 256)
    assert _descriptor(artifact, "tl.vmi.vstore") == (0, 320, 2, 256)
    # The descriptor keeps the original view storage and instruction offsets
    # carry the source offset once: the first accessed element stays at 256.
    assert load_ptr.args[1].same_as(view_data)
    assert store_ptr.args[1].same_as(view_data)
    assert _const(load_ptr.args[2]) + _const(load_call.args[1]) == 256
    assert _const(store_ptr.args[2]) + _const(store_call.args[2]) == 256


def test_padded_group_negative_stride_counts_elem_offset_once():
    """The descriptor origin carries elem_offset; the relative offset excludes it."""
    view_data = {}

    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            view = tvm.tirx.decl_buffer((2, 128), F32, data=work.data, strides=(256, 1), elem_offset=16, scope="shared.dyn")
            view_data["var"] = view.data
            with T.SimdVF():
                v = T.vmi.vload(view[1, 0], size=128, group=2, stride=-256)
                T.vmi.vstore(v, view[1, 0], group=2, stride=-256)
            T.copy(work, C)

    artifact = _lower_pto(main)
    func, call = _single(artifact, "tl.vmi.vload")
    ptr = _resolve_binding(call.args[0], func)
    assert _descriptor(artifact, "tl.vmi.vload") == (16, 320, 1, 256)
    assert _descriptor(artifact, "tl.vmi.vstore") == (16, 320, 2, 256)
    assert ptr.args[1].same_as(view_data["var"])
    # elem_offset (16) + relative (256) still starts at element 272.
    assert _const(ptr.args[2]) + _const(call.args[1]) == 272


def _sync_events(ir):
    """Ordered copy/sync/load events in evaluation order (one IR pass)."""
    from testing.ascend._ir import calls as ir_calls
    from testing.ascend._ir import statements as ir_statements

    events = []
    for stmt, _ in ir_statements(ir):
        if not isinstance(stmt, (tvm.tirx.Evaluate, tvm.tirx.Bind)):
            continue
        for call in ir_calls(stmt, "tl.tileop.ascend_copy"):
            destination = call.args[1].args[0]
            events.append(("copy", tuple(int(index) for index in destination.indices)))
        for call in ir_calls(stmt, "tl.ascend_set_flag"):
            events.append(("set", str(getattr(call.args[0], "value", call.args[0]))))
        for call in ir_calls(stmt, "tl.ascend_wait_flag"):
            events.append(("wait", str(getattr(call.args[0], "value", call.args[0]))))
        if ir_calls(stmt, "tl.vmi.vload"):
            events.append(("vload", None))
    return events


def test_padded_low_group_dma_synchronizes_before_vmi_load():
    """Only the low group is initialized, so no other DMA can mask a missing pair."""

    @T.prim_func
    def main(A: T.Tensor((64,), F32), C: T.Tensor((128,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            out = T.alloc_shared((128,), F32)
            view = T.StridedTensor((2, 128), (256, 1), F32, data=work.data, scope="shared.dyn")
            T.copy(A, view[0, 0:64])
            with T.SimdVF():
                mask = T.vmi.create_mask(128, size=128)
                value = T.vmi.vload(view[1, 0], size=128, group=2, stride=-256)
                T.vmi.vstore(value, out[0], mask)
            T.copy(out, C)

    _, observer = _lower_pto_traced(main, capture_finish="tl.InsertSync")
    events = _sync_events(observer.finished_module)
    dma = [index for index, event in enumerate(events) if event == ("copy", (0, 0))]
    assert len(dma) == 1, events
    loads = [index for index, event in enumerate(events) if event[0] == "vload"]
    assert len(loads) == 1, events
    sets = [index for index, event in enumerate(events) if event == ("set", "MTE2_V")]
    waits = [index for index, event in enumerate(events) if event == ("wait", "MTE2_V")]
    assert sets and waits, f"missing MTE2->V pair for the low-group DMA: {events}"
    set_index = next(index for index in sets if index > dma[0])
    wait_index = next(index for index in waits if index > set_index)
    assert dma[0] < set_index < wait_index < loads[0], events


# ---------------------------------------------------------------------------
# Pipeline placement and the codegen guard for the descriptor pass
# ---------------------------------------------------------------------------
def test_descriptor_pipeline_order():
    """The descriptor runs once after InjectAssumes and before VerifyBufferInit.

    AutoSchedule only controls the later scheduling passes; it never gates or
    duplicates descriptor legalization.
    """
    expected = [
        "tl.LegalizeNegativeIndex",
        "tl.InjectAssumes",
        "tl.NormalizePTOAllocMarkers",
        "tl.LegalizePTOVMIMemoryAccess",
        "tl.VerifyBufferInit",
    ]
    for auto_schedule in (True, False):
        artifact, observer = _lower_pto_traced(
            _load_store_kernel(),
            auto_schedule=auto_schedule,
            capture_start="tl.VerifyBufferInit" if auto_schedule else None,
        )
        names = observer.names
        assert names.count("tl.LegalizePTOVMIMemoryAccess") == 1, (auto_schedule, names)
        positions = [names.index(name) for name in expected]
        assert positions == sorted(positions), (auto_schedule, positions)
        if auto_schedule:
            assert "tl.MaterializeScheduleUnits" in names, names
            assert names.index("tl.VerifyBufferInit") < names.index("tl.MaterializeScheduleUnits")
        else:
            assert "tl.MaterializeScheduleUnits" not in names, names
            assert "tl.InsertSync" not in names, names
        assert "pto.vmi.vload(" in artifact.kernel_source
        if auto_schedule:
            module = observer.started_module
            assert module is not None
            func = next(iter(module.functions.values()))
            loads = []

            def collect_vload(node, loads=loads):
                if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.vmi.vload":
                    loads.append(node)

            tvm.tirx.stmt_functor.post_order_visit(
                func.body,
                collect_vload,
            )
            assert len(loads) == 1, loads
            address = loads[0].args[0]
            assert isinstance(address, tvm.tirx.Call) and address.op.name == "tl.access_ptr", address
            assert _const(address.args[1]) == 64
            assert _const(address.args[2]) == 1


def test_descriptor_pass_is_pto_only():
    """Plain AscendC lowering must not run the PTO descriptor or marker passes."""

    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            T.copy(work, C)

    artifact, observer = _lower_pto_traced(main, target_name="ascend")
    assert "tl.LegalizePTOVMIMemoryAccess" not in observer.names
    assert "tl.NormalizePTOAllocMarkers" not in observer.names
    assert "tl.VerifyBufferInit" in observer.names
    assert artifact.kernel_source


def test_vmi_store_does_not_fake_a_read(capfd):
    """A write-only VMI store is not reported as a read by VerifyBufferInit."""
    artifact = _lower_pto(_store_only_kernel())
    stderr = capfd.readouterr().err
    assert "Buffer read before initialization" not in stderr, stderr
    # The merged UB layout places the source first; only the rw mask matters here.
    assert _descriptor(artifact, "tl.vmi.vstore") == (_merged_target_offset(F32, F32, count=64), 64, 2, 0)


def test_one_lane_vmi_value_remains_vmi_data():
    """A size=1 VMI value keeps its scalar TIR dtype but is still VMI data.

    Lane count is not a provenance check: the pass must admit the value the
    frontend produced instead of rejecting it as a non-vector scalar.
    """
    artifact = _lower_pto(_lane_one_store_kernel())
    load = _descriptor(artifact, "tl.vmi.vload")
    store = _descriptor(artifact, "tl.vmi.vstore")
    assert load[1:] == (1, 1, 0), load
    assert store[1:] == (1, 2, 0), store
    func, call = _single(artifact, "tl.vmi.vstore")
    value = _resolve_binding(call.args[0], func)
    assert isinstance(value, tvm.tirx.Call) and value.op.name == "tl.vmi.vload", value


def test_formal_vmi_read_does_not_hide_a_later_scalar_read(capfd):
    """A read-only VMI descriptor is not a writer for VerifyBufferInit.

    The raw-address form made ``ForEachOpaqueWrite`` treat the direct
    ``BufferLoad`` argument as a possible writer, which hid the existing
    uninitialized-read warning for the later scalar read.  With the descriptor
    in place, the later scalar read of ``work`` is diagnosed.  This does not
    claim detection of VMI-only uninitialized loads: the unchanged verifier
    skips read descriptors on opaque calls.
    """
    _lower_pto(_uninitialized_vmi_read_kernel())
    stderr = capfd.readouterr().err
    assert "Buffer read before initialization" in stderr, stderr
    assert "`work`" in stderr, stderr


def _store_only_kernel():
    @T.prim_func
    def main(A: T.Tensor((64,), F32), C: T.Tensor((64,), F32)):
        with T.Kernel(1):
            source = T.alloc_shared((64,), F32)
            dest = T.alloc_shared((64,), F32)
            T.copy(A, source)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                v = T.vmi.vload(source[0], size=64)
                T.vmi.vstore(v, dest[0], mask)
            T.copy(source, C)

    return main


def _lane_one_store_kernel():
    @T.prim_func
    def main(A: T.Tensor((64,), F32), C: T.Tensor((64,), F32)):
        with T.Kernel(1):
            source = T.alloc_shared((64,), F32)
            dest = T.alloc_shared((64,), F32)
            T.copy(A, source)
            with T.SimdVF():
                value = T.vmi.vload(source[0], size=1, group=1, stride=1)
                T.vmi.vstore(value, dest[0], group=1, stride=1)
            T.copy(dest, C)

    return main


def _uninitialized_vmi_read_kernel():
    @T.prim_func
    def main(C: T.Tensor((64,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((64,), F32)  # never written
            out = T.alloc_shared((64,), F32)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                v = T.vmi.vload(work[0], size=64)
                scalar = work[0]  # ordinary scalar read of the same buffer
                T.vmi.vstore(T.vmi.vadd(v, T.vmi.vbrc(scalar, size=64), mask), out[0], mask)
            T.copy(out, C)

    return main


_CODEGEN_SOURCE = "vmi_unlowered_address_case.py"


def _raw_vload_address_func():
    work = tvm.tirx.decl_buffer((256,), F32, name="work", scope="shared.dyn")
    raw = tvm.tirx.BufferLoad(work, [tvm.tirx.IntImm("int32", 64)])
    span = tvm.ir.Span(tvm.ir.SourceName(_CODEGEN_SOURCE), 12, 12, 7, 24)
    call = tvm.tirx.call_intrin(
        "float32x64",
        tvm.ir.Op.get("tl.vmi.vload"),
        raw,
        tvm.tirx.IntImm("int32", 0),
        annotations={"size": 64},
        span=span,
    )
    return tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(call)).with_attr("global_symbol", "main")


def _raw_vstore_address_func():
    work = tvm.tirx.decl_buffer((256,), F32, name="work", scope="shared.dyn")
    out = tvm.tirx.decl_buffer((256,), F32, name="out", scope="shared.dyn")
    value = tvm.tirx.call_intrin(
        "float32x64",
        tvm.ir.Op.get("tl.vmi.vload"),
        tvm.tirx.tvm_access_ptr(
            tvm.tirx.type_annotation(F32),
            work.data,
            tvm.tirx.IntImm("int32", 0),
            tvm.tirx.IntImm("int32", 64),
            tvm.tirx.IntImm("int32", 1),
        ),
        tvm.tirx.IntImm("int32", 0),
        annotations={"size": 64},
    )
    raw = tvm.tirx.BufferLoad(out, [tvm.tirx.IntImm("int32", 128)])
    span = tvm.ir.Span(tvm.ir.SourceName(_CODEGEN_SOURCE), 20, 20, 9, 30)
    store = tvm.tirx.call_intrin(
        "void",
        tvm.ir.Op.get("tl.vmi.vstore"),
        value,
        raw,
        tvm.tirx.IntImm("int32", 0),
        annotations={"size": 64},
        span=span,
    )
    body = tvm.tirx.SeqStmt([tvm.tirx.AllocBuffer(work), tvm.tirx.AllocBuffer(out), tvm.tirx.Evaluate(store)])
    return tvm.tirx.PrimFunc([], body).with_attr("global_symbol", "main")


def _inner_index_address_func():
    data = tvm.tirx.decl_buffer((256,), F32, name="data", scope="shared.dyn")
    index = tvm.tirx.decl_buffer((2,), "int32", name="index", scope="shared.dyn")
    offset = tvm.tirx.BufferLoad(index, [tvm.tirx.IntImm("int32", 0)])
    pointer = tvm.tirx.tvm_access_ptr(
        tvm.tirx.type_annotation(F32),
        data.data,
        offset,
        tvm.tirx.IntImm("int32", 64),
        tvm.tirx.IntImm("int32", 1),
    )
    value = tvm.tirx.call_intrin(
        "float32x64",
        tvm.ir.Op.get("tl.vmi.vload"),
        pointer,
        tvm.tirx.IntImm("int32", 0),
        annotations={"size": 64},
    )
    body = tvm.tirx.SeqStmt([tvm.tirx.AllocBuffer(data), tvm.tirx.AllocBuffer(index), tvm.tirx.Evaluate(value)])
    return tvm.tirx.PrimFunc([], body).with_attr("global_symbol", "main")


def test_codegen_rejects_unlowered_outermost_vmi_address():
    """Only the actual address operand is diagnosed, with op, pass and span."""
    with pytest.raises(tvm.error.TVMError) as excinfo:
        _pto_source(_raw_vload_address_func())
    message = str(excinfo.value)
    assert "LegalizePTOVMIMemoryAccess" in message, message
    assert "tl.vmi.vload address operand 0" in message, message
    assert f"{_CODEGEN_SOURCE}:12:7" in message, message

    with pytest.raises(tvm.error.TVMError) as excinfo:
        _pto_source(_raw_vstore_address_func())
    message = str(excinfo.value)
    assert "LegalizePTOVMIMemoryAccess" in message, message
    assert "tl.vmi.vstore address operand 1" in message, message
    assert f"{_CODEGEN_SOURCE}:20:9" in message, message


def test_codegen_allows_a_nested_index_read_in_a_formal_address():
    """The scalar read inside an explicit access_ptr is not the address operand."""
    source = _pto_source(_inner_index_address_func())
    assert "pto.vmi.vload(" in source, source
    assert "pto.load(index, 0)" in source, source


# ---------------------------------------------------------------------------
# block_stride
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dtype", "size", "block_stride", "expected_extent"),
    [
        (F32, 64, 1, 64),  # one full b32 vector, contiguous blocks
        (F32, 64, 2, 120),  # [64, 184)
        (F32, 128, 2, 248),  # two full vectors: [64, 312)
        (F16, 128, 2, 240),  # b16: 8 blocks of 16 elements
    ],
    ids=["b32-single-vl-s1", "b32-single-vl-s2", "b32-double-vl-s2", "b16-single-vl-s2"],
)
def test_block_stride_envelope_uses_32b_blocks(dtype, size, block_stride, expected_extent):
    # The destination at 320 spans up to 320 + expected_extent = 568 (b32
    # double-VL) / 560 (b16); allocate and initialize the real footprint.
    @T.prim_func
    def main(A: T.Tensor((768,), dtype), C: T.Tensor((768,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((768,), dtype)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(size, size=size)
                v = T.vmi.vload(work[64], size=size, block_stride=block_stride)
                T.vmi.vstore(v, work[320], mask, block_stride=block_stride)
            T.copy(work, C)

    artifact = _lower_pto(main)
    assert _descriptor(artifact, "tl.vmi.vload") == (64, expected_extent, 1, 0)
    assert _descriptor(artifact, "tl.vmi.vstore") == (320, expected_extent, 2, 0)


@pytest.mark.parametrize(
    ("dynamic", "reason"),
    [
        (True, "block stride has no proven bounds"),
        (False, "negative block-stride control words have no proven physical bounds"),
    ],
    ids=["dynamic", "negative"],
)
def test_block_stride_without_proven_bounds_is_rejected(dynamic, reason):
    def build():
        @T.prim_func
        def main(A: T.Tensor((768,), F32), C: T.Tensor((768,), F32), block_stride: T.int32):
            with T.Kernel(1):
                work = T.alloc_shared((768,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    mask = T.vmi.create_mask(64, size=64)
                    if dynamic:
                        v = T.vmi.vload(work[64], size=64, block_stride=block_stride)
                    else:
                        v = T.vmi.vload(work[64], size=64, block_stride=-1)
                    T.vmi.vstore(v, work[320], mask)
                T.copy(work, C)

        return main

    _assert_descriptor_error(build, "T.vmi.vload", reason, descriptor=False)


# ---------------------------------------------------------------------------
# Genuine FP4
# ---------------------------------------------------------------------------
def _fp4_kernel(index=64, size=128):
    @T.prim_func
    def main(A: T.Tensor((256,), FP4), C: T.Tensor((256,), FP4)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), FP4)
            out = T.alloc_shared((256,), FP4)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(size // 2, size=size // 2)
                v = T.vmi.vload(work[index], size=size)
                T.vmi.vstore(v, out[index], mask)
            T.copy(out, C)

    return main


def test_fp4_descriptor_stays_logical_then_packed_once():
    with _capture_rewrite_input() as captured:
        artifact = _lower_pto(_fp4_kernel())
    before = captured["module"]

    # Logical FP4 element units before the rewrite.
    assert _descriptor_in_module(before, "tl.vmi.vload") == (64, 128, 1, 0)
    assert _descriptor_in_module(before, "tl.vmi.vstore") == (64, 128, 2, 0)

    # Exactly one physical packing: the single rewrite pass halves both.
    rewritten = ascend_transform.RewriteFp4ToFp4x2()(before)
    assert _descriptor_in_module(rewritten, "tl.vmi.vload") == (32, 64, 1, 0)
    assert _descriptor_in_module(rewritten, "tl.vmi.vstore") == (32, 64, 2, 0)

    source = artifact.kernel_source
    load_line = next(ln.strip() for ln in source.splitlines() if "vmi.vload" in ln)
    assert "addptr" in load_line and "), 32)" in load_line, load_line
    assert load_line.rstrip().endswith("size=(128 // 2))"), load_line


def test_fp4_odd_start_is_still_rejected():
    """The logical descriptor keeps the odd start; the later FP4 pack rejects it."""
    with pytest.raises(tvm.error.TVMError, match="fp4 access_ptr offset must be even"):
        _lower_pto(_fp4_kernel(index=65))


# ---------------------------------------------------------------------------
# Packed (PK) stores keep logical target-element extents
# ---------------------------------------------------------------------------
def _bits(dtype) -> int:
    return int(tvm.DataType(dtype).bits)


def _merged_target_offset(src, dst, count=512) -> int:
    """Offset of a target buffer merged after ``count`` source elements.

    MergeUBAllocations packs the source span first in these two-buffer
    kernels; descriptors are in destination-element units.  Deriving the
    expectation keeps the assertion on the absolute may-write start while
    making the layout assumption explicit.
    """
    source_bytes = count * _bits(src) // 8
    return source_bytes // (_bits(dst) // 8)


@pytest.mark.parametrize(
    ("src", "dst", "vcvt_kwargs"),
    [
        (F32, F16, {"rounding": "R", "saturate": "SAT"}),
        (F32, F8_E4M3, {"rounding": "R", "saturate": "SAT"}),
        ("int16", "int8", {"saturate": "NOSAT"}),
    ],
    ids=["pk-b32", "pk4-b32", "pk-b16"],
)
def test_packed_store_declares_logical_target_elements(src, dst, vcvt_kwargs):
    size = 128

    @T.prim_func
    def main(A: T.Tensor((512,), src), C: T.Tensor((512,), dst)):
        with T.Kernel(1):
            source = T.alloc_shared((512,), src)
            target = T.alloc_shared((512,), dst)
            T.copy(A, source)
            with T.SimdVF():
                mask = T.vmi.create_mask(size, size=size)
                v = T.vmi.vload(source[64], size=size)
                narrowed = T.vmi.vcvt(v, to_dtype=dst, **vcvt_kwargs)
                T.vmi.vstore(narrowed, target[64], mask)
            T.copy(target, C)

    artifact = _lower_pto(main)
    target_offset = _merged_target_offset(src, dst) + 64
    # The may-write span stays the logical target element count (128), not a
    # physical vreg capacity or a PK-derived fraction.
    assert _descriptor(artifact, "tl.vmi.vstore") == (target_offset, size, 2, 0)
    src_extent = 128 if src == F32 else size
    assert _descriptor(artifact, "tl.vmi.vload") == (64, src_extent, 1, 0)

    kernel = _compile_pto(main)
    assert "pto.vmi.vcvt(" in kernel.get_kernel_source()


# ---------------------------------------------------------------------------
# Explicit access_ptr + extra offset
# ---------------------------------------------------------------------------
def _explicit_pointer_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                p = T.access_ptr(work[32], "r", extent=128)
                v = T.vmi.vload(p, 16, size=64)
                T.vmi.vstore(v, work[128], mask)
            T.copy(work, C)

    return main


def test_explicit_pointer_keeps_declaration_and_extra_offset():
    artifact = _lower_pto(_explicit_pointer_kernel())
    func, call = _single(artifact, "tl.vmi.vload")
    ptr = _resolve_binding(call.args[0], func)
    # The caller-declared descriptor is not overwritten (offset 32, extent 128).
    assert _const(ptr.args[2]) == 32
    assert _const(ptr.args[3]) == 128
    # The extra offset is preserved, not zeroed or folded twice.
    assert _const(call.args[1]) == 16

    source = artifact.kernel_source
    lines = [ln.strip() for ln in source.splitlines()]
    definition = next(ln for ln in lines if ln.startswith("p = "))
    assert "pto.addptr(" in definition and ", 32)" in definition, definition
    # The VMI instruction offset stays 16: the base is counted once.
    assert "vload(p, 16, size=64)" in source, source


def test_explicit_pointer_requires_scalar_integer_instruction_offset():
    """The explicit-pointer instruction offset stays a frontend type check.

    Range/coverage proofs moved to the descriptor pass, but the offset selects
    a position inside the caller-declared descriptor and must remain a scalar
    integer at the API boundary.
    """

    def build():
        @T.prim_func
        def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    p = T.access_ptr(work[0], "r", extent=64)
                    value = T.vmi.vload(p, 0.5, size=64)
                    T.vmi.vstore(value, work[64])
                T.copy(work, C)

        return main

    with pytest.raises(TypeError, match="scalar integer instruction offset"):
        build()


def test_explicit_pointer_scope_validation_is_preserved():
    """Recovering the element type from an immediate access_ptr does not
    bypass the UB scope check for indirect-address operations."""

    def build():
        @T.prim_func
        def main(A: T.Tensor((256,), F32), I: T.Tensor((64,), "uint32"), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                index = T.alloc_shared((64,), "uint32")
                T.copy(A, work)
                T.copy(I, index)
                with T.SimdVF():
                    mask = T.vmi.create_mask(64, size=64)
                    offsets = T.vmi.vload(index[0], size=64)
                    p = T.access_ptr(A[0], "r", extent=64)
                    v = T.vmi.vgather(p, offsets, mask)
                    T.vmi.vstore(v, work[192], mask)
                T.copy(work, C)

        return main

    with pytest.raises(TypeError, match="requires a UB pointer"):
        build()


# ---------------------------------------------------------------------------
# Ordinary logical coverage: local descriptors and explicit pointers
# ---------------------------------------------------------------------------
def _ordinary_access_kernel(shape, starts, size, *, is_load, strides=None, explicit=False, extent=None, mask_lanes=None):
    """Build one ordinary UB view access over a tight or transposed view.

    The opposite side of the access always uses the tight root buffer, so only
    the tested address can make a case fail.  ``mask_lanes`` adds a small store
    mask; masks never shrink the may-write envelope.
    """
    if explicit and extent is None:
        extent = size

    @T.prim_func
    def main(A: T.Tensor((1024,), F32), C: T.Tensor((1024,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((1024,), F32)
            T.copy(A, work)
            if strides is None:
                view = tvm.tirx.decl_buffer(shape, F32, data=work.data, scope="shared.dyn")
            else:
                view = T.StridedTensor(shape, strides, F32, data=work.data, scope="shared.dyn")
            with T.SimdVF():
                if mask_lanes is None:
                    mask = None
                else:
                    mask = T.vmi.create_mask(mask_lanes, size=size)
                if is_load:
                    if explicit:
                        value = T.vmi.vload(T.access_ptr(view[starts], "r", extent=extent), 0, size=size)
                    else:
                        value = T.vmi.vload(view[starts], size=size)
                    T.vmi.vstore(value, work[0], mask)
                else:
                    value = T.vmi.vload(work[0], size=size)
                    if explicit:
                        T.vmi.vstore(value, T.access_ptr(view[starts], "w", extent=extent), mask)
                    else:
                        T.vmi.vstore(value, view[starts], mask)
            T.copy(work, C)

    return main


@pytest.mark.parametrize(
    ("shape", "starts", "size", "is_load", "expected"),
    [
        ((4, 128), (0, 32), 64, True, (32, 64, 1, 0)),
        ((4, 128), (3, 0), 128, True, (384, 128, 1, 0)),
        ((4, 128), (1, 0), 128, True, (128, 128, 1, 0)),
        ((2, 128), (0, 64), 128, True, None),
        ((4, 128), (0, 64), 128, True, None),
        ((4, 128), (0, 64), 128, False, None),
        ((4, 128), (0, 64), 80, True, None),
    ],
    ids=["in-row", "last-row", "first-row", "cross-row-tight-2x128", "cross-row", "store-cross-row", "rounded-cross-row"],
)
def test_ordinary_local_descriptor_requires_complete_row_major_coverage(shape, starts, size, is_load, expected):
    """A local descriptor keeps its own start, so the existing row-major carry
    must include the whole physical envelope; there is no whole-buffer fallback.
    The rounded case checks the physical envelope (128), not the logical size.
    """
    if expected is None:
        _assert_descriptor_error(
            lambda: _ordinary_access_kernel(shape, starts, size, is_load=is_load),
            "T.vmi.vload" if is_load else "T.vmi.vstore",
            "existing row-major carry",
        )
        return
    assert _vload_descriptor(_ordinary_access_kernel(shape, starts, size, is_load=is_load)) == expected


@pytest.mark.parametrize("explicit", [False, True], ids=["store-direct", "store-explicit"])
def test_cross_row_small_mask_store_is_rejected(explicit):
    """A one-lane store mask never shrinks the may-write envelope, so a
    cross-row store address is rejected on the direct and explicit paths."""
    _assert_descriptor_error(
        lambda: _ordinary_access_kernel((2, 128), (0, 64), 128, is_load=False, explicit=explicit, mask_lanes=1),
        "T.vmi.vstore",
        "existing row-major carry",
        explicit=explicit,
    )


@pytest.mark.parametrize("is_load", [True, False], ids=["load-direct", "store-direct"])
def test_transposed_view_access_is_rejected(is_load):
    """A local coordinate of a non-tight view has no local coverage proof; the
    store form is not a bypass."""
    _assert_descriptor_error(
        lambda: _ordinary_access_kernel((128, 2), (64, 0), 128, is_load=is_load, strides=(1, 128)),
        "T.vmi.vload" if is_load else "T.vmi.vstore",
        "not proven tight row-major",
    )


@pytest.mark.parametrize(
    ("shape", "starts", "extent", "size", "expected"),
    [
        ((4, 128), (0, 64), 192, 128, (64, 192, 1, 0)),
        ((2, 128), (0, 32), 64, 64, (32, 64, 1, 0)),
        ((4, 128), (0, 0), 256, 128, (0, 256, 1, 0)),
        ((2, 4, 128), (0, 0, 0), 512, 256, (0, 512, 1, 0)),
        ((2, 128), (0, 64), 128, 128, None),
        ((4, 128), (0, 96), 64, 64, None),
        ((2, 4, 128), (0, 1, 0), 512, 256, None),
    ],
    ids=[
        "two-row",
        "in-row",
        "origin-256",
        "3d-512",
        "cross-row-2x128",
        "carry-start-96",
        "3d-high-axis-carry-512",
    ],
)
def test_explicit_descriptor_must_cover_the_ordinary_access(shape, starts, extent, size, expected):
    """An explicit access_ptr is caller-declared, so it is never rewritten: its
    extent must contain the complete physical envelope and its declared region
    must still cover the access under the existing logical analysis.
    """

    def build():
        return _ordinary_access_kernel(shape, starts, size, is_load=True, explicit=True, extent=extent)

    if expected is None:
        _assert_descriptor_error(build, "T.vmi.vload", "does not cover the complete access", explicit=True)
        return
    assert _vload_descriptor(build()) == expected
    if shape == (4, 128) and extent == 192:
        # One representative case goes through PTODSL -> PTOAS -> Bisheng.
        kernel = _compile_pto(build())
        assert "pto.vmi.vload(" in kernel.get_kernel_source()


def test_explicit_pointer_whole_dense_permutation_is_accepted():
    """An exact whole dense permutation has no over-approximation or aliasing."""

    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            view = T.StridedTensor((128, 2), (1, 128), F32, data=work.data, scope="shared.dyn")
            with T.SimdVF():
                p = T.access_ptr(view[0, 0], "r", extent=256)
                v = T.vmi.vload(p, size=256)
                T.vmi.vstore(v, work[0])
            T.copy(work, C)

    assert _vload_descriptor(main) == (0, 256, 1, 0)


def test_explicit_pointer_partial_permutation_is_rejected():
    """A non-tight view needs an exact whole dense permutation, not a prefix."""

    def build():
        @T.prim_func
        def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                T.copy(A, work)
                view = T.StridedTensor((128, 2), (1, 128), F32, data=work.data, scope="shared.dyn")
                with T.SimdVF():
                    p = T.access_ptr(view[0, 0], "r", extent=128)
                    v = T.vmi.vload(p, size=128)
                    T.vmi.vstore(v, work[0])
                T.copy(work, C)

        return main

    _assert_descriptor_error(build, "T.vmi.vload", "does not cover the complete access", explicit=True)


def test_explicit_pointer_envelope_and_direction_diagnostics():
    def kernel(body):
        @T.prim_func
        def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    body(work)
                T.copy(work, C)

        return main

    def too_small(work):
        p = T.access_ptr(work[0], "r", extent=32)
        return T.vmi.vstore(T.vmi.vload(p, 0, size=64), work[128])

    _assert_descriptor_error(
        lambda: kernel(too_small),
        "T.vmi.vload",
        "instruction_offset + envelope_high <= descriptor_extent cannot be proven",
        explicit=True,
    )

    def negative_offset(work):
        p = T.access_ptr(work[64], "r", extent=128)
        return T.vmi.vstore(T.vmi.vload(p, -16, size=64), work[128])

    _assert_descriptor_error(
        lambda: kernel(negative_offset),
        "T.vmi.vload",
        "0 <= instruction_offset + envelope_low cannot be proven",
        explicit=True,
    )

    def wrong_direction(work):
        p = T.access_ptr(work[0], "w", extent=64)
        return T.vmi.vstore(T.vmi.vload(p, 0, size=64), work[128])

    # Direction is an API contract, not a footprint proof: it stays frontend.
    with pytest.raises(TypeError, match="declare read access"):
        kernel(wrong_direction)()

    def zero_extent(work):
        p = T.access_ptr(work[0], "r", extent=0)
        return T.vmi.vstore(T.vmi.vload(p, 0, size=64), work[128])

    _assert_descriptor_error(
        lambda: kernel(zero_extent),
        "T.vmi.vload",
        "descriptor extent must be positive",
        explicit=True,
    )


def test_ordinary_address_rejects_unrecoverable_handles():
    def raw_data():
        @T.prim_func
        def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    v = T.vmi.vload(work.data, 0, size=64)
                    T.vmi.vstore(v, work[128])
                T.copy(work, C)

        return main

    with pytest.raises(TypeError, match="no recoverable definition"):
        raw_data()

    def scalar_handle(k):
        @T.prim_func
        def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    v = T.vmi.vload(k, 0, size=64)
                    T.vmi.vstore(v, work[128])
                T.copy(work, C)

        return main

    # A numeric handle is never followed to its definition and treated as a
    # pointer.
    with pytest.raises(TypeError, match="got a scalar int32 variable"):
        scalar_handle(tvm.tirx.Var("unused", "int32"))()

    def multi_level():
        from tilelang.ascend.language import vmi as vmi_language
        from tilelang.language.frame import clear_let_values, register_let_value

        first = tvm.tirx.Var("probe_first", "handle")
        second = tvm.tirx.Var("probe_second", "handle")
        register_let_value(first, second)
        try:
            vmi_language._parse_vmi_address(first, is_load=True, context="T.vmi.vload(...)")
        finally:
            clear_let_values()

    with pytest.raises(TypeError, match="multi-level pointer aliases"):
        multi_level()


@pytest.mark.parametrize("is_load", [True, False], ids=["load", "store"])
def test_ordinary_accesses_require_a_ub_buffer(is_load):
    """Direct and explicit ordinary addresses share the UB scope check."""

    def build():
        @T.prim_func
        def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    if is_load:
                        v = T.vmi.vload(A[0], size=64)
                        T.vmi.vstore(v, work[128])
                    else:
                        v = T.vmi.vload(work[0], size=64)
                        T.vmi.vstore(v, C[0], T.vmi.create_mask(64, size=64))
                T.copy(work, C)

        return main

    with pytest.raises(TypeError, match="requires a UB pointer"):
        build()


def test_ordinary_dynamic_outer_row_is_accepted():
    """The outermost axis has no outward carry to prove, so a legal dynamic
    row coordinate stays expressible without a general bounds proof."""

    @T.prim_func
    def main(A: T.Tensor((1024,), F32), C: T.Tensor((1024,), F32), k: T.int32):
        with T.Kernel(1):
            work = T.alloc_shared((1024,), F32)
            T.copy(A, work)
            view = tvm.tirx.decl_buffer((4, 128), F32, data=work.data, scope="shared.dyn")
            with T.SimdVF():
                v = T.vmi.vload(view[k, 0], size=256)
                T.vmi.vstore(v, work[512])
            T.copy(work, C)

    artifact = _lower_pto(main)
    func, call = _single(artifact, "tl.vmi.vload")
    ptr = _resolve_binding(call.args[0], func)
    assert isinstance(ptr, tvm.tirx.Call) and ptr.op.name.endswith("tvm_access_ptr")
    assert _const(ptr.args[3]) == 256
    assert _const(ptr.args[4]) == 1
    assert _const(call.args[1]) == 0


def test_vstore_requires_matching_value_and_destination_element_dtype():
    """The descriptor extent and the value lanes must use the same element unit."""

    def build():
        @T.prim_func
        def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((256,), F32)
                out = T.alloc_shared((256,), F16)
                T.copy(A, work)
                with T.SimdVF():
                    v = T.vmi.vload(work[0], size=64)
                    T.vmi.vstore(v, out[128])
                T.copy(work, C)

        return main

    with pytest.raises(TypeError, match="destination element dtype float16"):
        build()


# ---------------------------------------------------------------------------
# Ordinary dynamic indices: proof from active loop domains and stable lets
# ---------------------------------------------------------------------------
def _serial_in_row():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((2, 128), F32)
            T.copy(A, work)
            with T.SimdVF():
                for i in T.serial(2):
                    col = 64 * i
                    T.vmi.vstore(T.vmi.vload(work[0, col], size=64), work[1, 0])
            T.copy(work, C)

    return main


def _chained_let_in_row():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((2, 128), F32)
            T.copy(A, work)
            with T.SimdVF():
                for i in T.serial(1, 3):
                    base = i - 1
                    col = 64 * base
                    T.vmi.vstore(T.vmi.vload(work[0, col], size=64), work[1, 0])
            T.copy(work, C)

    return main


def _nested_in_row():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((2, 128), F32)
            T.copy(A, work)
            with T.SimdVF():
                for i in T.serial(2):
                    for j in T.serial(2):
                        col = 32 * i + 16 * j
                        T.vmi.vstore(T.vmi.vload(work[0, col], size=64), work[1, 0])
            T.copy(work, C)

    return main


@pytest.mark.parametrize(
    "factory",
    [_serial_in_row, _chained_let_in_row, _nested_in_row],
    ids=["serial", "chained-let", "nested"],
)
def test_ordinary_dynamic_in_row_indices_are_accepted(factory):
    """Loop domains and stable lets prove that the inner start stays in its row.

    The descriptor keeps the symbolic start, so only the extent and the zero
    instruction offset are asserted structurally.
    """
    artifact = _lower_pto(factory())
    func, call = _single(artifact, "tl.vmi.vload")
    ptr = _resolve_binding(call.args[0], func)
    assert isinstance(ptr, tvm.tirx.Call) and ptr.op.name.endswith("tvm_access_ptr")
    assert _const(ptr.args[3]) == 64
    assert _const(ptr.args[4]) == 1
    assert _const(call.args[1]) == 0


def _serial_wrap():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((2, 128), F32)
            T.copy(A, work)
            with T.SimdVF():
                for i in T.serial(3):
                    col = 64 * i
                    T.vmi.vstore(T.vmi.vload(work[0, col], size=64), work[1, 0])
            T.copy(work, C)

    return main


def _mutable_snapshot():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((2, 128), F32)
            T.copy(A, work)
            with T.SimdVF():
                index = T.alloc_var("int32", init=0)
                snapshot = index
                col = 64 * snapshot
                T.vmi.vstore(T.vmi.vload(work[0, col], size=64), work[1, 0])
            T.copy(work, C)

    return main


def _unknown_inner_start():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32), k: T.int32):
        with T.Kernel(1):
            work = T.alloc_shared((2, 128), F32)
            T.copy(A, work)
            with T.SimdVF():
                T.vmi.vstore(T.vmi.vload(work[0, k], size=64), work[1, 0])
            T.copy(work, C)

    return main


@pytest.mark.parametrize(
    "factory",
    [_serial_wrap, _mutable_snapshot, _unknown_inner_start],
    ids=["serial-3", "mutable-snapshot", "unknown-scalar"],
)
def test_ordinary_dynamic_row_crossing_is_rejected(factory):
    """An unprovable start keeps the diagnostic; a mutable read or an unknown
    scalar is never bound as an equality in the local proof environment."""
    _assert_descriptor_error(factory, "T.vmi.vload", "existing row-major carry")


def test_ordinary_dynamic_shape_in_row_is_accepted():
    """A symbolic inner length is accepted when the static access extent stays
    inside that innermost row."""

    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            with T.SimdVF():
                for i in T.serial(2):
                    n = 64 * (i + 1)
                    view = tvm.tirx.decl_buffer((2, n), F32, data=work.data, scope="shared.dyn")
                    v = T.vmi.vload(view[0, 0], size=64)
                    T.vmi.vstore(v, work[2 * n])
            T.copy(work, C)

    artifact = _lower_pto(main)
    func, call = _single(artifact, "tl.vmi.vload")
    ptr = _resolve_binding(call.args[0], func)
    assert _const(ptr.args[3]) == 64
    assert _const(call.args[1]) == 0


def test_ordinary_dynamic_shape_wrap_is_rejected():
    """A symbolic row length fails when the start is not provably inside it."""

    def build():
        @T.prim_func
        def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
            with T.Kernel(1):
                work = T.alloc_shared((512,), F32)
                T.copy(A, work)
                with T.SimdVF():
                    for i in T.serial(2):
                        n = 32 * i
                        view = tvm.tirx.decl_buffer((2, n), F32, data=work.data, scope="shared.dyn")
                        v = T.vmi.vload(view[0, 16], size=64)
                        T.vmi.vstore(v, work[2 * n])
                T.copy(work, C)

        return main

    _assert_descriptor_error(build, "T.vmi.vload", "non-constant dimension 1")


def _symbolic_extent_kernel(col):
    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            view = tvm.tirx.decl_buffer((4, 128), F32, data=work.data, scope="shared.dyn")
            with T.SimdVF():
                for i in T.serial(2):
                    count = 64 * (i + 1)
                    p = T.access_ptr(view[0, col], "r", extent=count)
                    v = T.vmi.vload(p, size=64)
                    T.vmi.vstore(v, work[384])
            T.copy(work, C)

    return main


@pytest.mark.parametrize(("col", "accepted"), [(0, True), (64, False)], ids=["in-row", "cross-row"])
def test_explicit_symbolic_extent_must_stay_in_the_innermost_row(col, accepted):
    """A non-constant descriptor extent is provable only inside the innermost
    row: the same extent from a cross-row start is rejected."""
    if not accepted:
        _assert_descriptor_error(
            lambda: _symbolic_extent_kernel(col),
            "T.vmi.vload",
            "non-constant extent must stay inside the innermost dimension",
            explicit=True,
        )
        return
    artifact = _lower_pto(_symbolic_extent_kernel(col))
    func, call = _single(artifact, "tl.vmi.vload")
    ptr = _resolve_binding(call.args[0], func)
    assert isinstance(ptr, tvm.tirx.Call) and ptr.op.name.endswith("tvm_access_ptr")
    assert _const(call.args[1]) == 0
    kernel = _compile_pto(_symbolic_extent_kernel(col))
    assert "pto.vmi.vload(" in kernel.get_kernel_source()


# ---------------------------------------------------------------------------
# Gather/scatter non-zero bases (no footprint analysis)
# ---------------------------------------------------------------------------
def _gather_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), I: T.Tensor((64,), "uint32"), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            index = T.alloc_shared((64,), "uint32")
            T.copy(A, work)
            T.copy(I, index)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                offsets = T.vmi.vload(index[0], size=64)
                v = T.vmi.vgather(work[64], offsets, mask)
                T.vmi.vstore(v, work[192], mask)
            T.copy(work, C)

    return main


def _scatter_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), F32), I: T.Tensor((64,), "uint32"), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            index = T.alloc_shared((64,), "uint32")
            T.copy(A, work)
            T.copy(I, index)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                offsets = T.vmi.vload(index[0], size=64)
                value = T.vmi.vload(work[128], size=64)
                T.vmi.vscatter(value, work[64], offsets, mask)
            T.copy(work, C)

    return main


def _negative_view_kernel():
    @T.prim_func
    def main(A: T.Tensor((4, 64), F32), I: T.Tensor((64,), "uint32"), C: T.Tensor((4, 64), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((4, 64), F32)
            index = T.alloc_shared((64,), "uint32")
            T.copy(A, work)
            T.copy(I, index)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                offsets = T.vmi.vload(index[0], size=64)
                v = T.vmi.vgather(work[2, -32], offsets, mask)
                T.vmi.vstore(v, work[3, -64], mask)
            T.copy(work, C)

    return main


def test_indexed_frontend_keeps_raw_address():
    for kernel, op_name, address_index in (
        (_gather_kernel, "tl.vmi.vgather", 0),
        (_scatter_kernel, "tl.vmi.vscatter", 1),
    ):
        found = []

        def visit(node, found=found, op_name=op_name):
            if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == op_name:
                found.append(node)

        tvm.tirx.stmt_functor.post_order_visit(kernel().body, visit)
        assert len(found) == 1, (op_name, found)
        address = found[0].args[address_index]
        assert isinstance(address, tvm.tirx.BufferLoad), (op_name, address)
        assert address.buffer.scope() in ("shared", "shared.dyn")


def test_gather_descriptor_uses_owner_base():
    artifact = _lower_pto(_gather_kernel())
    _, call = _single(artifact, "tl.vmi.vgather")
    ptr = call.args[0]
    assert _const(ptr.args[2]) == 0, "the descriptor origin is the real owner origin"
    assert _const(ptr.args[3]) == 256, "the descriptor covers the whole owner"
    assert _const(ptr.args[4]) == 1
    assert _const(call.args[1]) == 64
    line = next(ln.strip() for ln in artifact.kernel_source.splitlines() if "vmi.vgather" in ln)
    assert "addptr" in line and ", 64)" in line, line

    negative_artifact = _lower_pto(_negative_view_kernel())
    _, negative_call = _single(negative_artifact, "tl.vmi.vgather")
    negative_ptr = negative_call.args[0]
    assert _const(negative_ptr.args[2]) == 0
    assert _const(negative_ptr.args[3]) == 256
    assert _const(negative_ptr.args[4]) == 1
    assert _const(negative_call.args[1]) == 160
    negative_line = next(ln.strip() for ln in negative_artifact.kernel_source.splitlines() if "vmi.vgather" in ln)
    assert "addptr" in negative_line and ", 160)" in negative_line, negative_line


def test_scatter_descriptor_uses_owner_base():
    artifact = _lower_pto(_scatter_kernel())
    _, call = _single(artifact, "tl.vmi.vscatter")
    ptr = call.args[1]
    assert _const(ptr.args[2]) == 0
    assert _const(ptr.args[3]) == 256
    assert _const(ptr.args[4]) == 2
    assert _const(call.args[2]) == 64
    line = next(ln.strip() for ln in artifact.kernel_source.splitlines() if "vmi.vscatter" in ln)
    assert "addptr" in line and ", 64)" in line, line


# ---------------------------------------------------------------------------
# Removed VMI interface parameters
# ---------------------------------------------------------------------------
def _vmi_body_kernel(body):
    @T.prim_func
    def main(A: T.Tensor((256,), F32), C: T.Tensor((256,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((256,), F32)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                body(work, mask)
            T.copy(work, C)

    return main


@pytest.mark.parametrize(
    ("case_id", "body", "message"),
    [
        (
            "vload-repeat-stride",
            lambda work, mask: T.vmi.vload(work[0], size=64, repeat_stride=0),
            "unexpected keyword argument 'repeat_stride'",
        ),
        (
            "vstore-repeat-stride",
            lambda work, mask: T.vmi.vstore(T.vmi.vload(work[0], size=64), work[0], mask, repeat_stride=0),
            "unexpected keyword argument 'repeat_stride'",
        ),
        (
            "vload-to-dtype",
            lambda work, mask: T.vmi.vload(work[0], size=64, to_dtype=F16),
            "unexpected keyword argument 'to_dtype'",
        ),
        (
            "vstore-pmode",
            lambda work, mask: T.vmi.vstore(T.vmi.vload(work[0], size=64), work[0], mask, pmode="zero"),
            "unexpected keyword argument 'pmode'",
        ),
    ],
)
def test_removed_vmi_parameters_fail_in_legal_scope(case_id, body, message):
    with pytest.raises(TypeError, match=message):
        _vmi_body_kernel(body)
