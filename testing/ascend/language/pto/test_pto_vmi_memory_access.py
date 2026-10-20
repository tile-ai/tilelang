"""VMI memory-access descriptors and interface adaptation (structural).

Hardware independent.  Each case lowers the real TileLang PTO pipeline (no
device execution) and inspects the produced TIR/PTODSL.  Cases that the
backend can express also call ``tilelang.compile``, which runs PTODSL -> PTOAS
(VPTO) -> Bisheng; that is compile evidence, not numerical evidence (device
results live in ``test_pto_vmi_memory_access_run.py``).

Contracts verified here:

* ``T.vmi.vload``/``T.vmi.vstore`` access descriptors carry the true start
  address and a per-access-mode effective extent.  The VMI instruction offset
  is zero unless a relocation candidate moves the source position into it;
  descriptor base + instruction offset always equals the original access
  address after the normal negative-index legalization, so the scheduler sees
  the actual accessed range.
* Ordinary continuous loads round up to full physical vectors; ordinary
  broadcast keeps the conservative extent; ``dintlv`` loads declare both
  results and a two-input ``dintlv`` store declares the full 2*N span (its
  PTODSL attribute is emitted as ``intlv``); group envelopes cover all
  groups (slot/broadcast count one element per group, a static negative
  stride relocates the lowest address and compensates with the instruction
  offset, a view whose strides are at least dense row-major declares its
  logical prefix from the buffer origin, and a whole tight buffer is used
  only when its envelope and every axis start are proven compatible with
  the buffer).  A group or group stride without proven bounds is rejected
  at compile time; a static negative stride is rejected only when no
  complete descriptor can be constructed without changing the address,
  and a single group's dynamic stride stays supported because it produces
  no inter-group displacement.  ``block_stride`` extents are measured in
  32B blocks; a dynamic or negative block stride is likewise rejected
  instead of falling back to a fabricated range;
  genuine FP4 descriptors stay logical and are packed exactly once (odd
  starts are rejected); packed (PK) store extents stay logical target
  elements; an explicit ``access_ptr`` keeps the caller-declared descriptor
  plus the extra offset and the UB scope check; gather/scatter preserve
  non-zero bases.
* Masks never shrink the may-write extent, and a logical ``size=80`` load
  declares both physical reads.
* Removed interface parameters (``repeat_stride``, vload ``unpack`` /
  ``to_dtype``, vstore ``pmode``) fail in a legal scope.

``size=80`` VMI vectors are not constructible by the PTODSL VMI API
(lane counts are 1/2/4/8/64/128/256).  Their real lowering is verified with a
direct PTOAS compile of the PTO form the descriptor describes; the TileLang
diagnostics/descriptors are verified structurally.
"""

from __future__ import annotations

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
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower

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


def _assert_descriptor_error(build, op, reason):
    """Assert an unprovable footprint is rejected with the right op and reason."""
    with pytest.raises(ValueError) as excinfo:
        build()
    message = str(excinfo.value)
    assert f"{op} cannot construct a descriptor covering the complete physical access" in message, message
    assert reason in message, message
    return message


def _ptr_position(call, op_name):
    if op_name == "tl.vmi.vstore" and "dist_mode" in call.annotations:
        return 2
    return 1 if op_name == "tl.vmi.vstore" else 0


def _descriptor(artifact, op_name):
    """Return (descriptor_offset, extent, rw_mask, instruction_offset)."""
    func, call = _single(artifact, op_name)
    ptr_pos = _ptr_position(call, op_name)
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


def test_elem_offset_counted_exactly_once():
    """A view with elem_offset=16 loaded at index 32 starts at element 48."""
    artifact = _lower_pto(_elem_offset_kernel())
    assert _descriptor(artifact, "tl.vmi.vload") == (48, 64, 1, 0)
    assert _descriptor(artifact, "tl.vmi.vstore") == (48, 64, 2, 0)


# ---------------------------------------------------------------------------
# Ordinary broadcast
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dtype", "size"),
    [("int8", 256), (F16, 128), (F32, 128)],
    ids=["b8", "b16", "b32"],
)
def test_ordinary_broadcast_keeps_conservative_extent(dtype, size):
    """brc reads one source element; the conservative ``size`` extent is kept."""

    # b8 full-VL broadcast at 64 reads [64,320) and stores [128,384); size
    # the storage (and its initialization) for the real footprint.
    @T.prim_func
    def main(A: T.Tensor((512,), dtype), C: T.Tensor((512,), dtype)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), dtype)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(size, size=size)
                v = T.vmi.vload(work[64], size=size, dist_mode="brc")
                T.vmi.vstore(v, work[128], mask)
            T.copy(work, C)

    artifact = _lower_pto(main)
    # The declaration is not re-expressed as a broadcast output width: it stays
    # `size` address elements (a conservative over-report).
    assert _descriptor(artifact, "tl.vmi.vload") == (64, size, 1, 0)
    assert 'dist_mode="brc"' in artifact.kernel_source

    kernel = _compile_pto(main)
    assert "pto.vmi.vload(" in kernel.get_kernel_source()


# ---------------------------------------------------------------------------
# Physical-vector rounding of ordinary continuous loads
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
def test_continuous_load_covers_full_physical_vectors(dtype, size, expected):
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
# dintlv load / two-input store
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [64, 128], ids=["single-vl", "multi-vl"])
def test_dintlv_load_declares_both_results(size):
    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            with T.SimdVF():
                # Each result has `size` lanes: the store mask must match.
                mask = T.vmi.create_mask(size, size=size)
                even, odd = T.vmi.vload(work[64], size=size, dist_mode="dintlv")
                T.vmi.vstore(even, work[320], mask)
                T.vmi.vstore(odd, work[384], mask)
            T.copy(work, C)

    artifact = _lower_pto(main)
    assert _descriptor(artifact, "tl.vmi.vload") == (64, 2 * size, 1, 0)


@pytest.mark.parametrize("mask_lanes", [0, 10, 64], ids=["all-false", "partial", "all-true"])
def test_dintlv_store_declares_full_two_input_span(mask_lanes):
    """The store may-write span is 2*N regardless of the ineffective mask."""

    @T.prim_func
    def main(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(mask_lanes, size=64)
                even, odd = T.vmi.vload(work[64], size=64, dist_mode="dintlv")
                T.vmi.vstore((even, odd), work[192], mask, dist_mode="dintlv")
            T.copy(work, C)

    artifact = _lower_pto(main)
    assert _descriptor(artifact, "tl.vmi.vstore") == (192, 128, 2, 0)

    source = artifact.kernel_source
    vstore_line = next(ln.strip() for ln in source.splitlines() if "vmi.vstore" in ln)
    # IR/API keep "dintlv"; only the PTODSL attribute is mapped to "intlv".
    assert 'dist_mode="intlv"' in vstore_line, vstore_line
    assert 'dist_mode="dintlv"' not in vstore_line, vstore_line
    _, store_call = _single(artifact, "tl.vmi.vstore")
    assert store_call.annotations["dist_mode"].value == "dintlv"


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


def test_group_slot_and_broadcast_use_one_element_per_group():
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

    @T.prim_func
    def group_brc(A: T.Tensor((512,), F32), C: T.Tensor((512,), F32)):
        with T.Kernel(1):
            work = T.alloc_shared((512,), F32)
            T.copy(A, work)
            with T.SimdVF():
                v = T.vmi.vload(work[64], size=128, group=2, stride=128, dist_mode="brc")
                T.vmi.vstore(v, work[256], group=2, stride=128)
            T.copy(work, C)

    assert _vload_descriptor(group_brc) == (64, 129, 1, 0)


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
    _assert_descriptor_error(build, op, "group stride has no proven bounds")


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
def test_group_negative_stride_nd_negative_start_is_rejected(is_load):
    """work[2,-32] legalizes to 160; relocating the descriptor would start at 96."""

    def build():
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

        return main

    op = "T.vmi.vload" if is_load else "T.vmi.vstore"
    _assert_descriptor_error(build, op, "negative-index legalization")


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

    from testing.ascend._ir import schedule as ir_schedule

    events = _sync_events(ir_schedule(main, through="InsertSync", auto_schedule=True))
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


@pytest.mark.parametrize(
    ("shape", "strides", "indices", "accepted"),
    [
        ((2, 128), (256, 1), (1, 0), True),
        ((2, 128), (256, 1), (0, 0), False),
        ((2, 128), (1, 2), (1, 0), False),
        ((2, 128), (64, 1), (1, 0), False),
        ((4, 128), (256, 1), (-1, 512), False),
    ],
    ids=["padded", "unproven-low", "transposed", "compressed", "negative-start"],
)
def test_padded_origin_plan_static_layout_boundaries(shape, strides, indices, accepted):
    """The origin plan requires non-negative starts and an envelope above the origin.

    A rejected source makes this candidate return ``None`` so the caller keeps
    trying other constructions; only after every candidate fails does the
    caller emit the unified diagnostic.
    """
    from tilelang.ascend.language import vmi as vmi_language

    buffer = tvm.tirx.decl_buffer(shape, F32, name="view", strides=strides, scope="shared.dyn")
    source = tvm.tirx.BufferLoad(buffer, list(indices))
    plan = vmi_language._padded_origin_plan(source, low=-256, high=64)
    if accepted:
        assert plan is not None
        assert plan[0] == 320
        assert plan[1] is not None and plan[1].same_as(buffer)
        assert _const(plan[2]) == 256
    else:
        assert plan is None


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

    _assert_descriptor_error(build, "T.vmi.vload", reason)


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
    with pytest.raises(Exception, match="fp4 access_ptr offset must be even"):
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


def test_gather_descriptor_preserves_nonzero_base():
    artifact = _lower_pto(_gather_kernel())
    _, call = _single(artifact, "tl.vmi.vgather")
    ptr = call.args[0]
    assert _const(ptr.args[2]) == 64
    # Lanes extent: generic index footprints are explicitly out of scope.
    assert _const(ptr.args[3]) == 64
    assert _const(call.args[1]) == 0
    line = next(ln.strip() for ln in artifact.kernel_source.splitlines() if "vmi.vgather" in ln)
    assert "addptr" in line and ", 64)" in line, line


def test_scatter_descriptor_preserves_nonzero_base():
    artifact = _lower_pto(_scatter_kernel())
    _, call = _single(artifact, "tl.vmi.vscatter")
    assert _const(call.args[1].args[2]) == 64
    assert _const(call.args[2]) == 0
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
            "vload-unpack",
            lambda work, mask: T.vmi.vload(work[0], size=64, dist_mode="unpack"),
            "does not support dist_mode='unpack'",
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
