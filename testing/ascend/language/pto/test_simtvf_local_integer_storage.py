"""SIMT local integer buffers use signless LLVM stack storage.

This file verifies that:

* scalar-element SIMT local buffers of the admitted integer types
  int8/uint8/int32/uint32 allocate with a same-width *signless* PTO element
  type (``pto.i8`` / ``pto.i32``);
* the four access sites -- scalar load, scalar store, contiguous
  builtin-vector load and contiguous builtin-vector store -- adapt between
  the signless backing and TileLang's logical signed/unsigned type via
  ``pto.cast``;
* the adapter is scoped to SIMT local storage only: VMI, ``local.var``,
  shared and global accesses keep their existing types;
* vector-element local slots use the same-width signless element backing in
  the allocation branch;
* the admitted dtype set is unchanged: int16/int64 stay rejected.
"""

import re

import pytest
import torch  # noqa: F401  (torch/torch_npu before tilelang/ptodsl)
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend.target import normalize_pto_target


def _lower(func):
    """Lower with an explicit PTO target context (required for fragments)."""
    target = normalize_pto_target("pto")
    with target:
        return tilelang.lower(func, target=target)


# ---------------------------------------------------------------------------
# SIMT local integer kernel: scalar and contiguous access sites
# ---------------------------------------------------------------------------
def _simt_local_integer_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype)):
        with T.Kernel(1), T.SimtVF(threads=1):
            local = T.alloc_local((8,), dtype)
            local[0] = A[0]
            B[0] = local[0]
            local[T.Ramp(1, 1, 4)] = A[T.Ramp(0, 1, 4)]
            B[T.Ramp(1, 1, 4)] = local[T.Ramp(1, 1, 4)]

    return main


@pytest.mark.parametrize(
    ("dtype", "logical_type", "backing_type"),
    [
        ("int8", "pto.si8", "pto.i8"),
        ("uint8", "pto.ui8", "pto.i8"),
        ("int32", "pto.si32", "pto.i32"),
        ("uint32", "pto.ui32", "pto.i32"),
    ],
)
def test_simtvf_local_integer_scalar_and_contiguous_accesses_use_signless_storage(dtype, logical_type, backing_type):
    source = _lower(_simt_local_integer_kernel(dtype)).kernel_source

    # Allocation: same-width signless backing.
    assert re.search(rf"pto\.alloc_buffer\(\(8,\), {re.escape(backing_type)}\)", source)
    assert "pto.alloc_buffer((8,), pto.si8)" not in source
    assert "pto.alloc_buffer((8,), pto.ui8)" not in source
    assert "pto.alloc_buffer((8,), pto.si32)" not in source
    assert "pto.alloc_buffer((8,), pto.ui32)" not in source

    # Scalar load restores the logical type; scalar store returns to backing.
    assert re.search(rf"pto\.cast\(pto\.load\(local, 0\), {re.escape(logical_type)}\)", source)
    assert re.search(
        rf"pto\.store\(pto\.cast\(pto\.load\(A, 0\), {re.escape(backing_type)}\), local, 0\)",
        source,
    )

    # Contiguous builtin-vector load restores; contiguous store returns.
    assert re.search(
        rf"pto\.cast\(pto\.load\(local, 1, contiguous=4\), {re.escape(logical_type)}\)",
        source,
    )
    assert re.search(
        rf"pto\.store\(pto\.cast\(pto\.load\(A, 0, contiguous=4\), "
        rf"{re.escape(backing_type)}\), local, 1, contiguous=4\)",
        source,
    )

    # Live allocation survives optimization: one allocation and the local
    # buffer is actually read/written by all four access sites.
    assert source.count("pto.alloc_buffer(") == 1
    assert len(re.findall(r"pto\.load\(local,", source)) == 2
    assert len(re.findall(r", local,", source)) == 2


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "uint32"])
def test_simtvf_local_integer_kernel_compiles_through_backend(dtype):
    """All four admitted dtypes compile through the full PTO backend."""
    target = normalize_pto_target("pto")
    with target:
        kernel = tilelang.compile(_simt_local_integer_kernel(dtype), target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    backing = "pto.i8" if dtype in ("int8", "uint8") else "pto.i32"
    assert f"pto.alloc_buffer((8,), {backing})" in source


# ---------------------------------------------------------------------------
# Vector-element local slot allocation
# ---------------------------------------------------------------------------
def _simt_local_integer_vector_kernel(alloc=T.alloc_local):
    @T.prim_func
    def main(A: T.Tensor((4,), "int8"), B: T.Tensor((4,), "int8")):
        with T.Kernel(1), T.SimtVF(threads=1):
            local = alloc((1,), "int8x4")
            local[0] = T.int8x4(A[0], A[1], A[2], A[3])
            B[T.Ramp(0, 1, 4)] = local[0]

    return main


@pytest.mark.parametrize(
    ("scope_name", "alloc"),
    [("local", T.alloc_local), ("local.fragment", T.alloc_fragment)],
    ids=["local", "local.fragment"],
)
def test_simtvf_vector_integer_local_allocation_is_signless(scope_name, alloc):
    """Vector-element local slots allocate with the same-width signless i8 backing."""
    source = _lower(_simt_local_integer_vector_kernel(alloc)).kernel_source

    assert "pto.alloc_buffer((1,), pto.i8)" in source
    assert "pto.alloc_buffer((1,), pto.si8)" not in source


# ---------------------------------------------------------------------------
# Admission boundaries stay closed
# ---------------------------------------------------------------------------
def _unsupported_local_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((4,), dtype), B: T.Tensor((4,), dtype)):
        with T.Kernel(1), T.SimtVF(threads=1):
            local = T.alloc_local((4,), dtype)
            local[0] = A[0]
            B[0] = local[0]

    return main


@pytest.mark.parametrize("dtype", ["int16", "int64"])
def test_simtvf_local_integer_admission_boundary_is_unchanged(dtype):
    with pytest.raises(Exception, match="PTO SIMT local allocation supports"):
        _lower(_unsupported_local_kernel(dtype))


# ---------------------------------------------------------------------------
# Scope controls: local.var, shared/global and VMI keep their contracts
# ---------------------------------------------------------------------------
def _local_var_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((8,), dtype)
            dst = T.alloc_shared((8,), dtype)
            T.copy(A, src)
            T.copy(A, dst)
            with T.SimtVF(threads=1):
                addr = T.alloc_var(dtype)
                addr = src[3]
                dst[3] = addr
            T.copy(dst, B)

    return main


def test_simtvf_local_var_integer_keeps_logical_type():
    source = _lower(_local_var_kernel("int32")).kernel_source
    # local.var is a Python SSA value typed with the logical dtype; it is not
    # a signless-backed local buffer.
    assert "addr = pto.const(0, dtype=pto.si32)" in source
    assert "pto.alloc_buffer((1,), pto.i32)" not in source


def _shared_control_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((8,), dtype)
            dst = T.alloc_shared((8,), dtype)
            T.copy(A, src)
            T.copy(A, dst)
            with T.SimtVF(threads=1):
                dst[0] = src[0]
            T.copy(dst, B)

    return main


@pytest.mark.parametrize("dtype", ["uint8", "int32"])
def test_simtvf_shared_integer_accesses_do_not_use_local_adapter(dtype):
    source = _lower(_shared_control_kernel(dtype)).kernel_source
    line = next(line for line in source.splitlines() if "pto.store(pto.load(" in line)
    assert "pto.cast(" not in line, line


def _vmi_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "int32"), C: T.Tensor((256,), "int32")):
        with T.Kernel(1):
            work = T.alloc_shared((256,), "int32")
            T.copy(A, work)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                value = T.vmi.vload(work[0], size=64)
                T.vmi.vstore(value, work[64], mask)
            T.copy(work, C)

    return main


def test_vmi_integer_carrier_keeps_signed_type():
    artifact = _lower(_vmi_kernel())
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.vmi.vload":
            found.append(node)

    for func in artifact.device_mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, visit)

    assert found, "no VMI load in the lowered module"
    assert any(str(call.dtype) == "int32x64" for call in found), [str(call.dtype) for call in found]
