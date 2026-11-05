"""Mutable pointer cells retain native pointer types and actual address state."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as I

import tilelang
import tilelang.cpu.language as T
from tilelang.language import _native
from tilelang.language.eager.builder import PointerRef


def _run_cpu(function, source, expected):
    target = "c"
    kernel = tilelang.compile(function, target=target, target_host=target, execution_backend="tvm_ffi")

    def aligned_copy(array):
        storage = np.empty(array.nbytes + 63, dtype="uint8")
        offset = -storage.ctypes.data % 64
        result = storage[offset : offset + array.nbytes].view(array.dtype).reshape(array.shape)
        result[...] = array
        return result

    output = aligned_copy(np.zeros(expected.shape, dtype="int32"))
    kernel(aligned_copy(source), output)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("round_trip", [False, True])
def test_mutable_pointer_across_if_and_loop_preserves_nonzero_offsets(round_trip):
    @T.prim_func
    def kernel(A: T.Tensor((12,), "int32"), B: T.Tensor((6,), "int32")):
        pointer = T.alloc_var("handle", T.access_ptr(A[1], "r", extent=1))
        first = T.make_tensor(pointer, (1,), "int32")
        B[0] = first[0]
        for i in T.serial(5):
            if i % 2 == 0:
                pointer = T.access_ptr(A[2 * i + 1], "r", extent=1)
            else:
                pointer = T.access_ptr(A[i + 2], "r", extent=1)
            view = T.make_tensor(pointer, (1,), "int32")
            B[i + 1] = view[0]

    if round_trip:
        kernel = ir.load_json(ir.save_json(kernel))
    cells = []
    tvm_ffi.structural_walk(
        kernel,
        (ir.Bind, lambda node: cells.append(node.var) if tirx.is_tensor_var(node.var) and str(node.var.ty.scope) == "local.var" else None),
    )
    assert len(cells) == 1 and str(cells[0].ty.dtype) == "uint64"
    source = np.arange(12, dtype="int32") * 11
    _run_cpu(kernel, source, source[[1, 1, 3, 5, 5, 9]])


def test_two_frontend_aliases_update_the_same_pointer_cell():
    with IRBuilder() as builder, I.function():
        I.func_name_("pointer_alias")
        source = I.arg_("A", tirx.TensorType([5], "int32"))
        output = I.arg_("B", tirx.TensorType([2], "int32"))
        pointer = T.alloc_var("handle", T.access_ptr(source[1], "r", extent=1))
        alias = pointer
        first = _native.bind(_native.match_buffer(pointer[0], (1,), "int32"))
        _native.buffer_store(output, first[0], 0)
        alias[0] = T.access_ptr(source[3], "r", extent=1)
        second = _native.bind(_native.match_buffer(pointer[0], (1,), "int32"))
        _native.buffer_store(output, second[0], 1)
        assert alias.buffer.same_as(pointer.buffer)
    function = builder.get()
    source = np.arange(5, dtype="int32") * 13
    _run_cpu(function, source, source[[1, 3]])


@pytest.mark.parametrize("scope", ["global", "shared.dyn", "shared"])
def test_pointer_cell_load_restores_pointee_and_address_space(scope):
    with IRBuilder(), I.function():
        pointer = ir.Var("pointer", ir.PtrType("float16", scope))
        cell = T.alloc_var("handle", pointer)
        assert isinstance(cell, PointerRef)
        assert str(cell.buffer.ty.dtype) == "uint64"
        value = cell.load()
        assert isinstance(value.ty, ir.PtrType)
        assert str(value.ty.element_type) == "float16"
        assert str(value.ty.scope) == scope
        copied = T.alloc_var("handle", cell)
        assert tvm_ffi.structural_equal(copied.load().ty, value.ty)
        other_scope = "shared" if scope != "shared" else "global"
        with pytest.raises(TypeError, match="address space"):
            cell.store(ir.Var("other", ir.PtrType("float16", other_scope)))
        with pytest.raises(TypeError, match="pointer expression"):
            cell.store(ir.const(1.5, "float32"))
        with pytest.raises(IndexError, match="index 0"):
            cell[1]


def test_simd_pointer_requires_its_declaring_builder():
    import tilelang.ascend.language as A

    with IRBuilder(), I.function():
        source = A.alloc_shared((128,), "uint16")
        cell = A.simd.make_ubuf_ptr(source[0], "uint16")
        value = cell[0]
    with IRBuilder(), I.function(), pytest.raises(ValueError, match="current IRBuilder"):
        A.simd.vld(value, post_inc=0)


def test_simd_pointer_update_results_preserve_the_declared_type():
    import tilelang.ascend.language as A

    with IRBuilder(), I.function():
        source = I.arg_("A", tirx.TensorType([256], "uint16", scope="shared.dyn"))
        pointer = A.simd.make_ubuf_ptr(source[0], "uint16")
        pair = A.simd.vld(pointer, post_inc=128)
        vector, advanced = pair
        assert tvm_ffi.structural_equal(advanced.ty, pointer.pointer_type)
        pointer[0] = advanced
        updated = A.simd.vsstb(vector, pointer, 1, update=True)
        assert tvm_ffi.structural_equal(updated.ty, pointer.pointer_type)
        pointer[0] = updated
        unregistered = T.alloc_var("handle", tirx.tensor_data_ptr(source))
        with pytest.raises(ValueError, match="declared with make_ubuf_ptr"):
            A.simd.vld(unregistered[0], post_inc=1)


@pytest.mark.parametrize("post_load", [False, True])
def test_simd_float32_mutable_pointer_codegen(post_load):
    import tilelang.ascend.language as A
    from tilelang.engine import lower

    @A.prim_func
    def kernel(src: A.Tensor((128,), "float32"), dst: A.Tensor((128,), "float32")):
        with A.Kernel(1):
            source = A.alloc_shared((128,), "float32")
            destination = A.alloc_shared((128,), "float32")
            A.copy(src, source)
            with A.SimdVF():
                source_pointer = A.simd.make_ubuf_ptr(A.access_ptr(source[0], "r", extent=128), "float32")
                destination_pointer = A.simd.make_ubuf_ptr(destination[0], "float32")
                if post_load:
                    first, source_pointer = A.simd.vld(source_pointer, post_inc=64)
                    second, source_pointer = A.simd.vld(source_pointer, post_inc=0)
                else:
                    first = A.simd.vld(source[0])
                    second = A.simd.vld(source[64])
                destination_pointer = A.simd.vsstb(first, destination_pointer, 1, update=True)
                destination_pointer = A.simd.vsstb(second, destination_pointer, 1, update=True)
            A.copy(destination, dst)

    source = lower(kernel, target="ascend").kernel_source
    assert source.count("simd_inst::vsstb(") == 2
    assert "uintptr_t" in source and "uint64_t" in source
    assert "__ubuf__" in source
    if post_load:
        assert source.count("vlds_norm_postupdate<") == 2
