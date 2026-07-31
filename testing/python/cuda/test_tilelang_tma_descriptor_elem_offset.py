import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower_to_host_device_ir
from tilelang.jit.adapter.nvrtc.wrapper import TLNVRTCSourceWrapper
from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper


def _descriptor_base_byte_offsets(func, arch):
    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target, tvm.transform.PassContext(config=pass_configs):
        host_mod, device_mod, _, _, _ = lower_to_host_device_ir(func, target=target)

    handle_add = tvm.ir.Op.get("tirx.handle_add_byte_offset")
    offsets = []
    descriptors = []
    for mod in (host_mod, device_mod):
        for global_var in mod.get_global_vars():
            args_map = mod[global_var].attrs.get("tma_descriptor_args")
            if args_map:
                descriptors.extend(args_map.items())

    for _, args in descriptors:
        base = args[4]
        assert isinstance(base, tvm.tirx.Call)
        assert base.op.same_as(handle_add)
        offset = tvm.arith.Analyzer().simplify(base.args[1])
        assert isinstance(offset, tvm.tirx.IntImm)
        offsets.append(int(offset.value))
    return sorted(set(offsets)), descriptors


def _bulk_copy_program():
    @T.prim_func
    def main(src_handle: T.handle, dst_handle: T.handle):
        src = T.match_buffer(src_handle, (16, 80), T.float16, elem_offset=16)
        dst = T.match_buffer(dst_handle, (16, 80), T.float16, elem_offset=32)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((8, 64), T.float16)
            mbar = T.alloc_barrier(1)
            T.tma_copy(src[0:8, 0:64], shared, barrier=mbar)
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(shared, dst[0:8, 0:64])

    return main


def _gather_scatter_program():
    @T.prim_func
    def main(src_handle: T.handle, rows: T.Tensor((4,), T.int32), dst_handle: T.handle):
        src = T.match_buffer(src_handle, (64, 64), T.float16, elem_offset=16)
        dst = T.match_buffer(dst_handle, (64, 64), T.float16, elem_offset=32)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((4, 64), T.float16)
            mbar = T.alloc_barrier(1)
            row_indices = [rows[0], rows[1], rows[2], rows[3]]
            T.tma_gather4(src, shared, 0, row_indices, barrier=mbar)
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.tma_scatter4(shared, dst, 0, row_indices)

    return main


def _im2col_program():
    @T.prim_func
    def main(src_handle: T.handle):
        src = T.match_buffer(src_handle, (1, 8, 8, 64), T.float16, elem_offset=16)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 32), T.float16)
            T.im2col(src, shared, 0, 0, 3, 1, 1, 1)

    return main


def _tma_atomic_program():
    @T.prim_func
    def main(dst_handle: T.handle):
        dst = T.match_buffer(dst_handle, (16, 16), T.float32, elem_offset=8)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 16), T.float32)
            T.atomic_add(dst, shared, use_tma=True)

    return main


def _auto_copy_program(elem_offset):
    if elem_offset == "symbolic":

        @T.prim_func
        def main(src_handle: T.handle, offset: T.int32):
            src = T.match_buffer(src_handle, (16, 80), T.float16, elem_offset=offset)
            with T.Kernel(1, threads=128):
                shared = T.alloc_shared((8, 64), T.float16)
                T.copy(src[0:8, 0:64], shared)

    else:

        @T.prim_func
        def main(src_handle: T.handle):
            src = T.match_buffer(src_handle, (16, 80), T.float16, elem_offset=elem_offset)
            with T.Kernel(1, threads=128):
                shared = T.alloc_shared((8, 64), T.float16)
                T.copy(src[0:8, 0:64], shared)

    return main


@pytest.mark.parametrize(
    ("program", "arch", "expected_offsets"),
    [
        (_bulk_copy_program, "sm_90", [32, 64]),
        (_gather_scatter_program, "sm_100a", [32, 64]),
        (_im2col_program, "sm_90", [32]),
        (_tma_atomic_program, "sm_90", [32]),
    ],
)
def test_tma_descriptor_base_includes_buffer_elem_offset(program, arch, expected_offsets):
    offsets, descriptors = _descriptor_base_byte_offsets(program(), arch)
    assert offsets == expected_offsets

    c_wrapper = object.__new__(TLCUDASourceWrapper)
    nvrtc_wrapper = object.__new__(TLNVRTCSourceWrapper)
    for desc_var, args in descriptors:
        address = args[4]
        desc_name_map = {desc_var.name: address.args[0].name}
        desc_name_var_map = {desc_var.name: desc_var}
        c_wrapper.tma_descriptor_args = {desc_var: args}
        nvrtc_wrapper.tma_descriptor_args = {desc_var: args}
        c_init = c_wrapper.generate_tma_descriptor_args(desc_name_map, desc_name_var_map)
        python_init = nvrtc_wrapper.generate_tma_descriptor_args(desc_name_map, desc_name_var_map)
        assert "handle_add_byte_offset" not in c_init
        assert "_globalAddress= (void*)((char*)" in c_init
        assert "handle_add_byte_offset" not in python_init
        assert ".data_ptr() +" in python_init


@pytest.mark.parametrize("elem_offset", [1, "symbolic"])
def test_auto_copy_falls_back_for_unproven_descriptor_alignment(elem_offset):
    offsets, descriptors = _descriptor_base_byte_offsets(_auto_copy_program(elem_offset), "sm_90")
    assert offsets == []
    assert descriptors == []
