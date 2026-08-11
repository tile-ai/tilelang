import re

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.engine.lower import lower_to_host_device_ir
from tilelang.layout import make_linear_layout
from tilelang.jit.adapter.cutedsl.wrapper import TLCuTeDSLSourceWrapper
from tilelang.jit.adapter.nvrtc.wrapper import TLNVRTCSourceWrapper
from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper


def _descriptor_base_byte_offsets(func, arch):
    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    context = create_backend_context(target, execution_backend="tvm_ffi")
    with context.target, tvm.transform.PassContext(config=pass_configs):
        host_mod, device_mod, _, _, _ = lower_to_host_device_ir(func, context)

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
        if isinstance(base, tvm.tirx.Var):
            offsets.append(0)
        else:
            assert isinstance(base, tvm.tirx.Call)
            assert base.op.same_as(handle_add)
            offset = tvm.arith.Analyzer().simplify(base.args[1])
            if isinstance(offset, tvm.tirx.IntImm):
                offsets.append(int(offset.value))
            else:
                offsets.append(offset)
    if all(isinstance(offset, int) for offset in offsets):
        offsets = sorted(set(offsets))
    return offsets, descriptors


def _bulk_copy_program(src_elem_offset=64, dst_elem_offset=128, annotate_linear=False):
    @T.prim_func
    def main(src_handle: T.handle, dst_handle: T.handle):
        src = T.match_buffer(src_handle, (16, 80), T.float16, elem_offset=src_elem_offset)
        dst = T.match_buffer(dst_handle, (16, 80), T.float16, elem_offset=dst_elem_offset)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((8, 64), T.float16)
            if annotate_linear:
                T.annotate_layout({shared: make_linear_layout(shared)})
            mbar = T.alloc_barrier(1)
            T.tma_copy(src[0:8, 0:64], shared, barrier=mbar)
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(shared, dst[0:8, 0:64])

    return main


def _gather_scatter_program():
    @T.prim_func
    def main(src_handle: T.handle, rows: T.Tensor((4,), T.int32), dst_handle: T.handle):
        src = T.match_buffer(src_handle, (64, 64), T.float16, elem_offset=64)
        dst = T.match_buffer(dst_handle, (64, 64), T.float16, elem_offset=128)
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
        src = T.match_buffer(src_handle, (1, 8, 8, 64), T.float16, elem_offset=64)
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


def _symbolically_aligned_tma_copy_program():
    @T.prim_func
    def main(src_handle: T.handle, offset: T.int32):
        src = T.match_buffer(src_handle, (16, 80), T.float16, elem_offset=offset * 8)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((8, 64), T.float16)
            T.copy(src[0:8, 0:64], shared, prefer_instruction="tma")

    return main


def _device_local_1d_tma_copy_program():
    @T.prim_func
    def main(src_handle: T.handle):
        src = T.match_buffer(src_handle, (1024,), T.float16)
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((1024,), T.float16)
            src_view = T.decl_buffer((1024,), T.float16, data=src.data, elem_offset=tx * 8)
            T.copy(src_view[0:1024], shared, prefer_instruction="tma")

    return main


def _unaligned_region_1d_tma_copy_program():
    @T.prim_func
    def main(src_handle: T.handle):
        src = T.match_buffer(src_handle, (1025,), T.float16)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((1024,), T.float16)
            T.copy(src[1:1025], shared, prefer_instruction="tma")

    return main


def _aligned_effective_region_1d_tma_copy_program():
    @T.prim_func
    def main(src_handle: T.handle):
        src = T.match_buffer(src_handle, (1031,), T.float16, elem_offset=1)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((1024,), T.float16)
            T.copy(src[7:1031], shared, prefer_instruction="tma")

    return main


def _fp4_unpacked_copy_program(elem_offset=32):
    @T.prim_func
    def main(src_handle: T.handle):
        src = T.match_buffer(src_handle, (16, 80), T.float4_e2m1fn, elem_offset=elem_offset)
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((8, 64), T.float4_e2m1_unpacked)
            mbar = T.alloc_barrier(128)
            T.tma_copy(src[0:8, 0:64], shared, barrier=mbar)

    return main


@pytest.mark.parametrize(
    ("program", "arch", "expected_offsets"),
    [
        (_bulk_copy_program, "sm_90", [128, 256]),
        (_gather_scatter_program, "sm_100a", [128, 256]),
        (_im2col_program, "sm_90", [128]),
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

        cutedsl_wrapper = object.__new__(TLCuTeDSLSourceWrapper)
        cutedsl_wrapper.tma_descriptor_args = {desc_var: args}
        cutedsl_desc_names = cutedsl_wrapper.generate_tma_descriptor_args(desc_name_map, desc_name_var_map, {})
        cutedsl_tensors, cutedsl_tensor_map = cutedsl_wrapper._process_tma_descriptors(cutedsl_desc_names)
        cutedsl_init = cutedsl_wrapper._generate_tma_init_func(cutedsl_desc_names, cutedsl_tensors, cutedsl_tensor_map, [])
        assert cutedsl_tensors == [address.args[0].name]
        assert f"{address.args[0].name}_ptr + (" in cutedsl_init
        assert "handle_add_byte_offset" not in cutedsl_init


def test_swizzled_tma_copy_rejects_sub_128_byte_global_offset():
    with pytest.raises(Exception, match=re.escape("T.tma_copy")):
        _descriptor_base_byte_offsets(_bulk_copy_program(16, 32), "sm_90")


def test_linear_tma_copy_keeps_dtype_alignment_requirement():
    offsets, descriptors = _descriptor_base_byte_offsets(_bulk_copy_program(16, 32, annotate_linear=True), "sm_90")
    assert offsets == [32, 64]
    assert len(descriptors) == 2


def test_cutedsl_descriptor_aliases_are_kernel_local():
    wrapper = object.__new__(TLCuTeDSLSourceWrapper)
    wrapper.target = None
    wrapper.pdl_sync_map = {}
    wrapper.use_cooperative_groups = {}

    kernel_metadata = []
    for kernel_index in range(2):
        alias = f"__tma_{kernel_index}_A_desc"
        kernel_metadata.append(
            {
                "function_name": f"kernel_{kernel_index}",
                "call_args": [("A", "buffer"), ("A_desc", "None")],
                "desc_names": [alias],
                "descriptor_aliases": {"A_desc": alias},
                "function_info": {
                    "grid_info": [1, 1, 1],
                    "block_info": [32, 1, 1],
                    "dynamic_smem_buf": 0,
                },
            }
        )

    launch_0 = wrapper._generate_kernel_launch(kernel_metadata[0], 0, ["__tma_0_A_desc", "__tma_1_A_desc"])
    launch_1 = wrapper._generate_kernel_launch(kernel_metadata[1], 1, ["__tma_0_A_desc", "__tma_1_A_desc"])
    assert "&tma_descs[0]" in launch_0
    assert "&tma_descs[1]" in launch_1

    descriptor_info = {
        "is_img2col": False,
        "dtype": 0,
        "global_dim": [32, 32],
        "global_stride": [1, 32],
        "box_dim": [16, 16],
        "element_strides": [1, 1],
        "interleave": 0,
        "swizzle": 0,
        "l2Promotion": 0,
        "oobFill": 0,
    }
    wrapper.tma_desc_info = {
        "__tma_0_A_desc": descriptor_info,
        "__tma_1_A_desc": descriptor_info,
    }
    cubin = wrapper._generate_cubin_gen_code(
        kernel_metadata,
        [{"name": "A", "type": "buffer", "dtype": "cutlass.Float16"}],
        ["A"],
        ["__tma_0_A_desc", "__tma_1_A_desc"],
        ["A"],
        {
            "__tma_0_A_desc": ("A", 0),
            "__tma_1_A_desc": ("A", 1),
        },
        True,
        "def kernel_0(A, A_desc):\n  pass\ndef kernel_1(A, A_desc):\n  pass",
    )
    assert cubin.count("A_desc: cutlass.GridConstant[cuda.TensorMap]") == 2
    assert "__tma_0_A_desc = cuda.create_tensor_map_tiled" in cubin
    assert "__tma_1_A_desc = cuda.create_tensor_map_tiled" in cubin
    assert "kernel_0(A_, __tma_0_A_desc)" in cubin
    assert "kernel_1(A_, __tma_1_A_desc)" in cubin


@pytest.mark.parametrize("wrapper_cls", [TLCUDASourceWrapper, TLNVRTCSourceWrapper])
def test_cuda_wrapper_descriptor_aliases_are_kernel_local(wrapper_cls):
    func = _bulk_copy_program()
    _, descriptors = _descriptor_base_byte_offsets(func, "sm_90")
    first_desc_var, first_desc_args = descriptors[0]
    _, second_desc_args = descriptors[1]
    second_desc_var = tvm.tirx.Var("src_desc_1", "handle")
    module = tvm.IRModule.from_expr(func)
    src_data = func.buffer_map[func.params[0]].data

    function_informations = {
        "kernel_0": {
            "block_info": [1, 1, 1],
            "grid_info": [1, 1, 1],
            "dynamic_smem_buf": 0,
            "function_params": [src_data, first_desc_var],
        },
        "kernel_1": {
            "block_info": [1, 1, 1],
            "grid_info": [1, 1, 1],
            "dynamic_smem_buf": 0,
            "function_params": [src_data, second_desc_var],
        },
    }
    code = (
        "__global__ void kernel_0(half_t* src, CUtensorMap src_desc) { int x; }\n"
        "__global__ void kernel_1(half_t* src, CUtensorMap src_desc) { int x; }"
    )

    wrapper = object.__new__(wrapper_cls)
    wrapper.mod = module
    wrapper.tma_descriptor_args = {
        first_desc_var: first_desc_args,
        second_desc_var: second_desc_args,
    }
    wrapper.l2_persistent_map = {}
    if wrapper_cls is TLCUDASourceWrapper:
        wrapper.use_cooperative_groups = {"kernel_0": False, "kernel_1": False}
        wrapper.cluster_dims = {"kernel_0": None, "kernel_1": None}
    else:
        wrapper.pdl_sync_map = {}

    generated = wrapper.create_dispatch_func(code, function_informations)
    assert "__tma_0_src_desc" in generated
    assert "__tma_1_src_desc" in generated
    if wrapper_cls is TLCUDASourceWrapper:
        assert "cudaLaunchKernelEx(&config, kernel_0, src, __tma_0_src_desc)" in generated
        assert "cudaLaunchKernelEx(&config, kernel_1, src, __tma_1_src_desc)" in generated
    else:
        assert "arg_values = src.data_ptr(), __tma_0_src_desc" in generated
        assert "arg_values = src.data_ptr(), __tma_1_src_desc" in generated


@pytest.mark.parametrize("elem_offset", [1, "symbolic"])
def test_auto_copy_falls_back_for_unproven_descriptor_alignment(elem_offset):
    offsets, descriptors = _descriptor_base_byte_offsets(_auto_copy_program(elem_offset), "sm_90")
    assert offsets == []
    assert descriptors == []


def test_preferred_tma_lowers_symbolically_aligned_host_offset():
    offsets, descriptors = _descriptor_base_byte_offsets(_symbolically_aligned_tma_copy_program(), "sm_90")
    assert len(descriptors) == 1
    assert len(offsets) == 1
    assert not isinstance(offsets[0], int)


def test_cutedsl_initializer_threads_symbolic_tma_address_offset():
    func = _symbolically_aligned_tma_copy_program()
    _, descriptors = _descriptor_base_byte_offsets(func, "sm_90")
    desc_var, desc_args = descriptors[0]
    address = desc_args[4]

    wrapper = object.__new__(TLCuTeDSLSourceWrapper)
    wrapper.tma_descriptor_args = {desc_var: desc_args}
    wrapper.tma_desc_info = {}
    wrapper.pdl_sync_map = {}
    wrapper.use_cooperative_groups = {}
    desc_name_map = {desc_var.name: address.args[0].name}
    desc_name_var_map = {desc_var.name: desc_var}
    desc_names = wrapper.generate_tma_descriptor_args(desc_name_map, desc_name_var_map, {})
    tensors, tensor_arg_map = wrapper._process_tma_descriptors(desc_names)

    generated = wrapper._generate_cpp_launcher(
        [
            {
                "function_name": "main",
                "function_info": {
                    "grid_info": [1, 1, 1],
                    "block_info": [128, 1, 1],
                    "dynamic_smem_buf": 0,
                },
                "call_args": [("src", "buffer"), ("src_desc", "None"), ("offset", "cutlass.Int32")],
                "desc_names": desc_names,
            }
        ],
        [
            {"name": "src", "type": "buffer", "dtype": "cutlass.Float16"},
            {"name": "offset", "type": "cutlass.Int32"},
        ],
        tensors,
        desc_names,
        tensor_arg_map,
    )

    assert "CUresult tma_init(CUtensorMap* tma_descs, uint64_t src_ptr, int32_t offset)" in generated
    assert "reinterpret_cast<void*>((src_ptr + ((int64_t)offset * 16)))" in generated
    assert "result = tma_init(tma_descs, src_ptr, offset);" in generated


def test_preferred_tma_keeps_device_local_1d_copy_descriptorless():
    offsets, descriptors = _descriptor_base_byte_offsets(_device_local_1d_tma_copy_program(), "sm_90")
    assert offsets == []
    assert descriptors == []


def test_unaligned_region_start_does_not_use_descriptorless_1d_tma():
    offsets, descriptors = _descriptor_base_byte_offsets(_unaligned_region_1d_tma_copy_program(), "sm_90")
    assert offsets == [0]
    assert len(descriptors) == 1


def test_aligned_effective_region_start_uses_descriptorless_1d_tma():
    offsets, descriptors = _descriptor_base_byte_offsets(_aligned_effective_region_1d_tma_copy_program(), "sm_90")
    assert offsets == []
    assert descriptors == []


def test_fp4_unpacked_copy_rejects_16_byte_offset():
    with pytest.raises(Exception, match=re.escape("T.tma_copy")):
        _descriptor_base_byte_offsets(_fp4_unpacked_copy_program(), "sm_100a")


def test_fp4_unpacked_copy_rejects_subbyte_offset():
    with pytest.raises(Exception, match=re.escape("T.tma_copy")):
        _descriptor_base_byte_offsets(_fp4_unpacked_copy_program(63), "sm_100a")
