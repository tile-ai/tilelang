"""Ordinary CUDA vectors honor actual view addresses and physical carrier sizes."""

from functools import lru_cache
import ctypes
import json
import shutil
import subprocess

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, prim, target, tirx

import tilelang
import tilelang.language as T
import tilelang.testing


def _raw_function(dtype="float32", lanes=8, alignment=1, base=0, *, volatile=False, vector_element=False, scope="global"):
    element = f"{dtype}x{lanes}" if vector_element else dtype
    source_ptr = ir.Var("source_ptr", ir.PtrType(ir.PrimType(element), scope))
    target_ptr = ir.Var("target_ptr", ir.PtrType(ir.PrimType(element), scope))
    ty = tirx.TensorType((256,), element, scope=scope, data_alignment=alignment)
    source, output = ir.Var("source", ty), ir.Var("output", ty)
    attributes = ir.DictAttrs({"tirx.volatile": True}) if volatile else None
    if isinstance(base, int):
        base = ir.const(base, "int32")
    index = base if vector_element else prim.Ramp(base, 1, lanes)
    statements = [
        ir.Bind(source, ir.Call("tirx.decl_tensor", [source_ptr], ty_args=[ty], attrs=attributes)),
        ir.Bind(output, ir.Call("tirx.decl_tensor", [target_ptr], ty_args=[ty], attrs=attributes)),
        ir.TensorStore(output, [index], ir.TensorLoad(source, [index])),
    ]
    function = tirx.Function([source_ptr, target_ptr], statements, ret_type=ir.PrimType("void"))
    function = function.with_attr("global_symbol", "vector_copy").with_attr("calling_conv", ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    assert tirx.analysis.check_well_formed(function)
    return function


def _raw_source(*args, **kwargs):
    function = _raw_function(*args, **kwargs)
    original = ir.save_json(function)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        ir.IRModule({"vector_copy": function}), target.Target({"kind": "cuda", "arch": "sm_100a"})
    )
    assert ir.save_json(function) == original
    return module.inspect_source("cuda")


@pytest.mark.parametrize("lanes", [2, 4, 8])
@pytest.mark.parametrize("alignment,base,guard", [(1, 0, True), (64, 0, False), (64, 8, False), (64, 1, True)])
def test_static_guarantee_needs_both_backing_and_offset_alignment(lanes, alignment, base, guard):
    source = _raw_source(lanes=lanes, alignment=alignment, base=base)
    assert ("tl::load_vector<" in source) == guard
    assert ("tl::store_vector<" in source) == guard
    if not guard and lanes == 8:
        assert "tl::load_global_256(" in source and "tl::store_global_256(" in source


@pytest.mark.parametrize("vector_element", [False, True])
def test_volatile_and_local_views_do_not_borrow_an_unestablished_alignment(vector_element):
    volatile = _raw_source(alignment=64, volatile=True, vector_element=vector_element)
    assert "tl::load_vector<" in volatile
    assert "const volatile unsigned char*" in volatile
    assert "volatile unsigned char*" in volatile
    local = _raw_source(alignment=64, scope="local", vector_element=vector_element)
    assert "tl::load_vector<" in local and "tl::store_vector<" in local


def test_effectful_base_is_evaluated_once_per_memory_operation():
    source = _raw_source(base=tirx.call_extern("next_index", ty="int32"))
    assert source.count("next_index()") == 2


def _nvcc(source, tmp_path, *, ptx=False):
    from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compilation requires nvcc")
    source_file = tmp_path / "vector.cu"
    output_file = tmp_path / ("vector.ptx" if ptx else "vector.cubin")
    source_file.write_text(source)
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "--ptx" if ptx else "--cubin",
        "-arch=sm_100a",
        f"-I{TILELANG_TEMPLATE_PATH}",
        f"-I{CUTLASS_INCLUDE_DIR}",
        str(source_file),
        "-o",
        str(output_file),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (tmp_path / "nvcc.json").write_text(json.dumps({"command": command, "exit_code": result.returncode}, indent=2))
    (tmp_path / "nvcc.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return output_file


@pytest.mark.parametrize(
    "dtype,lanes",
    [
        ("float32", 2),
        ("float32", 3),
        ("float32", 4),
        ("float32", 8),
        ("float16", 16),
        ("bfloat16", 16),
        ("int4", 64),
        ("uint4", 64),
        ("float4_e2m1fn", 64),
        ("float8_e4m3fn", 32),
        ("float6_e2m3fn", 4),
        ("bool", 2),
        ("bool", 3),
        ("bool", 4),
    ],
)
@pytest.mark.parametrize("volatile", [False, True])
def test_cuda_accepts_exact_carriers_and_volatile_accesses(dtype, lanes, volatile, tmp_path):
    _nvcc(_raw_source(dtype, lanes, volatile=volatile), tmp_path)


def test_nvcc_retains_byte_slow_path_and_aligned_256_bit_instruction(tmp_path):
    ptx = _nvcc(_raw_source(), tmp_path, ptx=True).read_text()
    assert "ld.global.v4.u64" in ptx and "st.global.v4.u64" in ptx
    assert "ld.volatile" in ptx and "st.volatile" in ptx
    assert ".b8" in ptx or ".u8" in ptx


@lru_cache(None)
def _copy_kernel(dtype, extent):
    @T.prim_func
    def main(A: T.Tensor((extent,), dtype), B: T.Tensor((extent,), dtype)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(extent):
                B[i] = A[i]

    return tilelang.compile(main, target="cuda")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype,extent",
    [
        ("float32", 256),
        ("float32", 512),
        ("float32", 1024),
        ("float16", 2048),
        ("bfloat16", 2048),
        ("float64", 512),
        ("int8", 4096),
        ("uint8", 4096),
        ("int16", 1024),
        ("int32", 1024),
        ("int64", 512),
        ("float8_e4m3fn", 4096),
        ("bool", 512),
    ],
)
def test_tilelang_parallel_preserves_bits_and_sentinels_in_sliced_storage(dtype, extent, tmp_path):
    kernel = _copy_kernel(dtype, extent)
    source_code = kernel.get_kernel_source()
    (tmp_path / "kernel.cu").write_text(source_code)
    helper = "tl::load_bool_vector<" if dtype == "bool" else "tl::load_vector<"
    assert helper in source_code
    torch_dtype = getattr(torch, dtype)
    source_storage = torch.empty(extent + 64, dtype=torch_dtype, device="cuda")
    target_storage = torch.empty_like(source_storage)
    pattern = (torch.arange(source_storage.numel() * source_storage.element_size(), device="cuda", dtype=torch.int64) * 17 + 39).to(
        torch.uint8
    )
    if dtype == "bool":
        pattern &= 1
    source_storage.view(torch.uint8).copy_(pattern)
    for source_offset, target_offset in [(0, 0), (1, 3), (3, 1), (7, 9)]:
        target_storage.view(torch.uint8).fill_(0xA5)
        source = source_storage[source_offset : source_offset + extent]
        output = target_storage[target_offset : target_offset + extent]
        kernel(source, output)
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.uint8), source.view(torch.uint8))
        expected = torch.full_like(target_storage.view(torch.uint8), 0xA5)
        begin = target_offset * output.element_size()
        expected[begin : begin + extent * output.element_size()] = source.view(torch.uint8)
        assert torch.equal(target_storage.view(torch.uint8), expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype,lanes,vector_element",
    [
        ("float32", 2, False),
        ("float32", 4, False),
        ("float32", 8, False),
        ("float32", 4, True),
        ("float32", 8, True),
        ("int4", 64, False),
        ("uint4", 64, False),
        ("float4_e2m1fn", 64, False),
        ("float8_e4m3fn", 32, False),
        ("float6_e2m3fn", 4, False),
        ("bool", 4, False),
    ],
)
@pytest.mark.parametrize("volatile", [False, True])
@tilelang.testing.requires_cuda_compute_version_eq(10, 0)
def test_native_emitter_copies_exact_bytes_at_every_address_residue(dtype, lanes, vector_element, volatile, tmp_path):
    from tilelang.backend.module import get_backend

    get_backend("cuda")
    torch.cuda.set_device(0)
    function = _raw_function(dtype, lanes, volatile=volatile, vector_element=vector_element)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda")(
        ir.IRModule({"vector_copy": function}), target.Target({"kind": "cuda", "arch": "sm_100a"})
    )
    (tmp_path / "kernel.cu").write_text(module.inspect_source("cuda"))
    byte_count = lanes if dtype in ("bool", "float6_e2m3fn") else (tvm_ffi.dtype(dtype).bits * lanes) // 8
    storage_size = 256 * 32 + 64
    source = ((torch.arange(storage_size, device="cuda", dtype=torch.int64) * 17 + 39) & 255).to(torch.uint8)
    if dtype == "bool":
        source &= 1
    if dtype == "float6_e2m3fn":
        source &= 63
    output = torch.empty_like(source)
    # Input and output residues differ, so a guard on only one address cannot
    # accidentally pass this oracle. Include every residue modulo 32.
    for source_offset in range(32):
        output_offset = (source_offset * 7 + 3) % 32
        output.fill_(0xA5)
        module["vector_copy"](ctypes.c_void_p(source.data_ptr() + source_offset), ctypes.c_void_p(output.data_ptr() + output_offset))
        torch.cuda.synchronize()
        expected = torch.full_like(output, 0xA5)
        expected[output_offset : output_offset + byte_count] = source[source_offset : source_offset + byte_count]
        assert torch.equal(output, expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("volatile", [False, True])
@tilelang.testing.requires_cuda_compute_version_eq(10, 0)
def test_shared_views_use_the_actual_byte_address(volatile, tmp_path):
    from tilelang.backend.module import get_backend

    get_backend("cuda")
    torch.cuda.set_device(0)
    dtype = ir.PrimType("float32")
    source_ptr = ir.Var("source_ptr", ir.PtrType(dtype))
    output_ptr = ir.Var("output_ptr", ir.PtrType(dtype))
    offset = ir.Var("offset", "int32")
    global_ty = tirx.TensorType((8,), dtype, data_alignment=1)
    source, output = ir.Var("source", global_ty), ir.Var("output", global_ty)
    owner = ir.Var("storage", tirx.TensorType((128,), "uint8", scope="shared", data_alignment=64))
    view_ty = tirx.TensorType((8,), dtype, scope="shared", data_alignment=1)
    view = ir.Var("view", view_ty)
    attributes = ir.DictAttrs({"tirx.volatile": True}) if volatile else None
    pointer = tirx.ptr_byte_offset(tirx.tensor_data_ptr(owner), offset, ty=ir.PtrType(dtype, "shared"))
    index = prim.Ramp(ir.const(0, "int32"), 1, 8)
    statements = [
        ir.Bind(source, ir.Call("tirx.decl_tensor", [source_ptr], ty_args=[global_ty])),
        ir.Bind(output, ir.Call("tirx.decl_tensor", [output_ptr], ty_args=[global_ty])),
        ir.Bind(owner, ir.Call("tirx.alloc_tensor", [], ty_args=[owner.ty], attrs=attributes)),
        ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view_ty])),
        ir.TensorStore(view, [index], ir.TensorLoad(source, [index])),
        ir.TensorStore(output, [index], ir.TensorLoad(view, [index])),
    ]
    function = tirx.Function([source_ptr, output_ptr, offset], statements, ret_type=ir.PrimType("void"))
    function = function.with_attr("global_symbol", "shared_copy").with_attr("calling_conv", ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    assert tirx.analysis.check_well_formed(function)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda")(
        ir.IRModule({"shared_copy": function}), target.Target({"kind": "cuda", "arch": "sm_100a"})
    )
    source_code = module.inspect_source("cuda")
    (tmp_path / "kernel.cu").write_text(source_code)
    assert "tl::store_vector<ulonglong4, 32, false>" in source_code
    assert "tl::load_vector<ulonglong4, 32, false>" in source_code
    source = (torch.arange(128, device="cuda", dtype=torch.int32) * 17 + 39).to(torch.uint8)
    output = torch.empty_like(source)
    for byte_offset in range(32):
        output.fill_(0xA5)
        module["shared_copy"](ctypes.c_void_p(source.data_ptr() + 1), ctypes.c_void_p(output.data_ptr() + 3), byte_offset)
        torch.cuda.synchronize()
        expected = torch.full_like(output, 0xA5)
        expected[3:35] = source[1:33]
        assert torch.equal(output, expected)
