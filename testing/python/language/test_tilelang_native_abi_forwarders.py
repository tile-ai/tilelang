"""Public ABI spellings execute with native typed pointers and packed calls."""

import ctypes
import inspect
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
import tvm_ffi.cpp
from tvm_tirx import codegen, ir, libinfo, runtime, tirx
from tvm_tirx.target import Target
from tvm_tirx.tirx.compilation_pipeline import finalize_host_passes

import tilelang
import tilelang.language as T
from tilelang.env import TILELANG_TEMPLATE_PATH


VOID = ir.TupleType([])
HANDLE = ir.PtrType("void")
# Exercise the enabled TileLang C host pipeline, including its packed binder.
BACKENDS = ["tilelang_c"]


@pytest.fixture(scope="module")
def descriptor_probe(tmp_path_factory):
    root = tmp_path_factory.mktemp("descriptor_probe")
    source = Path(__file__).resolve().parents[2] / "cpp/native_descriptor_probe.cc"
    runtime_lib = libinfo.find_libtvm_runtime()
    return tvm_ffi.cpp.load(
        "abi154_descriptor_probe",
        sources=[str(source)],
        extra_cflags=["-std=c++20"],
        extra_include_paths=[libinfo.find_include_path()],
        extra_ldflags=[runtime_lib, "-Wl,-rpath," + str(Path(runtime_lib).parent)],
        build_directory=str(root),
    )


def _function(name, params, body):
    return tirx.Function(params, [*body, ir.Return(ir.Tuple([]))], VOID, ir.DictAttrs({"global_symbol": name}))


def _build(function, backend, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = Target({"kind": "llvm", "mcpu": "generic"} if backend == "llvm" else "c")
    function = function.with_attr("target", target.with_host(target))
    assert tirx.analysis.check_well_formed(function)
    before = ir.IRModule({function.attrs["global_symbol"]: function})
    lowered = tilelang.transform.MakePackedAPI()(before)
    lowered = finalize_host_passes()(lowered)
    assert tirx.analysis.check_well_formed(lowered)
    (directory / "input.json").write_text(ir.save_json(before))
    (directory / "lowered.json").write_text(ir.save_json(lowered))
    key = "target.build.llvm" if backend == "llvm" else "target.build.c" if backend == "native_c" else "target.build.tilelang_c_host"
    module = tvm_ffi.get_global_func(key)(lowered, target)
    text = module.inspect_source("ll" if backend == "llvm" else "c")
    (directory / ("generated.ll" if backend == "llvm" else "generated.c")).write_text(text)
    library = directory / "entry.so"
    # Link the selected runtime explicitly, making helper resolution independent
    # of pre-existing global symbol visibility in the test process.
    options = ["-std=c++17", "-I" + libinfo.find_include_path(), "-I" + TILELANG_TEMPLATE_PATH]
    if os.environ.get("TL_TEMPLATE_PATH"):
        options.append("-I" + os.environ["TL_TEMPLATE_PATH"])
    runtime_lib = libinfo.find_libtvm_runtime()
    options += [runtime_lib, "-Wl,-rpath," + str(Path(runtime_lib).parent)]
    codegen.Executable(module).export_library(str(library), workspace_dir=str(directory), cc="c++", options=options)
    loaded = runtime.load_module(str(library))
    return loaded[function.attrs["global_symbol"]], library, lowered


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("tvm_stack_alloca", "dtype_str,num"),
        ("tvm_stack_make_shape", "args"),
        ("tvm_stack_make_array", "data,shape,strides,ndim,arr_dtype,elem_offset"),
        ("tvm_throw_last_error", ""),
        ("TVMBackendAllocWorkspace", "device_type,device_id,nbytes,dtype_code_hint,dtype_bits_hint"),
        ("TVMBackendFreeWorkspace", "device_type,device_id,ptr"),
    ],
)
def test_public_signatures(name, parameters):
    assert ",".join(inspect.signature(getattr(T, name)).parameters) == parameters


def test_native_pointer_and_effect_contracts():
    handle = ir.Var("handle", HANDLE)
    calls = [
        (T.tvm_stack_alloca("shape", 3), "tirx.stack_alloca", ir.PtrType("int64")),
        (T.tvm_stack_alloca("tvm_ffi_any", 2), "tirx.stack_alloca", HANDLE),
        (T.tvm_stack_make_shape(2, 3), "tirx.stack_make_shape", ir.PtrType("int64")),
        (T.tvm_stack_make_array(handle, handle, 0, 1, T.float32(0), 1), "tirx.stack_make_dltensor", HANDLE),
        (T.tvm_throw_last_error(), "tirx.throw_last_error", ir.PrimType("void")),
        (T.TVMBackendAllocWorkspace(1, 0, 128, 2, 32), "tirx.alloc_workspace", HANDLE),
        (T.TVMBackendFreeWorkspace(1, 0, handle), "tirx.free_workspace", ir.PrimType("int32")),
    ]
    for call, name, ty in calls:
        assert call.op.name == name and call.ty == ty
        assert len(call.ty_args) == 0
        call.validate()
        assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(call)), call, map_free_vars=True)
        effect = ir.CallEffectKind.Opaque
        assert int(call.op.get_attr("TCallEffectKind")) == int(effect)


@pytest.mark.parametrize("backend", BACKENDS)
def test_stack_descriptor_offset_strides_and_lifetime(backend, tmp_path):
    handle = ir.Var("data", HANDLE)
    descriptor = T.tvm_stack_make_array(handle, T.tvm_stack_make_shape(2, 3), T.tvm_stack_make_shape(5, 1), 2, T.float32(0), 1)
    captured = []

    def consume(value):
        # Tensor.numpy() intentionally requires compact storage. Gather through
        # the actual descriptor's strides with a separate address oracle.
        assert value.shape == (2, 3) and value.strides == (5, 1)
        assert value.data_ptr() == data.ctypes.data + 4
        memory = (ctypes.c_float * 8).from_address(value.data_ptr())
        captured.append([[memory[i * value.strides[0] + j * value.strides[1]] for j in range(3)] for i in range(2)])

    tvm_ffi.register_global_func("tl.testing.abi106.descriptor", consume, override=True)
    context = ir.RegionStmt(
        "tirx.device_context",
        [ir.const(1), ir.const(0)],
        [],
        None,
        [ir.Evaluate(T.call_packed("tl.testing.abi106.descriptor", descriptor))],
        [],
    )
    function = _function("descriptor", [handle], [context])
    run, _, _ = _build(function, backend, tmp_path / "descriptor")
    data = np.arange(20, dtype="float32")
    run(ctypes.c_void_p(data.ctypes.data))
    np.testing.assert_array_equal(captured, [[[1, 2, 3], [6, 7, 8]]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_workspace_allocate_write_read_free_and_error(backend, tmp_path):
    @T.prim_func
    def workspace():
        ptr = T.TVMBackendAllocWorkspace(1, 0, T.uint64(64), 2, 32)
        if T.isnullptr(ptr):
            T.tvm_throw_last_error()
        view = T.decl_buffer((4,), "float32", data=ptr)
        view[2] = T.float32(17.5)
        T.call_packed("tl.testing.abi106.workspace_value", view[2])
        T.evaluate(T.TVMBackendFreeWorkspace(1, 0, ptr))

    observed = []
    tvm_ffi.register_global_func("tl.testing.abi106.workspace_value", observed.append, override=True)
    run, _, _ = _build(workspace, backend, tmp_path / "workspace")
    run()
    assert observed == [17.5]


@pytest.mark.parametrize("backend", BACKENDS)
def test_public_stack_alloca_can_be_written_and_read(backend, tmp_path):
    @T.prim_func
    def stack_values():
        storage = T.tvm_stack_alloca("shape", 2)
        view = T.decl_buffer((2,), "int64", data=storage)
        view[0] = T.int64(29)
        view[1] = T.int64(-17)
        T.call_packed("tl.testing.abi106.stack_values", view[0], view[1])

    observed = []
    tvm_ffi.register_global_func("tl.testing.abi106.stack_values", lambda a, b: observed.append((a, b)), override=True)
    run, _, _ = _build(stack_values, backend, tmp_path / "stack")
    run()
    assert observed == [(29, -17)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_throw_last_error_propagates_workspace_failure(backend, tmp_path):
    @T.prim_func
    def invalid_workspace():
        ptr = T.TVMBackendAllocWorkspace(12, 0, T.uint64(64), 2, 32)
        if T.isnullptr(ptr):
            T.tvm_throw_last_error()
        T.call_packed("tl.testing.abi106.must_not_run")

    called = []
    tvm_ffi.register_global_func("tl.testing.abi106.must_not_run", lambda: called.append(True), override=True)
    run, _, _ = _build(invalid_workspace, backend, tmp_path / "throw")
    with pytest.raises((ValueError, RuntimeError)):
        run()
    assert called == []


DESCRIPTOR_CASES = [
    pytest.param((), None, 3, id="rank0"),
    pytest.param((2, 3), (3, 1), 0, id="explicit-compact"),
    pytest.param((2, 3), (5, 1), 1, id="explicit-noncompact"),
    pytest.param((2, 3), None, 2, id="implicit-compact-offset"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape,strides,offset", DESCRIPTOR_CASES)
def test_stack_descriptor_raw_fields(descriptor_probe, backend, shape, strides, offset, tmp_path):
    # Direct packed-call operands are consumed while their descriptor is live.
    # Nested packed-call return expressions have separate legacy limitations.
    extension = descriptor_probe
    tvm_ffi.register_global_func("tl.testing.abi154.observe", extension.abi154_observe, override=True)
    handle = ir.Var("data", HANDLE)
    descriptor = T.tvm_stack_make_array(
        handle,
        T.tvm_stack_make_shape(*shape),
        0 if strides is None else T.tvm_stack_make_shape(*strides),
        len(shape),
        T.float32(0),
        offset,
    )
    call = T.call_packed("tl.testing.abi154.observe", descriptor, 0)
    context = ir.RegionStmt("tirx.device_context", [ir.const(1), ir.const(0)], [], None, [ir.Evaluate(call)], [])
    run, _, _ = _build(_function("descriptor_direct", [handle], [context]), backend, tmp_path / "descriptor_direct")
    input_data = np.arange(64, dtype="float32")
    observations = []
    for increment in [0, 100]:
        input_data[:] = np.arange(64, dtype="float32") + increment
        extension.abi154_clear()
        run(ctypes.c_void_p(input_data.ctypes.data))
        records = list(extension.abi154_records())
        raw_records = [
            [int(value) for value in record[:12]] + [list(record[12]), None if record[13] is None else list(record[13]), list(record[14])]
            for record in records
        ]
        (tmp_path / f"raw-descriptor-observations-{increment}.json").write_text(json.dumps(raw_records, indent=2))
        assert len(records) == 1
        record = records[0]
        header = [int(value) for value in record[:12]]
        actual_shape = tuple(record[12])
        actual_steps = None if record[13] is None else tuple(record[13])
        actual_values = list(record[14])
        assert header[0] == 0 and header[1] != 0 and header[2] == input_data.ctypes.data
        assert header[5:12] == [len(shape), 1, 0, 2, 32, 1, offset * 4]
        assert actual_shape == shape
        assert bool(header[4]) == (strides is not None)
        assert actual_steps == strides
        if shape:
            assert header[3] != 0
        effective_steps = strides
        if effective_steps is None:
            effective_steps = tuple(int(np.prod(shape[i + 1 :])) for i in range(len(shape)))
        expected = [
            float(input_data[offset + sum(index * stride for index, stride in zip(indices, effective_steps, strict=True))])
            for indices in itertools.product(*(range(dim) for dim in shape))
        ]
        assert actual_values == expected
        observations.append({"input_increment": increment, "header": header, "shape": shape, "strides": strides, "values": actual_values})
    (tmp_path / "descriptor-observations.json").write_text(json.dumps(observations, indent=2))
