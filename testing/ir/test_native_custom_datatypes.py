"""Custom host storage keeps value, definition/use, callback, and packed ABI contracts."""

import ctypes
from contextlib import contextmanager

import pytest
import tilelang
import tilelang.testing
import tvm_ffi
import tvm_tirx
from tilelang.backend.module import create_backend_context
from tilelang.engine.lower import host_codegen
from tilelang.testing.ir import collect_nodes
from tvm_tirx import ir, tirx
from tvm_tirx.codegen import Executable


DTYPES = [("custom[tfloat32]32", 32), ("custom[float4_e2m1_unpacked]8", 8)]
HOST = tvm_tirx.target.Target("c")
LLVM_HOST = tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"})


def _function(dtype, variant="copy", target=HOST):
    ap, bp = ir.Var("a", ir.PtrType("void")), ir.Var("b", ir.PtrType("void"))
    a = ir.Var("a_tensor", tirx.TensorType([4], dtype))
    b = ir.Var("b_tensor", tirx.TensorType([4], dtype))
    body = [
        ir.Bind(a, ir.Call("tirx.decl_tensor", [ap], ty_args=[a.ty])),
        ir.Bind(b, ir.Call("tirx.decl_tensor", [bp], ty_args=[b.ty])),
    ]
    i = ir.Var("i", "int32")
    value = ir.TensorLoad(a, [i])
    if variant == "select":
        value = ir.prim.Select(i < 2, value, ir.TensorLoad(a, [3 - i]))
    elif variant == "add":
        value = ir.prim.Add(value, ir.TensorLoad(a, [1]))
    elif variant == "let":
        local = ir.Var("selected", dtype)
        value = ir.prim.Let(local, value, local)
    if variant == "local":
        scratch = ir.Var("scratch", tirx.TensorType([4], dtype, scope="global"))
        body += [
            ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty])),
            ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, ir.TensorStore(scratch, [i], value)),
            *[ir.TensorStore(b, [j], ir.TensorLoad(scratch, [j])) for j in range(4)],
        ]
    elif variant == "bind":
        local = ir.Var("selected", dtype)
        body += [ir.Bind(local, ir.TensorLoad(a, [3])), *[ir.TensorStore(b, [j], local) for j in range(4)]]
    elif variant in ("broadcast", "shuffle"):
        indices = ir.prim.Ramp(0, 1, 4)
        vector = (
            ir.prim.Broadcast(ir.TensorLoad(a, [3]), 4)
            if variant == "broadcast"
            else ir.prim.Shuffle([ir.TensorLoad(a, [indices])], [3, 2, 1, 0])
        )
        body.append(ir.TensorStore(b, [indices], vector))
    else:
        body.append(ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, ir.TensorStore(b, [i], value)))
    body.append(ir.Return(ir.const(0, "int32")))
    return tirx.Function([ap, bp], body, ir.PrimType("int32")).with_attr({"global_symbol": "main", "target": target})


def _host_module(function):
    module = ir.IRModule({"main": function})
    assert tirx.analysis.check_well_formed(module)
    target = function.attrs["target"]
    context = create_backend_context(target, target, "tvm_ffi")
    return host_codegen(module, context)


def _storage(bits):
    return (ctypes.c_uint32, [0, 1, 0xFFFFFFFF, 0x7F812345]) if bits == 32 else (ctypes.c_uint8, [0, 1, 255, 173])


def _check_native_containers(function):
    original = ir.save_json(function)
    module = ir.IRModule({"main": function})
    lowered = tilelang.transform.LowerCustomDatatypes()(module)
    assert tirx.analysis.check_well_formed(lowered)
    assert ir.save_json(function) == original
    assert tirx.analysis.check_well_formed(ir.load_json(ir.save_json(lowered)))
    primitive_values = [node for node in collect_nodes(lowered, ir.Expr) if isinstance(node.ty, ir.PrimType)]
    assert primitive_values and all(int(node.ty.dtype.type_code) < 129 for node in primitive_values)


def _check_storage_export(function, bits, variant, tmp_path):
    output = tmp_path / "custom-storage.so"
    Executable(_host_module(function)).export_library(str(output))
    loaded = ctypes.CDLL(str(output))
    entry = loaded.main
    entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    ctype, values = _storage(bits)
    source, destination = (ctype * 4)(*values), (ctype * 4)(*[37] * 4)
    assert entry(source, destination) == 0
    expected = {
        "select": [values[0], values[1], values[1], values[0]],
        "bind": [values[3]] * 4,
        "broadcast": [values[3]] * 4,
        "shuffle": values[::-1],
    }.get(variant, values)
    assert list(destination) == expected


@pytest.mark.parametrize("dtype,bits", DTYPES)
@pytest.mark.parametrize("variant", ["copy", "local", "select", "let", "bind"])
def test_custom_storage_survives_native_containers_and_export(dtype, bits, variant, tmp_path):
    function = _function(dtype, variant)
    _check_native_containers(function)
    _check_storage_export(function, bits, variant, tmp_path)


@pytest.mark.parametrize("dtype,bits", DTYPES)
@pytest.mark.parametrize("variant", ["broadcast", "shuffle"])
def test_custom_vector_storage_preserves_native_containers(dtype, bits, variant):
    # Check the shared lowering even in builds without a vector-capable host.
    _check_native_containers(_function(dtype, variant))


@pytest.mark.parametrize("dtype,bits", DTYPES)
@pytest.mark.parametrize("variant", ["broadcast", "shuffle"])
@tilelang.testing.requires_llvm
def test_custom_vector_storage_survives_llvm_export(dtype, bits, variant, tmp_path):
    # Raw host vectors use the LLVM emitter. The C packed-wrapper emitter has
    # no uint32_t4/uint8_t4 ABI; CPU device-vector tests exercise a different emitter.
    function = _function(dtype, variant, target=LLVM_HOST)
    _check_storage_export(function, bits, variant, tmp_path)


@contextmanager
def _callback(name, function):
    previous = tvm_ffi.get_global_func(name, allow_missing=True)
    tvm_ffi.register_global_func(name, function, override=True)
    try:
        yield
    finally:
        if previous is None:
            tvm_ffi.remove_global_func(name)
        else:
            tvm_ffi.register_global_func(name, previous, override=True)


def _custom_operation(operation):
    dtype = "custom[tfloat32]32"
    pointer = ir.Var("data", ir.PtrType("void"))
    tensor = ir.Var("data_tensor", tirx.TensorType([4], dtype))
    body = [ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty]))]
    value = operation(ir.TensorLoad(tensor, [0]), ir.TensorLoad(tensor, [1]))
    body += [ir.Evaluate(value), ir.Return(ir.const(0, "int32"))]
    return tirx.Function([pointer], body, ir.PrimType("int32")).with_attr({"global_symbol": "main", "target": HOST})


def test_custom_add_callback_gets_logical_result_and_physical_operands():
    observed = []

    def lower(op):
        observed.append((op.ty, op.a.ty, op.b.ty))
        return ir.prim.Add(op.a, op.b)

    function = _custom_operation(ir.prim.Add)
    with _callback("tvm.datatype.lower.c.Add.tfloat32", lower):
        lowered = tilelang.transform.LowerCustomDatatypes()(ir.IRModule({"main": function}))
    assert observed == [(ir.PrimType("custom[tfloat32]32"), ir.PrimType("uint32"), ir.PrimType("uint32"))]
    assert tirx.analysis.check_well_formed(lowered)
    assert str(collect_nodes(lowered, ir.prim.Add)[0].ty) == "uint32"


def test_custom_add_callback_executes_its_registered_arithmetic(tmp_path):
    function = _function("custom[tfloat32]32", "add")
    path = tmp_path / "custom-add.so"
    with _callback("tvm.datatype.lower.c.Add.tfloat32", lambda op: ir.prim.Add(op.a, op.b)):
        Executable(_host_module(function)).export_library(str(path))
    loaded = ctypes.CDLL(str(path))
    entry = loaded.main
    entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    ctype, values = _storage(32)
    source, destination = (ctype * 4)(*values), (ctype * 4)()
    assert entry(source, destination) == 0
    assert list(destination) == [(value + values[1]) & 0xFFFFFFFF for value in values]


def test_custom_cast_callback_can_define_arithmetic_conversion():
    observed = []

    def lower(op):
        observed.append((str(op.ty), str(op.value.ty)))
        return ir.prim.Cast("float32", op.value)

    function = _custom_operation(lambda a, b: ir.prim.Cast("float32", a))
    with _callback("tvm.datatype.lower.c.Cast.float.tfloat32", lower):
        lowered = tilelang.transform.LowerCustomDatatypes()(ir.IRModule({"main": function}))
    assert observed == [("float32", "uint32")]
    assert tirx.analysis.check_well_formed(lowered)


@pytest.mark.parametrize("dtype,bits", DTYPES)
def test_custom_cast_without_callback_keeps_the_old_error(dtype, bits):
    # Neither built-in storage tag defines a host numeric conversion. Missing
    # callbacks must stay errors rather than silently using integer storage.
    pointer = ir.Var("data", ir.PtrType("void"))
    tensor = ir.Var("data_tensor", tirx.TensorType([4], dtype))
    function = tirx.Function(
        [pointer],
        [
            ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty])),
            ir.Evaluate(ir.prim.Cast("float32", ir.TensorLoad(tensor, [0]))),
            ir.Return(ir.const(0, "int32")),
        ],
        ir.PrimType("int32"),
    ).with_attr({"global_symbol": "main", "target": HOST})
    with pytest.raises(tilelang.errors.InternalError, match="Cast lowering function.*not found"):
        _host_module(function)


@pytest.mark.parametrize("result", [lambda op: ir.const(1, "int32"), lambda op: 1])
def test_custom_callback_must_return_the_physical_native_type(result):
    function = _custom_operation(ir.prim.Add)
    with _callback("tvm.datatype.lower.c.Add.tfloat32", result), pytest.raises(TypeError, match="Custom datatype lowering callback"):
        tilelang.transform.LowerCustomDatatypes()(ir.IRModule({"main": function}))


class _Device(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("index", ctypes.c_int32)]


class _DType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int32),
        ("dtype", _DType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


@pytest.mark.parametrize("dtype,bits", DTYPES)
def test_physical_host_storage_keeps_packed_logical_dtype_checks(dtype, bits, tmp_path):
    a, b = [ir.Var(name, tirx.TensorType([4], dtype)) for name in ("input", "output")]
    function = tirx.Function([a, b], [ir.TensorStore(b, [j], ir.TensorLoad(a, [j])) for j in range(4)])
    target = tvm_tirx.target.Target(HOST, host=HOST)
    function = function.with_attr({"global_symbol": "main", "target": target})
    packed = tilelang.transform.MakePackedAPI()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(packed)
    # The pass must preserve explicit abi_field_get result types used by the
    # packed checks, even though those calls have no inference callback.
    typed_storage = tilelang.transform.LowerCustomDatatypes()(packed)
    assert tirx.analysis.check_well_formed(typed_storage)
    context = create_backend_context(target, HOST, "tvm_ffi")
    runtime = host_codegen(packed, context)
    output = tmp_path / "packed-custom.so"
    Executable(runtime).export_library(str(output))
    loaded = tvm_ffi.load_module(str(output))
    ctype, values = _storage(bits)
    backing = [(ctypes.c_uint8 * 128)() for _ in range(2)]
    pointers = [ctypes.addressof(owner) + (-ctypes.addressof(owner) % 64) for owner in backing]
    arrays = [(ctype * 4).from_address(pointer) for pointer in pointers]
    arrays[0][:] = values
    shape = (ctypes.c_int64 * 1)(4)
    logical = tvm_ffi.dtype(dtype)
    descriptors = [_DLTensor(pointer, _Device(1, 0), 1, _DType(logical.type_code, bits, 1), shape, None, 0) for pointer in pointers]
    arguments = [ctypes.c_void_p(ctypes.addressof(descriptor)) for descriptor in descriptors]
    loaded.main(*arguments)
    assert list(arrays[1]) == values
    for index in range(2):
        descriptors[index].dtype.code = 1
        with pytest.raises(RuntimeError, match="dtype"):
            loaded.main(*arguments)
        descriptors[index].dtype.code = logical.type_code


def test_custom_storage_lowering_requires_packed_tensor_boundary():
    tensor = ir.Var("input", tirx.TensorType([4], "custom[tfloat32]32"))
    function = tirx.Function([tensor], ir.Evaluate(ir.TensorLoad(tensor, [0]))).with_attr("target", HOST)
    with pytest.raises(TypeError, match="after the logical packed ABI"):
        tilelang.transform.LowerCustomDatatypes()(ir.IRModule({"main": function}))
