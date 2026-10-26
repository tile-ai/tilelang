"""TileLang CUDA lowering registrations, separate from the native registry."""

import functools
import inspect

import tvm_ffi
from tilelang._compat import ir
from tvm_tirx.backend.cuda.codegen.registry import CODEGEN_REGISTRY as NATIVE_CODEGEN_REGISTRY

CODEGEN_REGISTRY = {}
DTYPE_CODEGEN_REGISTRY = {}


def register_dtype_codegen(dtype, scalar_type, source):
    primitive = dtype if isinstance(dtype, ir.PrimType) else ir.PrimType(dtype)
    ir.custom_dtype_storage(primitive)
    if primitive.dtype.type_code < 129 or primitive.dtype.lanes != 1:
        raise ValueError("CUDA custom dtype registration requires a scalar custom dtype")
    if not scalar_type or not source:
        raise ValueError("CUDA custom dtype registration requires a type and its source")
    key = (primitive.dtype.type_code, primitive.dtype.bits)
    entry = (scalar_type, source)
    if key in DTYPE_CODEGEN_REGISTRY and DTYPE_CODEGEN_REGISTRY[key] != entry:
        raise ValueError(f"CUDA custom dtype is already registered: {dtype}")
    DTYPE_CODEGEN_REGISTRY[key] = entry


@tvm_ffi.register_global_func("tl.codegen.cuda.get_dtype_codegen")
def get_dtype_codegen(dtype):
    return list(DTYPE_CODEGEN_REGISTRY[(dtype.type_code, dtype.bits)])


@tvm_ffi.register_global_func("tl.codegen.cuda.get_codegen")
def get_codegen(operation):
    if operation in CODEGEN_REGISTRY:
        return CODEGEN_REGISTRY[operation]
    native = NATIVE_CODEGEN_REGISTRY.get(operation)
    if native is None:
        return None
    parameters = inspect.signature(native).parameters
    if "attrs" in parameters or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return native

    def operand_only(arguments, attrs=None):
        return native(arguments)

    return operand_only


def register_codegen(operation, *, namespace="tl"):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(arguments, attrs=None):
            result = function(*arguments, **({} if attrs is None else {"attrs": attrs}))
            return result if isinstance(result, tuple) else (result, [])

        key = namespace + "." + operation
        if key in CODEGEN_REGISTRY:
            raise ValueError(f"Duplicate TileLang CUDA lowering: {key}")
        CODEGEN_REGISTRY[key] = wrapper
        return wrapper

    return decorator
