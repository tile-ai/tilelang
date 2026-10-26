"""TileLang IR services over unchanged native TIRx objects."""

import tvm_ffi
import inspect
from tvm_tirx import tirx as _tirx
from tvm_tirx import ir as _native
from tvm_tirx.ir import *  # noqa: F401,F403


def const(value, ty=None, loc=None, *, dtype=None):
    if dtype is not None:
        if ty is not None:
            raise TypeError("const accepts only one of ty and dtype")
        ty = dtype
    options = {} if loc is None else {"loc": loc}
    if ty is not None:
        primitive = ty if isinstance(ty, _native.PrimType) else _native.PrimType(ty)
        if primitive.dtype.type_code >= 129:
            scalar = _native.FloatImm(_native.PrimType(primitive.dtype.with_lanes(1)), float(value), **options)
            return scalar if primitive.dtype.lanes == 1 else _native.prim.Broadcast(scalar, primitive.dtype.lanes, **options)
    return _native.const(value, ty, **options)


def cast(value, dtype=None, *, ty=None, loc=None):
    if ty is not None:
        if dtype is not None:
            raise TypeError("cast accepts only one of dtype and ty")
        dtype = ty
    if dtype is None:
        raise TypeError("cast requires a destination type")
    primitive = dtype if isinstance(dtype, _native.PrimType) else _native.PrimType(dtype)
    if not _native.is_prim_expr(value):
        return const(value, primitive, loc=loc)
    location = _native.UnknownLoc() if loc is None else loc
    return tvm_ffi.get_global_func("tl.ir.Cast")(value, primitive, location)


def register_custom_dtype(name, code, storage_dtype):
    storage = storage_dtype if isinstance(storage_dtype, _native.PrimType) else _native.PrimType(storage_dtype)
    tvm_ffi.get_global_func("tl.ir.RegisterCustomDType")(name, code, storage)


def custom_dtype_storage(dtype):
    primitive = dtype if isinstance(dtype, _native.PrimType) else _native.PrimType(dtype)
    return tvm_ffi.get_global_func("tl.ir.CustomDTypeStorage")(primitive)


def prim_type_from_dlpack(encoding):
    return _native.PrimType(tvm_ffi.dtype.from_dlpack_data_type(encoding))


def dtype_imm_from_dlpack(encoding):
    return _native.DataTypeImm(tvm_ffi.dtype.from_dlpack_data_type(encoding))


def index_map(mapping, ndim=None, inverse_index_map=None, *, index_dtype="int64"):
    """Construct a native IndexMap with explicit types for every bound index."""
    parameters = inspect.signature(mapping).parameters
    positional = []
    keywords = []
    variadic = None
    for name, parameter in parameters.items():
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional.append(name)
        elif parameter.kind == parameter.KEYWORD_ONLY:
            keywords.append(name)
        elif parameter.kind == parameter.VAR_POSITIONAL:
            variadic = name
        else:
            raise ValueError("IndexMap cannot accept variadic keyword arguments")
    if variadic is not None:
        if ndim is None or ndim < len(positional) + len(keywords):
            raise ValueError("ndim must include every IndexMap argument when *args is used")
        positional += [f"{variadic}_{axis}" for axis in range(ndim - len(positional) - len(keywords))]
    names = positional + keywords
    dtypes = [index_dtype] * len(names) if isinstance(index_dtype, str) else list(index_dtype)
    if len(dtypes) != len(names):
        raise ValueError("index_dtype must provide one dtype per input iterator")
    variables = [_native.Var(name, dtype) for name, dtype in zip(names, dtypes)]
    outputs = mapping(*variables[: len(positional)], **dict(zip(keywords, variables[len(positional) :])))
    outputs = [outputs] if _native.is_prim_expr(outputs) else list(outputs)
    if any(not _native.is_prim_expr(value) for value in outputs):
        raise TypeError("IndexMap must return primitive expressions")
    if callable(inverse_index_map):
        inverse_index_map = index_map(inverse_index_map)
    return _tirx.IndexMap(variables, outputs, inverse_index_map)


def script(value, **options):
    """Print native IR with TileLang's lossless dtype and IndexMap adapters."""
    from tvm_tirx.script.printer import PrinterConfig

    return tvm_ffi.get_global_func("tl.ir.Script")(value, PrinterConfig(**options))


def renew_def(function):
    return tvm_ffi.get_global_func("tl.ir.RenewDef")(function)


def check_well_formed(value, assert_mode=True):
    return tvm_ffi.get_global_func("tl.ir.CheckWellFormed")(value, assert_mode)


def structural_equal(first, second, map_free_vars=False):
    return tvm_ffi.get_global_func("tl.ir.StructuralEqual")(first, second, map_free_vars)


def assert_structural_equal(first, second, map_free_vars=False):
    if not structural_equal(first, second, map_free_vars):
        raise ValueError("TileLang IR structures are not equal")


def structural_hash(value, map_free_vars=False):
    return tvm_ffi.get_global_func("tl.ir.StructuralHash")(value, map_free_vars)


def save_json(value):
    """Serialize native IR with exact dtype encodings using TileLang codecs."""
    return tvm_ffi.get_global_func("tl.ir.SaveJSON")(value)


def load_json(value):
    """Read native or TileLang JSON without modifying native type codecs."""
    return tvm_ffi.get_global_func("tl.ir.LoadJSON")(value)


def __getattr__(name):
    return getattr(_native, name)
