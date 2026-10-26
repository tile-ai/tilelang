"""CUDA helper lowering for TileLang operations on native TIRx values."""

import tvm_ffi
from tvm_tirx import ir, tirx
from .cuda_registry import get_codegen, register_codegen, register_dtype_codegen
from . import cuda_async as cuda_async, cuda_header as cuda_header
from tvm_tirx.backend.cuda.op import func_call
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

from .target import native_backend_target


@tvm_ffi.register_global_func("tl.cuda.resolve_backend_config")
def _resolve_backend_config(local, defaults, target):
    return tvm_ffi.get_global_func("cuda.resolve_backend_config")(local, defaults, native_backend_target(target))


register_dtype_codegen("custom[tfloat32]", "tl::tfloat32_t", '#include "tl_templates/cuda/common.h"\n')
register_dtype_codegen("custom[float4_e2m1_unpacked]8", "float4_e2m1_unpacked_t", '#include "tl_templates/cuda/cuda_fp4.h"\n')


def _cuda_vector_type(dtype, lanes):
    if dtype in ("float16", "bfloat16"):
        base = "half" if dtype == "float16" else "nv_bfloat16"
        if lanes <= 4:
            return base + (str(lanes) if lanes > 1 else "")
        return f"uint{lanes // 2}" if lanes <= 8 else f"ulonglong{lanes // 4}" if lanes in (12, 16) else None
    if dtype == "float32":
        return "float" + (str(lanes) if lanes > 1 else "") if lanes <= 4 else "ulonglong4" if lanes == 8 else f"tvm_float32x{lanes}"
    if dtype == "float64":
        return "double" + (str(lanes) if lanes > 1 else "") if lanes <= 4 else None
    encodings = {
        "float8_e4m3": (8, "e4m3"),
        "float8_e4m3fn": (8, "e4m3"),
        "float8_e5m2": (8, "e5m2"),
        "float8_e8m0fnu": (8, "e8m0"),
        "float4_e2m1fn": (4, "e2m1"),
    }
    if dtype not in encodings:
        return None
    bits, encoding = encodings[dtype]
    if lanes <= 4:
        return f"__nv_fp{bits}{'x' + str(lanes) if lanes > 1 else ''}_{encoding}"
    words = lanes * bits // 32
    return "uint" if words == 1 else f"uint{words}" if words in (2, 4) else "ulonglong4" if words == 8 else None


def _lower_vector_cast(cast, allow_fp4_fold=True):
    value = cast.value
    destination = str(cast.ty.dtype.with_lanes(1))
    source = str(value.ty.dtype.with_lanes(1))
    lanes = cast.ty.dtype.lanes
    if allow_fp4_fold and destination in ("float8_e4m3", "float8_e4m3fn") and isinstance(value, prim.Cast) and source == "float32":
        if str(value.value.ty.dtype.with_lanes(1)) == "float4_e2m1fn":
            value = value.value
            source = "float4_e2m1fn"
    result_type = _cuda_vector_type(destination, lanes)
    if result_type is None or lanes not in (1, 2, 4, 8, 16, 32, 64):
        return None
    chunk = 2
    extra = ""
    header = "common"
    pair_types = {"float16": "half2", "bfloat16": "__nv_bfloat162", "float32": "float2", "float64": "double2"}
    fp8_formats = {"float8_e4m3": "__NV_E4M3", "float8_e4m3fn": "__NV_E4M3", "float8_e5m2": "__NV_E5M2"}
    ordinary = {
        ("float32", "float16"): "__float22half2_rn",
        ("float16", "float32"): "__half22float2",
        ("float32", "bfloat16"): "__float22bfloat162_rn",
        ("bfloat16", "float32"): "tl::__tl_cvt_bf16x2_to_f32x2",
    }
    if source == "float4_e2m1fn" and destination in ("float8_e4m3", "float8_e4m3fn"):
        chunk = min(lanes, 4)
        suffix = "" if chunk == 1 else f"x{chunk}"
        helper = f"__tl_cvt_e2m1{suffix}_to_e4m3{suffix}"
        pair_type = f"__nv_fp4{suffix}_storage_t"
        header = "cuda_fp4"
    elif lanes == 1:
        return None
    elif (source, destination) in ordinary:
        helper = ordinary[source, destination]
        pair_type = pair_types[source]
    elif destination in fp8_formats and source in ("float32", "float16", "bfloat16"):
        helper = {"float32": "__nv_cvt_float2_to_fp8x2", "float16": "__tl_cvt_half2_to_fp8x2", "bfloat16": "__tl_cvt_bfloat162_to_fp8x2"}[
            source
        ]
        extra = (", __NV_SATFINITE" if source == "float32" else "") + ", " + fp8_formats[destination]
        pair_type, header = pair_types[source], "cuda_fp8"
    elif source in fp8_formats and destination in ("float32", "float16", "bfloat16"):
        encoding = "e4m3" if fp8_formats[source] == "__NV_E4M3" else "e5m2"
        helper = {
            "float32": "__tl_cvt_fp8x2_to_float2",
            "float16": "__tl_cvt_fp8x2_to_half2",
            "bfloat16": f"__tl_cvt_{encoding}x2_to_bfloat162",
        }[destination]
        extra = "" if destination == "bfloat16" else ", " + fp8_formats[source]
        pair_type, header = "__nv_fp8x2_storage_t", "cuda_fp8"
    elif source == "float8_e8m0fnu" and destination == "bfloat16":
        helper, pair_type, header = "__tl_cvt_e8m0x2_to_bfloat162", "__nv_fp8x2_storage_t", "cuda_fp8"
    elif destination == "float8_e8m0fnu" and source in ("float32", "float64", "bfloat16"):
        suffix = {"float32": "float2", "float64": "double2", "bfloat16": "bfloat162"}[source]
        helper, pair_type, header = f"__tl_cvt_{suffix}_to_e8m0x2", pair_types[source], "cuda_fp8"
    elif source == "float4_e2m1fn" and destination in pair_types:
        suffix = {"float16": "half2", "float32": "float2", "float64": "double2", "bfloat16": "bfloat162"}[destination]
        helper, pair_type, header = f"__tl_cvt_fp4x2_to_{suffix}", "__nv_fp4x2_storage_t", "cuda_fp4"
    elif destination == "float4_e2m1fn" and source in pair_types:
        suffix = {"float16": "half2", "float32": "float2", "float64": "double2", "bfloat16": "bfloat162"}[source]
        helper, pair_type, header = f"__tl_cvt_{suffix}_to_fp4x2", pair_types[source], "cuda_fp4"
    else:
        return None
    name = f"tl_cast_{source}_to_{destination}_{lanes}"
    statements = []
    for index in range(lanes // chunk):
        statements.append(f"  auto converted_{index} = {helper}(reinterpret_cast<const {pair_type}*>(&value)[{index}]{extra});")
        statements.append(f"  reinterpret_cast<decltype(converted_{index})*>(&result)[{index}] = converted_{index};")
    source_code = f"""#include "tl_templates/cuda/{header}.h"
template <typename Result, typename Source>
__device__ __forceinline__ Result {name}(Source value) {{
  Result result;
{chr(10).join(statements)}
  return result;
}}
"""
    return func_call(f"{name}<{result_type}>", value, source_code, ty=cast.ty)


tvm_ffi.register_global_func("tl.codegen.cuda.lower_cast", _lower_vector_cast)


@register_codegen("cast_with_rounding", namespace="tl")
def _lower_cast_with_rounding(value, random_bits=None, attrs=None):
    mode = str(attrs["round"])
    dtype = str(attrs["dtype"])
    lanes = value.ty.dtype.lanes
    if not mode and attrs["sat"]:
        result_ty = ir.PrimType(dtype if lanes == 1 else f"{dtype}x{lanes}")
        lowered = _lower_vector_cast(prim.Cast(result_ty, value), allow_fp4_fold=False)
        if lowered is not None:
            return lowered
        result_type = _cuda_vector_type(dtype, lanes)
        if lanes == 1 and result_type is not None:
            source = """#include "tl_templates/cuda/common.h"
template <typename Result, typename Source>
__device__ __forceinline__ Result tl_annotated_cast(Source value) {
  return static_cast<Result>(value);
}
"""
            return func_call(f"tl_annotated_cast<{result_type}>", value, source, ty=result_ty)
    if mode != "rs":
        raise ValueError(f"round='{mode}' is not supported for a cast from {value.ty} to {dtype}")
    if not attrs["sat"]:
        raise ValueError("sat=false is not supported for stochastic rounding")
    formats = {
        "float16": ("f16", "common", 2),
        "bfloat16": ("bf16", "common", 2),
        "float8_e4m3fn": ("e4m3", "cast_fp8", 4),
        "float8_e5m2": ("e5m2", "cast_fp8", 4),
        "float4_e2m1fn": ("e2m1", "cast_fp4", 4),
    }
    if dtype not in formats or value.ty.dtype.with_lanes(1) != "float32":
        raise TypeError(f"round='rs' is not supported for a cast from {value.ty} to {dtype}")
    if random_bits is None or str(random_bits.ty) not in ("int32", "uint32"):
        raise TypeError("Stochastic rounding requires scalar int32 or uint32 random bits")
    if lanes not in (1, 2, 4, 8) or (dtype == "float4_e2m1fn" and lanes > 4):
        raise ValueError(f"Unsupported stochastic rounding vector width: {lanes}")
    encoding, header, maximum_chunk = formats[dtype]
    chunk = min(lanes, maximum_chunk)
    source_type = "float" if chunk == 1 else f"float{chunk}"
    if dtype in ("float16", "bfloat16"):
        result_type = "half" if dtype == "float16" else "nv_bfloat16"
        result_type = result_type if lanes == 1 else f"{result_type}{lanes}" if lanes <= 4 else f"uint{lanes // 2}"
        namespace = "tl::"
    else:
        bits = 4 if dtype == "float4_e2m1fn" else 8
        vector_suffix = "" if lanes == 1 else f"x{lanes}"
        result_type = f"__nv_fp{bits}{vector_suffix}_{encoding}" if lanes <= 4 else f"uint{lanes // 4}"
        namespace = ""
    helper = f"{namespace}__tl_cvt_f32x{chunk}_to_{encoding}x{chunk}_rs_sat"
    name = f"tl_cast_rs_{encoding}_{lanes}"
    source = f"""#include "tl_templates/cuda/{header}.h"
template <typename Result, typename Source>
__device__ __forceinline__ Result {name}(Source value, unsigned int random_bits) {{
  Result result;
  #pragma unroll
  for (int chunk = 0; chunk < {lanes // chunk}; ++chunk) {{
    auto converted = {helper}(reinterpret_cast<const {source_type}*>(&value)[chunk], random_bits);
    reinterpret_cast<decltype(converted)*>(&result)[chunk] = converted;
  }}
  return result;
}}
"""
    result_ty = ir.PrimType(dtype if lanes == 1 else f"{dtype}x{lanes}")
    return func_call(f"{name}<{result_type}>", value, random_bits, source, ty=result_ty)


@register_codegen("sync_grid", namespace="tl")
def _lower_grid_sync(attrs=None):
    return func_call("cooperative_groups::this_grid().sync", "#include <cooperative_groups.h>\n")


@register_codegen("device_assert_with_msg", namespace="tl")
def _lower_device_assert_with_msg(condition, message, attrs=None):
    return func_call("device_assert_with_msg", condition, message, '#include "tl_templates/cuda/debug.h"\n')


@register_codegen("device_assert", namespace="tl")
def _lower_device_assert(condition, attrs=None):
    return func_call("device_assert", condition, '#include "tl_templates/cuda/debug.h"\n')


@register_codegen("set_max_nreg", namespace="tl")
def _lower_register_reconfiguration(registers, increase, attrs=None):
    from tvm_tirx.backend.cuda.ptx import PTXNamespace

    count, action = int(registers), int(increase)
    if count < 24 or count > 256 or count % 8 or action not in (0, 1):
        raise ValueError("set_max_nreg requires a multiple of 8 in [24, 256] and a 0/1 action")
    instruction = PTXNamespace()[f"setmaxnreg.{'inc' if action else 'dec'}.sync.aligned.u32"](count)
    lowered = get_codegen(instruction.op.name)(instruction.args, instruction.attrs)
    return lowered[0], list(lowered[1])


def _register_global_memory(bits, store):
    name = f"{'stg' if store else 'ldg'}{bits}"

    @register_codegen(name, namespace="tl")
    def lower(pointer, *arguments, attrs=None):
        if not isinstance(pointer.ty, ir.PtrType):
            raise TypeError(f"tl.{name} requires a typed pointer")
        if len(arguments) not in ((1, 2) if store else (0, 1)):
            raise ValueError(f"tl.{name} has an invalid operand count")
        packed = ir.PrimType("uint32" if bits == 32 else f"uint32x{bits // 32}")
        predicated = len(arguments) == (2 if store else 1)
        if store and arguments[0].ty != packed:
            raise TypeError(f"tl.{name} requires {packed} storage bits")
        if predicated and arguments[-1].ty != ir.PrimType("bool"):
            raise TypeError(f"tl.{name} requires a scalar boolean predicate")
        header = "copy_sm100.h" if bits == 256 else "copy.h"
        helper = f"tl::{'store' if store else 'load'}_global_{bits}"
        if predicated:
            helper += "_conditional"
        if bits == 256 and not store:
            pointer = ir.Call("tirx.reinterpret", [pointer], ty=ir.PtrType(packed, scope="global"))
        return func_call(helper, pointer, *arguments, f'#include "tl_templates/cuda/{header}"\n', ty=None if store else packed)


for _bits in (32, 64, 128, 256):
    _register_global_memory(_bits, False)
    _register_global_memory(_bits, True)
del _bits


def _register_shared_memory(bits, store):
    name = f"{'sts' if store else 'lds'}{bits}"

    @register_codegen(name, namespace="tl")
    def lower(pointer, *arguments, attrs=None):
        if not isinstance(pointer.ty, ir.PtrType) or not pointer.ty.scope.name.startswith("shared"):
            raise TypeError(f"tl.{name} requires a shared-memory pointer")
        packed = ir.PrimType("uint32" if bits == 32 else f"uint32x{bits // 32}")
        if len(arguments) != int(store) or (store and arguments[0].ty != packed):
            raise TypeError(f"tl.{name} requires {'one ' + str(packed) + ' value' if store else 'only a pointer'}")
        helper = f"tl::{'store' if store else 'load'}_shared_{bits}"
        return func_call(helper, pointer, *arguments, '#include "tl_templates/cuda/copy.h"\n', ty=None if store else packed)


for _bits in (32, 64, 128):
    _register_shared_memory(_bits, False)
    _register_shared_memory(_bits, True)
del _bits


@register_codegen("ptx_cp_async", namespace="tl")
def _lower_async_copy(destination, source, elements, *predicate, attrs=None):
    if not isinstance(destination.ty, ir.PtrType) or not isinstance(source.ty, ir.PtrType):
        raise TypeError("tl.ptx_cp_async requires typed source and destination pointers")
    dtype = source.ty.element_type
    if not isinstance(dtype, ir.PrimType) or destination.ty.element_type != dtype:
        raise TypeError("tl.ptx_cp_async requires matching primitive pointer element types")
    bits = int(elements) * dtype.dtype.bits * dtype.dtype.lanes
    if bits not in (32, 64, 128):
        raise ValueError("tl.ptx_cp_async requires a 4, 8, or 16-byte transfer")
    if len(predicate) > 1:
        raise ValueError("tl.ptx_cp_async accepts at most one zero-fill predicate")
    if dtype.dtype.bits < 8:
        destination = ir.Call("tirx.reinterpret", [destination], ty=ir.PtrType("uint8", scope=destination.ty.scope))
        source = ir.Call("tirx.reinterpret", [source], ty=ir.PtrType("uint8", scope=source.ty.scope))
    result = get_codegen("tirx.s_tir.cp_async_raw")([destination, ir.const(0), source, ir.const(0), ir.const(bits // 8), *predicate])
    return result[0], list(result[1])


def _register_packed_math(name, arity):
    @register_codegen(name, namespace="tl")
    def lower(*arguments, attrs=None):
        if len(arguments) != arity:
            raise ValueError(f"tl.{name} requires {arity} operands")
        result_type = arguments[0].ty
        supported = ("float16x2", "bfloat16x2") if name.endswith("_nan") else ("float16x2", "bfloat16x2", "float32x2")
        if str(result_type) not in supported or any(value.ty != result_type for value in arguments):
            raise TypeError(f"tl.{name} requires matching packed floating-point operands")
        helper = name
        if name == "add2":
            for product, addend in (arguments, arguments[::-1]):
                if isinstance(product, ir.Call) and isinstance(product.op, ir.Op) and product.op.name == "tl.mul2":
                    helper = "fma2"
                    arguments = (*product.args, addend)
                    break
        return func_call(f"tl::{helper}", *arguments, '#include "tl_templates/cuda/common.h"\n', ty=result_type)


for _name, _arity in (
    ("add2", 2),
    ("sub2", 2),
    ("mul2", 2),
    ("fma2", 3),
    ("max2", 2),
    ("min2", 2),
    ("abs2", 1),
    ("max2_nan", 2),
    ("min2_nan", 2),
):
    _register_packed_math(_name, _arity)
del _name, _arity


def _register_fast_math(name):
    @register_codegen(name, namespace="tl")
    def lower(value, attrs=None):
        dtype = value.ty.dtype
        if dtype.lanes != 1:
            raise TypeError(f"tl.{name} requires a scalar operand")
        if str(dtype) == "float32":
            symbol = name + "f"
        elif str(dtype) == "float64":
            symbol = name[2:]
        elif str(dtype) in ("float16", "bfloat16"):
            symbol = "h" + name[2:]
        else:
            raise TypeError(f"tl.{name} does not support {dtype}")
        return func_call(symbol, value, '#include "tl_templates/cuda/common.h"\n', ty=value.ty)


for _name in ("__exp", "__exp10", "__log", "__log2", "__log10", "__tan", "__cos", "__sin"):
    _register_fast_math(_name)
del _name


@register_codegen("fast_rcp", namespace="tl")
def _lower_fast_rcp(value, attrs=None):
    if value.ty != ir.PrimType("float32"):
        raise TypeError("tl.fast_rcp requires float32")
    return func_call("tl::fast_rcp", value, '#include "tl_templates/cuda/common.h"\n', ty=value.ty)


def _register_ieee_math(name, intrinsic, arity):
    @register_codegen(name, namespace="tl")
    def lower(*arguments, attrs=None):
        operands = arguments[:arity]
        mode = "rn" if name == "ieee_frsqrt" else arguments[arity].value
        if mode not in ("rn", "rz", "ru", "rd"):
            raise ValueError(f"Invalid IEEE rounding mode: {mode}")
        result_type = operands[0].ty
        if any(operand.ty != result_type for operand in operands):
            raise TypeError("IEEE math requires matching operand types")
        dtype = str(result_type)
        if dtype == "float32":
            symbol = f"__{intrinsic}_{mode}"
        elif dtype == "float64":
            if name == "ieee_frsqrt":
                raise TypeError("IEEE frsqrt is not supported for float64 in CUDA")
            symbol = f"__fma_{mode}" if name == "ieee_fmaf" else f"__d{intrinsic[1:]}_{mode}"
        elif dtype in ("float16", "bfloat16"):
            if mode != "rn":
                raise ValueError("Only rounding mode 'rn' is available for half precision")
            symbol = {
                "fmaf": "__hfma",
                "fsqrt": "hsqrt",
                "frcp": "hrcp",
                "frsqrt": "hrsqrt",
                "fdiv": "__hdiv",
            }.get(intrinsic, f"__h{intrinsic[1:]}_rn")
        else:
            raise TypeError(f"IEEE {intrinsic} is not supported for dtype {dtype}")
        if dtype == "bfloat16" and intrinsic == "fmaf":
            return func_call(
                "tl_ieee_bf16_fma",
                *operands,
                """#include <cuda_bf16.h>
__device__ __forceinline__ nv_bfloat16 tl_ieee_bf16_fma(nv_bfloat16 left, nv_bfloat16 right, nv_bfloat16 addend) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hfma(left, right, addend);
#else
  return __float2bfloat16_rn(__fmaf_rn(__bfloat162float(left), __bfloat162float(right), __bfloat162float(addend)));
#endif
}
""",
                ty=result_type,
            )
        return func_call(symbol, *operands, '#include "tl_templates/cuda/common.h"\n', ty=result_type)


for _name, _intrinsic, _arity in (
    ("ieee_add", "fadd", 2),
    ("ieee_sub", "fsub", 2),
    ("ieee_mul", "fmul", 2),
    ("ieee_fmaf", "fmaf", 3),
    ("ieee_frcp", "frcp", 1),
    ("ieee_fsqrt", "fsqrt", 1),
    ("ieee_frsqrt", "frsqrt", 1),
    ("ieee_fdiv", "fdiv", 2),
):
    _register_ieee_math(_name, _intrinsic, _arity)
del _name, _intrinsic, _arity


def _register_round_nearest_math(name, scalar_name, packed_name, arity):
    @register_codegen(name, namespace="tl")
    def lower(*operands, attrs=None):
        if len(operands) != arity or any(value.ty != operands[0].ty for value in operands[1:]):
            raise TypeError(f"tl.{name} requires {arity} matching floating-point operands")
        dtype = operands[0].ty.dtype
        lanes = dtype.lanes
        scalar_dtype = str(dtype.with_lanes(1))
        target = Target.current()
        architecture = str(target.attrs.get("arch", "")) if target is not None else ""
        compute = int("".join(character for character in architecture if character.isdigit()) or "0")
        packed = lanes % 2 == 0 and (
            (scalar_dtype == "float16" and compute >= 53)
            or (scalar_dtype == "bfloat16" and compute >= 80)
            or (scalar_dtype == "float32" and compute >= 100)
        )
        if lanes == 1:
            result, tags = get_codegen(f"tl.{scalar_name}")([*operands, ir.StringImm("rn")])
            return result, list(tags)
        width = 2 if packed else 1
        helper = packed_name if packed else scalar_name
        arguments = [ir.Var(f"operand_{index}", ir.PrimType(dtype.with_lanes(width))) for index in range(arity)]
        if not packed:
            arguments.append(ir.StringImm("rn"))
        result, tags = get_codegen(f"tl.{helper}")(arguments)
        element_type = _cuda_vector_type(scalar_dtype, width)
        symbol = result.args[0].value
        helper_name = f"tl_{name}_{scalar_dtype}x{lanes}_{width}"
        parameters = ", ".join(f"Vector operand_{index}" for index in range(arity))
        lines = [
            result.args[-1].value,
            "template <typename Vector>",
            f"__device__ __forceinline__ Vector {helper_name}({parameters}) {{",
            "  Vector result;",
        ]
        for index in range(lanes // width):
            values = ", ".join(f"reinterpret_cast<const {element_type}*>(&operand_{operand})[{index}]" for operand in range(arity))
            lines.append(f"  reinterpret_cast<{element_type}*>(&result)[{index}] = {symbol}({values});")
        lines.extend(["  return result;", "}"])
        return func_call(helper_name, *operands, "\n".join(lines) + "\n", ty=operands[0].ty), list(tags)


_register_round_nearest_math("fmul", "ieee_mul", "mul2", 2)
_register_round_nearest_math("fma", "ieee_fmaf", "fma2", 3)


def _register_atomic(name, helper, returns_value=False):
    @register_codegen(name, namespace="tl")
    def lower(pointer, *arguments, attrs=None):
        if not isinstance(pointer.ty, ir.PtrType) or not isinstance(pointer.ty.element_type, ir.PrimType):
            raise TypeError(f"tl.{name} requires a typed primitive pointer")
        if helper != "AtomicLoad" and arguments and arguments[0].ty != pointer.ty.element_type:
            arguments = (prim.Cast(pointer.ty.element_type, arguments[0]), *arguments[1:])
        result_type = pointer.ty.element_type if returns_value else None
        return func_call(helper, pointer, *arguments, '#include "tl_templates/cuda/atomic.h"\n', ty=result_type)


for _operation in ("add", "max", "min"):
    _helper = "Atomic" + _operation.title()
    _register_atomic(f"atomic_{_operation}_elem_op", _helper)
    _register_atomic(f"atomic_{_operation}_ret_elem_op", _helper + "Ret", True)
_register_atomic("atomic_or_elem_op", "AtomicOr")
_register_atomic("atomic_load_elem_op", "AtomicLoad", True)
_register_atomic("atomic_store_elem_op", "AtomicStore")
del _operation, _helper


def _register_vector_atomic(width, returns_value):
    name = f"atomic_addx{width}_{'ret_' if returns_value else ''}elem_op"

    @register_codegen(name, namespace="tl")
    def lower(destination, value, order=0, attrs=None):
        if not isinstance(destination.ty, ir.PtrType):
            raise TypeError(f"tl.{name} requires a typed destination pointer")
        element_type = destination.ty.element_type
        dtype = str(element_type)
        if dtype not in ("float32", "float16", "bfloat16"):
            raise TypeError(f"tl.{name} requires float32, float16, or bfloat16 storage")
        result_type = ir.PrimType(f"{dtype}x{width}") if returns_value else None
        helper = f"AtomicAddx{width}{'Ret' if returns_value else ''}"
        header = '#include "tl_templates/cuda/atomic.h"\n'
        if dtype == "float32":
            return func_call(helper, destination, value, order, header, ty=result_type)
        source_type = value.ty.element_type if isinstance(value.ty, ir.PtrType) else value.ty
        source_dtype = str(source_type.dtype).split("x")[0]
        source_ctype = {"float16": "half", "bfloat16": "nv_bfloat16", "float32": "float"}.get(source_dtype)
        if source_ctype is None:
            raise TypeError(f"Unsupported vector atomic source type: {source_type}")
        ctype = "half" if dtype == "float16" else "nv_bfloat16"
        storage_type = "half_t" if dtype == "float16" else "bfloat16_t"
        source_pointer = "value" if isinstance(value.ty, ir.PtrType) else f"reinterpret_cast<const {source_ctype}*>(&value)"
        helper_name = f"tl_native_{name}_{dtype}_{source_dtype}_{'ptr' if isinstance(value.ty, ir.PtrType) else 'value'}"
        return_type = f"{ctype}{width}" if returns_value else "void"
        call = f"{helper}(reinterpret_cast<{storage_type}*>(destination), {source_pointer}, order)"
        body = (
            f"auto previous = {call}; {return_type} result; static_assert(sizeof(result) == sizeof(previous)); memcpy(&result, &previous, sizeof(result)); return result;"
            if returns_value
            else f"{call};"
        )
        source = (
            header
            + f"template<typename Value> __device__ __forceinline__ {return_type} {helper_name}({ctype}* destination, Value value, int order) {{ {body} }}\n"
        )
        return func_call(helper_name, destination, value, order, source, ty=result_type)


for _width in (2, 4):
    for _returns_value in (False, True):
        _register_vector_atomic(_width, _returns_value)
del _width, _returns_value


def _constant(value, name, minimum, maximum):
    if not isinstance(value, ir.IntImm) or not minimum <= int(value.value) <= maximum:
        raise ValueError(f"{name} must be a constant integer in [{minimum}, {maximum}]")
    return int(value.value)


def _descriptor_pointer(descriptor):
    if not isinstance(descriptor, ir.TensorLoad) or descriptor.ty != ir.PrimType("uint64"):
        raise TypeError("A shared-memory descriptor must be a uint64 tensor element")
    return ir.Call("tirx.address_of", [descriptor], ty=ir.PtrType("uint64", scope=descriptor.source.scope()))


@register_codegen("initialize_wgmma_descriptor", namespace="tl")
def _lower_wgmma_descriptor(descriptor, pointer, layout, leading, stride, attrs=None):
    layout = _constant(layout, "WGMMA descriptor layout", 0, 3)
    leading = _constant(leading, "WGMMA descriptor leading offset", 0, 16383)
    stride = _constant(stride, "WGMMA descriptor stride offset", 0, 16383)
    return func_call(
        f"tl::initialize_wgmma_descriptor<{layout}, {leading}, {stride}>",
        _descriptor_pointer(descriptor),
        pointer,
        '#include "tl_templates/cuda/common.h"\n',
    )


@register_codegen("initialize_tcgen05_descriptor", namespace="tl")
def _lower_tcgen05_descriptor(descriptor, pointer, leading, stride, base, absolute, swizzle, attrs=None):
    return func_call(
        "tl::initialize_tcgen05_descriptor",
        _descriptor_pointer(descriptor),
        pointer,
        leading,
        stride,
        base,
        absolute,
        swizzle,
        '#include "tl_templates/cuda/common.h"\n',
    )


@register_codegen("increase_descriptor_offset", namespace="tl")
def _lower_descriptor_offset(descriptor, offset, attrs=None):
    return func_call(
        "tl::increase_descriptor_offset",
        _descriptor_pointer(descriptor),
        offset,
        '#include "tl_templates/cuda/common.h"\n',
    )


@register_codegen("warpgroup_fence_operand", namespace="tl")
def _lower_warpgroup_fence_operand(dtype, pointer, offset, count, attrs=None):
    if not isinstance(pointer.ty, ir.PtrType) or not isinstance(pointer.ty.element_type, ir.PrimType):
        raise TypeError("WGMMA register fences require a typed primitive pointer")
    element_dtype = pointer.ty.element_type.dtype
    byte_offset = ir.PrimType("int64")(offset) * (element_dtype.bits * element_dtype.lanes) // 8
    advanced = ir.Call("tirx.ptr_byte_offset", [pointer, byte_offset], ty=pointer.ty)
    storage = "float32" if dtype.value in ("fp32", "tf32", "float32") else "uint32"
    registers = ir.Call("tirx.reinterpret", [advanced], ty=ir.PtrType(storage, scope=pointer.ty.scope))
    return func_call("tl::warpgroup_fence_operand", registers, count, '#include "tl_templates/cuda/intrin.h"\n')


@register_codegen("warpgroup_wait", namespace="tl")
def _lower_warpgroup_wait(pending, attrs=None):
    count = int(pending)
    if count < 0 or count > 7:
        raise ValueError("WGMMA pending group count must be in [0, 7]")
    return func_call(f"tl::warpgroup_wait<{count}>", '#include "tl_templates/cuda/intrin.h"\n')


@register_codegen("get_warp_idx_sync", namespace="tl")
def _lower_warp_index(warp_size=None, attrs=None):
    arguments = [] if warp_size is None else [warp_size]
    return func_call("tl::get_warp_idx_sync", *arguments, '#include "tl_templates/cuda/intrin.h"\n', ty="int32")


@register_codegen("get_lane_idx", namespace="tl")
def _lower_lane_index(warp_size=None, attrs=None):
    arguments = [] if warp_size is None else [warp_size]
    return func_call("tl::get_lane_idx", *arguments, '#include "tl_templates/cuda/intrin.h"\n', ty="int32")


@register_codegen("get_warp_idx", namespace="tl")
def _lower_unsynchronized_warp_index(warp_size=None, attrs=None):
    arguments = [] if warp_size is None else [warp_size]
    return func_call("tl::get_warp_idx", *arguments, '#include "tl_templates/cuda/intrin.h"\n', ty="int32")


@register_codegen("get_warp_group_idx", namespace="tl")
def _lower_warp_group_index(warp_size=None, warps_per_group=None, attrs=None):
    arguments = [] if warp_size is None else [warp_size]
    if warps_per_group is not None:
        if warp_size is None:
            raise ValueError("get_warp_group_idx requires warp_size when specifying warps_per_group")
        arguments.append(warps_per_group)
    return func_call("tl::get_warp_group_idx", *arguments, '#include "tl_templates/cuda/intrin.h"\n', ty="int32")


@register_codegen("__ldg", namespace="tl")
def _lower_readonly_load(value, attrs=None):
    if not isinstance(value, ir.TensorLoad):
        raise TypeError("__ldg requires a tensor load")
    pointer = tirx.address_of(value)
    return func_call("__ldg", pointer, "", ty=value.ty)


def _register_shuffle(name):
    @register_codegen(name, namespace="tl")
    def lower(mask, value, lane, width, attrs=None):
        return func_call(
            f"__{name}",
            ir.PrimType("uint32")(mask),
            value,
            lane,
            width,
            '#include "tl_templates/cuda/common.h"\n',
            ty=value.ty,
        )


for _name in ("shfl_sync", "shfl_up_sync", "shfl_down_sync", "shfl_xor_sync"):
    _register_shuffle(_name)
del _name


@register_codegen("sync_warp", namespace="tl")
def _lower_sync_warp(mask=None, attrs=None):
    mask = ir.const(0xFFFFFFFF, "uint32") if mask is None else ir.PrimType("uint32")(mask)
    return func_call("__syncwarp", mask, "")


def _register_warp_vote(name, masked):
    @register_codegen(name, namespace="tl")
    def lower(*arguments, attrs=None):
        predicate = arguments[-1]
        predicate = prim.NE(predicate, ir.const(0, predicate.ty))
        operands = [ir.PrimType("uint32")(arguments[0]), predicate] if masked else [predicate]
        return func_call(f"__{name}", *operands, "", ty="int32")


for _name in ("any_sync", "all_sync"):
    _register_warp_vote(_name, True)
for _name in ("syncthreads_count", "syncthreads_and", "syncthreads_or"):
    _register_warp_vote(_name, False)
del _name


@register_codegen("ballot_sync", namespace="tl")
def _lower_ballot_sync(mask, predicate, attrs=None):
    predicate = prim.NE(predicate, ir.const(0, predicate.ty))
    return func_call("__ballot_sync", ir.PrimType("uint32")(mask), predicate, "", ty="uint64")


@register_codegen("ballot", namespace="tl")
def _lower_ballot(predicate, attrs=None):
    predicate = prim.NE(predicate, ir.const(0, predicate.ty))
    return func_call("__ballot_sync", ir.const(0xFFFFFFFF, "uint32"), predicate, "", ty="uint64")


@register_codegen("activemask", namespace="tl")
def _lower_activemask(attrs=None):
    return func_call("__activemask", "", ty="uint64")


@register_codegen("__ffs", namespace="tl")
def _lower_ffs(value, attrs=None):
    bits = value.ty.dtype.bits
    if bits > 64 or value.ty.dtype.lanes != 1:
        raise TypeError("__ffs requires a scalar integer of at most 64 bits")
    return func_call("__ffsll" if bits > 32 else "__ffs", ir.PrimType("int64" if bits > 32 else "int32")(value), "", ty="int32")


@register_codegen("__fns", namespace="tl")
def _lower_fns(mask, base, offset, attrs=None):
    return func_call("__fns", ir.PrimType("uint32")(mask), ir.PrimType("uint32")(base), ir.PrimType("int32")(offset), "", ty="uint32")


@register_codegen("match_any_sync", namespace="tl")
def _lower_match_any(mask, value, attrs=None):
    return func_call("__match_any_sync", ir.PrimType("uint32")(mask), value, "", ty="uint32")


@register_codegen("match_all_sync", namespace="tl")
def _lower_match_all(mask, value, attrs=None):
    source = """template <typename Value>
__device__ __forceinline__ unsigned int tl_match_all_sync(unsigned int mask, Value value) {
  int predicate;
  return __match_all_sync(mask, value, &predicate);
}
"""
    return func_call("tl_match_all_sync", ir.PrimType("uint32")(mask), value, source, ty="uint32")


@register_codegen("ptx_arrive_cluster_barrier", namespace="tl")
def _lower_cluster_barrier_arrive(barrier, cta_id, attrs=None):
    return func_call(
        "tl::mbarrier_arrive",
        barrier,
        cta_id,
        ir.const(1, "uint32"),
        '#include "tl_templates/cuda/barrier.h"\n',
    )


def _register_helper(name, header, arity, helper=None, dtype=None):
    @register_codegen(name, namespace="tl")
    def lower(*arguments, attrs=None):
        if len(arguments) != arity:
            raise ValueError(f"tl.{name} requires {arity} operands")
        return func_call(f"tl::{helper or name}", *arguments, f'#include "tl_templates/cuda/{header}.h"\n', ty=dtype)


def _register_warp_reduce(name):
    @register_codegen(name, namespace="tl")
    def lower(value, attrs=None):
        return func_call(f"tl::{name}", value, '#include "tl_templates/cuda/reduce.h"\n', ty=value.ty)


for _name in ("warp_reduce_sum", "warp_reduce_max", "warp_reduce_min", "warp_reduce_bitand", "warp_reduce_bitor"):
    _register_warp_reduce(_name)
del _name


@register_codegen("ptx_st_bulk_shared", namespace="tl")
def _lower_shared_bulk_store(pointer, byte_count, value, attrs=None):
    size = _constant(byte_count, "bulk-store byte count", 8, 2**32 - 1)
    _constant(value, "bulk-store initialization value", 0, 0)
    if size % 8:
        raise ValueError("Shared bulk-store byte count must be divisible by eight")
    return func_call(f"tl::st_bulk_shared<{size}, 0>", pointer, '#include "tl_templates/cuda/common.h"\n')


_register_helper("prefetch_tma_descriptor", "copy_sm90", 1)
_register_helper("fence_proxy_async", "barrier", 0)
_register_helper("warpgroup_arrive", "intrin", 0)
_register_helper("warpgroup_commit_batch", "intrin", 0)
_register_helper("tma_store_arrive", "barrier", 0)
_register_helper("mbarrier_init", "barrier", 2)
_register_helper("mbarrier_arrive", "barrier", 1)
_register_helper("mbarrier_arrive_expect_tx", "barrier", 2)
_register_helper("mbarrier_expect_tx", "barrier", 2)
_register_helper("mbarrier_wait_parity", "barrier", 2, "mbarrier_wait")
_register_helper("ptx_cp_async_barrier_noinc", "barrier", 1, "mbarrier_cp_async_arrive_noinc")
_register_helper("ptx_fence_barrier_init", "barrier", 0, "fence_barrier_init")
_register_helper("tcgen05_before_thread_sync", "tcgen_05", 0)
_register_helper("tcgen05_after_thread_sync", "tcgen_05", 0)
_register_helper("ptx_tcgen05_sf_warp_transpose", "tcgen_05", 1, "tcgen05_sf_warp_transpose")
_register_helper("cluster_arrive_relaxed", "cluster", 0)
_register_helper("cluster_arrive", "cluster", 0)
_register_helper("cluster_wait", "cluster", 0)
_register_helper("cluster_sync", "cluster", 0)
_register_helper("block_rank_in_cluster", "cluster", 0, dtype="int32")
_register_helper("clc_try_cancel", "cluster", 2)
_register_helper("clc_try_cancel_multicast", "cluster", 2)
_register_helper("clc_is_canceled", "cluster", 1, dtype="int32")
_register_helper("clc_get_first_ctaid_x", "cluster", 1, dtype="uint32")
_register_helper("clc_get_first_ctaid_y", "cluster", 1, dtype="uint32")
_register_helper("clc_get_first_ctaid_z", "cluster", 1, dtype="uint32")


@register_codegen("pdl_trigger", namespace="tl")
def _lower_pdl_trigger(attrs=None):
    return func_call("cudaTriggerProgrammaticLaunchCompletion", "#include <cuda_runtime.h>\n")


@register_codegen("pdl_sync", namespace="tl")
def _lower_pdl_sync(attrs=None):
    return func_call("cudaGridDependencySynchronize", "#include <cuda_runtime.h>\n")


@register_codegen("ptx_tcgen05_cp_warpx4", namespace="tl")
def _lower_tcgen05_copy(smem_pointer, tmem_pointer, column_offset, attrs=None):
    header = '#include "tl_templates/cuda/tcgen_05.h"\n'
    descriptor = func_call("tl::make_sf_smem_desc", smem_pointer, header, ty="uint64")
    address = tmem_pointer
    if isinstance(address.ty, ir.PtrType):
        address = func_call("tl::load_scalar", address, '#include "tl_templates/cuda/common.h"\n', ty="uint32")
    if address.ty != ir.PrimType("uint32"):
        raise TypeError("TMEM copies require a uint32 address token")
    column = ir.PrimType("uint32")(column_offset)
    two_ctas = bool(attrs.get("use_2cta", False)) if attrs is not None else False
    return func_call(f"tl::tcgen05_cp<{str(two_ctas).lower()}>", descriptor, address + column, header)


def _is_tensor_map(value):
    value_type = value.ty
    if isinstance(value_type, ir.PtrType):
        value_type = value_type.element_type
    return isinstance(value_type, tirx.TensorMapType)


def _register_tma_load(name):
    @register_codegen(name, namespace="tl")
    def lower(*arguments, attrs=None):
        if len(arguments) < 5:
            raise ValueError(f"tl.{name} requires a descriptor or global pointer, barrier, shared pointer, and transfer geometry")
        eviction = _constant(arguments[-1], "TMA eviction policy", 0, 2)
        two_ctas = bool(attrs.get("use_2cta", False)) if attrs is not None else False
        helper = "tma_load_2sm" if two_ctas and name == "tma_load" else name
        header = "copy_sm100" if two_ctas or name == "tma_load_gather4" else "copy_sm90"
        if eviction:
            if not _is_tensor_map(arguments[0]) and name == "tma_load":
                raise ValueError("One-dimensional bulk loads do not support cache eviction hints")
            policy = ("EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST")[eviction]
            helper += f"<tl::CacheHintSm{'100' if two_ctas else '90'}::{policy}>"
        return func_call(f"tl::{helper}", *arguments[:-1], f'#include "tl_templates/cuda/{header}.h"\n')


for _load in ("tma_load", "tma_load_multicast", "tma_load_im2col", "tma_load_gather4"):
    _register_tma_load(_load)
del _load


@register_codegen("tma_store", namespace="tl")
def _lower_tma_store(*arguments, attrs=None):
    if not 5 <= len(arguments) <= 9:
        raise ValueError("tl.tma_store requires descriptor, shared pointer, 1..5 coordinates, reduce, and eviction policy")
    reduce = _constant(arguments[-2], "TMA reduction flag", 0, 1)
    eviction = _constant(arguments[-1], "TMA eviction policy", 0, 2)
    if reduce and eviction:
        raise ValueError("TMA reduction does not support a cache eviction hint")
    helper = "tl::tma_store_add" if reduce else "tl::tma_store"
    if eviction:
        policy = ("EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST")[eviction]
        helper += f"<tl::CacheHintSm90::{policy}>"
    return func_call(helper, *arguments[:-2], '#include "tl_templates/cuda/copy_sm90.h"\n')


@register_codegen("tma_store_cluster", namespace="tl")
def _lower_tma_store_cluster(destination, source, rank, byte_count, barrier, attrs=None):
    constant = byte_count
    while isinstance(constant, prim.Cast):
        constant = constant.value
    if isinstance(constant, ir.IntImm) and int(constant) % 16:
        raise ValueError("Cluster TMA store byte count must be a multiple of 16")
    return func_call(
        "tl::tma_store_cluster",
        destination,
        source,
        ir.PrimType("int32")(rank),
        ir.PrimType("uint32")(byte_count),
        barrier,
        '#include "tl_templates/cuda/copy_sm90.h"\n',
    )


@register_codegen("ptx_cluster_store", namespace="tl")
def _lower_ptx_cluster_store(buffer, value, rank, index, attrs=None):
    source = """#include <cooperative_groups.h>
template <typename Element, typename Value, typename Index>
__device__ __forceinline__ void tl_cluster_store(Element* buffer, Value value, int rank, Index index) {
  auto cluster = cooperative_groups::this_cluster();
  auto* destination = cluster.map_shared_rank(buffer + index, rank);
  *destination = static_cast<Element>(value);
}
"""
    return func_call("tl_cluster_store", buffer, value, ir.PrimType("int32")(rank), index, source)


@register_codegen("tma_store_scatter4", namespace="tl")
def _lower_tma_scatter4(*arguments, attrs=None):
    if len(arguments) != 8:
        raise ValueError("TMA scatter4 requires a descriptor, shared pointer, five coordinates, and eviction policy")
    eviction = _constant(arguments[-1], "TMA eviction policy", 0, 2)
    policy = ("EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST")[eviction]
    return func_call(
        f"tl::tma_store_scatter4<tl::CacheHintSm90::{policy}>",
        *arguments[:-1],
        '#include "tl_templates/cuda/copy_sm100.h"\n',
    )


@register_codegen("tma_store_wait", namespace="tl")
def _lower_tma_store_wait(count, read=None, attrs=None):
    count = _constant(count, "TMA wait group count", 0, 7)
    read = 1 if read is None else _constant(read, "TMA wait read flag", 0, 1)
    return func_call(f"tl::tma_store_wait<{count}, {'true' if read else 'false'}>", '#include "tl_templates/cuda/barrier.h"\n')


@register_codegen("tl_shuffle_elect", namespace="tl")
def _lower_shuffle_elect(extent, attrs=None):
    extent = _constant(extent, "Shuffle election extent", 0, 1024)
    return func_call(f"tl::tl_shuffle_elect<{extent}>", '#include "tl_templates/cuda/intrin.h"\n', ty=ir.PrimType("bool"))


def _register_tmem_lifetime(name, helper):
    @register_codegen(name, namespace="tl")
    def lower(pointer, columns, attrs=None):
        count = _constant(columns, "TMEM column count", 32, 512)
        if count & (count - 1):
            raise ValueError("TMEM allocations require a power-of-two column count")
        if not isinstance(pointer.ty, ir.PtrType) or pointer.ty.element_type != ir.PrimType("uint32"):
            raise TypeError("TMEM allocation handles must be uint32 pointers")
        two_ctas = bool(attrs.get("use_2cta", False)) if attrs is not None else False
        return func_call(f"tl::{helper}<{str(two_ctas).lower()}>", pointer, columns, '#include "tl_templates/cuda/tcgen_05.h"\n')


_register_tmem_lifetime("ptx_init_tensor_memory", "tmem_allocate")
_register_tmem_lifetime("ptx_deallocate_tensor_memory", "tmem_deallocate")


@register_codegen("tcgen05_mma_arrive", namespace="tl")
def _lower_tcgen05_arrive(pointer, attrs=None):
    two_ctas = bool(attrs.get("use_2cta", False)) if attrs is not None else False
    return func_call(
        f"tl::tcgen05_mma_arrive<{str(two_ctas).lower()}>",
        pointer,
        '#include "tl_templates/cuda/tcgen_05.h"\n',
    )


def _register_tmem_transfer(name):
    @register_codegen(name, namespace="tl")
    def lower(bits, chunks, packed, address, offset, *operands, attrs=None):
        width = _constant(bits, "TMEM instruction width", 32, 256)
        count = _constant(chunks, "TMEM transfer chunks", 1, 512)
        pack = _constant(packed, "TMEM packing flag", 0, 1)
        if len(operands) == 1:
            datapaths, pointer = 32, operands[0]
        elif len(operands) == 2:
            datapaths = _constant(operands[0], "TMEM datapaths", 16, 32)
            pointer = operands[1]
        else:
            raise ValueError(f"tl.{name} requires six or seven operands")
        if width not in (32, 64, 128, 256) or datapaths not in (16, 32) or (datapaths == 16 and width == 32):
            raise ValueError("Unsupported TMEM instruction geometry")
        if address.ty != ir.PrimType("uint32") or not isinstance(pointer.ty, ir.PtrType):
            raise TypeError("TMEM transfers require a uint32 address token and a typed register pointer")
        helper = f"tl::{name}_{datapaths}dp{width}bNx<{count}, {str(bool(pack)).lower()}>"
        return func_call(helper, address, offset, pointer, '#include "tl_templates/cuda/copy_sm100.h"\n')


_register_tmem_transfer("tcgen05_ld")
_register_tmem_transfer("tcgen05_st")
