"""Execute native host control flow with CUDA Driver API kernel launches."""

import ctypes
import math
import operator
import struct

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.ir import prim


_CTYPES = {
    "bool": ctypes.c_bool,
    "int8": ctypes.c_int8,
    "uint8": ctypes.c_uint8,
    "int16": ctypes.c_int16,
    "uint16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}
_BINARY = {
    prim.Add: operator.add,
    prim.Sub: operator.sub,
    prim.Mul: operator.mul,
    prim.FloorDiv: operator.floordiv,
    prim.FloorMod: operator.mod,
    prim.Min: min,
    prim.Max: max,
    prim.EQ: operator.eq,
    prim.NE: operator.ne,
    prim.LT: operator.lt,
    prim.LE: operator.le,
    prim.GT: operator.gt,
    prim.GE: operator.ge,
    prim.LShift: operator.lshift,
    prim.RShift: operator.rshift,
    prim.BitwiseAnd: operator.and_,
    prim.BitwiseOr: operator.or_,
    prim.BitwiseXor: operator.xor,
}


def _check(result):
    from cuda.bindings import driver

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA Driver API failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def _cast(value, native_type):
    if isinstance(native_type, ir.PtrType):
        return value.value or 0 if isinstance(value, ctypes.c_void_p) else value
    dtype = str(native_type.dtype)
    if dtype == "void":
        return None
    if dtype == "float16":
        return struct.unpack("e", struct.pack("e", value))[0]
    if dtype not in _CTYPES:
        raise TypeError(f"Unsupported NVRTC host scalar type: {native_type}")
    return _CTYPES[dtype](value).value


class _TensorMap:
    def __init__(self):
        self.value = None
        self._storage = ctypes.create_string_buffer(255)
        self.address = (ctypes.addressof(self._storage) + 127) & -128

    def packed_pointer(self):
        if self.value is not None:
            ctypes.memmove(self.address, self.value.getPtr(), 128)
        return ctypes.c_void_p(self.address)

    def load_packed_value(self):
        from cuda.bindings import driver

        self.value = driver.CUtensorMap()
        ctypes.memmove(self.value.getPtr(), self.address, 128)


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class NativeHostProgram:
    """Keep native lexical bindings and control flow without parsing printed IR."""

    def __init__(self, serialized_function):
        self.function = ir.load_json(serialized_function)

    def __call__(self, kernels, arguments, stream=0):
        if len(arguments) + 1 != len(self.function.params):
            raise TypeError(f"Expected {len(self.function.params) - 1} host arguments, got {len(arguments)}")
        values = {}
        for parameter, value in zip(self.function.params, (*arguments, stream), strict=True):
            if isinstance(parameter.ty, ir.PtrType) and hasattr(value, "data_ptr"):
                value = value.data_ptr()
            values[parameter] = _cast(value, parameter.ty)
        execution = _Execution(kernels, values)
        try:
            execution.statement(self.function.body)
        except _Return as result:
            return result.value


class _Execution:
    def __init__(self, kernels, values):
        self.kernels = kernels
        self.values = values

    def expression(self, node):
        if isinstance(node, ir.Var):
            return self.values[node]
        if isinstance(node, (ir.IntImm, ir.FloatImm, ir.StringImm)):
            return node.value
        if isinstance(node, ir.DataTypeImm):
            return node.value
        if isinstance(node, ir.Tuple):
            return tuple(self.expression(value) for value in node.fields)
        if isinstance(node, prim.Cast):
            return _cast(self.expression(node.value), node.ty)
        if isinstance(node, prim.And):
            return self.expression(node.a) and self.expression(node.b)
        if isinstance(node, prim.Or):
            return self.expression(node.a) or self.expression(node.b)
        if isinstance(node, prim.Not):
            return not self.expression(node.a)
        if isinstance(node, prim.BitwiseNot):
            return _cast(~self.expression(node.a), node.ty)
        if isinstance(node, prim.Select):
            return self.expression(node.true_value if self.expression(node.condition) else node.false_value)
        if isinstance(node, prim.Let):
            self.values[node.var] = self.expression(node.value)
            return self.expression(node.body)
        if isinstance(node, (prim.Div, prim.Mod)):
            left, right = self.expression(node.a), self.expression(node.b)
            if str(node.ty.dtype).startswith("float"):
                value = left / right if isinstance(node, prim.Div) else math.fmod(left, right)
            else:
                quotient = abs(left) // abs(right) * (-1 if (left < 0) != (right < 0) else 1)
                value = quotient if isinstance(node, prim.Div) else left - quotient * right
            return _cast(value, node.ty)
        if type(node) in _BINARY:
            return _cast(_BINARY[type(node)](self.expression(node.a), self.expression(node.b)), node.ty)
        if isinstance(node, ir.Call):
            return self.call(node)
        raise TypeError(f"Unsupported native NVRTC host expression: {type(node).__name__}")

    def statement(self, node):
        if isinstance(node, ir.SeqStmt):
            for child in node.seq:
                self.statement(child)
        elif isinstance(node, ir.Bind):
            self.values[node.var] = self.expression(node.value)
        elif isinstance(node, ir.Evaluate):
            self.expression(node.value)
        elif isinstance(node, ir.If):
            branch = node.then_case if self.expression(node.condition) else node.else_case
            if branch is not None:
                self.statement(branch)
        elif isinstance(node, ir.For):
            begin, end, step = (self.expression(value) for value in (node.begin, node.end, node.step))
            if step == 0:
                raise ValueError("Native host loop has zero step")
            current = begin
            while current < end if step > 0 else current > end:
                self.values[node.var] = _cast(current, node.var.ty)
                try:
                    self.statement(node.body)
                except _Break:
                    break
                except _Continue:
                    pass
                current += step
        elif isinstance(node, ir.While):
            while self.expression(node.condition):
                try:
                    self.statement(node.body)
                except _Break:
                    break
                except _Continue:
                    pass
        elif isinstance(node, ir.Break):
            raise _Break()
        elif isinstance(node, ir.Continue):
            raise _Continue()
        elif isinstance(node, ir.Return):
            raise _Return(self.expression(node.value))
        elif isinstance(node, ir.AssertStmt):
            if not self.expression(node.condition):
                raise RuntimeError("".join(str(self.expression(value)) for value in node.message_parts))
        elif isinstance(node, ir.RegionStmt) and node.op.name in ("tirx.scope", "tirx.compute_scope", "tl.scope"):
            self.statement(node.body)
        else:
            raise TypeError(f"Unsupported native NVRTC host statement: {type(node).__name__}")

    def call(self, node):
        name = node.op.name
        if name == "tirx.call_ffi_kernel":
            return self.launch(node)
        if name == "tirx.address_of":
            load = node.args[0]
            if not isinstance(load, ir.TensorLoad):
                raise TypeError("Native host address_of requires a tensor element")
            tensor_type = load.source.ty
            if tensor_type.layout is None:
                offset = 0
                for index, extent in zip(load.indices, tensor_type.shape, strict=True):
                    offset = offset * self.expression(extent) + self.expression(index)
            else:
                coordinates = tensor_type.layout.apply(*load.indices, shape=tensor_type.shape)
                if set(coordinates) != {"m"}:
                    raise ValueError("Native host tensor addresses require a memory-only layout")
                offset = self.expression(coordinates["m"])
            dtype = tensor_type.dtype.dtype
            bit_offset = offset * dtype.bits * dtype.lanes
            if bit_offset % 8:
                raise ValueError("Native host tensor address is not byte-aligned")
            return self.expression(load.source) + bit_offset // 8
        arguments = [self.expression(argument) for argument in node.args]
        if name in ("tirx.decl_tensor", "tirx.tensor_data_ptr", "tirx.reinterpret"):
            return arguments[0]
        if name in ("tirx.ptr_byte_offset", "tirx.handle_add_byte_offset"):
            return arguments[0] + arguments[1]
        if name == "tirx.stack_alloca" and arguments == ["tensormap", 1]:
            return _TensorMap()
        if name == "tirx.cuda.tensormap_encode_tiled":
            return self.encode_tiled(node, arguments)
        if name in ("tirx.assume", "prim.assume"):
            if not arguments[0]:
                raise ValueError("Native host assumption was violated")
            return None
        if name == "tirx.call_packed":
            packed_arguments = []
            for expression, value in zip(node.args[1:], arguments[1:], strict=True):
                if isinstance(value, _TensorMap):
                    value = value.packed_pointer()
                elif isinstance(expression.ty, ir.PtrType) and not isinstance(value, ctypes.c_void_p):
                    value = ctypes.c_void_p(value)
                packed_arguments.append(value)
            result = tvm_ffi.get_global_func(arguments[0])(*packed_arguments)
            if arguments[0] in ("__tvm_tensormap_create_im2col", "__tvm_tensormap_create_tiled"):
                arguments[1].load_packed_value()
            return _cast(result, node.ty) if isinstance(node.ty, (ir.PrimType, ir.PtrType)) else result
        if name in ("tirx.call_extern", "tirx.call_pure_extern") and arguments[0] in ("sqrt", "sqrtf", "ceil", "floor"):
            return getattr(math, arguments[0].removesuffix("f"))(*arguments[1:])
        raise NotImplementedError(f"NVRTC host operation is not supported: {name}")

    def encode_tiled(self, node, arguments):
        from cuda.bindings import driver

        attributes = node.attrs
        rank = attributes.rank
        dtype_names = {
            "uint8": "UINT8",
            "uint16": "UINT16",
            "uint32": "UINT32",
            "uint64": "UINT64",
            "int32": "INT32",
            "int64": "INT64",
            "float16": "FLOAT16",
            "float32": "FLOAT32",
            "float64": "FLOAT64",
            "bfloat16": "BFLOAT16",
        }
        dtype = str(attributes.descriptor_dtype)
        encoding = attributes.force_cu_dtype
        if encoding < 0:
            encoding = getattr(driver.CUtensorMapDataType, "CU_TENSOR_MAP_DATA_TYPE_" + dtype_names[dtype])
        dimensions = arguments[2 : 2 + rank]
        strides = arguments[2 + rank : 1 + 2 * rank]
        box = arguments[1 + 2 * rank : 1 + 3 * rank]
        element_strides = arguments[1 + 3 * rank : 1 + 4 * rank]
        if any(value < 0 for value in (*dimensions, *strides, *box, *element_strides)):
            raise ValueError("Negative tensor-map dimension or stride")
        if any(value > 0xFFFFFFFF for value in (*box, *element_strides)):
            raise ValueError("Tensor-map dimension or stride exceeds uint32")
        arguments[0].value = _check(
            driver.cuTensorMapEncodeTiled(
                driver.CUtensorMapDataType(encoding),
                driver.cuuint32_t(rank),
                arguments[1],
                [driver.cuuint64_t(value) for value in dimensions],
                [driver.cuuint64_t(value) for value in strides],
                [driver.cuuint32_t(value) for value in box],
                [driver.cuuint32_t(value) for value in element_strides],
                driver.CUtensorMapInterleave(attributes.interleave),
                driver.CUtensorMapSwizzle(attributes.swizzle),
                driver.CUtensorMapL2promotion(attributes.l2_promotion),
                driver.CUtensorMapFloatOOBfill(attributes.oob_fill),
            )
        )
        return 0

    def launch(self, node):
        from cuda.bindings import driver

        arguments = [self.expression(argument) for argument in node.args]
        count = node.attrs.num_kernel_args
        fields = dict(zip(node.attrs.launch_fields, arguments[1 + count :], strict=True))
        kernel = self.kernels[arguments[0]]
        config = driver.CUlaunchConfig()
        for axis in "xyz":
            grid = int(fields.pop(f"grid.{axis}", 1))
            block = int(fields.pop(f"block.{axis}", 1))
            if not 0 < grid <= 0xFFFFFFFF or not 0 < block <= 0xFFFFFFFF:
                raise ValueError("CUDA grid and block dimensions must be positive uint32 values")
            setattr(config, "gridDim" + axis.upper(), grid)
            setattr(config, "blockDim" + axis.upper(), block)
        config.hStream = fields.pop("stream", 0)
        config.sharedMemBytes = int(fields.pop("dynamic_smem_bytes", 0))
        device = _check(driver.cuCtxGetDevice())
        if config.sharedMemBytes:
            _check(
                driver.cuKernelSetAttribute(
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    config.sharedMemBytes,
                    kernel,
                    device,
                )
            )
        attributes = []
        for prefix, enum_name, field_name in (
            ("cluster", "CLUSTER_DIMENSION", "clusterDim"),
            ("preferred_cluster", "PREFERRED_CLUSTER_DIMENSION", "preferredClusterDim"),
        ):
            if any(f"{prefix}.{axis}" in fields for axis in "xyz"):
                attribute = driver.CUlaunchAttribute()
                attribute.id = getattr(driver.CUlaunchAttributeID, "CU_LAUNCH_ATTRIBUTE_" + enum_name)
                for axis in "xyz":
                    setattr(getattr(attribute.value, field_name), axis, int(fields.pop(f"{prefix}.{axis}", 1)))
                attributes.append(attribute)
                _check(
                    driver.cuKernelSetAttribute(
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                        1,
                        kernel,
                        device,
                    )
                )
        for field, enum_name, member in (
            ("programmatic_stream_serialization", "PROGRAMMATIC_STREAM_SERIALIZATION", "programmaticStreamSerializationAllowed"),
            ("cooperative", "COOPERATIVE", "cooperative"),
        ):
            if field in fields:
                attribute = driver.CUlaunchAttribute()
                attribute.id = getattr(driver.CUlaunchAttributeID, "CU_LAUNCH_ATTRIBUTE_" + enum_name)
                setattr(attribute.value, member, int(fields.pop(field)))
                attributes.append(attribute)
        if fields:
            raise NotImplementedError(f"Unsupported native CUDA launch fields: {sorted(fields)}")
        config.attrs = attributes
        config.numAttrs = len(attributes)
        values, ctypes_types = [], []
        for expression, value in zip(node.args[1 : 1 + count], arguments[1 : 1 + count], strict=True):
            if isinstance(value, _TensorMap):
                if value.value is None:
                    raise ValueError("Uninitialized CUDA tensor map")
                values.append(value.value)
                ctypes_types.append(None)
            elif isinstance(expression.ty, ir.PtrType):
                values.append(value)
                ctypes_types.append(ctypes.c_void_p)
            else:
                dtype = str(expression.ty.dtype)
                if dtype == "float16":
                    value = int.from_bytes(struct.pack("e", value), "little")
                    dtype = "uint16"
                values.append(value)
                ctypes_types.append(_CTYPES[dtype])
        _check(driver.cuLaunchKernelEx(config, kernel, (tuple(values), tuple(ctypes_types)), 0))
        return 0
