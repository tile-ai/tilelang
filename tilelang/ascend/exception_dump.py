"""Preserve host tensor extents for opt-in Ascend exception dumps."""

from __future__ import annotations

from tvm import DataType, IRModule, arith, ir, tirx

EXCEPTION_DUMP_ATTR = "tl.ascend_exception_dump_tensors"
EXCEPTION_DUMP_TAG = "tl.ascend_exception_dump.v1"
MAX_DUMP_TENSORS = 16
# Each attribute entry is [pointer_arg_index, numel_arg_index, acl_dtype, bits].
# Indices refer to the split device signature, whose order can differ from the
# user function. The final by-value ExceptionDumpInfo is added by codegen later.

# Values from aclDataType. Unsupported formats fail while compiling instead of
# emitting a dump with a misleading dtype or an incorrect allocation size.
_ACL_DTYPES = {
    "float32": 0,
    "float16": 1,
    "int8": 2,
    "int32": 3,
    "uint8": 4,
    "int16": 6,
    "uint16": 7,
    "uint32": 8,
    "int64": 9,
    "uint64": 10,
    "float64": 11,
    "bool": 12,
    "bfloat16": 27,
}


def annotate_exception_dump(mod: IRModule) -> IRModule:
    """Append host-computed tensor lengths after SplitHostDevice.

    The host still owns its buffer map here, including dynamic shapes. Device
    pointers alone no longer retain that information after LowerAccessPtr.
    Keeping lengths as ordinary scalar kernel arguments lets the existing host
    codegen, launch lowering and serialized FunctionInfo carry them unchanged.
    """
    analyzer = arith.Analyzer()
    plans = {}
    updates = {}

    for global_var, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            continue
        # Match by data Var identity, not printed names. DMA-only device bodies
        # may already have lost Buffer nodes, while the host map still has shape.
        buffers = {buffer.data: buffer for buffer in func.buffer_map.values()}

        def rewrite_call(node, buffers=buffers):
            if not isinstance(node, tirx.Call) or not isinstance(node.op, ir.GlobalVar):
                return None
            callee = mod[node.op]
            if not isinstance(callee, tirx.PrimFunc) or not callee.attrs:
                return None
            target = callee.attrs.get("target")
            if target is None or target.kind.name != "ascend":
                return None
            if "code_block_source" in callee.attrs:
                raise ValueError("Ascend exception dump does not support source-code kernels")
            metadata = []
            lengths = []
            for pointer_index, argument in enumerate(node.args):
                buffer = buffers.get(argument) if isinstance(argument, tirx.Var) else None
                if buffer is None:
                    continue
                dtype = str(buffer.dtype)
                if dtype not in _ACL_DTYPES:
                    raise ValueError(f"Ascend exception dump does not support tensor dtype {dtype}")
                # Canonical lowering may materialize dense strides explicitly.
                # Only reject layouts with holes or an unknown starting offset.
                if buffer.strides:
                    expected_stride = tirx.const(1, "int64")
                    for stride, extent in zip(reversed(buffer.strides), reversed(buffer.shape)):
                        if not analyzer.can_prove_equal(stride, expected_stride):
                            raise ValueError("Ascend exception dump requires contiguous tensor buffers")
                        expected_stride = expected_stride * tirx.Cast("int64", extent)
                if not analyzer.can_prove_equal(buffer.elem_offset, 0):
                    raise ValueError("Ascend exception dump requires a zero tensor element offset")
                elements = tirx.const(1, "uint64")
                for dimension in buffer.shape:
                    # Widen each factor before multiplying; casting only the
                    # product would preserve an earlier int32 overflow.
                    elements = elements * tirx.Cast("uint64", dimension)
                elements = analyzer.simplify(elements)
                bits = 8 if dtype == "bool" else DataType(dtype).bits
                metadata.append((pointer_index, len(callee.params) + len(lengths), _ACL_DTYPES[dtype], bits))
                lengths.append(elements)
            if len(metadata) > MAX_DUMP_TENSORS:
                raise ValueError(f"Ascend exception dump supports at most {MAX_DUMP_TENSORS} tensors per kernel")
            previous = plans.get(node.op)
            # Calls to one device function share its parameter layout, but may
            # supply different runtime lengths for their dynamic buffers.
            if previous is not None and previous != metadata:
                raise ValueError("Ascend exception dump requires consistent tensor arguments at each kernel call")
            plans[node.op] = metadata
            return tirx.Call(node.dtype, node.op, [*node.args, *lengths], annotations=node.annotations, span=node.span)

        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite_call, ["tirx.Call"])
        updates[global_var] = func.with_body(body, span=func.span)

    for global_var, metadata in plans.items():
        func = updates[global_var]
        name = str(func.attrs.get("global_symbol", global_var.name_hint))
        if len(name.encode("utf-8")) >= 128:
            raise ValueError("Ascend exception dump kernel names must contain fewer than 128 UTF-8 bytes")
        params = [*func.params, *(tirx.Var(f"__tl_dump_numel_{i}", "uint64") for i in range(len(metadata)))]
        func = tirx.PrimFunc(params, func.body, func.ret_type, func.buffer_map, func.attrs, func.span)
        updates[global_var] = func.with_attr(EXCEPTION_DUMP_ATTR, [list(entry) for entry in metadata])

    result = mod.clone()
    for global_var, func in updates.items():
        result.update_func(global_var, func)
    return result
