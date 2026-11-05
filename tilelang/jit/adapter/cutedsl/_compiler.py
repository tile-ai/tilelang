"""Lazy CuTeDSL compilation of one native device signature.

This module is imported only on a compilation miss or when the optional CUTLASS
host launcher is requested. Cached provider launches do not import CUTLASS.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import shlex
import sys
import uuid

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass import dsl_user_op
from cutlass._mlir.dialects import cuda as cuda_dialect
from cutlass.cute.runtime import make_fake_stream, make_ptr, nullptr
from cuda.bindings.driver import CUstream


@dsl_user_op
def tensor_map_from_pointer(pointer, *, loc=None, ip=None):
    """Borrow a native host descriptor; the launch copies its 128-byte value."""
    return cuda.TensorMap(cuda_dialect.cast(cuda.TensorMap._get_mlir_type(), pointer.llvm_ptr, loc=loc, ip=ip))


def annotate_tensor_map_arguments(function, indices, arity):
    """Apply native TensorMap parameter types without parsing a source signature."""
    parameters = list(inspect.signature(function).parameters)
    if len(parameters) != arity:
        raise ValueError("CuTeDSL codegen signature differs from its native device Function")
    while function is not None:
        annotations = dict(function.__annotations__)
        for index in indices:
            annotations[parameters[index]] = cutlass.GridConstant[cuda.TensorMap]
        function.__annotations__ = annotations
        function = getattr(function, "__wrapped__", None)


def dtype_for(parameter):
    dtype = parameter["dtype"]
    if dtype == "bool":
        name = "Boolean" if parameter["kind"] == "scalar" else "Uint8"
    elif dtype.startswith("uint"):
        name = "Uint" + dtype[4:]
    elif dtype.startswith("int"):
        name = "Int" + dtype[3:]
    else:
        names = {
            "float16": "Float16",
            "float32": "Float32",
            "float64": "Float64",
            "bfloat16": "BFloat16",
            "float8_e4m3": "Float8E4M3FN",
            "float8_e4m3fn": "Float8E4M3FN",
            "float8_e5m2": "Float8E5M2",
            "float8_e8m0fnu": "Float8E8M0FNU",
            "float6_e3m2fn": "Float6E3M2FN",
            "float6_e2m3fn": "Float6E2M3FN",
            "float4_e2m1fn": "Float4E2M1FN",
            "custom[tf32]32": "Float32",
            "tfloat32": "Float32",
        }
        if dtype in ("custom[float4_e2m1_unpacked]8", "float4_e2m1_unpacked"):
            from tilelang.contrib.cutedsl import Float4E2M1FN_unpack

            return Float4E2M1FN_unpack
        name = names.get(dtype)
    result = getattr(cutlass, name, None) if name else None
    if result is None:
        raise TypeError(f"CuTeDSL does not support native parameter dtype {dtype!r}")
    return result


def _fake(parameter):
    if parameter["kind"] == "stream":
        return make_fake_stream()
    dtype = dtype_for(parameter)
    if parameter["kind"] == "scalar":
        return dtype(1)
    return nullptr(dtype, assumed_align=parameter["alignment"])


def runtime_arguments(parameters, arguments, stream):
    """Fallback arguments for CUTLASS versions without a convertible FFI handle."""
    from tilelang.jit.adapter.device_py_host import as_pointer

    result = []
    for parameter, value in zip(parameters, arguments):
        if parameter["kind"] == "stream":
            result.append(CUstream(as_pointer(value).value or 0))
        elif parameter["kind"] == "scalar":
            result.append(dtype_for(parameter)(value))
        else:
            result.append(make_ptr(dtype_for(parameter), as_pointer(value).value or 0, assumed_align=parameter["alignment"]))
    return [*result, CUstream(stream)]


def launch_source(spec):
    """Generate the CUTLASS launch options from typed call-site metadata."""
    fields = {field["name"]: field for field in spec["fields"]}
    unsupported = set(fields)

    def field(name, default=1):
        unsupported.discard(name)
        value = fields.get(name)
        if value is None:
            return repr(default)
        if value["constant"] is not None:
            return repr(value["constant"])
        return f"_tl_launch_{value['position']}"

    def dims(prefix):
        return "[" + ", ".join(field(f"{prefix}.{axis}") for axis in "xyz") + "]"

    options = [f"grid={dims('grid')}", f"block={dims('block')}", f"smem={field('dynamic_smem_bytes', 0)}", "min_blocks_per_mp=1"]
    has_cluster = any(name.startswith("cluster.") for name in fields)
    has_preferred = any(name.startswith("preferred_cluster.") for name in fields)
    if has_preferred:
        options.extend([f"cluster={dims('preferred_cluster')}", f"fallback_cluster={dims('cluster')}"])
    elif has_cluster:
        options.append(f"cluster={dims('cluster')}")
    options.append(f"stream={field('stream') if 'stream' in fields else '_tl_stream'}")
    options.extend([f"use_pdl={field('programmatic_stream_serialization', False)}", f"cooperative={field('cooperative', False)}"])
    for native, cute_name in (
        ("preferred_shared_memory_carveout", "preferred_smem_carveout"),
        ("programmatic_event.event", "programmatic_event"),
        ("programmatic_event.flags", "programmatic_event_flags"),
        ("programmatic_event.trigger_at_block_start", "programmatic_event_trigger_at_block_start"),
        ("launch_completion_event.event", "launch_completion_event"),
        ("launch_completion_event.flags", "launch_completion_event_flags"),
    ):
        if native in fields:
            options.append(f"{cute_name}={field(native)}")
    # Required-block launches have CUDA 13 cluster-grid encoding in the
    # provider. CUTLASS's host API does not expose that encoding contract.
    if spec["kernel_attrs"].get("required_block_size"):
        unsupported.add("required_block_size")
    return ",\n        ".join(options), sorted(unsupported)


def compile_kernel(source, spec, arch, compile_flags, output):
    """Return a real compiled handle and the cubin's compiler-reported symbol."""
    symbol = spec["symbol"]
    if not symbol.isidentifier():
        raise ValueError(f"Invalid native CuTeDSL kernel symbol: {symbol!r}")
    output = Path(output)
    parameters = spec["parameters"]
    launch_parameters = [dict(parameter) for parameter in spec["launch_parameters"]]
    for field in spec["fields"]:
        if field["name"] == "stream" and field["position"] is not None:
            launch_parameters[field["position"]] = {"kind": "stream", "dtype": "handle"}
    argument_names = [f"_tl_arg_{index}" for index in range(len(parameters))]
    launch_names = [
        f"_tl_launch_{index}" + (": CUstream" if parameter["kind"] == "stream" else "") for index, parameter in enumerate(launch_parameters)
    ]
    device_arguments = []
    tensor_maps = []
    for index, parameter in enumerate(parameters):
        name = argument_names[index]
        if parameter["kind"] == "tensormap":
            tensor_maps.append(index)
            device_arguments.append(f"tensor_map_from_pointer({name})")
        elif parameter["kind"] == "pointer":
            device_arguments.append(f"cute.make_tensor({name}, cute.make_layout(1))")
        else:
            device_arguments.append(name)
    launch, unsupported = launch_source(spec)
    host_source = (
        "import cutlass\nimport cutlass.cute as cute\nimport cutlass.experimental.cuda as cuda\n"
        "from cuda.bindings.driver import CUstream\nimport tilelang.contrib.cutedsl as tl\n"
        "from tilelang.jit.adapter.cutedsl._compiler import tensor_map_from_pointer, annotate_tensor_map_arguments\n\n"
        + source
        + "\n\n"
        + f"annotate_tensor_map_arguments({symbol}, {tensor_maps!r}, {len(parameters)})\n\n"
        + "@cute.jit\n"
        + f"def _tl_native_compile_host({', '.join([*argument_names, *launch_names, '_tl_stream: CUstream'])}):\n"
        + f"    {symbol}({', '.join(device_arguments)}).launch(\n        {launch},\n    )\n"
    )
    path = output / "compile.py"
    path.write_text(host_source)
    module_name = "tilelang_cutedsl_compile_" + uuid.uuid4().hex
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
        options = ["--enable-tvm-ffi", "--keep-cubin", f"--gpu-arch={arch}", f"--dump-dir={output}"]
        for flag in compile_flags or ():
            options.extend(shlex.split(flag))
        compiled = cute.compile(
            module._tl_native_compile_host,
            *(_fake(parameter) for parameter in [*parameters, *launch_parameters]),
            make_fake_stream(),
            options=shlex.join(options),
        )
    finally:
        sys.modules.pop(module_name, None)
    symbols = list(compiled.kernel_info)
    if len(symbols) != 1:
        raise RuntimeError(f"Expected one CuTeDSL device symbol, got {symbols}")
    cubins = list(output.glob("*.cubin"))
    if len(cubins) != 1:
        raise RuntimeError(f"Expected one CuTeDSL cubin under {output}, got {cubins}")
    return (
        compiled,
        cubins[0],
        {"symbol": symbols[0], "parameters": [*parameters, *launch_parameters], "unsupported_cutlass_options": unsupported},
    )
