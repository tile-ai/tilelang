"""Compile native CuTe kernels and execute their TIRx host program."""

import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import weakref

from tilelang.jit.adapter.nvrtc.runtime import NativeHostProgram, _check


_DTYPES = {
    "bool": "Uint8",
    "int4": "Int4",
    "uint4": "Uint4",
    "int8": "Int8",
    "uint8": "Uint8",
    "int16": "Int16",
    "uint16": "Uint16",
    "int32": "Int32",
    "uint32": "Uint32",
    "int64": "Int64",
    "uint64": "Uint64",
    "float16": "Float16",
    "bfloat16": "BFloat16",
    "float32": "Float32",
    "float64": "Float64",
    "float8_e4m3fn": "Float8E4M3FN",
    "float8_e5m2": "Float8E5M2",
    "float4_e2m1fn": "Float4E2M1FN",
}


def _dtype_name(dtype):
    try:
        return "cutlass." + _DTYPES[dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported native CuTeDSL dtype: {dtype}") from error


def _compilation_source(source, plans):
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    parameters = []
    fake_arguments = []
    statements = []
    for kernel_index, plan in enumerate(plans):
        kernel_arguments = []
        for parameter_index, parameter in enumerate(plan["parameters"]):
            name = f"kernel_{kernel_index}_argument_{parameter_index}"
            dtype = _dtype_name(parameter["dtype"])
            parameters.append(name)
            if parameter["kind"] in ("tensor", "tensormap"):
                fake_arguments.append(f"make_fake_compact_tensor({dtype}, (32,), assumed_align=16)")
            else:
                fake_arguments.append(f"{dtype}(1)")
            if parameter["kind"] == "tensormap":
                functions[plan["name"]].args.args[parameter_index].annotation = ast.parse(
                    "cutlass.GridConstant[cuda.TensorMap]", mode="eval"
                ).body
                descriptor = name + "_descriptor"
                rank = parameter["rank"]
                statements.append(
                    f"    {descriptor} = cuda.create_tensor_map_tiled("
                    f"global_address={name}.iterator.toint(), dtype={dtype}, "
                    f"global_dims={[32] * rank!r}, global_strides={[2] * (rank - 1)!r}, "
                    f"box_dims={[16] + [1] * (rank - 1)!r}, traversal_strides={[1] * rank!r}, "
                    f"swizzle=cuda.TensorMapSwizzle({parameter['swizzle']}))"
                )
                kernel_arguments.append(descriptor)
            else:
                kernel_arguments.append(name)
        fields = {}
        for field, value in plan["fields"].items():
            if value is None:
                argument = f"kernel_{kernel_index}_{field.replace('.', '_')}"
                parameters.append(argument)
                fake_arguments.append("cutlass.Int32(1)")
                fields[field] = argument
            else:
                fields[field] = str(value)
        block = ", ".join(fields.pop("block." + axis, "1") for axis in "xyz")
        shared = fields.pop("dynamic_smem_bytes", "0")
        pdl = fields.pop("programmatic_stream_serialization", "0")
        cooperative = fields.pop("cooperative", "0")
        cluster = ""
        if any(field.startswith("cluster.") for field in fields):
            dimensions = ", ".join(fields.pop("cluster." + axis, "1") for axis in "xyz")
            cluster = f", cluster=[{dimensions}]"
        for axis in "xyz":
            fields.pop("preferred_cluster." + axis, None)
        if fields:
            raise NotImplementedError(f"Unsupported native CuTe compilation fields: {sorted(fields)}")
        statements.append(
            f"    {plan['name']}({', '.join(kernel_arguments)}).launch("
            f"grid=[1, 1, 1], block=[{block}], smem={shared}, stream=stream, "
            f"use_pdl={pdl}, cooperative={cooperative}{cluster})"
        )
    if not statements:
        raise ValueError("CuTeDSL module has no device kernels")
    parameters.append("stream: CUstream")
    fake_arguments.append("make_fake_stream()")
    ast.fix_missing_locations(tree)
    return (
        "import cutlass\nimport cutlass.cute as cute\nimport cutlass.experimental.cuda as cuda\n"
        "import tilelang.contrib.cutedsl as tl\nfrom cuda.bindings.driver import CUstream\n"
        "from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream\n"
        + ast.unparse(tree)
        + "\n\n@cute.jit\ndef _compile_kernels("
        + ", ".join(parameters)
        + "):\n"
        + "\n".join(statements)
        + "\n\ndef compile_kernels(options):\n"
        + "    return cute.compile(_compile_kernels, "
        + ", ".join(fake_arguments)
        + ", options=options)\n"
    )


class CuTeHostProgram:
    def __init__(self, serialized_function, source, plans, architecture, module_path):
        self.host = NativeHostProgram(serialized_function)
        self.source = source
        self.plans = plans
        self.architecture = architecture
        self.cubin_path = Path(module_path).with_suffix(".cubin")
        self.kernels = None
        self.lock = threading.Lock()

    def _compile(self):
        if self.cubin_path.is_file():
            return
        code = _compilation_source(self.source, self.plans)
        with tempfile.TemporaryDirectory(prefix="cutedsl_", dir=self.cubin_path.parent) as directory:
            staging = Path(directory)
            source_path = staging / "compile_kernels.py"
            source_path.write_text(code)
            spec = importlib.util.spec_from_file_location("_tilelang_cutedsl_compile", source_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.compile_kernels(f"--enable-tvm-ffi --keep-cubin --gpu-arch={self.architecture} --dump-dir={staging.as_posix()}")
            cubins = list(staging.glob("*.cubin"))
            if len(cubins) != 1:
                raise RuntimeError(f"Expected one CuTe cubin, found {len(cubins)}")
            os.replace(cubins[0], self.cubin_path)

    def _load(self):
        from cuda.bindings import driver

        self._compile()
        library = _check(driver.cuLibraryLoadFromFile(os.fsencode(self.cubin_path), [], [], 0, [], [], 0))
        try:
            count = _check(driver.cuLibraryGetKernelCount(library))
            if count == 0:
                raise RuntimeError(
                    f"CuTe cubin for {self.architecture} contains no loadable kernels; "
                    "select an architecture compatible with the execution device"
                )
            available = {}
            for kernel in _check(driver.cuLibraryEnumerateKernels(count, library)):
                name = _check(driver.cuKernelGetName(kernel))
                available[name.decode() if isinstance(name, bytes) else name] = kernel
            kernels = {}
            for plan in sorted(self.plans, key=lambda plan: len(plan["name"]), reverse=True):
                matches = [name for name in available if plan["name"] in name]
                if len(matches) != 1:
                    raise RuntimeError(f"CuTe kernel {plan['name']} resolves to {matches}")
                kernels[plan["name"]] = available.pop(matches[0])
            self.kernels = kernels
        except BaseException:
            driver.cuLibraryUnload(library)
            raise
        self._finalize = weakref.finalize(self, driver.cuLibraryUnload, library)

    def __call__(self, arguments, stream=0, device_id=0):
        import torch

        with torch.cuda.device(device_id):
            if self.kernels is None:
                with self.lock:
                    if self.kernels is None:
                        self._load()
            return self.host(self.kernels, arguments, stream or 0)
