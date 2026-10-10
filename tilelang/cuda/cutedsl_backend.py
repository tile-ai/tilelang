"""CuTeDSL backend manifest sharing the CUDA lowering pipeline."""

import re
from importlib import metadata as _importlib_metadata
from importlib.util import find_spec as _find_spec
import os

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline


_CUTEDSL_PUBLIC_DIST = "nvidia-cutlass-dsl"
_CUTEDSL_MIN_VERSION = (4, 7, 0)
_CUTEDSL_BANNED_VERSIONS = {(4, 3, 4)}  # Known broken versions
_VERSION_TRIPLE_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _parse_version_triple(version_str: str) -> tuple[int, int, int] | None:
    """Parse a best-effort (major, minor, patch) triple from a version string.

    We intentionally avoid importing heavy/optional version parsers. For our
    minimum requirement (>= 4.7.0), a numeric triple comparison is sufficient.
    """
    m = _VERSION_TRIPLE_RE.search(version_str)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _min_version_str() -> str:
    return ".".join(map(str, _CUTEDSL_MIN_VERSION))


def _requirement_spec() -> str:
    spec = f"{_CUTEDSL_PUBLIC_DIST}>={_min_version_str()}"
    for banned in _CUTEDSL_BANNED_VERSIONS:
        spec += f",!={'.'.join(map(str, banned))}"
    return spec


def check_cutedsl_available() -> None:
    """Fail fast if the CuTeDSL backend cannot be used in this Python environment.

    Policy:
    - If the public distribution `nvidia-cutlass-dsl` is installed, require version >= a minimum supported version.
    - Regardless of distribution metadata, require that `cutlass.cute` is importable.

    This intentionally does not mention or special-case any internal distributions.
    """
    # 1) Version gate (only when the public dist metadata is present)
    try:
        dist_version = _importlib_metadata.version(_CUTEDSL_PUBLIC_DIST)
    except _importlib_metadata.PackageNotFoundError:
        dist_version = None
    except Exception:
        # Metadata is best-effort; don't block internal/nonstandard installs here.
        dist_version = None

    if dist_version is not None:
        parsed = _parse_version_triple(dist_version)
        if parsed is None or parsed < _CUTEDSL_MIN_VERSION:
            req = _requirement_spec()
            raise ImportError(
                f"CuTeDSL backend requires `{req}`, but found version `{dist_version}`. Please run: `pip install -U '{req}'`."
            )
        if parsed in _CUTEDSL_BANNED_VERSIONS:
            req = _requirement_spec()
            raise ImportError(
                f"CuTeDSL version `{dist_version}` is known to have compatibility issues and is not supported. Please run: `pip install -U '{req}'`."
            )

    # 2) Capability probe: keep it cheap.
    # Importing cutlass/cute can be expensive and defeats our lazy-import design,
    # especially on cache hits. We only require that the module is importable.
    cutlass_spec = _find_spec("cutlass")
    if cutlass_spec is None:
        req = _requirement_spec()
        raise ImportError(f"CuTeDSL backend requires the CUTLASS Python DSL with CuTe support (install via `pip install -U '{req}'`).")

    # Avoid find_spec("cutlass.cute") which can be surprisingly expensive.
    # Instead, check for a 'cute' submodule/package under cutlass's search locations.
    locs = getattr(cutlass_spec, "submodule_search_locations", None)
    has_cute = False
    if locs:
        for base in locs:
            if os.path.isdir(os.path.join(base, "cute")) or os.path.isfile(os.path.join(base, "cute.py")):
                has_cute = True
                break

    if not has_cute:
        req = _requirement_spec()
        raise ImportError(f"CuTeDSL backend requires the CUTLASS Python DSL with CuTe support (install via `pip install -U '{req}'`).")


CUTEDSL_TYPES: dict[str, str] = {
    "float32": "cutlass.Float32",
    "float16": "cutlass.Float16",
    "bfloat16": "cutlass.BFloat16",
    "float8_e4m3": "cutlass.Float8E4M3",
    "float8_e4m3fn": "cutlass.Float8E4M3FN",
    "float8_e5m2": "cutlass.Float8E5M2",
    "float4_e2m1fn": "cutlass.Float4E2M1FN",
    "float64": "cutlass.Float64",
    "int64": "cutlass.Int64",
    "int32": "cutlass.Int32",
    "uint32": "cutlass.Uint32",
    "bool": "cutlass.Uint8",  # CuTeDSL only supports i1 in rmem; use u8 for gmem
    "int8": "cutlass.Int8",
    "uint8": "cutlass.Uint8",
    "int16": "cutlass.Int16",
    "uint16": "cutlass.Uint16",
    "uchar": "cutlass.Uint8",
}


def compile_cutedsl(code, func, target):
    """Compile one device entry; host control flow remains in TileLang Host IR."""
    import importlib.util
    import inspect
    from pathlib import Path
    import re
    import tempfile

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from tvm import tirx

    from tilelang.contrib.nvcc import get_target_arch_and_code, get_target_code_list

    arch, target_code = get_target_arch_and_code(target)
    if target_code and get_target_code_list(target_code) != [f"sm_{arch}"]:
        raise ValueError("CuTeDSL tvm_ffi requires a single code target matching arch")

    args, names, descriptors = [], [], []
    setup = ""
    for index, param in enumerate(func.params):
        name = f"p{index}"
        if param.dtype == "handle":
            pointer = param.type_annotation
            if pointer.storage_scope == "grid_constant":
                descriptors.append(index)
                # Only the descriptor type is needed for device compilation.
                # The real descriptor is constructed by the compiled Host IR.
                setup += (
                    f"    {name} = cuda.create_tensor_map_tiled(global_address=cutlass.Int64(0), dtype=cutlass.Uint8, "
                    "global_dims=[16], global_strides=[], box_dims=[16], traversal_strides=[1])\n"
                )
                continue
            dtype = getattr(cutlass, CUTEDSL_TYPES[str(pointer.element_type.dtype)].removeprefix("cutlass."))
            args.append(make_fake_compact_tensor(dtype, (1,), stride_order=(0,), assumed_align=16))
        else:
            if str(param.dtype) not in ("int32", "uint32", "int64", "float32", "float64"):
                raise ValueError(f"CuTeDSL tvm_ffi does not support device scalar type {param.dtype}")
            args.append(getattr(cutlass, CUTEDSL_TYPES[str(param.dtype)].removeprefix("cutlass."))(0))
        names.append(name)
    extents = func.attrs["thread_extent"]
    block = tuple(extents.get(f"threadIdx.{axis}", 1) for axis in "xyz")
    if any(not isinstance(value, (int, tirx.IntImm)) for value in block):
        raise ValueError("CuTeDSL compilation requires constant thread extents")
    block = tuple(int(value) for value in block)
    symbol = str(func.attrs["global_symbol"])
    cluster = tuple(int(value) for value in func.attrs["cluster_dims"]) if "cluster_dims" in func.attrs else None
    use_pdl = bool(func.attrs.get("has_cuda_pdl_sync", False))
    source = (
        "import cutlass\nimport cutlass.cute as cute\nimport tilelang.contrib.cutedsl as tl\nfrom cuda.bindings.driver import CUstream\n"
    )
    if descriptors:
        source += "from cutlass.experimental import cuda\n"
    source += code + "\n@cute.jit\ndef compile_only(" + ", ".join(names + ["stream: CUstream"]) + "):\n"
    call_args = ", ".join(f"p{i}" for i in range(len(func.params)))
    source += (
        setup
        + f"    {symbol}({call_args}).launch(grid={cluster or (1, 1, 1)!r}, block={block!r}, cluster={cluster!r}, use_pdl={use_pdl}, stream=stream)\n"
    )
    with tempfile.TemporaryDirectory(prefix="tilelang_cutedsl_") as directory:
        path = Path(directory) / "device.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("tilelang_cutedsl_device", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if descriptors:
            from cutlass.experimental.cuda import TensorMap

            kernel = inspect.unwrap(getattr(module, symbol))
            parameters = list(inspect.signature(kernel).parameters)
            for index in descriptors:
                kernel.__annotations__[parameters[index]] = cutlass.GridConstant[TensorMap]
        compiled = cute.compile(
            module.compile_only,
            *args,
            make_fake_stream(),
            options=f"--enable-tvm-ffi --keep-ptx --gpu-arch=sm_{arch} --dump-dir={Path(directory).as_posix()}",
        )
        # Each compilation has exactly one entry. Rename its exact compiler
        # symbol in PTX so the shared CUDA runtime resolves the Host IR name.
        (device_symbol,) = compiled.kernel_info
        ptx = re.sub(r"\b" + re.escape(device_symbol) + r"\b", lambda _: symbol, compiled.__ptx__)
        return bytearray((ptx + "\0").encode())


BACKEND = register_backend(
    BackendModule(
        name="cutedsl",
        target_kinds=("cuda",),
        supports_target=codegen.is_cutedsl_target,
        pipelines={"cuda": pipeline.CUDA_PIPELINE},
        device_codegens={
            "cuda": DeviceCodegen(
                "cutedsl",
                build=codegen.build_cutedsl,
                build_without_compile=codegen.build_cutedsl_without_compile,
            )
        },
        execution_backends=execution_backend.CUTEDSL_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_cutedsl_compile": compile_cutedsl},
    )
)
