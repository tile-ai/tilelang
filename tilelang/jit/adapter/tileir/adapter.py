"""Execution adapter for TileLang kernels lowered through CUDA Tile IR."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from tvm import tirx
from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.backend.target import determine_target
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.jit.adapter.tileir.runtime import load_native_dispatchers, make_torch_func
from tilelang.tileir.assembly import target_arch
from tilelang.tileir.artifact import (
    TileIRArgumentRef,
    TileIRArtifactCompatibility,
    TileIRLoweringResult,
    TileIRTemporaryBuffer,
    deserialize_tileir_artifact,
    serialize_tileir_artifact,
)
from tilelang.tileir.checks import TileIRToolchain, check_tileir_available
from tilelang.tileir.launch import (
    _global_temporary_buffers,
    _split_grid_sync_primfunc,
    _split_host_orchestrated_primfunc,
    extract_launch_metadata,
)
from tilelang.tileir.lowering import lower_primfunc_to_tileir
from tilelang.transform import MaterializeKernelLaunch


class TileIRKernelAdapter(BaseKernelAdapter):
    """Execution adapter for kernels compiled through CUDA Tile IR."""

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        del verbose, compile_flags

        self.target = determine_target(target, return_object=True)
        self.toolchain = check_tileir_available()
        self.ir_module, self._prim_func = self._prepare_device_module(func_or_mod, self.target, pass_configs)
        self._validate_argument_names(self.prim_func)
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self._cache_param_metadata()

        self.native_dispatcher = None
        self.native_dispatchers = []
        self._host_kernel_source_path = None
        self._device_kernel_source_path = None
        self.tileir_artifact = replace(
            self._attach_argument_metadata(
                lower_primfunc_to_tileir(
                    self.prim_func,
                    self.target,
                    self.toolchain,
                    pass_configs=pass_configs,
                ),
                self.prim_func,
            ),
            compatibility=self._artifact_compatibility(self.target, self.toolchain),
        )
        self.device_kernel_source = self.get_kernel_source()
        self.host_kernel_source = self._format_host_source()
        self.kernel_global_source = self.device_kernel_source
        self.libpath = None
        self._load_native_dispatcher()
        self._post_init()

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_kernel_source: CachedTextSource,
        device_kernel_source: CachedTextSource,
        kernel_lib_path: str,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        del verbose, compile_flags

        adapter = cls.__new__(cls)
        adapter.target = determine_target(target, return_object=True)
        adapter.toolchain = check_tileir_available()
        adapter.ir_module, adapter._prim_func = adapter._prepare_device_module(func_or_mod, adapter.target, pass_configs)
        adapter._validate_argument_names(adapter.prim_func)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter._set_cached_text_source("host_kernel_source", "_host_kernel_source_path", host_kernel_source)
        device_kernel_source = adapter._set_cached_text_source(
            "device_kernel_source",
            "_device_kernel_source_path",
            device_kernel_source,
        )
        adapter._cache_param_metadata()
        adapter.tileir_artifact = cls._read_tileir_artifact(
            kernel_lib_path,
            adapter.prim_func,
            device_kernel_source.text,
        )
        expected_compatibility = cls._artifact_compatibility(adapter.target, adapter.toolchain)
        if adapter.tileir_artifact.compatibility != expected_compatibility:
            raise ValueError(
                "TileIR cache artifact compatibility does not match the active target/toolchain: "
                f"cached={adapter.tileir_artifact.compatibility!r}, active={expected_compatibility!r}."
            )
        cls._validate_cached_artifact_abi(adapter.tileir_artifact, adapter.prim_func)
        adapter.kernel_global_source = adapter.device_kernel_source
        adapter.libpath = kernel_lib_path
        adapter.native_dispatcher = None
        adapter.native_dispatchers = []
        adapter._load_native_dispatcher()
        adapter._post_init()
        return adapter

    def _convert_torch_func(self) -> Callable[..., Any]:
        return make_torch_func(self)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        del kernel_only
        source = self._load_cached_text_source("device_kernel_source", "_device_kernel_source_path")
        if source is not None:
            return source
        if self.tileir_artifact.tileir_source is not None:
            return self.tileir_artifact.tileir_source
        return str(self.prim_func)

    def get_host_source(self) -> str:
        source = self._load_cached_text_source("host_kernel_source", "_host_kernel_source_path")
        if source is not None:
            return source
        return self._format_host_source()

    def _format_host_source(self) -> str:
        if self.tileir_artifact.kernels:
            lines = ["TileIR backend uses the cuTile native dispatcher.", f"program={self.tileir_artifact.kernel_name}"]
            for index, kernel in enumerate(self.tileir_artifact.kernels):
                meta = kernel.launch_metadata
                lines.append(
                    f"kernel[{index}]={kernel.kernel_name} grid={meta.grid} block={meta.block} dynamic_smem_bytes={meta.dynamic_smem_bytes} "
                    f"scratch_bytes_per_block={kernel.scratch_bytes_per_block}"
                )
            if self.tileir_artifact.temporary_buffers:
                temps = ", ".join(f"{buffer.name}{buffer.shape}:{buffer.dtype}" for buffer in self.tileir_artifact.temporary_buffers)
                lines.append(f"temporary_buffers={temps}")
            return "\n".join(lines) + "\n"

        meta = self.tileir_artifact.launch_metadata
        return (
            "TileIR backend uses the cuTile native dispatcher.\n"
            f"kernel={self.tileir_artifact.kernel_name}\n"
            f"grid={meta.grid}\n"
            f"block={meta.block}\n"
            f"dynamic_smem_bytes={meta.dynamic_smem_bytes}\n"
            f"scratch_bytes_per_block={self.tileir_artifact.scratch_bytes_per_block}\n"
        )

    @staticmethod
    def _primary_prim_func(func_or_mod: tirx.PrimFunc | tvm.IRModule) -> tirx.PrimFunc:
        if isinstance(func_or_mod, tirx.PrimFunc):
            return func_or_mod
        funcs = [func for func in func_or_mod.functions.values() if isinstance(func, tirx.PrimFunc)]
        if len(funcs) != 1:
            raise ValueError("TileIR backend expects an IRModule with exactly one frontend TileLang PrimFunc.")
        return funcs[0]

    @property
    def prim_func(self) -> tirx.PrimFunc:
        """Returns the primary TileLang TIR function used for TileIR lowering."""
        return self._prim_func

    @staticmethod
    def _prepare_device_module(
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        target: Target,
        pass_configs: dict[str, Any] | None = None,
    ) -> tuple[tvm.IRModule, tirx.PrimFunc]:
        del target, pass_configs
        mod = func_or_mod
        if isinstance(func_or_mod, tirx.PrimFunc):
            mod = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})

        # T.Kernel traces the launch as a target-neutral kThreadBinding For-nest;
        # materialize it into the SIMT thread_extent form before the launch nest
        # is validated or read. Every SIMT backend runs this at the top of its
        # pipeline (e.g. tilelang/cuda/pipeline.py); the TileIR backend must too,
        # otherwise the launch stays as kThreadBinding loops that
        # _count_kernel_launches and extract_launch_metadata do not recognize.
        mod = MaterializeKernelLaunch()(mod)

        funcs = [func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc)]
        if len(funcs) != 1:
            raise ValueError("TileIR backend expects exactly one frontend TileLang PrimFunc.")
        TileIRKernelAdapter._validate_kernel_launches(funcs[0])
        return mod, funcs[0]

    @staticmethod
    def _count_kernel_launches(prim_func: tirx.PrimFunc) -> int:
        attrs = prim_func.attrs or {}
        if "thread_extent" in attrs:
            return int(any(str(tag).startswith("blockIdx.") for tag in attrs["thread_extent"]))

        launches = 0

        def visit(node):
            nonlocal launches
            if not isinstance(node, tirx.AttrStmt) or node.attr_key != "thread_extent":
                return
            thread_tag = str(getattr(node.node, "thread_tag", ""))
            if thread_tag == "blockIdx.x":
                launches += 1

        tirx.stmt_functor.post_order_visit(prim_func.body, visit)
        return launches

    @staticmethod
    def _validate_kernel_launches(prim_func: tirx.PrimFunc) -> None:
        launches = TileIRKernelAdapter._count_kernel_launches(prim_func)
        if launches < 1:
            kernel_name = str(prim_func.attrs.get("global_symbol", "main"))
            raise ValueError(
                f"TileIR backend expects at least one TileLang kernel launch per PrimFunc; found {launches} in `{kernel_name}`."
            )

    def _cache_param_metadata(self) -> None:
        self.param_dtypes = [param.torch_dtype() for param in self.params]
        self.param_shapes = [list(param.shape) for param in self.params]
        self.dynamic_symbolic_map: dict[tirx.Var, tuple[int, int]] = {}
        self._dynamic_symbolic_name_map: dict[str, tuple[int, int]] = {}

        primary = self._primary_prim_func(self.ir_module)
        for i, param in enumerate(primary.params):
            if param not in primary.buffer_map:
                continue
            buffer = primary.buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tirx.Var) and shape not in self.dynamic_symbolic_map:
                    self.dynamic_symbolic_map[shape] = (i, j)
                    self._dynamic_symbolic_name_map[shape.name] = (i, j)

    def _lookup_dynamic_symbolic(self, var: tirx.Var) -> tuple[int, int]:
        if var in self.dynamic_symbolic_map:
            return self.dynamic_symbolic_map[var]
        if var.name in self._dynamic_symbolic_name_map:
            return self._dynamic_symbolic_name_map[var.name]
        raise KeyError(f"Dynamic symbolic variable '{var.name}' not found in TileIR adapter metadata")

    @staticmethod
    def _read_load_image(path: str) -> bytes:
        with open(path, "rb") as file:
            return file.read()

    @staticmethod
    def _serialize_tileir_artifact(artifact: TileIRLoweringResult) -> bytes:
        return serialize_tileir_artifact(artifact)

    @staticmethod
    def _read_tileir_artifact(path: str, prim_func: tirx.PrimFunc, device_kernel_source: str | None) -> TileIRLoweringResult:
        del prim_func, device_kernel_source
        payload = TileIRKernelAdapter._read_load_image(path)
        return deserialize_tileir_artifact(payload)

    @staticmethod
    def _argument_names(prim_func: tirx.PrimFunc) -> tuple[str, ...]:
        return tuple(prim_func.buffer_map[param].name if param in prim_func.buffer_map else param.name for param in prim_func.params)

    @staticmethod
    def _validate_argument_names(prim_func: tirx.PrimFunc) -> tuple[str, ...]:
        argument_names = TileIRKernelAdapter._argument_names(prim_func)
        if len(set(argument_names)) != len(argument_names):
            raise ValueError(f"duplicate TileIR ABI argument name; argument names must be unique before lowering: {argument_names!r}.")
        return argument_names

    @staticmethod
    def _artifact_compatibility(target: Target, toolchain: TileIRToolchain) -> TileIRArtifactCompatibility:
        if toolchain.tileiras_version is None:
            raise ValueError("TileIR toolchain compatibility requires a declared tileiras version.")
        return TileIRArtifactCompatibility(
            target_arch=target_arch(target),
            cuda_tile_ir_version=toolchain.cuda_tile_ir_version,
            cuda_tile_runtime_version=toolchain.cuda_tile_runtime_version,
            tileiras_version=toolchain.tileiras_version,
        )

    @staticmethod
    def _expected_temporary_buffers(prim_func: tirx.PrimFunc) -> tuple[TileIRTemporaryBuffer, ...]:
        temporary_buffers = []
        for buffer in _global_temporary_buffers(prim_func):
            shape = []
            for dim in buffer.shape:
                if isinstance(dim, tirx.IntImm):
                    shape.append(int(dim))
                elif type(dim) is int:
                    shape.append(dim)
                else:
                    raise ValueError(
                        f"cached TileIR artifact ABI cannot validate dynamic temporary `{buffer.name}` with shape {buffer.shape}."
                    )
            temporary_buffers.append(TileIRTemporaryBuffer(buffer.name, tuple(shape), str(buffer.dtype)))
        return tuple(temporary_buffers)

    @staticmethod
    def _launch_metadata_matches(actual, expected) -> bool:
        if actual.block != expected.block or actual.dynamic_smem_bytes != expected.dynamic_smem_bytes:
            return False
        for actual_extent, expected_extent in zip(actual.grid, expected.grid):
            if isinstance(actual_extent, tirx.PrimExpr) or isinstance(expected_extent, tirx.PrimExpr):
                if not isinstance(actual_extent, tirx.PrimExpr) or not isinstance(expected_extent, tirx.PrimExpr):
                    return False
                actual_vars = tirx.analysis.undefined_vars(actual_extent)
                expected_vars = tirx.analysis.undefined_vars(expected_extent)
                actual_by_name = {var.name: var for var in actual_vars}
                expected_by_name = {var.name: var for var in expected_vars}
                if len(actual_by_name) != len(actual_vars) or len(expected_by_name) != len(expected_vars):
                    return False
                if actual_by_name.keys() != expected_by_name.keys():
                    return False
                if any(str(var.dtype) != str(expected_by_name[name].dtype) for name, var in actual_by_name.items()):
                    return False
                normalized_actual = tirx.stmt_functor.substitute(
                    actual_extent,
                    {var: expected_by_name[name] for name, var in actual_by_name.items()},
                )
                if not tvm.ir.structural_equal(normalized_actual, expected_extent):
                    return False
            elif actual_extent != expected_extent:
                return False
        return True

    @classmethod
    def _validate_cached_artifact_abi(cls, artifact: TileIRLoweringResult, prim_func: tirx.PrimFunc) -> None:
        """Validate cached launch and argument metadata against the active TIR."""

        from tilelang.tileir.scratch import scratch_bytes_for_primfunc

        prim_func = _split_grid_sync_primfunc(prim_func)

        def require_equal(field: str, actual, expected) -> None:
            if actual != expected:
                raise ValueError(f"cached TileIR artifact ABI mismatch for {field}: cached={actual!r}, active={expected!r}.")

        kernel_name = str((prim_func.attrs or {}).get("global_symbol", "main"))
        root_names = cls._validate_argument_names(prim_func)
        root_scalar_flags = tuple(param not in prim_func.buffer_map for param in prim_func.params)
        root_refs = tuple(TileIRArgumentRef("parameter", index) for index in range(len(prim_func.params)))
        require_equal("program symbol", artifact.kernel_name, kernel_name)
        require_equal("program argument names", artifact.argument_names, root_names)
        require_equal("program scalar flags", artifact.argument_scalar_flags, root_scalar_flags)
        require_equal("program argument references", artifact.argument_refs, root_refs)

        if not artifact.kernels:
            require_equal("per-block scratch bytes", artifact.scratch_bytes_per_block, scratch_bytes_for_primfunc(prim_func))
            require_equal("single-kernel temporary buffers", artifact.temporary_buffers, ())
            if not artifact.cubin:
                raise ValueError("cached TileIR artifact ABI mismatch: a single-kernel artifact requires a non-empty cubin.")
            expected_launch = extract_launch_metadata(prim_func)
            if not cls._launch_metadata_matches(artifact.launch_metadata, expected_launch):
                raise ValueError(
                    "cached TileIR artifact ABI mismatch for launch metadata: "
                    f"cached={artifact.launch_metadata!r}, active={expected_launch!r}."
                )
            return

        require_equal("multi-kernel program cubin", artifact.cubin, b"")
        require_equal("multi-kernel program scratch bytes", artifact.scratch_bytes_per_block, 0)
        expected_temporaries = cls._expected_temporary_buffers(prim_func)
        require_equal("temporary buffers", artifact.temporary_buffers, expected_temporaries)
        split_functions = _split_host_orchestrated_primfunc(prim_func)
        require_equal("kernel count", len(artifact.kernels), len(split_functions))

        ref_by_name: dict[str, TileIRArgumentRef] = {name: TileIRArgumentRef("parameter", index) for index, name in enumerate(root_names)}
        for index, temporary in enumerate(expected_temporaries):
            if temporary.name in ref_by_name:
                raise ValueError(f"cached TileIR artifact ABI has ambiguous active argument name {temporary.name!r}.")
            ref_by_name[temporary.name] = TileIRArgumentRef("temporary", index)

        for index, (kernel, split_func) in enumerate(zip(artifact.kernels, split_functions)):
            require_equal(f"kernel {index} per-block scratch bytes", kernel.scratch_bytes_per_block, scratch_bytes_for_primfunc(split_func))
            if kernel.kernels:
                raise ValueError(f"cached TileIR artifact ABI mismatch: kernel {index} must not contain nested kernels.")
            if not kernel.cubin:
                raise ValueError(f"cached TileIR artifact ABI mismatch: kernel {index} requires a non-empty cubin.")
            expected_name = str(split_func.attrs.get("global_symbol", f"{kernel_name}_{index}"))
            expected_names = cls._argument_names(split_func)
            expected_scalar_flags = tuple(param not in split_func.buffer_map for param in split_func.params)
            try:
                expected_refs = tuple(ref_by_name[name] for name in expected_names)
            except KeyError as exc:
                raise ValueError(f"cached TileIR artifact ABI cannot resolve active kernel argument {exc.args[0]!r}.") from exc
            require_equal(f"kernel {index} symbol", kernel.kernel_name, expected_name)
            require_equal(f"kernel {index} argument names", kernel.argument_names, expected_names)
            require_equal(f"kernel {index} scalar flags", kernel.argument_scalar_flags, expected_scalar_flags)
            require_equal(f"kernel {index} argument references", kernel.argument_refs, expected_refs)
            expected_launch = extract_launch_metadata(split_func)
            if not cls._launch_metadata_matches(kernel.launch_metadata, expected_launch):
                raise ValueError(
                    f"cached TileIR artifact ABI mismatch for kernel {index} launch metadata: "
                    f"cached={kernel.launch_metadata!r}, active={expected_launch!r}."
                )

    @staticmethod
    def _attach_argument_metadata(artifact: TileIRLoweringResult, prim_func: tirx.PrimFunc) -> TileIRLoweringResult:
        argument_names = TileIRKernelAdapter._validate_argument_names(prim_func)
        metadata_by_name: dict[str, tuple[TileIRArgumentRef, bool]] = {}

        def register(name: str, ref: TileIRArgumentRef, is_scalar: bool) -> None:
            if name in metadata_by_name:
                raise ValueError(f"duplicate TileIR ABI argument name {name!r}; argument names must be unique before lowering.")
            metadata_by_name[name] = (ref, is_scalar)

        for index, (param, name) in enumerate(zip(prim_func.params, argument_names)):
            register(name, TileIRArgumentRef("parameter", index), param not in prim_func.buffer_map)
        for index, temporary in enumerate(artifact.temporary_buffers):
            register(temporary.name, TileIRArgumentRef("temporary", index), False)

        def attach(current: TileIRLoweringResult, default_names: tuple[str, ...] = ()) -> TileIRLoweringResult:
            names = current.argument_names or default_names
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate TileIR ABI argument name in kernel `{current.kernel_name}`: {names!r}.")
            try:
                metadata = tuple(metadata_by_name[name] for name in names)
            except KeyError as exc:
                raise ValueError(f"TileIR kernel `{current.kernel_name}` references unknown ABI argument {exc.args[0]!r}.") from exc
            return replace(
                current,
                argument_names=names,
                argument_scalar_flags=tuple(is_scalar for _, is_scalar in metadata),
                argument_refs=tuple(ref for ref, _ in metadata),
                kernels=tuple(attach(kernel) for kernel in current.kernels),
            )

        return attach(artifact, argument_names)

    def _load_native_dispatcher(self) -> None:
        self.native_dispatcher, self.native_dispatchers = load_native_dispatchers(self)
