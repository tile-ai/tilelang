from __future__ import annotations
from typing import Any, Callable, Generic, Literal, TypeVar

# Python 3.9 compatibility for ParamSpec
try:
    from typing import ParamSpec
except ImportError:  # Python < 3.10
    from typing_extensions import ParamSpec

from tilelang.jit.adapter.utils import is_cutedsl_target, is_metal_target, is_cuda_target
from tvm.target import Target
from tvm.tir import PrimFunc

import tilelang
from tilelang import tvm
from tilelang import env
from tilelang.engine.param import CompiledArtifact, KernelParam
from tilelang.jit.adapter import (
    BaseKernelAdapter,
    CythonKernelAdapter,
    CuTeDSLKernelAdapter,
    TVMFFIKernelAdapter,
    MetalKernelAdapter,
)
from tilelang.profiler import Profiler, TensorSupplyType
from tilelang.utils.target import determine_target
from tilelang.contrib import nvcc as tl_nvcc
from tilelang.transform import PassConfigKey
import logging
import os

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class JITKernel(Generic[_P, _T]):
    """
    A wrapper class for compiling and invoking TileLang (TVM TIR) functions as PyTorch-compatible functions.

    Attributes
    ----------
    artifact : CompiledArtifact
        The compiled artifact containing the runtime module and parameters.
    adapter : BaseKernelAdapter
        The adapter for the compiled function.
    torch_function : Callable
        The compiled function that can be invoked as a PyTorch-compatible function.
    """

    prim_func: PrimFunc = None
    artifact: CompiledArtifact = None
    adapter: BaseKernelAdapter = None
    torch_function: Callable = None

    # tuner result
    latency: float = None
    config: dict[str, Any] = None
    ref_latency: float = None

    def __init__(
        self,
        func: PrimFunc = None,
        out_idx: list[int] | int = None,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] = "tvm_ffi",
        target: str | Target = "auto",
        target_host: str | Target = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        from_database: bool = False,
        compile_flags: list[str] | None = None,
    ):
        """
        Initializes a TorchFunction instance.

        Parameters
        ----------
        func : tvm.tir.PrimFunc, optional
            The TileLang TIR function to compile and wrap.
        out_idx : Union[List[int], int], optional
            Index(es) of the output tensors to return (default: None).
        execution_backend : Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"], optional
            Execution backend to use for kernel execution.
        target : Union[str, Target], optional
            Compilation target, either as a string or a TVM Target object (default: "auto").
        target_host : Union[str, Target], optional
            Target host for cross-compilation (default: None).
        verbose : bool, optional
            Whether to enable verbose output (default: False).
        pass_configs : dict, optional
            Additional keyword arguments to pass to the Compiler PassContext.
            Refer to `tilelang.PassConfigKey` for supported options.
        from_database : bool, optional
            Whether to create a TorchFunction from a database.
        """
        self.prim_func = func
        self.execution_backend = execution_backend
        self.target_host = target_host
        self.verbose = verbose

        if pass_configs is None:
            pass_configs = {}
        self.pass_configs = pass_configs

        self.compile_flags = [compile_flags] if isinstance(compile_flags, str) else compile_flags

        # Ensure the target is always a valid TVM Target object.
        self.target = determine_target(target, return_object=True)

        # Validate the execution backend.
        assert execution_backend in [
            "tvm_ffi",
            "cython",
            "nvrtc",
            "torch",
            "cutedsl",
        ], f"Invalid execution backend. {execution_backend}"
        if execution_backend == "cython":
            from tilelang.contrib.cc import get_cplus_compiler

            assert get_cplus_compiler() is not None, "Cython backend requires a C++ compiler, please install or use other backends."

        if from_database:
            return

        # Print log on compilation starts
        # NOTE(Chenggang): printing could let the training/inference framework easier to know
        # whether the communication timeout is from compilation
        if env.is_print_on_compilation_enabled():
            # assert func must have "global_symbol"
            func_name = func.attrs.get("global_symbol")
            assert func_name is not None, "func must have global_symbol"
            logger.info(f"TileLang begins to compile kernel `{func_name}` with `{out_idx=}`")

        # Compile the TileLang function and create a kernel adapter for execution.
        adapter = self._compile_and_create_adapter(func, out_idx)

        if env.is_print_on_compilation_enabled():
            func_name = func.attrs.get("global_symbol")
            assert func_name is not None, "func must have global_symbol"
            logger.info(f"TileLang completes to compile kernel `{func_name}`")

        # The adapter's function is assigned as the callable function for this instance.
        self.adapter = adapter
        self.torch_function = adapter.func

    @classmethod
    def from_database(
        cls,
        func: PrimFunc,
        host_kernel_source: str,
        device_kernel_source: str,
        kernel_lib_path: str,
        params: list[KernelParam],
        target: str | Target,
        target_host: str | Target,
        out_idx: list[int] | int,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch"],
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        """
        Alternative constructor to create a TorchFunction directly from a database.
        """
        instance = cls(
            func=func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            pass_configs=pass_configs,
            from_database=True,
            compile_flags=compile_flags,
        )

        instance.adapter = instance._create_adapter_from_database(
            func_or_mod=func,
            params=params,
            result_idx=out_idx,
            target=target,
            host_kernel_source=host_kernel_source,
            device_kernel_source=device_kernel_source,
            kernel_lib_path=kernel_lib_path,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )
        instance.torch_function = instance.adapter.func
        return instance

    def __call__(self, *args: _P.args, **kwds: _P.kwargs) -> _T:
        """
        Invokes the compiled function with the given arguments.

        Parameters
        ----------
        *args : Any
            Positional arguments for the function.
        **kwds : Any
            Keyword arguments for the function.

        Returns
        -------
        Any
            The result of the function execution.
        """
        return self.torch_function(*args, **kwds)

    def _compile_and_create_adapter(self, tilelang_func: PrimFunc, out_idx: list[int]) -> BaseKernelAdapter:
        """
        Compiles the given TileLang PrimFunc using TVM and creates a kernel adapter.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.

        Returns
        -------
        BaseKernelAdapter
            The compiled and ready-to-run kernel adapter.
        """
        verbose = self.verbose
        target = self.target
        target_host = self.target_host

        execution_backend = self.execution_backend
        pass_configs = self.pass_configs or {}

        compile_flags = self.compile_flags

        if compile_flags is not None:
            compile_flags_cfg = pass_configs.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS)
            pass_configs[PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = (
                compile_flags_cfg + compile_flags if compile_flags_cfg is not None else compile_flags
            )

        # Compile the function with TVM, optimizing with shared memory lowering.
        enable_host_codegen = execution_backend == "tvm_ffi"
        enable_device_compile = execution_backend == "tvm_ffi"
        with tvm.transform.PassContext(opt_level=3, config=pass_configs), self.target:
            artifact = tilelang.lower(
                tilelang_func,
                target=target,
                target_host=target_host,
                enable_host_codegen=enable_host_codegen,
                enable_device_compile=enable_device_compile,
            )

        self.artifact = artifact

        # Create an adapter based on the specified execution backend.
        if execution_backend == "tvm_ffi":
            # Use TVMFFIKernelAdapter for interoperability with PyTorch via DLPack.
            # But we need to ensure that the runtime is enabled and the runtime module is not None.
            assert artifact.rt_mod is not None, "tvm_ffi backend requires a runtime module."
            adapter = TVMFFIKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                rt_mod=artifact.rt_mod,
                device_kernel_source=artifact.kernel_source,
                verbose=verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "cython":
            adapter = CythonKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                device_kernel_source=artifact.kernel_source,
                verbose=verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "nvrtc":
            from tilelang.jit.adapter import NVRTCKernelAdapter

            adapter = NVRTCKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                device_kernel_source=artifact.kernel_source,
                verbose=verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "torch":
            assert is_metal_target(target)
            adapter = MetalKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                # target=target,
                func_or_mod=tilelang_func,
                # host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                kernel_global_source=artifact.kernel_source,
                verbose=verbose,
                # pass_configs=pass_configs,
                # compile_flags=compile_flags,
            )
        elif execution_backend == "cutedsl":
            assert is_cutedsl_target(target)
            adapter = CuTeDSLKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                device_kernel_source=artifact.kernel_source,
                verbose=verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        else:
            # Handle invalid backend.
            raise ValueError(f"Invalid execution backend: {execution_backend}")

        return adapter

    def _create_adapter_from_database(
        self,
        params: list[KernelParam],
        result_idx: list[int] | int,
        target: str | Target,
        func_or_mod: PrimFunc | tvm.runtime.Module,
        host_kernel_source: str,
        device_kernel_source: str,
        kernel_lib_path: str,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ) -> BaseKernelAdapter:
        target = self.target
        execution_backend = self.execution_backend

        # Create an adapter based on the specified execution backend.
        if execution_backend == "tvm_ffi":
            adapter = TVMFFIKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "cython":
            adapter = CythonKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                pass_configs=pass_configs,
            )
        elif execution_backend == "nvrtc":
            from tilelang.jit.adapter import NVRTCKernelAdapter

            adapter = NVRTCKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "cutedsl":
            adapter = CuTeDSLKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        else:
            # Handle invalid backend.
            raise ValueError(f"Invalid execution backend: {execution_backend}")

        return adapter

    @classmethod
    def from_tilelang_function(cls, tilelang_func: PrimFunc, **kwargs):
        """
        Alternative constructor to create a TorchFunction directly from a TileLang PrimFunc.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.
        **kwargs : dict
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        TorchFunction
            An instance of TorchFunction wrapping the compiled function.
        """
        return cls(func=tilelang_func, **kwargs)

    def get_profiler(self, tensor_supply_type: TensorSupplyType = TensorSupplyType.Auto) -> Profiler:
        """
        Creates a profiler to benchmark the compiled runtime module.

        Parameters
        ----------
        tensor_supply_type : TensorSupplyType, optional
            The type of input tensors to supply for profiling (default: TensorSupplyType.Auto).

        Returns
        -------
        Profiler
            A Profiler instance for benchmarking the runtime module.
        """
        return Profiler(self.params, self.out_idx, tensor_supply_type).with_default_adapter(self.adapter)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        """
        Returns the source code of the compiled kernel function.

        Returns
        -------
        str
            The source code of the compiled kernel function.
        """
        if self.execution_backend in {"cython", "nvrtc", "tvm_ffi", "cutedsl"}:
            return self.adapter.get_kernel_source(kernel_only=kernel_only)
        return self.artifact.kernel_source

    def get_host_source(self) -> str:
        """
        Returns the source code of the host function.
        """
        if self.execution_backend in {"cython", "nvrtc", "tvm_ffi", "cutedsl"}:
            return self.adapter.get_host_source()
        assert self.artifact.host_mod is not None, "host_mod is not available"
        return str(self.artifact.host_mod)

    def run_once(self, func: Callable | None = None) -> None:
        return self.get_profiler().run_once(func)

    def show_source(self, which: Literal["kernel", "host", "both"] = "kernel") -> None:
        """
        Print generated source code to stdout.

        Parameters
        ----------
        which : Literal["kernel", "host", "both"], optional
            Select which source to print. Defaults to "kernel".

        Examples
        --------
        >>> jit_kernel.show_source()            # print kernel source
        >>> jit_kernel.show_source("host")      # print host source
        >>> jit_kernel.show_source("both")      # print both sources
        """
        try:
            if which == "kernel":
                src = self.get_kernel_source()
                print(src)
            elif which == "host":
                src = self.get_host_source()
                # Host is generally C/C++
                print(src)
            elif which == "both":
                print("===== Kernel Source =====")
                ksrc = self.get_kernel_source()
                print(ksrc)
                print("===== Host Source =====")
                hsrc = self.get_host_source()
                print(hsrc)
            else:
                raise ValueError(f"Unknown option for 'which': {which}")
        except Exception as e:
            logger.error(f"Failed to show source code: {e}")

    def export_sources(self, kernel_path: str | None = None, host_path: str | None = None) -> None:
        """
        Export generated source code to files.

        Parameters
        ----------
        kernel_path : Optional[str]
            Destination file path to write the kernel source. If None, skips writing kernel code.
        host_path : Optional[str]
            Destination file path to write the host source. If None, skips writing host code.

        Examples
        --------
        >>> jit_kernel.export_sources(kernel_path="/tmp/kernel.cu")
        >>> jit_kernel.export_sources(host_path="/tmp/host.cc")
        >>> jit_kernel.export_sources(
        ...     kernel_path="/tmp/kernel.cu",
        ...     host_path="/tmp/host.cc",
        ... )
        """
        if kernel_path is None and host_path is None:
            raise ValueError("At least one of kernel_path or host_path must be provided.")
        try:
            if kernel_path is not None:
                dir_path = os.path.dirname(kernel_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(kernel_path, "w") as f:
                    f.write(self.get_kernel_source())
            if host_path is not None:
                dir_path = os.path.dirname(host_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(host_path, "w") as f:
                    f.write(self.get_host_source())
        except Exception as e:
            logger.error(f"Failed to export sources: {e}")

    # Backward compatibility alias (deprecated)
    def print_source_code(self, which: Literal["kernel", "host", "both"] = "kernel", file: str | None = None) -> None:
        """
        Deprecated: use show_source() or export_sources() instead.

        Parameters
        ----------
        which : Literal["kernel", "host", "both"], optional
            Kept for backward compatibility with printing behavior.
        file : Optional[str]
            If provided, behaves like export_sources(kernel_path=file).

        Examples
        --------
        >>> # New API (preferred)
        >>> jit_kernel.show_source("both")
        >>> jit_kernel.export_sources(kernel_path="/tmp/kernel.cu")

        >>> # Old API (still works but deprecated)
        >>> jit_kernel.print_source_code(file="/tmp/kernel.cu")
        """
        logger.warning("print_source_code is deprecated; use show_source() or export_sources() instead.")
        if file is not None:
            # Historical behavior wrote only kernel source when file provided
            self.export_sources(kernel_path=file)
        else:
            self.show_source(which=which)

    def update_tuner_result(self, latency: float, config: dict[str, Any], ref_latency: float) -> JITKernel:
        """
        Updates the tuning results for this kernel.

        Parameters
        ----------
        latency : float
            The measured latency of this kernel configuration.
        config : Dict[str, Any]
            The configuration parameters used for this kernel.
        ref_latency : float
            The reference latency to compare against.

        Returns
        -------
        None
        """
        self.latency = latency
        self.config = config
        self.ref_latency = ref_latency

        return self

    def get_tuner_result(self) -> dict[str, Any]:
        """
        Gets the tuning results for this kernel.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - latency: The measured latency of this kernel
            - config: The configuration parameters used
            - ref_latency: The reference latency for comparison
        """
        if self.latency is None:
            raise ValueError("Tuning results are not available. Please tune the kernel first.")

        return {
            "latency": self.latency,
            "config": self.config,
            "ref_latency": self.ref_latency,
        }

    @property
    def out_idx(self) -> list[int]:
        return self.adapter.result_idx

    @property
    def params(self) -> list[KernelParam]:
        return self.artifact.params if self.artifact else self.adapter.params

    @property
    def kernel_source(self) -> str:
        return self.artifact.kernel_source if self.artifact else self.adapter.kernel_global_source

    @property
    def host_source(self) -> str:
        return str(self.artifact.host_mod) if self.artifact else ""

    def export_library(self, kernel_file: str) -> None:
        """
        Exports the compiled kernel function to a shared library file.

        Parameters
        ----------
        kernel_file : str
            The path to the shared library file to create.
        """
        # rt_module: tvm.runtime.Module = None
        # rt_params: dict = None
        # adapter: BaseKernelAdapter = None
        # torch_function: Callable = None
        # rt_module: use export_library to export
        # rt_params: use cloudpickle to serialize

        # Export the compiled kernel function to a shared library file.
        self.rt_module.export_library(kernel_file)

    def _get_ptx(self, verbose: bool | None = None) -> str:
        """
        Compile and return PTX for the current kernel (CUDA only).

        Parameters
        ----------
        verbose : Optional[bool]
            Whether to enable verbose NVRTC logs. Defaults to self.verbose.

        Returns
        -------
        str
            The compiled PTX text.
        """
        if not is_cuda_target(self.target):
            raise ValueError("PTX is only available for CUDA targets.")
        # Prefer NVCC for PTX generation via contrib helper
        code = self.get_kernel_source()
        if verbose is None:
            verbose = self.verbose
        # Ensure target is set so nvcc picks correct arch via Target.current()
        with self.target:
            return tl_nvcc.get_ptx_from_source(code, compile_flags=self.compile_flags, verbose=verbose)

    def show_ptx(self) -> None:
        """
        Print compiled PTX for the kernel (CUDA only).

        Examples
        --------
        >>> jit_kernel.show_ptx()
        """
        try:
            ptx = self._get_ptx()
            print(ptx)
        except Exception as e:
            logger.error(f"Failed to show PTX: {e}")

    def export_ptx(self, path: str) -> None:
        """
        Export compiled PTX to a file (CUDA only).

        Parameters
        ----------
        path : str
            Destination file path to write PTX.

        Examples
        --------
        >>> jit_kernel.export_ptx("/tmp/kernel.ptx")
        """
        if not path:
            raise ValueError("path must be provided to export PTX")
        try:
            ptx = self._get_ptx()
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, "w") as f:
                f.write(ptx)
            logger.info(f"PTX saved to {os.path.abspath(path)}")
        except Exception as e:
            logger.error(f"Failed to export PTX: {e}")

    def _get_sass(self, verbose: bool | None = None) -> str:
        """
        Compile and return SASS for the current kernel (CUDA only).

        Parameters
        ----------
        verbose : Optional[bool]
            Whether to enable verbose tool logs. Defaults to self.verbose.

        Returns
        -------
        str
            The disassembled SASS text.
        """
        if not is_cuda_target(self.target):
            raise ValueError("SASS is only available for CUDA targets.")
        code = self.get_kernel_source()
        if verbose is None:
            verbose = self.verbose
        with self.target:
            return tl_nvcc.get_sass_from_source(code, compile_flags=self.compile_flags, verbose=verbose)

    def show_sass(self) -> None:
        """
        Print disassembled SASS for the kernel (CUDA only).

        Examples
        --------
        >>> jit_kernel.show_sass()
        """
        try:
            sass = self._get_sass()
            print(sass)
        except Exception as e:
            logger.error(f"Failed to show SASS: {e}")

    def export_sass(self, path: str) -> None:
        """
        Export disassembled SASS to a file (CUDA only).

        Parameters
        ----------
        path : str
            Destination file path to write SASS.

        Examples
        --------
        >>> jit_kernel.export_sass("/tmp/kernel.sass")
        """
        if not path:
            raise ValueError("path must be provided to export SASS")
        try:
            sass = self._get_sass()
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, "w") as f:
                f.write(sass)
            logger.info(f"SASS saved to {os.path.abspath(path)}")
        except Exception as e:
            logger.error(f"Failed to export SASS: {e}")
