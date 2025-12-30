"""The auto-tune parameters."""

from __future__ import annotations

import tilelang
from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target
from typing import Callable, Literal, Any
from dataclasses import dataclass
from pathlib import Path

from tilelang.jit import JITKernel
import cloudpickle
import os
from tilelang.engine.param import KernelParam
from tilelang import logger
import json
import hashlib
import uuid
from tilelang import env
from tvm.runtime import Executable

BEST_CONFIG_PATH = "best_config.json"
FUNCTION_PATH = "function.pkl"
LATENCY_PATH = "latency.json"

# Align file names with cache/kernel_cache.py
DEVICE_KERNEL_PATH = "device_kernel.cu"
HOST_KERNEL_PATH = "host_kernel.cu"
EXECUTABLE_PATH = "executable.so"
KERNEL_LIB_PATH = "kernel_lib.so"
KERNEL_CUBIN_PATH = "kernel.cubin"
KERNEL_PY_PATH = "kernel.py"
PARAMS_PATH = "params.pkl"


@dataclass(frozen=True)
class CompileArgs:
    """Compile arguments for the auto-tuner. Detailed description can be found in `tilelang.jit.compile`.
    Attributes:
        out_idx: List of output tensor indices.
        execution_backend: Execution backend to use for kernel execution (default: "auto").
        target: Compilation target, either as a string or a TVM Target object (default: "auto").
        target_host: Target host for cross-compilation (default: None).
        verbose: Whether to enable verbose output (default: False).
        pass_configs: Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.PassConfigKey` for supported options.
    """

    out_idx: list[int] | int | None = None
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch"] = "auto"
    target: Literal["auto", "cuda", "hip"] = "auto"
    target_host: str | Target = None
    verbose: bool = False
    pass_configs: dict[str, Any] | None = None

    def compile_program(self, program: PrimFunc):
        return tilelang.compile(
            program,
            out_idx=self.out_idx,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs,
        )

    def __hash__(self):
        data = {
            "execution_backend": self.execution_backend,
            "target": str(self.target),
            "target_host": str(self.target_host) if self.target_host else None,
            "verbose": self.verbose,
            "pass_configs": json.dumps(self.pass_configs, sort_keys=True) if self.pass_configs else None,
        }

        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8"))
        return int.from_bytes(hash_obj.digest(), byteorder="big")


@dataclass(frozen=True)
class ProfileArgs:
    """Profile arguments for the auto-tuner.

    Attributes:
        warmup: Number of warmup iterations.
        rep: Number of repetitions for timing.
        timeout: Maximum time per configuration.
        supply_type: Type of tensor supply mechanism.
        ref_prog: Reference program for correctness validation.
        supply_prog: Supply program for input tensors.
        out_idx: Union[List[int], int] = -1
        supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
        ref_prog: Callable = None
        supply_prog: Callable = None
        rtol: float = 1e-2
        atol: float = 1e-2
        max_mismatched_ratio: float = 0.01
        skip_check: bool = False
        manual_check_prog: Callable = None
        cache_input_tensors: bool = True
    """

    warmup: int = 25
    rep: int = 100
    timeout: int = 30
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
    ref_prog: Callable = None
    supply_prog: Callable = None
    rtol: float = 1e-2
    atol: float = 1e-2
    max_mismatched_ratio: float = 0.01
    skip_check: bool = False
    manual_check_prog: Callable = None
    cache_input_tensors: bool = True

    def __hash__(self):
        data = {
            "warmup": self.warmup,
            "rep": self.rep,
            "timeout": self.timeout,
            "supply_type": str(self.supply_type),
            "rtol": self.rtol,
            "atol": self.atol,
            "max_mismatched_ratio": self.max_mismatched_ratio,
        }
        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8"))
        return int.from_bytes(hash_obj.digest(), byteorder="big")


@dataclass(frozen=True)
class AutotuneResult:
    """Results from auto-tuning process.

    Attributes:
        latency: Best achieved execution latency.
        config: Configuration that produced the best result.
        ref_latency: Reference implementation latency.
        libcode: Generated library code.
        func: Optimized function.
        kernel: Compiled kernel function.
    """

    latency: float | None = None
    config: dict | None = None
    ref_latency: float | None = None
    libcode: str | None = None
    func: Callable | None = None
    kernel: Callable | None = None

    @staticmethod
    def _load_binary(path: str):
        with open(path, "rb") as file:
            binary = file.read()
        return binary

    @staticmethod
    def _safe_write_file(path: str, mode: str, operation: Callable[[Any], None]):
        # Random a temporary file within the same FS as the cache directory
        tmp_dir = env.TILELANG_TMP_DIR
        os.makedirs(tmp_dir, exist_ok=True)
        temp_path = os.path.join(tmp_dir, f"{os.getpid()}_{uuid.uuid4()}")
        with open(temp_path, mode) as temp_file:
            operation(temp_file)
        # Use atomic POSIX replace, so other processes cannot see a partial write
        os.replace(temp_path, path)

    @staticmethod
    def _safe_write_executable(executable: Executable, path: str):
        tmp_dir = env.TILELANG_TMP_DIR
        os.makedirs(tmp_dir, exist_ok=True)
        temp_path = os.path.join(tmp_dir, f"{os.getpid()}_{uuid.uuid4()}.so")
        executable.export_library(temp_path)
        os.replace(temp_path, path)

    def _save_kernel_to_disk(self, cache_path: Path, kernel: JITKernel, verbose: bool = False):
        """
        Persists a compiled kernel to disk cache.

        Args:
            cache_path (Path): The root path for the cache files.
            kernel (JITKernel): The compiled kernel to be saved.
            verbose (bool): Enable verbose log messages.

        Note:
            Saves the following files:
            - kernel.cu: The compiled kernel source code
            - wrapped_kernel.cu: The wrapped kernel source code
            - kernel_lib.so: The compiled kernel library
            - params.pkl: The serialized kernel parameters
        """
        os.makedirs(cache_path, exist_ok=True)  # Ensure directory exists

        # Save device kernel source code
        try:
            device_kernel_path = os.path.join(cache_path, DEVICE_KERNEL_PATH)
            if verbose:
                logger.debug(f"Saving kernel source code to file: {device_kernel_path}")
            if kernel.kernel_source is not None:
                self._safe_write_file(device_kernel_path, "w", lambda f: f.write(kernel.kernel_source))
        except Exception as e:
            logger.error(f"Error saving kernel source code to disk: {e}")

        # Save host kernel source code (wrapped)
        try:
            host_kernel_path = os.path.join(cache_path, HOST_KERNEL_PATH)
            if verbose:
                logger.debug(f"Saving wrapped kernel source code to file: {host_kernel_path}")
            # Match kernel_cache behavior: use host source for tvm_ffi, otherwise wrapped kernel
            if kernel.execution_backend == "tvm_ffi":
                self._safe_write_file(host_kernel_path, "w", lambda f: f.write(kernel.adapter.get_host_source()))
            else:
                self._safe_write_file(host_kernel_path, "w", lambda f: f.write(kernel.adapter.get_kernel_source()))
        except Exception as e:
            logger.error(f"Error saving wrapped kernel source code to disk: {e}")

        # Save kernel library (backend-specific)
        try:
            if kernel.execution_backend == "nvrtc":
                kernel_lib_file = KERNEL_CUBIN_PATH
            elif kernel.execution_backend == "tvm_ffi":
                kernel_lib_file = EXECUTABLE_PATH
            else:
                kernel_lib_file = KERNEL_LIB_PATH

            kernel_lib_path = os.path.join(cache_path, kernel_lib_file)

            if kernel.execution_backend == "nvrtc":
                # Save cubin and python helper file
                src_lib_path = kernel.adapter.libpath
                kernel_py_path = os.path.join(cache_path, KERNEL_PY_PATH)
                py_src_path = src_lib_path.replace(".cubin", ".py")
                if verbose:
                    logger.debug(f"Saving kernel nvrtc python code to file: {kernel_py_path}")
                self._safe_write_file(kernel_py_path, "wb", lambda f: f.write(self._load_binary(py_src_path)))
                if verbose:
                    logger.debug(f"Saving kernel library to file: {kernel_lib_path}")
                self._safe_write_file(kernel_lib_path, "wb", lambda f: f.write(self._load_binary(src_lib_path)))
            elif kernel.execution_backend == "tvm_ffi":
                executable = kernel.adapter.executable
                if verbose:
                    logger.debug(f"Saving kernel executable to file: {kernel_lib_path}")
                self._safe_write_executable(executable, kernel_lib_path)
            else:
                src_lib_path = kernel.adapter.libpath
                if verbose:
                    logger.debug(f"Saving kernel library to file: {kernel_lib_path}")
                self._safe_write_file(kernel_lib_path, "wb", lambda f: f.write(self._load_binary(src_lib_path)))

        except Exception as e:
            logger.error(f"Error saving kernel library to disk: {e}")

        # Save kernel parameters
        try:
            params_path = os.path.join(cache_path, PARAMS_PATH)
            if verbose:
                logger.debug(f"Saving kernel parameters to disk: {params_path}")
            self._safe_write_file(params_path, "wb", lambda f: cloudpickle.dump(kernel.params, f))
        except Exception as e:
            logger.error(f"Error saving kernel parameters to disk: {e}")

    def _load_kernel_from_disk(
        self,
        cache_path: Path,
        target: str | Target = "auto",
        target_host: str | Target = None,
        out_idx: list[int] | int | None = None,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch"] = "tvm_ffi",
        pass_configs: dict = None,
        compile_flags: list[str] | str | None = None,
        func: Callable = None,
        verbose: bool = False,
    ) -> JITKernel:
        """
        Loads a previously compiled kernel from disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            target (Union[str, Target]): Compilation target platform. Defaults to "auto".
            target_host (Union[str, Target], optional): Host target platform.
            out_idx (List[int], optional): Indices specifying which outputs to return.
            execution_backend (Literal): Backend type for execution. Defaults to "cython".
            pass_configs (dict, optional): Configuration for compiler passes.
            func (Callable, optional): The original function.
            verbose (bool): Enable verbose log messages.

        Returns:
            JITKernel: The loaded kernel if found, None otherwise.
        """

        if not os.path.exists(cache_path):
            return None

        # Resolve backend to pick correct file names
        if execution_backend == "nvrtc":
            kernel_lib_file = KERNEL_CUBIN_PATH
        elif execution_backend == "tvm_ffi":
            kernel_lib_file = EXECUTABLE_PATH
        else:
            kernel_lib_file = KERNEL_LIB_PATH

        device_kernel_path = os.path.join(cache_path, DEVICE_KERNEL_PATH)
        host_kernel_path = os.path.join(cache_path, HOST_KERNEL_PATH)
        kernel_lib_path = os.path.join(cache_path, kernel_lib_file)
        params_path = os.path.join(cache_path, PARAMS_PATH)

        if not all([os.path.exists(file) for file in (kernel_lib_path, params_path)]):
            return None

        device_kernel_source: str | None = None
        host_kernel_source: str | None = None
        kernel_params: list[KernelParam] | None = None

        # Load optional device kernel source
        try:
            if verbose:
                logger.debug(f"Loading kernel source code from file: {device_kernel_path}")
            with open(device_kernel_path) as f:
                device_kernel_source = f.read()
        except Exception as e:
            logger.error(f"Error loading kernel source code from disk: {e}")

        # Load optional host kernel source
        try:
            if verbose:
                logger.debug(f"Loading wrapped kernel source code from file: {host_kernel_path}")
            with open(host_kernel_path) as f:
                host_kernel_source = f.read()
        except Exception as e:
            logger.error(f"Error loading host kernel source code from disk: {e}")

        # Load kernel parameters
        try:
            if verbose:
                logger.debug(f"Loading kernel parameters from file: {params_path}")
            with open(params_path, "rb") as f:
                kernel_params = cloudpickle.load(f)
        except Exception as e:
            logger.error(f"Error loading kernel parameters from disk: {e}")

        if host_kernel_source and device_kernel_source and kernel_params:
            return JITKernel.from_database(
                func=func,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                params=kernel_params,
                target=target,
                target_host=target_host,
                out_idx=out_idx,
                execution_backend=execution_backend,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        else:
            return None

    def save_to_disk(self, path: Path, verbose: bool = False):
        if not os.path.exists(path):
            os.makedirs(path)

        # save best config (atomic)
        if verbose:
            logger.debug(f"Saving best config to file: {path / BEST_CONFIG_PATH}")
        self._safe_write_file(str(path / BEST_CONFIG_PATH), "w", lambda f: json.dump(self.config, f))

        # save function (atomic)
        if verbose:
            logger.debug(f"Saving function to file: {path / FUNCTION_PATH}")
        self._safe_write_file(str(path / FUNCTION_PATH), "wb", lambda f: cloudpickle.dump(self.func, f))

        # save ref latency (atomic)
        if verbose:
            logger.debug(f"Saving latency to file: {path / LATENCY_PATH}")
        self._safe_write_file(
            str(path / LATENCY_PATH),
            "w",
            lambda f: json.dump(
                {
                    "latency": self.latency,
                    "ref_latency": self.ref_latency,
                },
                f,
            ),
        )

        # save kernel
        self._save_kernel_to_disk(path, self.kernel)

    @classmethod
    def load_from_disk(cls, path: Path, compile_args: CompileArgs) -> AutotuneResult:
        if not os.path.exists(path):
            return None

        verbose = compile_args.verbose
        # Normalize target and resolve execution backend for loading
        from tilelang.utils.target import determine_target as _determine_target
        from tilelang.jit.execution_backend import resolve_execution_backend

        norm_target = Target(_determine_target(compile_args.target)) if isinstance(compile_args.target, str) else compile_args.target
        requested_backend = compile_args.execution_backend
        resolved_backend = resolve_execution_backend(requested_backend, norm_target)
        # load best config
        if verbose:
            logger.debug(f"Loading best config from file: {path / BEST_CONFIG_PATH}")
        with open(path / BEST_CONFIG_PATH) as f:
            config = json.load(f)

        # load function
        if verbose:
            logger.debug(f"Loading function from file: {path / FUNCTION_PATH}")
        with open(path / FUNCTION_PATH, "rb") as f:
            func = cloudpickle.load(f)

        # load latency
        if verbose:
            logger.debug(f"Loading latency from file: {path / LATENCY_PATH}")
        with open(path / LATENCY_PATH) as f:
            latency = json.load(f)
            latency, ref_latency = latency["latency"], latency["ref_latency"]

        kernel = cls._load_kernel_from_disk(
            cls,
            path,
            norm_target,
            compile_args.target_host,
            compile_args.out_idx,
            resolved_backend,
            compile_args.pass_configs,
            None,  # compile_flags not tracked here
            func,
        )
        if kernel is None:
            return None
        kernel.update_tuner_result(
            config=config,
            latency=latency,
            ref_latency=ref_latency,
        )
        result = cls(
            config=config,
            func=func,
            kernel=kernel,
            libcode=kernel.get_kernel_source(),
            latency=latency,
            ref_latency=ref_latency,
        )
        return result
