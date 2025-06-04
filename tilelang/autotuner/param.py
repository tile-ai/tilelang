# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The auto-tune parameters.
"""

import tilelang
from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target
from typing import Callable, List, Literal, Any, Optional, Union, Dict
from dataclasses import dataclass
from pathlib import Path

from tilelang.jit import JITKernel
import cloudpickle
import os
import shutil
from tilelang.engine.param import KernelParam
from tilelang import logger
import json
import hashlib

BEST_CONFIG_PATH = "best_config.json"
FUNCTION_PATH = "function.pkl"
LATENCY_PATH = "latency.json"
KERNEL_PATH = "kernel.cu"
WRAPPED_KERNEL_PATH = "wrapped_kernel.cu"
KERNEL_LIB_PATH = "kernel_lib.so"
PARAMS_PATH = "params.pkl"


@dataclass(frozen=True)
class CompileArgs:
    """Compile arguments for the auto-tuner. Detailed description can be found in `tilelang.jit.compile`.
    Attributes:
        out_idx: List of output tensor indices.
        execution_backend: Execution backend to use for kernel execution (default: "cython").
        target: Compilation target, either as a string or a TVM Target object (default: "auto").
        target_host: Target host for cross-compilation (default: None).
        verbose: Whether to enable verbose output (default: False).
        pass_configs: Additional keyword arguments to pass to the Compiler PassContext.
        Available options:
            "tir.disable_vectorize": bool, default: False
            "tl.disable_tma_lower": bool, default: False
            "tl.disable_warp_specialized": bool, default: False
            "tl.config_index_bitwidth": int, default: None
            "tl.disable_dynamic_tail_split": bool, default: False
            "tl.dynamic_vectorize_size_bits": int, default: 128
            "tl.disable_safe_memory_legalize": bool, default: False
    """

    out_idx: Union[List[int], int] = -1
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython"
    target: Literal['auto', 'cuda', 'hip'] = 'auto'
    target_host: Union[str, Target] = None
    verbose: bool = False
    pass_configs: Optional[Dict[str, Any]] = None

    def compile_program(self, program: PrimFunc):
        return tilelang.compile(
            program,
            out_idx=self.out_idx,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs)

    def __hash__(self):
        data = {
            "out_idx":
                self.out_idx,
            "execution_backend":
                self.execution_backend,
            "target":
                self.target,
            "target_host":
                str(self.target_host) if self.target_host else None,
            "verbose":
                self.verbose,
            "pass_configs":
                json.dumps(self.pass_configs, sort_keys=True) if self.pass_configs else None,
        }

        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8'))
        return int.from_bytes(hash_obj.digest(), byteorder='big')


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
        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8'))
        return int.from_bytes(hash_obj.digest(), byteorder='big')


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
    latency: float
    config: dict
    ref_latency: float
    libcode: str
    func: Callable
    kernel: Callable

    def _save_kernel_to_disk(self, cache_path: Path, kernel: JITKernel):
        """
        Persists a compiled kernel to disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            kernel (JITKernel): The compiled kernel to be saved.
            func (Callable, optional): The original function.

        Note:
            Saves the following files:
            - kernel.cu: The compiled kernel source code
            - wrapped_kernel.cu: The wrapped kernel source code
            - kernel_lib.so: The compiled kernel library
            - params.pkl: The serialized kernel parameters
        """
        os.makedirs(cache_path, exist_ok=True)  # Ensure directory exists

        # Save kernel source code
        try:
            kernel_path = os.path.join(cache_path, KERNEL_PATH)
            with open(kernel_path, "w") as f:
                f.write(kernel.artifact.kernel_source)
        except Exception as e:
            logger.error(f"Error saving kernel source code to disk: {e}")

        # Save wrapped kernel source code
        try:
            wrapped_kernel_path = os.path.join(cache_path, WRAPPED_KERNEL_PATH)
            with open(wrapped_kernel_path, "w") as f:
                f.write(kernel.adapter.get_kernel_source())
        except Exception as e:
            logger.error(f"Error saving wrapped kernel source code to disk: {e}")

        # Save kernel library
        try:
            kernel_lib_path = os.path.join(cache_path, KERNEL_LIB_PATH)
            src_lib_path = kernel.adapter.libpath
            shutil.copy(src_lib_path, kernel_lib_path)
        except Exception as e:
            logger.error(f"Error saving kernel library to disk: {e}")

        # Save kernel parameters
        try:
            params_path = os.path.join(cache_path, PARAMS_PATH)
            with open(params_path, "wb") as f:
                cloudpickle.dump(kernel.params, f)
        except Exception as e:
            logger.error(f"Error saving kernel parameters to disk: {e}")

    def _load_kernel_from_disk(
        self,
        cache_path: Path,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        out_idx: List[int] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        pass_configs: dict = None,
        func: Callable = None,
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

        Returns:
            JITKernel: The loaded kernel if found, None otherwise.
        """

        if not os.path.exists(cache_path):
            return None

        kernel_global_source: Optional[str] = None
        kernel_params: Optional[List[KernelParam]] = None

        try:
            wrapped_kernel_path = os.path.join(cache_path, WRAPPED_KERNEL_PATH)
            with open(wrapped_kernel_path, "r") as f:
                kernel_global_source = f.read()
        except Exception as e:
            logger.error(f"Error loading wrapped kernel source code from disk: {e}")

        kernel_lib_path = os.path.join(cache_path, KERNEL_LIB_PATH)

        # Load kernel parameters
        try:
            params_path = os.path.join(cache_path, PARAMS_PATH)
            with open(params_path, "rb") as f:
                kernel_params = cloudpickle.load(f)
        except Exception as e:
            logger.error(f"Error loading kernel parameters from disk: {e}")

        if kernel_global_source and kernel_params:
            return JITKernel.from_database(
                func=func,
                kernel_global_source=kernel_global_source,
                kernel_lib_path=kernel_lib_path,
                params=kernel_params,
                target=target,
                target_host=target_host,
                out_idx=out_idx,
                execution_backend=execution_backend,
                pass_configs=pass_configs,
            )
        else:
            return None

    def save_to_disk(self, path: Path):
        if not os.path.exists(path):
            os.makedirs(path)

        # save best config
        with open(path / BEST_CONFIG_PATH, "w") as f:
            json.dump(self.config, f)

        # save function
        with open(path / FUNCTION_PATH, "wb") as f:
            cloudpickle.dump(self.func, f)

        # save ref latency
        with open(path / LATENCY_PATH, "w") as f:
            json.dump({
                "latency": self.latency,
                "ref_latency": self.ref_latency,
            }, f)

        # save kernel
        self._save_kernel_to_disk(path, self.kernel)

    @classmethod
    def load_from_disk(cls, path: Path, compile_args: CompileArgs) -> 'AutotuneResult':
        if not os.path.exists(path):
            return None

        # load best config
        with open(path / BEST_CONFIG_PATH, "r") as f:
            config = json.load(f)

        # load function
        with open(path / FUNCTION_PATH, "rb") as f:
            func = cloudpickle.load(f)

        # load latency
        with open(path / LATENCY_PATH, "r") as f:
            latency = json.load(f)
            latency, ref_latency = latency["latency"], latency["ref_latency"]

        kernel = cls._load_kernel_from_disk(cls, path, compile_args.target,
                                            compile_args.target_host, compile_args.out_idx,
                                            compile_args.execution_backend,
                                            compile_args.pass_configs, func)
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
