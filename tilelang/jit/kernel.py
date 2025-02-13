# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Union, Any, Callable, Literal, Optional
from tvm.target import Target
import tilelang
from tilelang import tvm as tvm
from tvm.tir import PrimFunc

from tilelang.jit.adapter import TorchCPPKernelAdapter, TorchDLPackKernelAdapter, BaseKernelAdapter
from tilelang.utils.target import determine_target, AVALIABLE_TARGETS
from tilelang.profiler import Profiler, TensorSupplyType


class JITKernel(object):
    """
    A wrapper class for compiling and invoking TileLang (TVM TIR) functions as PyTorch-compatible functions.

    Attributes
    ----------
    rt_module : tvm.runtime.Module
        The runtime module compiled by TVM.
    rt_params : dict
        Parameters for the compiled runtime module (e.g., weights or constants).
    torch_function : Callable
        The compiled function that can be invoked as a PyTorch-compatible function.
    """
    rt_module: tvm.runtime.Module = None
    rt_params: dict = None
    adapter: BaseKernelAdapter = None
    torch_function: Callable = None

    def __init__(
        self,
        func: PrimFunc = None,
        out_idx: Union[List[int], int] = None,
        execution_backend: Literal["dlpack", "torch_cpp", "ctypes"] = "dlpack",
        target: Union[str, Target] = "auto",
        verbose: bool = False,
    ):
        """
        Initializes a TorchFunction instance.

        Parameters
        ----------
        func : tvm.tir.PrimFunc, optional
            The TileLang TIR function to compile and wrap.
        out_idx : Union[List[int], int], optional
            Index(es) of the output tensors to return (default: None).
        execution_backend : Literal["dlpack", "torch_cpp", "ctypes"], optional
            Execution backend to use for kernel execution (default: "dlpack").
        target : Union[str, Target], optional
            Compilation target, either as a string or a TVM Target object (default: "auto").
        verbose : bool, optional
            Whether to enable verbose output (default: False).
        """
        self.func = func
        self.out_idx = out_idx
        self.execution_backend = execution_backend
        self.target = target
        self.verbose = verbose

        # If the target is specified as a string, validate it and convert it to a TVM Target.
        if isinstance(target, str):
            assert target in AVALIABLE_TARGETS, f"Invalid target: {target}"
            target = determine_target(target)

        # Ensure the target is always a TVM Target object.
        target = Target(target)

        # Validate the execution backend.
        assert execution_backend in ["dlpack", "torch_cpp",
                                     "ctypes"], f"Invalid execution backend. {execution_backend}"

        # Compile the TileLang function and create a kernel adapter for execution.
        adapter = self._compile_and_create_adapter(func)

        # The adapter's function is assigned as the callable function for this instance.
        self.adapter = adapter
        self.torch_function = adapter.func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
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

    def _compile_and_create_adapter(self, tilelang_func: PrimFunc) -> BaseKernelAdapter:
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
        out_idx = self.out_idx
        execution_backend = self.execution_backend

        # Compile the function with TVM, optimizing with shared memory lowering.
        with tvm.transform.PassContext(opt_level=3):
            rt_mod, params = tilelang.lower(tilelang_func, target=target)

        # Store the runtime module and parameters for later use.
        self.rt_module = rt_mod
        self.rt_params = params

        # Create an adapter based on the specified execution backend.
        if execution_backend == "dlpack":
            # Use TorchDLPackKernelAdapter for interoperability with PyTorch via DLPack.
            adapter = TorchDLPackKernelAdapter(rt_mod, params=params, result_idx=out_idx)
        elif execution_backend == "torch_cpp":
            # Torch CPP backend adapter (not fully implemented yet).
            adapter = TorchCPPKernelAdapter(
                rt_mod,
                params=params,
                result_idx=out_idx,
                target=target,
                prim_func=tilelang_func,
                verbose=verbose,
            )
            raise NotImplementedError("Torch CPP backend is not fully implemented.")
        elif execution_backend == "ctypes":
            # CTYPES backend (not implemented yet).
            raise NotImplementedError("CTypes backend is not implemented.")
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

    def get_profiler(self,
                     tensor_supply_type: TensorSupplyType = TensorSupplyType.Integer) -> Profiler:
        """
        Creates a profiler to benchmark the compiled runtime module.

        Parameters
        ----------
        tensor_supply_type : TensorSupplyType, optional
            The type of input tensors to supply for profiling (default: TensorSupplyType.Integer).

        Returns
        -------
        Profiler
            A Profiler instance for benchmarking the runtime module.
        """
        return Profiler(self.rt_module, self.rt_params, self.out_idx, tensor_supply_type)

    def get_kernel_source(self) -> str:
        """
        Returns the source code of the compiled kernel function.

        Returns
        -------
        str
            The source code of the compiled kernel function.
        """
        return self.rt_module.imported_modules[0].get_source()

    def run_once(self, func: Optional[Callable] = None) -> None:
        return self.get_profiler().run_once(func)
