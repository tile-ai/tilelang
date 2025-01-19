# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

import torch
from typing import List, Union
from .base import BaseKernelAdapter
from pathlib import Path
from tvm.relay import TensorType
from tilelang.jit.core import load_cuda_ops
from tilelang.jit.env import (TILELANG_JIT_WORKSPACE_DIR)


def torch_cpp_cuda_compile(code, target, verbose):
    # TODO(lei): This is not fully implemented yet
    # TODO(lei): extract name and magic number from module
    name: str = "matmul"
    magic_number = 0x9f
    full_kernel_dir = TILELANG_JIT_WORKSPACE_DIR / Path(f"{name}_{magic_number}")
    full_kernel_dir.mkdir(parents=True, exist_ok=True)

    sources: List[Union[str, Path]] = []

    tmp_cuda_kernel_file = (full_kernel_dir / "kernel.cu")

    code = (
        code + r"""
        void kenrel_interface(void* A, void *B, void *C, int64_t cuda_stream) {    
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
            main_kernel<<<dim3(4, 4, 1), dim3(128, 1, 1), 0, stream>>>((half_t *)A, (half_t *)B, (half_t *)C);
        }
    """)
    with open(tmp_cuda_kernel_file, "w") as f:
        f.write(code)

    print(tmp_cuda_kernel_file)

    sources.append(tmp_cuda_kernel_file)

    tmp_host_file = (full_kernel_dir / "host.cpp")

    host_code = r"""
        #include <torch/extension.h>
        #include <stdio.h>
        #include <ATen/ATen.h>
        
        void kenrel_interface(void* A, void *B, void *C, int64_t cuda_stream);

        int dispather(at::Tensor& A, at::Tensor& B, at::Tensor& C, int64_t cuda_stream) {    
            kenrel_interface(
                A.data_ptr(),
                B.data_ptr(),
                C.data_ptr(),
                cuda_stream
            );
            return 0;
        }

        int dispather(at::Tensor& A, at::Tensor& B, at::Tensor& C, int64_t cuda_stream);

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("matmul", &dispather, "matmul");
            printf("Registering matmul\n");
        }
    """
    with open(tmp_host_file, "w") as f:
        f.write(host_code)

    sources.append(tmp_host_file)
    module = load_cuda_ops(name=name, sources=sources, verbose=verbose)
    return module.matmul


class TorchCPPKernelAdapter(BaseKernelAdapter):

    target = "cuda"
    prim_func = None

    def __init__(self,
                 mod,
                 params: List[TensorType],
                 result_idx: List[int],
                 target,
                 prim_func,
                 verbose: bool = False):
        self.target = target
        self.prim_func = prim_func
        self.verbose = verbose
        super().__init__(mod, params, result_idx)

    def _convert_torch_func(self) -> callable:

        target = self.target
        verbose = self.verbose
        code = self.get_kernel_source()
        torch_module = torch_cpp_cuda_compile(code, target, verbose)

        # raise NotImplementedError("Please implement this function")

        def func(*ins: List[torch.Tensor]):
            if len(ins) + len(self.result_idx) != len(self.params):
                raise ValueError(
                    f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
                )
            ins_idx = 0
            args = []

            # use the device of the first input tensor if available
            device = ins[0].device if len(ins) > 0 else torch.cuda.current_device()

            for i in range(len(self.params)):
                if i in self.result_idx:
                    dtype = torch.__getattribute__(str(self.params[i].dtype))
                    shape = list(map(int, self.params[i].shape))
                    tensor = torch.empty(*shape, dtype=dtype, device=device)
                else:
                    tensor = ins[ins_idx]
                    ins_idx += 1
                args.append(tensor)

            torch_module(*args, 0)

            if len(self.result_idx) == 1:
                return args[self.result_idx[0]]
            else:
                return [args[i] for i in self.result_idx]

        return func
