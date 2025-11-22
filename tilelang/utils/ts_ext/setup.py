from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import os
import torch

try:
    from torch.utils.cpp_extension import _get_torch_lib_dir
    torch_lib_dir = _get_torch_lib_dir()
except Exception:
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

include_dirs = []
if CUDA_HOME is not None:
    cuda_inc = os.path.join(CUDA_HOME, "include")
    if os.path.isdir(cuda_inc):
        include_dirs.append(cuda_inc)
    cuda_lib = os.path.join(CUDA_HOME, "lib64")
else:
    cuda_lib = None

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17", "-fPIC"],
    "nvcc": ["-O3", "-std=c++17", "-Xcompiler", "-fPIC"],
}

extra_link_args = [f"-Wl,-rpath,{torch_lib_dir}"]
runtime_library_dirs = [torch_lib_dir]
libraries = []
library_dirs = [torch_lib_dir]

if cuda_lib and os.path.isdir(cuda_lib):
    libraries.append("cudart")
    library_dirs.append(cuda_lib)
    extra_link_args.append(f"-Wl,-rpath,{cuda_lib}")

setup(
    name="tilescale_ext",
    packages=["tilescale_ext"],
    package_dir={"tilescale_ext": "."},
    ext_modules=[
        CUDAExtension(
            name="tilescale_ext._C",
            sources=[
                "ts_ext_bindings.cpp",
                "tensor.cpp",
                "ipc_ops.cpp",
            ],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            runtime_library_dirs=runtime_library_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
