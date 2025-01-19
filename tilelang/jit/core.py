# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import os
import re
from pathlib import Path
from typing import List, Union

import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock
from .env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH, TILELANG_JIT_DIR

class TileLangJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        # Add a StreamHandler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.addHandler(stream_handler)

    def info(self, msg):
        super().info("tilelang.jit: " + msg)


logger = TileLangJITLogger("tilelang.jit")


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        pass


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


remove_unwanted_pytorch_nvcc_flags()

sm90a_nvcc_flags = ["-gencode", "arch=compute_90a,code=sm_90a"]


def load_cuda_ops(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: List[str] = [],
    extra_cuda_cflags: List[str] = [],
    extra_ldflags=None,
    extra_include_paths=None,
    verbose=False,
):
    cflags = ["-O3", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
    ]
    cflags += extra_cflags
    cuda_cflags += extra_cuda_cflags
    logger.info(f"Loading JIT ops: {name}")
    check_cuda_arch()
    logger.info(f"check_cuda_arch passed")
    build_directory = TILELANG_JIT_DIR / name
    os.makedirs(build_directory, exist_ok=True)
    if extra_include_paths is None:
        extra_include_paths = [
            CUTLASS_INCLUDE_DIR,
            TILELANG_TEMPLATE_PATH,
        ]
    import time
    start = time.time()
    lock = FileLock(TILELANG_JIT_DIR / f"{name}.lock", thread_local=False)
    with lock:
        module = torch_cpp_ext.load(
            name,
            list(map(lambda _: str(_), sources)),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=list(map(lambda _: str(_), extra_include_paths)),
            build_directory=build_directory,
            verbose=verbose,
            with_cuda=True,
            keep_intermediates=False,
        )
    end = time.time()
    print(f"Time taken to load JIT ops: {end - start}")
    logger.info(f"Finished loading JIT ops: {name}")
    return module
