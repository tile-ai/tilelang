import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Union
import shutil
import sysconfig
import pybind11
import torch
import torch_npu
import functools
from ..engine import lower
from tilelang import tvm as tvm
from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll
from .kernel_npu import JitKernel_NPU
from tvm.tir import PrimFunc
from typing import (
    Any,
    List,
    Union,
)


class compiler_common:
    def __init__(self,
        out_idx: Union[List[int], int] = None,
    ):
      self.out_idx = out_idx

    def compile(self, mod: PrimFunc = None) -> JitKernel_NPU:
        self.metadata = {}
        self.mod = mod
        print(f"mod is in compiler_common: \n {mod}")
        # get grid message
        self._parse_grid()
        debug_enabled = os.environ.get("TILELANG_PRINT_COMMONIR", "0") in (
            "1",
            "true",
            "on",
        )

        instruments = [PrintAfterAll(), PrintBeforeAll()
                       ] if debug_enabled else []
        with tvm.transform.PassContext(instruments=instruments):
            mlir_path = lower(mod)
        self.mlir_content = mlir_path.kernel_source
        self.metadata["tl_params"] = mlir_path.params
        self.metadata["tl_out_idx"] = self.out_idx

        print(self.mlir_content)

        dump_ir = os.environ.get("DUMP_COMMON_IR", "0") == "1"
        if dump_ir:
            with tempfile.TemporaryDirectory() as tmpdir:
                dst_path = os.path.join(tmpdir, "kernel.commonir.mlir")
                print(dst_path)
                self._write_mlir_file(dst_path)
                if not os.path.exists("./tmp"):
                    os.makedirs("./tmp")
                shutil.copy(dst_path, "./tmp/kernel.commonir.mlir")

        self.constants = {}
        self.signature = self._parse_signature()
        print(f"self.signature is {self.signature}")

        from triton.backends.dicp_triton.commonir.compiler import (
            CommonIRCompiler,
            CommonIRSource,
        )

        commonir_compiler = CommonIRCompiler()
        source = CommonIRSource(
            self.mlir_content, self.metadata["grid"], self.signature
        )
        return JitKernel_NPU(commonir_compiler.compile(source), metadata=self.metadata)

    def _parse_grid(self):
        patterns = {
            "x": r'T\.launch_thread\("blockIdx\.x",\s*(\d+)\)',
            "y": r'T\.launch_thread\("blockIdx\.y",\s*(\d+)\)',
            "z": r'T\.launch_thread\("blockIdx\.z",\s*(\d+)\)'
        }
        block_indices = {"x": None, "y": None, "z": None}
        for dim, pattern in patterns.items():
            match = re.search(pattern, str(str(self.mod)))
            if match:
                block_indices[dim] = int(match.group(1))
        self.metadata['grid'] = [
            block_indices["x"] if block_indices["x"] is not None else 1,
            block_indices["y"] if block_indices["y"] is not None else 1,
            block_indices["z"] if block_indices["z"] is not None else 1
        ]

    def _write_mlir_file(self, file_path):
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(self.mlir_content)
            return True
        except FileNotFoundError:
            print(f"Error: Directory for '{file_path}' does not exist")
            return False
        except Exception as e:
            print(f"Error occurred while writing to the file: {e}")
            return False

    def _read_mlir_file(self, file_path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"Error: File '{file_path}' does not exist")
            return None
        except Exception as e:
            print(f"Error occurred while reading the file: {e}")
            return None

    def _parse_signature(self) -> dict:
        target_types = {
            "i1",
            "i8",
            "i16",
            "i32",
            "i64",
            "u32",
            "u64",
            "fp16",
            "bf16",
            "fp32",
            "f32",
            "fp64",
            "f16",
        }

        pattern = r"func\.func\s*@[^(]*\(([^)]*)\)"
        match = re.search(pattern, self.mlir_content)

        if not match:
            return {}

        params_str = match.group(1)

        params = []
        current_param = ""
        brace_count = 0
        angle_count = 0

        for char in params_str:
            if char == "," and brace_count == 0 and angle_count == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                elif char == "<":
                    angle_count += 1
                elif char == ">":
                    angle_count -= 1

        if current_param:
            params.append(current_param.strip())

        result = {}
        index = 0

        for param in params:
            if re.match(r"%args\d+", param.strip()):
                continue

            found_type = None
            for t_type in target_types:
                x_pattern = r"\bx" + t_type + r"\b"
                if re.search(x_pattern, param):
                    found_type = "*" + t_type
                    break
                elif re.search(r"\b" + t_type + r"\b", param):
                    found_type = t_type
                    break

            if found_type:
                if found_type == "f16":
                    found_type = "fp16"
                elif found_type == "*f16":
                    found_type = "*fp16"
                elif found_type == "f32":
                    found_type = "fp32"
                elif found_type == "*f32":
                    found_type = "*fp32"

                result[index] = found_type
                index += 1

        return result